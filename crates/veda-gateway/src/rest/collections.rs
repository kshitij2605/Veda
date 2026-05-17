use std::collections::HashMap;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use veda_core::metrics as m;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::local_shard::LocalShardSearcher;
use veda_storage::segment_manager::SegmentManager;

use super::state::AppState;

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub fields: Vec<FieldDef>,
    #[serde(default = "default_shard_count")]
    pub shard_count: u32,
    #[serde(default = "default_replication_factor")]
    pub replication_factor: u32,
}

fn default_shard_count() -> u32 {
    1
}

fn default_replication_factor() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
pub struct FieldDef {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(default)]
    pub filterable: bool,
    pub dimension: Option<u32>,
    pub metric: Option<String>,
    /// Optional embedding-model identifier for vector fields. When set,
    /// embed-insert calls must use the same model.
    pub model: Option<String>,
    /// Maximum number of vectors per record for multi-vector fields.
    /// 0 / absent = unbounded.
    pub max_vectors: Option<u32>,
    /// Matryoshka truncation ladder. When set, per-dim HNSW indexes are
    /// built so search with `truncate_to=N` (for N in this list) hits a
    /// real ANN index instead of brute-force scanning.
    pub mrl_dims: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<CollectionSummary>,
}

#[derive(Debug, Serialize)]
pub struct CollectionSummary {
    pub name: String,
    pub field_count: usize,
    pub shard_count: u32,
}

#[derive(Debug, Serialize)]
pub struct DescribeCollectionResponse {
    pub name: String,
    pub fields: Vec<FieldInfo>,
    pub shard_count: u32,
    pub replication_factor: u32,
}

#[derive(Debug, Serialize)]
pub struct FieldInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub filterable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DropCollectionResponse {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ── Handlers ─────────────────────────────────────────────────────────────

pub async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<(StatusCode, Json<CreateCollectionResponse>), (StatusCode, Json<ErrorResponse>)> {
    // Validate name
    if req.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "collection name must not be empty".into(),
            }),
        ));
    }

    // Check for duplicate
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&req.name) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("collection '{}' already exists", req.name),
                }),
            ));
        }
    }

    // Build schema
    let mut builder = CollectionSchema::builder(&req.name)
        .shard_count(req.shard_count)
        .replication_factor(req.replication_factor);

    for f in &req.fields {
        let field_def = parse_field_def(f).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;
        builder = builder.add_field(field_def);
    }

    let schema = builder.build().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Create segment managers and query executor for all shards
    let collection_dir = state.inner.data_dir.join(&req.name);
    let mut shard_managers: HashMap<u32, SegmentManager> = HashMap::new();
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..schema.shard_count() {
        let sm = SegmentManager::new(schema.clone(), shard_id, &collection_dir, 512 * 1024 * 1024)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("failed to create segment manager: {e}"),
                    }),
                )
            })?;
        let searcher = LocalShardSearcher::new(
            SegmentManager::new(schema.clone(), shard_id, &collection_dir, 512 * 1024 * 1024)
                .unwrap(),
        );
        executor.register_shard(shard_id, Box::new(searcher));
        shard_managers.insert(shard_id, sm);
    }

    // Store everything
    {
        let mut schemas = state.inner.schemas.write().await;
        schemas.insert(req.name.clone(), schema);
    }
    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.insert(req.name.clone(), shard_managers);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.insert(req.name.clone(), executor);
    }

    // Update active collections gauge.
    let collection_count = {
        let schemas = state.inner.schemas.read().await;
        schemas.len()
    };
    metrics::gauge!(m::ACTIVE_COLLECTIONS).set(collection_count as f64);

    Ok((
        StatusCode::CREATED,
        Json(CreateCollectionResponse {
            name: req.name,
            status: "created".into(),
        }),
    ))
}

pub async fn list_collections(
    State(state): State<AppState>,
) -> Json<ListCollectionsResponse> {
    let schemas = state.inner.schemas.read().await;
    let collections = schemas
        .values()
        .map(|s| CollectionSummary {
            name: s.name().to_owned(),
            field_count: s.fields().len(),
            shard_count: s.shard_count(),
        })
        .collect();

    Json(ListCollectionsResponse { collections })
}

pub async fn describe_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<DescribeCollectionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let schemas = state.inner.schemas.read().await;
    let schema = schemas.get(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    let fields = schema
        .fields()
        .iter()
        .map(field_to_info)
        .collect();

    Ok(Json(DescribeCollectionResponse {
        name: schema.name().to_owned(),
        fields,
        shard_count: schema.shard_count(),
        replication_factor: schema.replication_factor(),
    }))
}

pub async fn drop_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<DropCollectionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let removed = {
        let mut schemas = state.inner.schemas.write().await;
        schemas.remove(&name)
    };

    if removed.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        ));
    }

    // Clean up segment managers and executors
    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.remove(&name);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.remove(&name);
    }

    // Update active collections gauge.
    let collection_count = {
        let schemas = state.inner.schemas.read().await;
        schemas.len()
    };
    metrics::gauge!(m::ACTIVE_COLLECTIONS).set(collection_count as f64);

    Ok(Json(DropCollectionResponse {
        name,
        status: "dropped".into(),
    }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn parse_field_def(f: &FieldDef) -> Result<FieldDefinition, String> {
    match f.field_type.as_str() {
        "vector" => {
            let dim = f
                .dimension
                .ok_or_else(|| format!("field '{}': vector type requires 'dimension'", f.name))?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "euclidean" => DistanceMetric::Euclidean,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                "manhattan" => DistanceMetric::Manhattan,
                other => return Err(format!("field '{}': unknown metric '{other}'", f.name)),
            };
            let mut fd = match &f.model {
                Some(m) if !m.is_empty() => {
                    FieldDefinition::vector_with_model(&f.name, dim, metric, m)
                }
                _ => FieldDefinition::vector(&f.name, dim, metric),
            };
            if let Some(dims) = &f.mrl_dims {
                fd = fd.with_mrl_dims(dims.clone());
            }
            Ok(fd.with_filterable(f.filterable))
        }
        "multi_vector" | "multivector" => {
            let dim = f.dimension.ok_or_else(|| {
                format!("field '{}': multi_vector type requires 'dimension'", f.name)
            })?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "euclidean" => DistanceMetric::Euclidean,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                other => {
                    return Err(format!(
                        "field '{}': multi_vector unsupported metric '{other}'",
                        f.name
                    ));
                }
            };
            // max_vectors is optional; 0 = unbounded.
            let max_vectors = f.max_vectors.unwrap_or(0);
            Ok(FieldDefinition::multi_vector(&f.name, dim, max_vectors, metric)
                .with_filterable(f.filterable))
        }
        "sparse_vector" | "sparsevector" => {
            let dim = f.dimension.ok_or_else(|| {
                format!("field '{}': sparse_vector type requires 'dimension'", f.name)
            })?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                other => {
                    return Err(format!(
                        "field '{}': sparse_vector supports only 'cosine' or 'dot_product', got '{other}'",
                        f.name
                    ));
                }
            };
            Ok(FieldDefinition::sparse_vector(&f.name, dim, metric).with_filterable(f.filterable))
        }
        "blob" => Ok(FieldDefinition::blob(&f.name).with_filterable(f.filterable)),
        "string" => Ok(
            FieldDefinition::new(&f.name, FieldType::String).with_filterable(f.filterable),
        ),
        "int32" => Ok(
            FieldDefinition::new(&f.name, FieldType::Int32).with_filterable(f.filterable),
        ),
        "int64" => Ok(
            FieldDefinition::new(&f.name, FieldType::Int64).with_filterable(f.filterable),
        ),
        "float32" => Ok(
            FieldDefinition::new(&f.name, FieldType::Float32).with_filterable(f.filterable),
        ),
        "float64" => Ok(
            FieldDefinition::new(&f.name, FieldType::Float64).with_filterable(f.filterable),
        ),
        "bool" => Ok(
            FieldDefinition::new(&f.name, FieldType::Bool).with_filterable(f.filterable),
        ),
        "timestamp" => Ok(
            FieldDefinition::new(&f.name, FieldType::Timestamp).with_filterable(f.filterable),
        ),
        "json" => Ok(
            FieldDefinition::new(&f.name, FieldType::Json).with_filterable(f.filterable),
        ),
        "uuid" => Ok(
            FieldDefinition::new(&f.name, FieldType::Uuid).with_filterable(f.filterable),
        ),
        "array_string" | "stringarray" | "string_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::String)),
        )
        .with_filterable(f.filterable)),
        "array_int32" | "int32_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::Int32)),
        )
        .with_filterable(f.filterable)),
        "array_int64" | "int64_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::Int64)),
        )
        .with_filterable(f.filterable)),
        other => Err(format!("field '{}': unknown type '{other}'", f.name)),
    }
}

fn field_to_info(f: &FieldDefinition) -> FieldInfo {
    let (type_str, dimension, metric, model) = match f.field_type() {
        FieldType::Vector { dimension, metric, model_id, .. } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::Euclidean => "euclidean",
                DistanceMetric::DotProduct => "dot_product",
                DistanceMetric::Manhattan => "manhattan",
                DistanceMetric::Hamming => "hamming",
                DistanceMetric::Jaccard => "jaccard",
            };
            ("vector", Some(*dimension), Some(m.to_owned()), model_id.clone())
        }
        FieldType::Blob => ("blob", None, None, None),
        FieldType::String => ("string", None, None, None),
        FieldType::Int32 => ("int32", None, None, None),
        FieldType::Int64 => ("int64", None, None, None),
        FieldType::Float32 => ("float32", None, None, None),
        FieldType::Float64 => ("float64", None, None, None),
        FieldType::Bool => ("bool", None, None, None),
        FieldType::Timestamp => ("timestamp", None, None, None),
        FieldType::Json => ("json", None, None, None),
        FieldType::Uuid => ("uuid", None, None, None),
        FieldType::SparseVector { dimension, metric } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::DotProduct => "dot_product",
                _ => "cosine",
            };
            ("sparse_vector", Some(*dimension), Some(m.to_owned()), None)
        }
        FieldType::MultiVector { dimension, metric, .. } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::Euclidean => "euclidean",
                DistanceMetric::DotProduct => "dot_product",
                _ => "cosine",
            };
            ("multi_vector", Some(*dimension), Some(m.to_owned()), None)
        }
        FieldType::Array(_) => ("array", None, None, None),
    };

    FieldInfo {
        name: f.name().to_owned(),
        field_type: type_str.to_owned(),
        filterable: f.filterable(),
        dimension,
        metric,
        model,
    }
}
