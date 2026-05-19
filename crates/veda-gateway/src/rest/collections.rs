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
    /// Default physical layout for the stored-fields data of new
    /// sealed segments. `"jsonl"` (default — legacy + small records)
    /// or `"columnar"` (per-field column files, projection-friendly).
    /// Per-segment metadata still records the format that was
    /// actually written, so this can be changed later for new
    /// segments without rewriting existing ones.
    #[serde(default)]
    pub stored_format: Option<String>,
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
    /// Per-field auto-embed config. JSON form:
    /// `{"source": "content", "modality": "text"}`. When present, the
    /// process-document handler uses this to pick the source field and
    /// embedder for this vector field instead of the name-based
    /// heuristic.
    pub auto_embed: Option<veda_core::types::AutoEmbedConfig>,
    /// Build a full-text BM25 index for this field. Only meaningful for
    /// `string` fields. When true, the schema exposes a `bm25` query
    /// type and the `hybrid` (dense + BM25) query type.
    #[serde(default)]
    pub bm25: bool,
    /// Tokenizer for BM25. Recognized: `"cjk_bigram"` (default, good
    /// for mixed-script and CJK), `"whitespace"` (cheaper for
    /// ASCII-only). Unknown names fall back to the default.
    #[serde(default)]
    pub bm25_tokenizer: Option<String>,
    /// Vector index type. Recognized: `"hnsw"` (default for vector
    /// fields), `"diskann"`, `"ivfpq"`, `"flat"`, `"rabitq"` (1-bit
    /// quantized brute-force scan; ~32× memory savings at the cost of
    /// some recall, well-suited for ≥ 1 M-row collections). Unknown
    /// values fall back to `hnsw`.
    #[serde(default)]
    pub index: Option<String>,
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

    // Check for duplicate — in-memory state OR an existing on-disk
    // collection directory with our schema marker. The on-disk check
    // closes Gap D: previously, after a Veda restart with empty
    // in-memory state, POST /v1/collections silently orphaned the
    // existing sealed segments under data_dir/{name}/ by overwriting
    // the segment-manager view of the world.
    let collection_dir = state.inner.data_dir.join(&req.name);
    let schema_marker = collection_dir.join("collection.json");
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&req.name) || schema_marker.exists() {
            return Err((
                StatusCode::CONFLICT,
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
    if let Some(fmt) = req.stored_format.as_deref() {
        match fmt.to_lowercase().as_str() {
            "jsonl" | "columnar" => builder = builder.stored_format(fmt.to_lowercase()),
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!(
                            "stored_format must be \"jsonl\" or \"columnar\", got {other:?}"
                        ),
                    }),
                ));
            }
        }
    }

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

    // Persist the schema at the collection root so a restart can recover
    // both the in-memory state and the on-disk segments under each shard.
    // This is the source of truth for `recover_collections_from_disk`.
    std::fs::create_dir_all(&collection_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("failed to create collection dir: {e}"),
            }),
        )
    })?;
    let schema_json = serde_json::to_vec_pretty(&schema).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("serialize schema: {e}"),
            }),
        )
    })?;
    std::fs::write(&schema_marker, &schema_json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("persist schema: {e}"),
            }),
        )
    })?;

    // Raft mode: route the creation through the consensus log
    // BEFORE materializing locally. coord.execute returns once
    // the entry is committed AND applied; on every node the
    // ClusterStateMachine then carries the new collection.
    // After that, this node materializes the local Veda objects
    // (segment managers, executor). Followers reconcile via the
    // background task spawned in start_server.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cmd = veda_coordinator::raft_types::ClusterCommand::CreateCollection {
                name: req.name.clone(),
                schema_json: String::from_utf8(schema_json.clone()).unwrap_or_default(),
            };
            if let Err(e) = coord.execute(cmd).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("raft submit: {e}"),
                    }),
                ));
            }
            // Fall through to local materialization — same code
            // path the non-Raft mode uses, and the same code path
            // the periodic reconcile task uses on followers.
        }
    }

    if let Err((status, e)) = materialize_collection_locally(&state, &schema, &collection_dir).await
    {
        return Err((status, e));
    }

    Ok((
        StatusCode::CREATED,
        Json(CreateCollectionResponse {
            name: req.name,
            status: "created".into(),
        }),
    ))
}

/// Wire a parsed `CollectionSchema` into the local in-process
/// state: segment managers + query executor per shard, then insert
/// into `state.inner.schemas`. Idempotent — a collection already
/// present locally is left alone.
///
/// This is the shared code path between:
///   - direct REST creation (non-Raft mode)
///   - REST creation after a Raft commit (leader)
///   - the background reconcile task picking up entries from the
///     Raft state machine (followers)
pub(crate) async fn materialize_collection_locally(
    state: &AppState,
    schema: &CollectionSchema,
    collection_dir: &std::path::Path,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let name = schema.name().to_owned();
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&name) {
            return Ok(()); // already materialized
        }
    }
    let mut shard_managers: HashMap<u32, SegmentManager> = HashMap::new();
    let mut executor = DistributedQueryExecutor::new();
    for shard_id in 0..schema.shard_count() {
        let sm =
            SegmentManager::new(schema.clone(), shard_id, collection_dir, 512 * 1024 * 1024)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("failed to create segment manager: {e}"),
                        }),
                    )
                })?;
        let searcher = LocalShardSearcher::new(
            SegmentManager::new(schema.clone(), shard_id, collection_dir, 512 * 1024 * 1024)
                .unwrap(),
        );
        executor.register_shard(shard_id, Box::new(searcher));
        shard_managers.insert(shard_id, sm);
    }
    {
        let mut schemas = state.inner.schemas.write().await;
        schemas.insert(name.clone(), schema.clone());
    }
    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.insert(name.clone(), shard_managers);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.insert(name.clone(), executor);
    }
    let collection_count = state.inner.schemas.read().await.len();
    metrics::gauge!(m::ACTIVE_COLLECTIONS).set(collection_count as f64);
    Ok(())
}

/// Reconcile the local Veda state with the Raft state machine's
/// view. Materializes any collection that exists in Raft state
/// but not locally; drops any local collection no longer in Raft
/// state. Called by the periodic reconcile task in raft mode and
/// also (synchronously, fall-through) by the create/drop handlers
/// so the leader's response reflects the new state.
pub async fn reconcile_collections_from_raft(
    state: &AppState,
) -> Result<(), String> {
    let raft_state = {
        let guard = state.inner.raft.read().await;
        match guard.as_ref() {
            Some(c) => c.query_state().await.map_err(|e| format!("{e}"))?,
            None => return Ok(()),
        }
    };

    let local_names: std::collections::HashSet<String> = {
        let schemas = state.inner.schemas.read().await;
        schemas.keys().cloned().collect()
    };
    let raft_names: std::collections::HashSet<String> =
        raft_state.collections.keys().cloned().collect();

    // Materialize: in Raft, not in local.
    for name in raft_names.difference(&local_names) {
        let Some(schema_json) = raft_state.collections.get(name) else {
            continue;
        };
        let schema: CollectionSchema = match serde_json::from_str(schema_json) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    error = %e,
                    "reconcile: schema JSON parse failed; skipping",
                );
                continue;
            }
        };
        let collection_dir = state.inner.data_dir.join(name);
        // Persist the schema marker so a restart can recover the
        // collection from disk independently of the Raft log.
        let _ = std::fs::create_dir_all(&collection_dir);
        let _ = std::fs::write(
            collection_dir.join("collection.json"),
            schema_json.as_bytes(),
        );
        if let Err(e) = materialize_collection_locally(state, &schema, &collection_dir).await {
            tracing::warn!(
                collection = %name,
                error = ?e.1.0.error,
                "reconcile: materialize failed",
            );
        } else {
            tracing::info!(collection = %name, "reconcile: materialized from raft state");
        }
    }

    // Drop: in local, not in Raft.
    for name in local_names.difference(&raft_names) {
        tracing::info!(collection = %name, "reconcile: dropping (no longer in raft state)");
        let _ = state.inner.schemas.write().await.remove(name);
        let _ = state.inner.segment_managers.write().await.remove(name);
        let _ = state.inner.query_executors.write().await.remove(name);
    }
    Ok(())
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
    // Raft mode: submit the drop to the consensus log first.
    // After the entry commits + applies on every node, the local
    // reconcile (here or in the periodic task) tears down
    // segment managers + executors.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cmd = veda_coordinator::raft_types::ClusterCommand::DropCollection {
                name: name.clone(),
            };
            if let Err(e) = coord.execute(cmd).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("raft submit: {e}"),
                    }),
                ));
            }
        }
    }

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

    // Remove the on-disk collection dir so a subsequent POST with the
    // same name doesn't trip the dedup guard and so disk space is
    // actually reclaimed.
    let collection_dir = state.inner.data_dir.join(&name);
    if collection_dir.exists() {
        let _ = std::fs::remove_dir_all(&collection_dir);
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
            if let Some(cfg) = &f.auto_embed {
                fd = fd.with_auto_embed(cfg.clone());
            }
            if let Some(idx) = f.index.as_deref() {
                let idx_type = match idx.to_lowercase().as_str() {
                    "hnsw" => veda_core::types::IndexType::Hnsw,
                    "diskann" => veda_core::types::IndexType::DiskAnn,
                    "ivfpq" | "ivf_pq" => veda_core::types::IndexType::IvfPq,
                    "flat" => veda_core::types::IndexType::Flat,
                    "rabitq" | "rabit_q" => veda_core::types::IndexType::RaBitQ,
                    "rabitq_rerank" | "rabit_q_rerank" => {
                        veda_core::types::IndexType::RaBitQRerank
                    }
                    "hnsw_rabitq" | "hnsw_rabit_q" | "quantized_hnsw" => {
                        veda_core::types::IndexType::HnswRaBitQ
                    }
                    other => {
                        return Err(format!(
                            "field '{}': unknown vector index type '{other}'",
                            f.name
                        ));
                    }
                };
                fd = fd.with_index_type(idx_type);
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
            let mut fd = FieldDefinition::sparse_vector(&f.name, dim, metric);
            if let Some(cfg) = &f.auto_embed {
                fd = fd.with_sparse_auto_embed(cfg.clone());
            }
            Ok(fd.with_filterable(f.filterable))
        }
        "blob" => Ok(FieldDefinition::blob(&f.name).with_filterable(f.filterable)),
        "string" => Ok(
            FieldDefinition::new(&f.name, FieldType::String)
                .with_filterable(f.filterable)
                .with_bm25(f.bm25)
                .with_bm25_tokenizer(f.bm25_tokenizer.clone()),
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
        FieldType::SparseVector { dimension, metric, .. } => {
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
