use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use veda_core::record::{FieldValue, Record};
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::local_shard::LocalShardSearcher;

use super::collections::ErrorResponse;
use super::state::AppState;

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InsertRequest {
    pub records: Vec<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct InsertResponse {
    pub inserted: usize,
}

#[derive(Debug, Serialize)]
pub struct GetRecordResponse {
    pub id: String,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteRecordsRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteRecordsResponse {
    pub deleted: usize,
}

// ── Handlers ─────────────────────────────────────────────────────────────

pub async fn insert(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            )
        })?
    };

    let mut inserted = 0usize;
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers.get_mut(&name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "no segment managers for collection".into(),
            }),
        )
    })?;

    let shard_count = schema.shard_count();

    for raw in &req.records {
        let record = parse_record(raw, &schema).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;

        // Simple hash-based shard routing
        let shard_id = record.id().as_u128() as u32 % shard_count;
        let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("shard {shard_id} not found"),
                }),
            )
        })?;

        sm.insert(record).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("insert failed: {e}"),
                }),
            )
        })?;

        inserted += 1;
    }

    // Seal growing segments so data is on disk, then rebuild search executors
    for sm in shard_map.values_mut() {
        sm.seal_growing().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to seal segment: {e}"),
                }),
            )
        })?;
    }

    drop(managers);
    rebuild_executors(&state, &name).await?;

    Ok((StatusCode::CREATED, Json(InsertResponse { inserted })))
}

pub async fn get_record(
    State(state): State<AppState>,
    Path((name, id_str)): Path<(String, String)>,
) -> Result<Json<GetRecordResponse>, (StatusCode, Json<ErrorResponse>)> {
    let record_id: Uuid = id_str.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid UUID: '{id_str}'"),
            }),
        )
    })?;

    // Check collection exists
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    // Search all shards for the record
    for sm in shard_map.values() {
        // Check growing segment
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if let Some(rec) = gs.get_record(doc_id)
                && rec.id() == record_id
                && !gs.is_deleted(doc_id)
            {
                return Ok(Json(record_to_response(rec)));
            }
        }

        // Check sealed segments
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id)
                    && rec.id() == record_id
                {
                    return Ok(Json(record_to_response(&rec)));
                }
            }
        }
    }

    Err((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: format!("record '{record_id}' not found"),
        }),
    ))
}

pub async fn delete_records(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteRecordsRequest>,
) -> Result<Json<DeleteRecordsResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Check collection exists
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let mut deleted = 0usize;
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers.get_mut(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    for id_str in &req.ids {
        let record_id: Uuid = id_str.parse().map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("invalid UUID: '{id_str}'"),
                }),
            )
        })?;

        for sm in shard_map.values_mut() {
            if sm.delete_record(record_id).unwrap_or(false) {
                deleted += 1;
                break;
            }
        }
    }

    Ok(Json(DeleteRecordsResponse { deleted }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn parse_record(
    raw: &serde_json::Map<String, serde_json::Value>,
    schema: &veda_core::schema::CollectionSchema,
) -> Result<Record, String> {
    let mut record = Record::new();

    for field_def in schema.fields() {
        let name = field_def.name();
        let value = match raw.get(name) {
            Some(v) => v,
            None => {
                // Vector fields are required
                if matches!(field_def.field_type(), FieldType::Vector { .. }) {
                    return Err(format!("missing required vector field '{name}'"));
                }
                continue;
            }
        };

        let fv = json_to_field_value(value, field_def.field_type())
            .map_err(|e| format!("field '{name}': {e}"))?;
        record.set(name, fv);
    }

    Ok(record)
}

fn json_to_field_value(
    value: &serde_json::Value,
    field_type: &FieldType,
) -> Result<FieldValue, String> {
    match field_type {
        FieldType::Vector { .. } => {
            let arr = value
                .as_array()
                .ok_or("expected array for vector field")?;
            let floats: Vec<f32> = arr
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or("vector elements must be numbers")
                })
                .collect::<Result<_, _>>()?;
            Ok(FieldValue::Vector(DenseVector::new(floats)))
        }
        FieldType::String => {
            let s = value.as_str().ok_or("expected string")?;
            Ok(FieldValue::String(s.to_owned()))
        }
        FieldType::Int32 => {
            let n = value.as_i64().ok_or("expected integer")?;
            Ok(FieldValue::Int32(n as i32))
        }
        FieldType::Int64 => {
            let n = value.as_i64().ok_or("expected integer")?;
            Ok(FieldValue::Int64(n))
        }
        FieldType::Float32 => {
            let n = value.as_f64().ok_or("expected number")?;
            Ok(FieldValue::Float32(n as f32))
        }
        FieldType::Float64 => {
            let n = value.as_f64().ok_or("expected number")?;
            Ok(FieldValue::Float64(n))
        }
        FieldType::Bool => {
            let b = value.as_bool().ok_or("expected boolean")?;
            Ok(FieldValue::Bool(b))
        }
        FieldType::Timestamp => {
            let n = value.as_i64().ok_or("expected integer timestamp")?;
            Ok(FieldValue::Timestamp(n))
        }
        FieldType::Blob => {
            let s = value.as_str().ok_or("expected string for blob ref")?;
            Ok(FieldValue::BlobRef(s.to_owned()))
        }
        FieldType::Json => Ok(FieldValue::Json(value.clone())),
        FieldType::Uuid => {
            let s = value.as_str().ok_or("expected string for uuid")?;
            let u: uuid::Uuid = s.parse().map_err(|_| "invalid UUID")?;
            Ok(FieldValue::Uuid(u))
        }
        _ => Err(format!("unsupported field type: {field_type:?}")),
    }
}

fn record_to_response(rec: &Record) -> GetRecordResponse {
    let mut fields = serde_json::Map::new();
    for (key, value) in rec.fields() {
        fields.insert(key.clone(), field_value_to_json(value));
    }

    GetRecordResponse {
        id: rec.id().to_string(),
        fields,
    }
}

fn field_value_to_json(fv: &FieldValue) -> serde_json::Value {
    match fv {
        FieldValue::String(s) => serde_json::Value::String(s.clone()),
        FieldValue::Int32(n) => serde_json::json!(*n),
        FieldValue::Int64(n) => serde_json::json!(*n),
        FieldValue::Float32(n) => serde_json::json!(*n),
        FieldValue::Float64(n) => serde_json::json!(*n),
        FieldValue::Bool(b) => serde_json::Value::Bool(*b),
        FieldValue::Timestamp(t) => serde_json::json!(*t),
        FieldValue::Vector(v) => serde_json::json!(v.as_slice()),
        FieldValue::Json(j) => j.clone(),
        FieldValue::BlobRef(s) => serde_json::Value::String(s.clone()),
        FieldValue::Uuid(u) => serde_json::Value::String(u.to_string()),
        FieldValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(field_value_to_json).collect())
        }
        FieldValue::Null => serde_json::Value::Null,
    }
}

/// Rebuild the DistributedQueryExecutor for a collection after data changes.
///
/// Creates new LocalShardSearcher instances that load sealed segments from
/// disk, then builds vector indexes so search can find the data.
async fn rebuild_executors(
    state: &AppState,
    collection_name: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(collection_name).cloned().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "schema not found during rebuild".into(),
                }),
            )
        })?
    };

    let collection_dir = state.inner.data_dir.join(collection_name);
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..schema.shard_count() {
        let mut sm = veda_storage::segment_manager::SegmentManager::new(
            schema.clone(),
            shard_id,
            &collection_dir,
            512 * 1024 * 1024,
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to recreate segment manager: {e}"),
                }),
            )
        })?;

        // Load sealed segments from disk
        sm.load_sealed_segments().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to load sealed segments: {e}"),
                }),
            )
        })?;

        let mut searcher = LocalShardSearcher::new(sm);
        searcher.build_indexes().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to build indexes: {e}"),
                }),
            )
        })?;
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let mut executors = state.inner.query_executors.write().await;
    executors.insert(collection_name.to_owned(), executor);

    Ok(())
}
