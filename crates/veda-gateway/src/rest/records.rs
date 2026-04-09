use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;
use veda_core::metrics as m;
use veda_core::record::{FieldValue, Record};
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::local_shard::LocalShardSearcher;

use veda_storage::blob_manager::BlobManager;
use veda_storage::blob_store::ArcBlobStore;

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

#[instrument(name = "insert", skip(state, req), fields(collection = %name, record_count = req.records.len()))]
pub async fn insert(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
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

    // Get blob store for processing blob fields
    let blob_store = state.inner.blob_store.read().await.clone();

    for raw in &req.records {
        let mut record = parse_record(raw, &schema).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;

        if !schema.blob_fields().is_empty() {
            let blob_manager = BlobManager::new(Box::new(ArcBlobStore(blob_store.clone())), &name);
            blob_manager.process_insert(&mut record, &schema).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("blob processing failed: {e}"),
                    }),
                )
            })?;
        }

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
        let seal_start = Instant::now();
        sm.seal_growing().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to seal segment: {e}"),
                }),
            )
        })?;
        metrics::histogram!(m::SEAL_DURATION, "collection" => name.clone())
            .record(seal_start.elapsed().as_secs_f64());
    }

    // Run compaction if needed.
    {
        let config = veda_storage::compaction::CompactionConfig::default();
        for sm in shard_map.values_mut() {
            metrics::gauge!(m::COMPACTION_IN_PROGRESS, "collection" => name.clone()).set(1.0);
            let compact_start = Instant::now();
            match sm.compact_if_needed(&config) {
                Ok(removed) if removed > 0 => {
                    metrics::histogram!(m::COMPACTION_DURATION, "collection" => name.clone())
                        .record(compact_start.elapsed().as_secs_f64());
                    tracing::info!(removed_segments = removed, "compaction completed");
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "compaction failed");
                }
            }
            metrics::gauge!(m::COMPACTION_IN_PROGRESS, "collection" => name.clone()).set(0.0);
        }
        // Update segment count after compaction.
        let total_segments: usize = shard_map.values().map(|sm| sm.sealed_segments().len()).sum();
        metrics::gauge!(m::SEGMENTS_TOTAL, "collection" => name.clone()).set(total_segments as f64);
    }

    drop(managers);
    metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(1.0);
    let rebuild_start = Instant::now();
    rebuild_executors(&state, &name).await?;
    metrics::histogram!(m::INDEX_BUILD_DURATION, "collection" => name.clone(), "index_type" => "rebuild")
        .record(rebuild_start.elapsed().as_secs_f64());
    metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(0.0);

    metrics::histogram!(m::INSERT_DURATION, "collection" => name.clone())
        .record(start.elapsed().as_secs_f64());
    metrics::counter!(m::INSERT_TOTAL, "collection" => name.clone())
        .increment(inserted as u64);

    // Update record count gauge from segment managers.
    {
        let managers = state.inner.segment_managers.read().await;
        if let Some(shard_managers) = managers.get(&name) {
            let total: u64 = shard_managers.values().map(|sm| sm.total_records()).sum();
            metrics::gauge!(m::RECORDS_TOTAL, "collection" => name.clone()).set(total as f64);
        }
    }

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
            // User sends base64 data as a string; store as String so BlobManager
            // can detect it, decode it, store the bytes, and replace with BlobRef.
            let s = value.as_str().ok_or("expected base64 string for blob field")?;
            Ok(FieldValue::String(s.to_owned()))
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
#[instrument(name = "rebuild_executors", skip(state), fields(collection = %collection_name))]
pub async fn rebuild_executors(
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
