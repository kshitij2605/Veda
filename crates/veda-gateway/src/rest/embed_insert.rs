use std::collections::HashMap;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::metrics as m;
use veda_core::record::{FieldValue, Record};
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;

use super::collections::ErrorResponse;
use super::records::rebuild_executors;
use super::state::AppState;
use crate::tenant::{self, Tenant};

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbedInsertRequest {
    pub records: Vec<serde_json::Map<String, serde_json::Value>>,
    pub auto_embed: HashMap<String, AutoEmbedMapping>,
}

#[derive(Debug, Deserialize)]
pub struct AutoEmbedMapping {
    pub source_field: String,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedInsertResponse {
    pub inserted: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<RecordError>,
}

#[derive(Debug, Serialize)]
pub struct RecordError {
    pub index: usize,
    pub error: String,
}

// ── Handler ──────────────────────────────────────────────────────────────

#[instrument(name = "embed_insert", skip(state, req), fields(collection = %name, record_count = req.records.len()))]
pub async fn embed_insert(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<EmbedInsertRequest>,
) -> Result<(StatusCode, Json<EmbedInsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
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

    let registry = state.inner.embedding_registry.read().await;

    // Two-pass model resolution:
    // Pass 1: detect if ANY auto-select mapping targets a non-text modality.
    // If so, use the multimodal model for ALL auto-selected mappings (unified vector space).
    let has_multimodal = req.auto_embed.values().any(|m| {
        m.model.is_none()
            && !matches!(
                Modality::from_field_name(&m.source_field).unwrap_or(Modality::Text),
                Modality::Text
            )
    });

    // Pass 2: resolve model names.
    let mut resolved_models: HashMap<String, String> = HashMap::new();
    for (target_field, mapping) in &req.auto_embed {
        let modality = Modality::from_field_name(&mapping.source_field).unwrap_or(Modality::Text);

        let model_name = match &mapping.model {
            // Explicit override — always respected.
            Some(name) => name.clone(),
            None => {
                // If multimodal data is present, use multimodal model for everything
                // so all embeddings share the same vector space.
                let lookup_modality = if has_multimodal { Modality::Image } else { modality };
                let candidates = registry.find_by_modality(lookup_modality);
                if candidates.is_empty() {
                    let available: Vec<&str> = registry.list().iter().map(|c| c.name.as_str()).collect();
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!(
                                "no model registered for modality '{lookup_modality}'. Available models: {available:?}. Register one via EMBEDDING_* or MULTIMODAL_EMBEDDING_* env vars"
                            ),
                        }),
                    ));
                }
                candidates[0].name.clone()
            }
        };

        // Validate model exists.
        let model = registry.get(&model_name).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("model '{}' not found in registry", model_name),
                }),
            )
        })?;

        // Validate model supports the modality.
        registry.validate_modality(&model_name, modality).map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

        // Validate target field is a vector field.
        let target_def = schema.get_field(target_field).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("target field '{target_field}' not found in schema"),
                }),
            )
        })?;
        match target_def.field_type() {
            FieldType::Vector { dimension, model_id, .. } => {
                if model.output_dimension != *dimension {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!(
                                "model '{}' output dimension {} does not match field '{}' dimension {}",
                                model_name, model.output_dimension, target_field, dimension
                            ),
                        }),
                    ));
                }
                if let Some(declared) = model_id
                    && declared != &model_name
                {
                    return Err((
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(ErrorResponse {
                            error: format!(
                                "field '{target_field}' is bound to model '{declared}' \
                                but request would use '{model_name}'. Mixing models on the \
                                same vector field corrupts retrieval — create a new field \
                                or drop+re-embed the collection."
                            ),
                        }),
                    ));
                }
            }
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("target field '{target_field}' is not a vector field"),
                    }),
                ));
            }
        }

        resolved_models.insert(target_field.clone(), model_name);
    }

    // Process each record: generate embeddings, then insert.
    let mut inserted = 0usize;
    let mut errors = Vec::new();

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

    for (idx, raw) in req.records.iter().enumerate() {
        // Clone the raw map so we can inject vector fields.
        let mut augmented = raw.clone();

        // Generate embeddings for each auto_embed mapping.
        let mut embed_ok = true;
        for (target_field, mapping) in &req.auto_embed {
            let source_value = match raw.get(&mapping.source_field) {
                Some(v) => v,
                None => {
                    errors.push(RecordError {
                        index: idx,
                        error: format!("source field '{}' not present in record", mapping.source_field),
                    });
                    embed_ok = false;
                    break;
                }
            };

            let model_name = resolved_models.get(target_field).unwrap();
            let model_config = registry.get(model_name).unwrap();
            let client = EmbeddingClient::new(model_config.clone());
            let modality = Modality::from_field_name(&mapping.source_field).unwrap_or(Modality::Text);

            let vector = match modality {
                Modality::Text => {
                    let text = source_value.as_str().unwrap_or("");
                    match client.embed_text(text) {
                        Ok(v) => v,
                        Err(e) => {
                            errors.push(RecordError {
                                index: idx,
                                error: format!("embedding failed: {e}"),
                            });
                            embed_ok = false;
                            break;
                        }
                    }
                }
                _ => {
                    // For non-text modalities, expect base64-encoded data.
                    let b64 = source_value.as_str().unwrap_or("");
                    let bytes = base64_decode(b64).map_err(|e| {
                        (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: format!("invalid base64 in field '{}': {e}", mapping.source_field),
                            }),
                        )
                    })?;
                    match client.embed_bytes(&bytes, modality) {
                        Ok(v) => v,
                        Err(e) => {
                            errors.push(RecordError {
                                index: idx,
                                error: format!("embedding failed: {e}"),
                            });
                            embed_ok = false;
                            break;
                        }
                    }
                }
            };

            // Inject the vector into the augmented record.
            augmented.insert(
                target_field.clone(),
                serde_json::Value::Array(vector.iter().map(|f| serde_json::json!(*f)).collect()),
            );
        }

        if !embed_ok {
            continue;
        }

        // Parse the augmented record using existing logic.
        let record = match parse_record_for_embed(&augmented, &schema) {
            Ok(r) => r,
            Err(e) => {
                errors.push(RecordError {
                    index: idx,
                    error: e,
                });
                continue;
            }
        };

        let shard_id =
            veda_coordinator::shard_router::ShardRouter::assign_shard(&record.id(), shard_count);
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

    // Seal growing segments. Track whether any shard actually sealed.
    let mut any_sealed = false;
    for sm in shard_map.values_mut() {
        let seal_start = std::time::Instant::now();
        let sealed = sm.seal_growing().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to seal segment: {e}"),
                }),
            )
        })?;
        if sealed.is_some() {
            any_sealed = true;
            metrics::histogram!(m::SEAL_DURATION, "collection" => name.clone())
                .record(seal_start.elapsed().as_secs_f64());
        }
    }

    if any_sealed {
        let config = veda_storage::compaction::CompactionConfig::default();
        for sm in shard_map.values_mut() {
            metrics::gauge!(m::COMPACTION_IN_PROGRESS, "collection" => name.clone()).set(1.0);
            let compact_start = std::time::Instant::now();
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
        let total_segments: usize = shard_map.values().map(|sm| sm.sealed_segments().len()).sum();
        metrics::gauge!(m::SEGMENTS_TOTAL, "collection" => name.clone()).set(total_segments as f64);
    }

    drop(managers);
    drop(registry);

    if any_sealed {
        metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(1.0);
        let rebuild_start = std::time::Instant::now();
        rebuild_executors(&state, &name).await?;
        metrics::histogram!(m::INDEX_BUILD_DURATION, "collection" => name.clone(), "index_type" => "rebuild")
            .record(rebuild_start.elapsed().as_secs_f64());
        metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(0.0);
    }

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

    Ok((
        StatusCode::CREATED,
        Json(EmbedInsertResponse { inserted, errors }),
    ))
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Parse a JSON map into a Record, tolerating missing non-vector fields.
fn parse_record_for_embed(
    raw: &serde_json::Map<String, serde_json::Value>,
    schema: &veda_core::schema::CollectionSchema,
) -> Result<Record, String> {
    let mut record = Record::new();

    for field_def in schema.fields() {
        let name = field_def.name();
        let value = match raw.get(name) {
            Some(v) => v,
            None => {
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
            let arr = value.as_array().ok_or("expected array for vector field")?;
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

/// Simple base64 decoder (standard alphabet, with optional padding).
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const TABLE: [u8; 256] = {
        let mut t = [255u8; 256];
        let mut i = 0u8;
        while i < 26 {
            t[(b'A' + i) as usize] = i;
            t[(b'a' + i) as usize] = i + 26;
            i += 1;
        }
        let mut d = 0u8;
        while d < 10 {
            t[(b'0' + d) as usize] = d + 52;
            d += 1;
        }
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        t
    };

    let input = input.trim_end_matches('=');
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &b in bytes {
        let val = TABLE[b as usize];
        if val == 255 {
            return Err(format!("invalid base64 character: '{}'", b as char));
        }
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(out)
}
