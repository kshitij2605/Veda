//! Data quality and duplicate-detection evaluation endpoint.
//!
//! Wires the cheapest features advertised by Niriksha's Evaluation page:
//! - Duplicate-candidate detection via nearest-neighbour vector similarity.
//! - Per-field quality stats: missing rate, character-length min/avg/max.
//!
//! LLM-as-judge features (RAG/FAQ usability scoring, cross-modal alignment)
//! are intentionally NOT implemented here — they need an external endpoint
//! and a separate design decision.

use std::collections::HashMap;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use veda_core::record::{FieldValue, Record};
use veda_core::types::FieldType;
use veda_query::search::VectorSearchRequest;

use super::collections::ErrorResponse;
use super::state::AppState;

#[derive(Debug, Deserialize)]
pub struct EvaluateQuery {
    /// Vector field to use for duplicate detection. If absent, duplicate
    /// detection is skipped.
    pub vector_field: Option<String>,
    /// Distance below which two records are flagged as duplicates.
    /// Default: 0.01 (cosine).
    #[serde(default = "default_dup_threshold")]
    pub duplicate_threshold: f32,
    /// Maximum records to evaluate. Sampling is from the start.
    #[serde(default = "default_sample")]
    pub sample: usize,
}

fn default_dup_threshold() -> f32 {
    0.01
}
fn default_sample() -> usize {
    200
}

#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub collection: String,
    pub records_evaluated: u64,
    pub duplicate_pairs: Vec<DuplicatePair>,
    pub field_quality: HashMap<String, FieldQuality>,
}

#[derive(Debug, Serialize)]
pub struct DuplicatePair {
    pub id_a: String,
    pub id_b: String,
    pub distance: f32,
}

#[derive(Debug, Serialize, Default)]
pub struct FieldQuality {
    pub missing: u64,
    pub min_length: Option<u64>,
    pub max_length: Option<u64>,
    pub avg_length: f64,
    pub null_or_empty: u64,
}

pub async fn evaluate(
    State(state): State<AppState>,
    Path(name): Path<String>,
    axum::extract::Query(q): axum::extract::Query<EvaluateQuery>,
) -> Result<Json<EvaluateResponse>, (StatusCode, Json<ErrorResponse>)> {
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

    // Collect a sample of records from all shards.
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "no segment managers for collection".into(),
            }),
        )
    })?;

    let mut sample: Vec<Record> = Vec::with_capacity(q.sample);
    'outer: for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                sample.push(rec.clone());
                if sample.len() >= q.sample {
                    break 'outer;
                }
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    sample.push(rec);
                    if sample.len() >= q.sample {
                        break 'outer;
                    }
                }
            }
        }
    }
    drop(managers);

    let records_evaluated = sample.len() as u64;

    // Per-field quality metrics.
    let mut field_quality: HashMap<String, FieldQuality> = HashMap::new();
    for field in schema.fields() {
        let name = field.name().to_owned();
        let mut q = FieldQuality::default();
        let mut total_len: u64 = 0;
        let mut counted: u64 = 0;

        for rec in &sample {
            match rec.get(&name) {
                None | Some(FieldValue::Null) => q.missing += 1,
                Some(FieldValue::String(s)) => {
                    let l = s.chars().count() as u64;
                    counted += 1;
                    total_len += l;
                    if s.is_empty() {
                        q.null_or_empty += 1;
                    }
                    q.min_length = Some(q.min_length.map_or(l, |m| m.min(l)));
                    q.max_length = Some(q.max_length.map_or(l, |m| m.max(l)));
                }
                Some(_) => counted += 1,
            }
        }
        if counted > 0 {
            q.avg_length = total_len as f64 / counted as f64;
        }
        field_quality.insert(name, q);
    }

    // Duplicate detection (optional).
    let mut duplicate_pairs: Vec<DuplicatePair> = Vec::new();
    if let Some(ref vec_field) = q.vector_field {
        // Validate field exists and is a dense vector.
        let metric = schema
            .get_field(vec_field)
            .and_then(|f| match f.field_type() {
                FieldType::Vector { metric, .. } => Some(*metric),
                _ => None,
            })
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("'{vec_field}' is not a dense vector field"),
                    }),
                )
            })?;

        let executors = state.inner.query_executors.read().await;
        let executor = executors.get(&name).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "no query executor".into(),
                }),
            )
        })?;

        let mut seen_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for rec in &sample {
            let Some(FieldValue::Vector(dense)) = rec.get(vec_field) else {
                continue;
            };
            let request = VectorSearchRequest {
                field: vec_field.clone(),
                query_vector: dense.as_slice().to_vec(),
                k: 2,
                metric,
                filter: None,
                truncate_to: None,
            };
            let Ok((resp, _)) = executor.search(&request) else {
                continue;
            };
            for hit in resp.results {
                // Skip self-match.
                if hit.id == rec.id() {
                    continue;
                }
                if hit.distance > q.duplicate_threshold {
                    continue;
                }
                let mut ids = [rec.id().to_string(), hit.id.to_string()];
                ids.sort();
                let key = (ids[0].clone(), ids[1].clone());
                if seen_pairs.insert(key) {
                    duplicate_pairs.push(DuplicatePair {
                        id_a: ids[0].clone(),
                        id_b: ids[1].clone(),
                        distance: hit.distance,
                    });
                }
            }
        }
    }

    Ok(Json(EvaluateResponse {
        collection: name,
        records_evaluated,
        duplicate_pairs,
        field_quality,
    }))
}
