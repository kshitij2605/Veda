//! Cross-encoder reranker integration.
//!
//! Wraps any rerank API that follows the Voyage/Cohere/ZeroEntropy shape:
//!
//!   POST {RERANK_ENDPOINT}/v1/rerank
//!   {
//!     "model": "...",
//!     "query": "...",
//!     "documents": ["doc 1", "doc 2", ...],
//!     "top_k": N,
//!     "instruction": "..."  // optional, see below
//!   }
//!
//!   → { "data": [ { "index": i, "relevance_score": s }, ... ] }
//!
//! `instruction` is forwarded only when the caller provides it.
//! Required to unlock instruction-following rerankers — Voyage
//! rerank-2.5 (Aug 2025), Cohere rerank-3.5, Qwen3-Reranker.
//! Classic cross-encoders (older rerank-2, bge-reranker-v2-m3)
//! ignore unknown fields.
//!
//! Exposed two ways:
//! 1. **Standalone**: `POST /v1/collections/{name}/rerank` — caller sends the
//!    query and a list of candidate IDs; gateway fetches their text from a
//!    declared field, reranks, returns ordered IDs + scores.
//! 2. **Inline**: `POST /v1/collections/{name}/search` with
//!    `"rerank": { "field": "content", "top_k": 10, "instruction": "..." }`
//!    — the gateway runs the normal search (oversampled), then reranks
//!    results in place.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use veda_core::record::FieldValue;

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::tenant::{self, Tenant};

#[derive(Debug, Deserialize, Clone)]
pub struct RerankConfig {
    /// Scalar string field on the collection whose value is the document
    /// text fed to the reranker.
    pub field: String,
    /// The user query as text. Required because vector / sparse / colpali
    /// queries don't carry a text representation.
    pub query: String,
    /// Number of results returned after rerank.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional override of the rerank model name. Defaults to env RERANK_MODEL.
    pub model: Option<String>,
    /// Optional natural-language instruction to the reranker. Passed
    /// through as `instruction:` in the upstream body. Required to
    /// unlock instruction-following rerankers — Voyage rerank-2.5
    /// (Aug 2025), Cohere rerank-3.5 with `instructions:` field,
    /// Qwen3-Reranker with instruction prefix. Optional for
    /// classic cross-encoders (Cohere rerank-3, mxbai-rerank-v2)
    /// that ignore it.
    pub instruction: Option<String>,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    /// Either pre-supplied texts (we rerank them directly) ...
    pub documents: Option<Vec<String>>,
    /// ... or candidate record IDs whose `field` we fetch and rerank.
    pub candidates: Option<Vec<String>>,
    /// Field to read text from when `candidates` is used.
    pub field: Option<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub model: Option<String>,
    /// Optional natural-language instruction for the reranker. See
    /// `RerankConfig::instruction` for context.
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankHit>,
    pub took_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct RerankHit {
    pub index: usize,
    pub id: Option<String>,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

pub async fn rerank(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    let start = Instant::now();

    // Decide what we're reranking: literal documents or fetched-by-id.
    let (texts, ids): (Vec<String>, Vec<Option<String>>) = if let Some(docs) = req.documents {
        let ids = vec![None; docs.len()];
        (docs, ids)
    } else if let Some(cands) = req.candidates {
        let field = req.field.as_ref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "field is required when candidates are supplied".into(),
                }),
            )
        })?;
        fetch_candidate_texts(&state, &name, cands, field).await?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "either 'documents' or 'candidates' must be set".into(),
            }),
        ));
    };

    let scored = call_reranker(
        &req.query,
        &texts,
        req.top_k,
        req.model.as_deref(),
        req.instruction.as_deref(),
    )
    .await?;

    let results: Vec<RerankHit> = scored
        .into_iter()
        .map(|(idx, score)| RerankHit {
            index: idx,
            id: ids.get(idx).and_then(|o| o.clone()),
            score,
            document: None,
        })
        .collect();

    Ok(Json(RerankResponse {
        results,
        took_ms: start.elapsed().as_millis() as u64,
    }))
}

/// Look up `(text, id)` pairs from a collection's records by ID.
async fn fetch_candidate_texts(
    state: &AppState,
    collection: &str,
    candidates: Vec<String>,
    field: &str,
) -> Result<(Vec<String>, Vec<Option<String>>), (StatusCode, Json<ErrorResponse>)> {
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(collection).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{collection}' not found"),
            }),
        )
    })?;

    let mut texts = Vec::with_capacity(candidates.len());
    let mut ids = Vec::with_capacity(candidates.len());
    'next_cand: for id_str in candidates {
        let id: uuid::Uuid = match id_str.parse() {
            Ok(u) => u,
            Err(_) => continue,
        };
        for sm in shard_map.values() {
            let gs = sm.growing_segment();
            for doc_id in 0..gs.count() {
                if let Some(rec) = gs.get_record(doc_id)
                    && rec.id() == id
                {
                    if let Some(FieldValue::String(s)) = rec.get(field) {
                        texts.push(s.clone());
                        ids.push(Some(id_str.clone()));
                    }
                    continue 'next_cand;
                }
            }
            for handle in sm.sealed_segments() {
                for doc_id in 0..handle.metadata.row_count as u32 {
                    if handle.reader.delete_bitmap().is_deleted(doc_id) {
                        continue;
                    }
                    if let Ok(rec) = handle.reader.get_record(doc_id)
                        && rec.id() == id
                    {
                        if let Some(FieldValue::String(s)) = rec.get(field) {
                            texts.push(s.clone());
                            ids.push(Some(id_str.clone()));
                        }
                        continue 'next_cand;
                    }
                }
            }
        }
    }
    Ok((texts, ids))
}

/// Call the configured rerank endpoint. Returns `(original_index, score)`
/// pairs sorted by score descending, truncated to `top_k`.
///
/// `instruction` is forwarded to the upstream as `instruction:`.
/// Instruction-following rerankers (Voyage rerank-2.5, Cohere
/// rerank-3.5, Qwen3-Reranker) honour it; classic cross-encoders
/// (older rerank-2, bge-reranker-v2-m3) ignore unknown fields.
pub async fn call_reranker(
    query: &str,
    documents: &[String],
    top_k: usize,
    model_override: Option<&str>,
    instruction: Option<&str>,
) -> Result<Vec<(usize, f32)>, (StatusCode, Json<ErrorResponse>)> {
    let endpoint = std::env::var("RERANK_ENDPOINT").unwrap_or_default();
    if endpoint.is_empty() || documents.is_empty() {
        // No reranker configured — return the original order with score=0.
        return Ok(documents
            .iter()
            .enumerate()
            .take(top_k)
            .map(|(i, _)| (i, 0.0))
            .collect());
    }
    let api_key = std::env::var("RERANK_API_KEY").unwrap_or_default();
    let model = model_override
        .map(|s| s.to_owned())
        .or_else(|| std::env::var("RERANK_MODEL").ok())
        .unwrap_or_else(|| "rerank-2".to_owned());

    let url = format!("{}/v1/rerank", endpoint.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let mut body = serde_json::json!({
        "model": model,
        "query": query,
        "documents": documents,
        "top_k": top_k,
    });
    if let Some(inst) = instruction
        && !inst.is_empty()
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert("instruction".to_owned(), serde_json::Value::String(inst.to_owned()));
    }

    let mut req_builder = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30));
    if !api_key.is_empty() {
        req_builder = req_builder.bearer_auth(&api_key);
    }
    let resp = req_builder.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: format!("rerank request failed: {e}"),
            }),
        )
    })?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: format!("reranker returned {status}: {body}"),
            }),
        ));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: format!("rerank response parse failed: {e}"),
            }),
        )
    })?;

    let arr = json["data"].as_array().ok_or_else(|| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: "rerank response missing 'data' array".into(),
            }),
        )
    })?;
    let mut out: Vec<(usize, f32)> = arr
        .iter()
        .filter_map(|entry| {
            let idx = entry["index"].as_u64()? as usize;
            let score = entry["relevance_score"]
                .as_f64()
                .or_else(|| entry["score"].as_f64())? as f32;
            Some((idx, score))
        })
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_k);
    Ok(out)
}
