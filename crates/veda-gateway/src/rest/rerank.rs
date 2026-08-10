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

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
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
    // Resolve the alias like `/search` does — the `candidates` path fetches
    // record text out of this collection, and ids handed back by a search
    // (which follows the alias) only exist in the alias target. Without this
    // every candidate misses and the handler returns an empty 200.
    let name = state.inner.aliases.resolve(&name).await;
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
#[tracing::instrument(
    name = "rerank_call",
    skip(query, documents, model_override, instruction),
    fields(n_documents = documents.len(), top_k)
)]
#[allow(clippy::type_complexity)]
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

    let client = reqwest::Client::new();
    // Body carries BOTH the Cohere/Voyage key (`documents`) and the
    // TEI / HuggingFace text-embeddings-inference key (`texts`) so the
    // same request works against either flavour of reranker. Unknown
    // keys are ignored by both.
    let mut body = serde_json::json!({
        "model": model,
        "query": query,
        "documents": documents,
        "texts": documents,
        "top_k": top_k,
        "return_documents": false,
    });
    if let Some(inst) = instruction
        && !inst.is_empty()
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert(
            "instruction".to_owned(),
            serde_json::Value::String(inst.to_owned()),
        );
    }

    // Try the Cohere/Voyage path (`/v1/rerank`) first, then fall back to
    // the TEI path (`/rerank`) when the former 404s. TEI rerankers
    // (e.g. ruri-v3-reranker) only serve `/rerank`.
    let base = endpoint.trim_end_matches('/');
    let mut last_err: Option<(StatusCode, String)> = None;
    let mut parsed: Option<Vec<(usize, f32)>> = None;
    for url in [format!("{base}/v1/rerank"), format!("{base}/rerank")] {
        let mut trace_headers = reqwest::header::HeaderMap::new();
        veda_core::trace_propagation::inject_current_context(&mut trace_headers);
        let mut req_builder = client
            .post(&url)
            .headers(trace_headers)
            .json(&body)
            .timeout(std::time::Duration::from_secs(30));
        if !api_key.is_empty() {
            req_builder = req_builder.bearer_auth(&api_key);
        }
        let resp = match req_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some((
                    StatusCode::BAD_GATEWAY,
                    format!("rerank request failed: {e}"),
                ));
                continue;
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let rbody = resp.text().await.unwrap_or_default();
            // 404 on the first (Cohere) path → try the TEI path.
            last_err = Some((
                StatusCode::BAD_GATEWAY,
                format!("reranker returned {status}: {rbody}"),
            ));
            continue;
        }
        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(e) => {
                last_err = Some((
                    StatusCode::BAD_GATEWAY,
                    format!("rerank response parse failed: {e}"),
                ));
                continue;
            }
        };
        match parse_rerank_response(&json) {
            Some(v) => {
                parsed = Some(v);
                break;
            }
            None => {
                last_err = Some((
                    StatusCode::BAD_GATEWAY,
                    "rerank response missing 'data'/array of {index,score}".into(),
                ));
            }
        }
    }
    let mut out = match parsed {
        Some(v) => v,
        None => {
            let (status, msg) =
                last_err.unwrap_or((StatusCode::BAD_GATEWAY, "rerank failed".into()));
            return Err((status, Json(ErrorResponse { error: msg })));
        }
    };
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_k);
    Ok(out)
}

/// Parse a rerank response into `(original_index, score)` pairs,
/// tolerating both common reranker response shapes:
///   - Cohere/Voyage: `{ "data": [ { "index": i, "relevance_score": s }, ... ] }`
///     (also accepts `results` as the array key and `score` as the score key).
///   - TEI / text-embeddings-inference: a bare array
///     `[ { "index": i, "score": s }, ... ]`.
///
/// Returns `None` when neither shape matches.
fn parse_rerank_response(json: &serde_json::Value) -> Option<Vec<(usize, f32)>> {
    let arr = json
        .as_array()
        .or_else(|| json["data"].as_array())
        .or_else(|| json["results"].as_array())?;
    let out: Vec<(usize, f32)> = arr
        .iter()
        .filter_map(|entry| {
            let idx = entry["index"].as_u64()? as usize;
            let score = entry["relevance_score"]
                .as_f64()
                .or_else(|| entry["score"].as_f64())? as f32;
            Some((idx, score))
        })
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

#[cfg(test)]
mod tests {
    use super::parse_rerank_response;

    #[test]
    fn parses_cohere_data_shape() {
        let j = serde_json::json!({
            "data": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.1}
            ]
        });
        let out = parse_rerank_response(&j).expect("cohere shape");
        assert_eq!(out, vec![(0, 0.9), (1, 0.1)]);
    }

    #[test]
    fn parses_tei_bare_array_shape() {
        // ruri-v3-reranker / HF text-embeddings-inference shape.
        let j = serde_json::json!([
            {"index": 0, "score": 0.0067712096},
            {"index": 1, "score": 0.00019716943}
        ]);
        let out = parse_rerank_response(&j).expect("tei shape");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 0);
        assert!((out[0].1 - 0.0067712096).abs() < 1e-6);
    }

    #[test]
    fn parses_results_array_key() {
        let j = serde_json::json!({
            "results": [ {"index": 2, "score": 0.5} ]
        });
        let out = parse_rerank_response(&j).expect("results key");
        assert_eq!(out, vec![(2, 0.5)]);
    }

    #[test]
    fn rejects_unknown_shape() {
        let j = serde_json::json!({"unexpected": true});
        assert!(parse_rerank_response(&j).is_none());
    }
}
