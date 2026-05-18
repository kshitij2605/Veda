//! Per-collection search query log + coverage rollup.
//!
//! Phase 5.7 — every successful /search appends one JSON line to
//! `{data_dir}/{collection}/_query_log.jsonl` recording the doc ids
//! that came back. `GET /v1/collections/{name}/coverage?window=7d`
//! then rolls the log up so the dashboard can show:
//!   - total queries in the window
//!   - which chunks were never retrieved (orphans)
//!   - which chunks dominate top-k (monoculture risk)
//!
//! The log writer never blocks /search — appends happen on a
//! tokio::spawn and a write failure is logged, not surfaced.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use super::state::AppState;
use crate::rest::collections::ErrorResponse;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QueryLogEntry {
    /// Unix milliseconds when the search returned.
    pub ts_ms: u64,
    pub top_k: usize,
    pub doc_ids: Vec<String>,
    pub took_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn log_path(data_dir: &std::path::Path, collection: &str) -> PathBuf {
    data_dir.join(collection).join("_query_log.jsonl")
}

/// Best-effort append. Never panics; logs on error.
pub fn append_blocking(
    data_dir: &std::path::Path,
    collection: &str,
    entry: &QueryLogEntry,
) {
    let path = log_path(data_dir, collection);
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = match serde_json::to_string(entry) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "query_log: serialize failed");
            return;
        }
    };
    match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                tracing::warn!(error = %e, path = %path.display(), "query_log: write failed");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "query_log: open failed");
        }
    }
}

/// Spawn a non-blocking write so /search returns immediately.
pub fn record_search(
    state: &AppState,
    collection: &str,
    top_k: usize,
    doc_ids: Vec<String>,
    took_ms: u64,
) {
    let entry = QueryLogEntry {
        ts_ms: now_ms(),
        top_k,
        doc_ids,
        took_ms,
    };
    let data_dir = state.inner.data_dir.clone();
    let collection = collection.to_owned();
    tokio::task::spawn_blocking(move || {
        append_blocking(&data_dir, &collection, &entry);
    });
}

#[derive(Deserialize)]
pub struct CoverageQuery {
    /// Window string like "7d", "24h", "30m". Defaults to 7d.
    #[serde(default)]
    pub window: Option<String>,
}

fn parse_window_secs(s: Option<&str>) -> u64 {
    let raw = s.unwrap_or("7d").trim();
    if raw.is_empty() {
        return 7 * 24 * 3600;
    }
    let (num, unit) = raw.split_at(raw.len().saturating_sub(1));
    let n: u64 = num.parse().unwrap_or(7);
    match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 24 * 3600,
        _ => raw.parse().unwrap_or(7 * 24 * 3600),
    }
}

#[derive(Serialize)]
pub struct CoverageHit {
    pub doc_id: String,
    pub count: u64,
    pub fraction: f64,
}

#[derive(Serialize)]
pub struct CoverageResponse {
    pub window_secs: u64,
    pub total_queries: u64,
    pub total_chunks: u64,
    pub retrieved_chunks: u64,
    pub orphan_count: u64,
    /// Top-10 most-retrieved doc ids (monoculture indicator).
    pub top_hits: Vec<CoverageHit>,
    /// Doc ids that exist in the corpus but were not retrieved
    /// in the window. Capped at 50 to keep responses small.
    pub orphan_sample: Vec<String>,
}

fn read_window_entries(
    data_dir: &std::path::Path,
    collection: &str,
    window_secs: u64,
) -> Vec<QueryLogEntry> {
    let path = log_path(data_dir, collection);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let cutoff = now_ms().saturating_sub(window_secs * 1000);
    text.lines()
        .filter_map(|line| serde_json::from_str::<QueryLogEntry>(line).ok())
        .filter(|e| e.ts_ms >= cutoff)
        .collect()
}

pub async fn coverage(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<CoverageQuery>,
) -> Result<Json<CoverageResponse>, (StatusCode, Json<ErrorResponse>)> {
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

    let total_chunks: u64 = {
        let segs = state.inner.segment_managers.read().await;
        segs.get(&name)
            .map(|m| m.values().map(|sm| sm.total_records()).sum())
            .unwrap_or(0)
    };

    let window_secs = parse_window_secs(q.window.as_deref());
    let data_dir = state.inner.data_dir.clone();
    let collection_name = name.clone();
    let entries = tokio::task::spawn_blocking(move || {
        read_window_entries(&data_dir, &collection_name, window_secs)
    })
    .await
    .unwrap_or_default();

    let total_queries = entries.len() as u64;

    let mut counts: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for e in &entries {
        for id in &e.doc_ids {
            *counts.entry(id.clone()).or_insert(0) += 1;
        }
    }
    let retrieved_chunks = counts.len() as u64;
    let orphan_count = total_chunks.saturating_sub(retrieved_chunks);

    let mut sorted: Vec<(String, u64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let denom = entries
        .iter()
        .map(|e| e.doc_ids.len() as u64)
        .sum::<u64>()
        .max(1);
    let top_hits: Vec<CoverageHit> = sorted
        .iter()
        .take(10)
        .map(|(id, c)| CoverageHit {
            doc_id: id.clone(),
            count: *c,
            fraction: *c as f64 / denom as f64,
        })
        .collect();

    // Build orphan_sample: walk up to N records across shards, keep
    // the first ~50 ids that don't appear in the retrieved set.
    let retrieved_ids: std::collections::HashSet<String> =
        sorted.iter().map(|(id, _)| id.clone()).collect();
    let mut orphan_sample: Vec<String> = Vec::with_capacity(50);
    {
        let segs = state.inner.segment_managers.read().await;
        if let Some(shard_map) = segs.get(&name) {
            'outer: for sm in shard_map.values() {
                let gs = sm.growing_segment();
                for doc_id in 0..gs.count() {
                    if gs.is_deleted(doc_id) {
                        continue;
                    }
                    if let Some(rec) = gs.get_record(doc_id) {
                        let id = rec.id().to_string();
                        if !retrieved_ids.contains(&id) {
                            orphan_sample.push(id);
                            if orphan_sample.len() >= 50 {
                                break 'outer;
                            }
                        }
                    }
                }
                for handle in sm.sealed_segments() {
                    for doc_id in 0..handle.metadata.row_count as u32 {
                        if handle.reader.delete_bitmap().is_deleted(doc_id) {
                            continue;
                        }
                        if let Ok(rec) = handle.reader.get_record(doc_id) {
                            let id = rec.id().to_string();
                            if !retrieved_ids.contains(&id) {
                                orphan_sample.push(id);
                                if orphan_sample.len() >= 50 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Json(CoverageResponse {
        window_secs,
        total_queries,
        total_chunks,
        retrieved_chunks,
        orphan_count,
        top_hits,
        orphan_sample,
    }))
}
