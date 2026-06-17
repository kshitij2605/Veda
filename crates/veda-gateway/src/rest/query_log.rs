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
use crate::tenant::{self, Tenant};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QueryLogEntry {
    /// Unix milliseconds when the search returned.
    pub ts_ms: u64,
    pub top_k: usize,
    pub doc_ids: Vec<String>,
    pub took_ms: u64,
    /// OTel trace id (32-hex) of the search request, when a valid
    /// trace context was active. `#[serde(default)]` keeps older
    /// log files (written before this field existed) loadable.
    #[serde(default)]
    pub trace_id: Option<String>,
    // ── APM Phase 1 enrichment ──────────────────────────────────────
    // Every field below carries `#[serde(default)]` so log lines
    // written before these existed (older `_query_log.jsonl` files)
    // still deserialize, defaulting the missing attributes.
    /// User-facing collection name (tenant prefix stripped).
    #[serde(default)]
    pub collection: String,
    /// Query kind: dense | sparse | bm25 | match_phrase | hybrid |
    /// colpali | multi_vector. Drives correlation grouping.
    #[serde(default)]
    pub query_type: String,
    /// Whether a scalar filter (raw or RLS-injected) was applied.
    #[serde(default)]
    pub has_filter: bool,
    /// Whether a cross-encoder rerank stage ran.
    #[serde(default)]
    pub reranked: bool,
    /// Number of results actually returned to the caller.
    #[serde(default)]
    pub result_count: u32,
    /// "ok" for a successful search, "error" for a failed one.
    /// Defaults to "ok" so old entries (which were only ever written
    /// on success) read back as successes.
    #[serde(default = "default_status")]
    pub status: String,
    /// Coarse error variant name when `status == "error"`, else None.
    #[serde(default)]
    pub error_kind: Option<String>,
    /// Index type backing the queried vector field (best-effort), e.g.
    /// "hnsw" / "ivf_pq" / "flat". None for lexical-only queries or when
    /// it can't be resolved from the schema.
    #[serde(default)]
    pub index_type: Option<String>,
    /// Shard ids touched by the query (best-effort; may be empty).
    #[serde(default)]
    pub shard_ids: Vec<u32>,
}

fn default_status() -> String {
    "ok".to_owned()
}

/// Request-time metadata captured by the search handler and threaded
/// into the query log on both the success and error paths. Kept
/// separate from [`QueryLogEntry`] so the handler can build it up front
/// (before the search runs) and the logger stamps `ts_ms` / fills the
/// outcome fields exactly once per request.
#[derive(Clone, Debug, Default)]
pub struct QueryMeta {
    pub query_type: String,
    pub has_filter: bool,
    pub reranked: bool,
    pub top_k: usize,
    pub index_type: Option<String>,
    pub shard_ids: Vec<u32>,
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

/// Spawn a non-blocking write of an already-built entry so /search
/// returns immediately. The query-log write is UNCONDITIONAL — it is
/// never gated on trace sampling, because the correlation advisor and
/// coverage rollups depend on a complete log.
fn spawn_append(state: &AppState, collection: &str, entry: QueryLogEntry) {
    let data_dir = state.inner.data_dir.clone();
    let collection = collection.to_owned();
    tokio::task::spawn_blocking(move || {
        append_blocking(&data_dir, &collection, &entry);
    });
}

/// Log a SUCCESSFUL search. `scoped` is the tenant-scoped collection key
/// (the on-disk log directory); `display` is the user-facing name stored
/// in the entry (`collection` field) for the correlation analyzer.
/// Exactly one of [`record_success`] / [`record_error`] is called per
/// request, so success and error counts never double-count.
pub fn record_success(
    state: &AppState,
    scoped: &str,
    display: &str,
    meta: &QueryMeta,
    doc_ids: Vec<String>,
    took_ms: u64,
) {
    let (trace_id, _span_id) = crate::logs::current_trace_ids();
    let result_count = doc_ids.len() as u32;
    let entry = QueryLogEntry {
        ts_ms: now_ms(),
        top_k: meta.top_k,
        doc_ids,
        took_ms,
        trace_id,
        collection: display.to_owned(),
        query_type: meta.query_type.clone(),
        has_filter: meta.has_filter,
        reranked: meta.reranked,
        result_count,
        status: "ok".to_owned(),
        error_kind: None,
        index_type: meta.index_type.clone(),
        shard_ids: meta.shard_ids.clone(),
    };
    spawn_append(state, scoped, entry);
}

/// Log a FAILED search. `error_kind` is a coarse variant name (e.g.
/// the HTTP-status family). `took_ms` is measured to the point of
/// failure. Mutually exclusive with [`record_success`].
pub fn record_error(
    state: &AppState,
    scoped: &str,
    display: &str,
    meta: &QueryMeta,
    error_kind: &str,
    took_ms: u64,
) {
    let (trace_id, _span_id) = crate::logs::current_trace_ids();
    let entry = QueryLogEntry {
        ts_ms: now_ms(),
        top_k: meta.top_k,
        doc_ids: Vec::new(),
        took_ms,
        trace_id,
        collection: display.to_owned(),
        query_type: meta.query_type.clone(),
        has_filter: meta.has_filter,
        reranked: meta.reranked,
        result_count: 0,
        status: "error".to_owned(),
        error_kind: Some(error_kind.to_owned()),
        index_type: meta.index_type.clone(),
        shard_ids: meta.shard_ids.clone(),
    };
    spawn_append(state, scoped, entry);
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

pub(crate) fn read_window_entries(
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
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Query(q): Query<CoverageQuery>,
) -> Result<Json<CoverageResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_log_entry_roundtrips_with_trace_id() {
        let entry = QueryLogEntry {
            ts_ms: 123,
            top_k: 5,
            doc_ids: vec!["a".into(), "b".into()],
            took_ms: 7,
            trace_id: Some("0123456789abcdef0123456789abcdef".into()),
            collection: "docs".into(),
            query_type: "dense".into(),
            has_filter: true,
            reranked: true,
            result_count: 2,
            status: "ok".into(),
            error_kind: None,
            index_type: Some("hnsw".into()),
            shard_ids: vec![0, 1],
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: QueryLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.trace_id, entry.trace_id);
        assert_eq!(back.doc_ids, entry.doc_ids);
        assert_eq!(back.collection, "docs");
        assert_eq!(back.query_type, "dense");
        assert!(back.has_filter);
        assert!(back.reranked);
        assert_eq!(back.result_count, 2);
        assert_eq!(back.status, "ok");
        assert_eq!(back.index_type.as_deref(), Some("hnsw"));
        assert_eq!(back.shard_ids, vec![0, 1]);
    }

    #[test]
    fn error_entry_roundtrips() {
        let entry = QueryLogEntry {
            ts_ms: 1,
            top_k: 10,
            doc_ids: Vec::new(),
            took_ms: 3,
            trace_id: None,
            collection: "docs".into(),
            query_type: "hybrid".into(),
            has_filter: false,
            reranked: false,
            result_count: 0,
            status: "error".into(),
            error_kind: Some("internal".into()),
            index_type: None,
            shard_ids: Vec::new(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: QueryLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, "error");
        assert_eq!(back.error_kind.as_deref(), Some("internal"));
        assert_eq!(back.result_count, 0);
    }

    #[test]
    fn old_entry_without_trace_id_still_deserializes() {
        // A line written before the `trace_id` field existed.
        let old = r#"{"ts_ms":1,"top_k":3,"doc_ids":["x"],"took_ms":2}"#;
        let parsed: QueryLogEntry = serde_json::from_str(old).unwrap();
        assert_eq!(parsed.trace_id, None);
        assert_eq!(parsed.top_k, 3);
    }

    #[test]
    fn old_entry_defaults_apm_fields() {
        // A line written before ANY of the APM enrichment fields
        // existed (only the original four). Must deserialize with sane
        // defaults: status="ok", empty collection/query_type, no filter,
        // not reranked, zero result_count, None index/error, no shards.
        let old = r#"{"ts_ms":42,"top_k":7,"doc_ids":["a","b"],"took_ms":11}"#;
        let parsed: QueryLogEntry = serde_json::from_str(old).unwrap();
        assert_eq!(parsed.collection, "");
        assert_eq!(parsed.query_type, "");
        assert!(!parsed.has_filter);
        assert!(!parsed.reranked);
        assert_eq!(parsed.result_count, 0);
        assert_eq!(parsed.status, "ok", "old entries default to a success");
        assert_eq!(parsed.error_kind, None);
        assert_eq!(parsed.index_type, None);
        assert!(parsed.shard_ids.is_empty());
        // original fields still parse.
        assert_eq!(parsed.top_k, 7);
        assert_eq!(parsed.doc_ids, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(parsed.took_ms, 11);
    }
}
