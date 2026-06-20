//! Admin REST API for Log Insights.
//!
//! Admin-only — registered under the same protected router as
//! `/v1/admin/insights`, with an extra `rbac::require_admin` route layer.
//!
//!   GET /v1/admin/logs            — recent captured log records
//!                                    (level + substring filters).
//!   GET /v1/admin/logs/categories — semantic categories of recent
//!                                    WARN/ERROR log messages.
//!   GET /v1/admin/slow-queries    — slowest searches across collections,
//!                                    from the per-collection query log.

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};

use crate::logs::categorize::CategorizeOptions;
use crate::logs::{LogCategory, LogRecord};
use crate::rest::query_log::read_window_entries;
use crate::rest::state::AppState;
use crate::tenant::{self, Tenant};

/// Default number of records returned by `GET /v1/admin/logs`.
const DEFAULT_LOG_LIMIT: usize = 200;
/// Hard cap on the `limit` query param for `GET /v1/admin/logs`.
const MAX_LOG_LIMIT: usize = 1000;
/// Window read from each per-collection query log for slow queries.
const SLOW_QUERY_WINDOW_SECS: u64 = 24 * 3600;
/// Default number of slow queries returned.
const DEFAULT_SLOW_LIMIT: usize = 20;

#[derive(Debug, Serialize)]
pub struct CategoriesResponse {
    pub categories: Vec<LogCategory>,
}

/// `GET /v1/admin/logs/categories` — cluster recent WARN+ERROR log
/// messages into semantic categories. Uses the configured text
/// embedding model when available, else an embedder-free fallback.
pub async fn list_categories(State(state): State<AppState>) -> Json<CategoriesResponse> {
    let registry = state.embedding_registry().read().await;
    let categories = state
        .log_categorizer()
        .categorize(state.log_buffer(), &registry, CategorizeOptions::default())
        .await;
    Json(CategoriesResponse { categories })
}

#[derive(Debug, Deserialize)]
pub struct LogsQuery {
    /// Minimum severity to keep (`ERROR`, `WARN`, `INFO`, ...). Records
    /// below this level are excluded. Omitted → all levels.
    #[serde(default)]
    pub level: Option<String>,
    /// Case-insensitive substring filter over message/target.
    #[serde(default)]
    pub q: Option<String>,
    /// Max records to return (default 200, capped at 1000).
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct LogsResponse {
    pub logs: Vec<LogRecord>,
}

/// `GET /v1/admin/logs?level=&q=&limit=` — most-recent captured log
/// records (newest-first), filtered by minimum severity and an
/// optional case-insensitive substring.
pub async fn list_logs(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> Json<LogsResponse> {
    let limit = q.limit.unwrap_or(DEFAULT_LOG_LIMIT).clamp(1, MAX_LOG_LIMIT);
    let level = q.level.as_deref().filter(|s| !s.is_empty());
    let contains = q.q.as_deref().filter(|s| !s.is_empty());
    let logs = state.log_buffer().recent(limit, level, contains);
    Json(LogsResponse { logs })
}

#[derive(Debug, Deserialize)]
pub struct SlowQueriesQuery {
    /// Restrict to a single collection (user-facing name). Omitted →
    /// all collections visible to the caller's tenant.
    #[serde(default)]
    pub collection: Option<String>,
    /// Max slow queries to return (default 20).
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SlowQuery {
    /// User-facing collection name (tenant prefix stripped).
    pub collection: String,
    pub ts_ms: u64,
    pub took_ms: u64,
    pub top_k: usize,
    pub trace_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SlowQueriesResponse {
    pub slow_queries: Vec<SlowQuery>,
}

/// `GET /v1/admin/slow-queries?collection=&limit=` — slowest searches
/// over the last 24h, read from each collection's `_query_log.jsonl`.
/// Tenant-scoped: only collections under the caller's tenant are read,
/// and names are returned without the tenant prefix.
pub async fn list_slow_queries(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Query(q): Query<SlowQueriesQuery>,
) -> Json<SlowQueriesResponse> {
    let limit = q.limit.unwrap_or(DEFAULT_SLOW_LIMIT).max(1);

    // Resolve the set of (scoped_key, display_name) collections to read.
    let targets: Vec<(String, String)> = {
        let schemas = state.inner.schemas.read().await;
        match q.collection.as_deref() {
            Some(name) => {
                let scoped = tenant::scope(&tenant, name);
                if schemas.contains_key(&scoped) {
                    vec![(scoped, name.to_owned())]
                } else {
                    Vec::new()
                }
            }
            None => schemas
                .keys()
                .filter_map(|key| {
                    tenant::strip_prefix(&tenant, key)
                        .map(|display| (key.clone(), display.to_owned()))
                })
                .collect(),
        }
    };

    let data_dir = state.inner.data_dir.clone();
    let mut slow_queries: Vec<SlowQuery> = tokio::task::spawn_blocking(move || {
        let mut out: Vec<SlowQuery> = Vec::new();
        for (scoped, display) in targets {
            for e in read_window_entries(&data_dir, &scoped, SLOW_QUERY_WINDOW_SECS) {
                out.push(SlowQuery {
                    collection: display.clone(),
                    ts_ms: e.ts_ms,
                    took_ms: e.took_ms,
                    top_k: e.top_k,
                    trace_id: e.trace_id,
                });
            }
        }
        out
    })
    .await
    .unwrap_or_default();

    slow_queries.sort_by(|a, b| b.took_ms.cmp(&a.took_ms));
    slow_queries.truncate(limit);
    Json(SlowQueriesResponse { slow_queries })
}
