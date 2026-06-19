//! Searchable semantic LLM trace store admin API
//! (`GET /v1/admin/llm-traces`, `GET /v1/admin/llm-traces/clusters`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::correlations`]
//! and [`crate::rest::incidents`].
//!
//!   * `GET /v1/admin/llm-traces?q=&limit=` — semantic search over the
//!     tenant's persisted converse runs (NL query → relevant past runs with
//!     faithfulness / cost / cache / guardrail signals).
//!   * `GET /v1/admin/llm-traces/clusters?limit=&faithfulness_max=` —
//!     failure clustering: recent (optionally low-faithfulness) runs
//!     collapsed into semantic groups.
//!
//! Both are tenant/RLS-scoped and empty-safe: a fresh server (no
//! `_llm_traces` collection, no embedder) returns an empty result with
//! `200`, never an error.

use axum::extract::{Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::agent::trace_store::{self, TraceCluster, TraceHit};
use crate::rest::state::AppState;
use crate::tenant::Tenant;

#[derive(Debug, Deserialize)]
pub struct TraceSearchQuery {
    /// Natural-language search query. Empty → empty result.
    #[serde(default)]
    pub q: Option<String>,
    /// Max hits to return (default [`trace_store::DEFAULT_LIMIT`]).
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct TraceSearchResponse {
    pub hits: Vec<TraceHit>,
}

#[derive(Debug, Deserialize)]
pub struct TraceClustersQuery {
    /// Max recent traces to consider (default [`trace_store::DEFAULT_LIMIT`]).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Only cluster runs with `faithfulness < faithfulness_max` (the
    /// "failure" view). Omitted → cluster all recent runs.
    #[serde(default)]
    pub faithfulness_max: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct TraceClustersResponse {
    pub clusters: Vec<TraceCluster>,
}

/// `GET /v1/admin/llm-traces?q=&limit=` — semantic search over the tenant's
/// persisted LLM traces. Empty `q` (or no collection / embedder) → empty.
pub async fn search_traces(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Query(q): Query<TraceSearchQuery>,
) -> Json<TraceSearchResponse> {
    let query = q.q.unwrap_or_default();
    let limit = q.limit.unwrap_or(trace_store::DEFAULT_LIMIT);
    let hits = trace_store::search_traces(&state, &tenant, &query, limit).await;
    Json(TraceSearchResponse { hits })
}

/// `GET /v1/admin/llm-traces/clusters?limit=&faithfulness_max=` — failure
/// clustering over the tenant's recent traces. Empty (no collection /
/// embedder / matching traces) → empty.
pub async fn cluster_traces(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Query(q): Query<TraceClustersQuery>,
) -> Json<TraceClustersResponse> {
    let limit = q.limit.unwrap_or(trace_store::DEFAULT_LIMIT);
    let clusters =
        trace_store::cluster_traces(&state, &tenant, limit, q.faithfulness_max).await;
    Json(TraceClustersResponse { clusters })
}
