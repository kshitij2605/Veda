//! Admin REST API for slow-query / error correlation analysis
//! (`GET /v1/admin/correlations`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::slos`] and
//! [`crate::rest::logs`].
//!
//! Reads the per-collection `_query_log.jsonl` over a recent window and
//! groups the entries by each correlation *dimension* (collection,
//! query_type, has_filter, reranked, index_type, plus a result-count
//! bucket and a top_k bucket). For every bucket it computes count,
//! p50/p95 latency, slow count (>1000ms), error count and error rate.
//! Buckets within each dimension are ranked so the worst drivers of
//! latency / errors surface first.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::rest::query_log::{read_window_entries, QueryLogEntry};
use crate::rest::state::AppState;
use crate::tenant::{self, Tenant};

/// Default analysis window (seconds) — matches the advisor snapshot.
const DEFAULT_WINDOW_SECS: u64 = 300;
/// A query slower than this many milliseconds counts as "slow"
/// (matches `advisor::snapshot::SLOW_QUERY_MS`).
const SLOW_QUERY_MS: u64 = 1000;

#[derive(Debug, Deserialize)]
pub struct CorrelationsQuery {
    /// Restrict to a single collection (user-facing name). Omitted →
    /// all collections under the caller's tenant.
    #[serde(default)]
    pub collection: Option<String>,
    /// Analysis window in seconds (default 300).
    #[serde(default)]
    pub window_secs: Option<u64>,
    /// Single dimension to analyze. Omitted → analyze all dimensions.
    /// One of: collection | query_type | has_filter | reranked |
    /// index_type | result_count | top_k.
    #[serde(default)]
    pub dimension: Option<String>,
}

/// Per-bucket rolled-up statistics within a dimension.
#[derive(Debug, Clone, Serialize)]
pub struct Bucket {
    /// The bucket label (e.g. a collection name, "dense", "filter=true",
    /// "hnsw", or a count range like "11-100").
    pub key: String,
    pub count: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub slow_count: u64,
    pub error_count: u64,
    /// Fraction of queries in this bucket that errored, in [0, 1].
    pub error_rate: f64,
}

/// One correlation dimension and its ranked buckets.
#[derive(Debug, Clone, Serialize)]
pub struct Dimension {
    /// Dimension name (collection | query_type | ...).
    pub dimension: String,
    /// Buckets ranked worst-first by (error_rate, p95_ms).
    pub buckets: Vec<Bucket>,
}

/// Overall window summary across every entry analyzed.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub window_secs: u64,
    pub total_count: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub slow_count: u64,
    pub error_count: u64,
    pub error_rate: f64,
}

#[derive(Debug, Serialize)]
pub struct CorrelationsResponse {
    pub summary: Summary,
    pub dimensions: Vec<Dimension>,
}

/// Accumulator for one bucket: keeps every `took_ms` (for percentiles)
/// plus the error tally. Mirrors the HashMap-accumulator pattern in
/// `rest::aggregate`.
#[derive(Default)]
struct Acc {
    took: Vec<u64>,
    errors: u64,
}

impl Acc {
    fn push(&mut self, e: &QueryLogEntry) {
        self.took.push(e.took_ms);
        if e.status == "error" {
            self.errors += 1;
        }
    }

    fn finish(mut self, key: String) -> Bucket {
        self.took.sort_unstable();
        let count = self.took.len() as u64;
        let slow_count = self.took.iter().filter(|&&t| t > SLOW_QUERY_MS).count() as u64;
        let error_rate = if count == 0 {
            0.0
        } else {
            self.errors as f64 / count as f64
        };
        Bucket {
            key,
            count,
            p50_ms: percentile(&self.took, 0.50),
            p95_ms: percentile(&self.took, 0.95),
            slow_count,
            error_count: self.errors,
            error_rate,
        }
    }
}

/// Nearest-rank percentile over a pre-sorted slice (`q` in [0, 1]).
/// 0.0 for an empty slice. Same definition as `advisor::snapshot`.
fn percentile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = (q * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)] as f64
}

/// Coarse bucket label for a result / top_k count.
fn count_bucket(n: u64) -> &'static str {
    match n {
        0 => "0",
        1..=10 => "1-10",
        11..=100 => "11-100",
        101..=1000 => "101-1000",
        _ => "1000+",
    }
}

/// Build a single dimension by grouping `entries` with the given key fn,
/// then ranking buckets worst-first (highest error_rate, then p95).
fn build_dimension<F>(name: &str, entries: &[QueryLogEntry], key_of: F) -> Dimension
where
    F: Fn(&QueryLogEntry) -> String,
{
    let mut groups: HashMap<String, Acc> = HashMap::new();
    for e in entries {
        groups.entry(key_of(e)).or_default().push(e);
    }
    let mut buckets: Vec<Bucket> = groups.into_iter().map(|(k, acc)| acc.finish(k)).collect();
    buckets.sort_by(|a, b| {
        b.error_rate
            .partial_cmp(&a.error_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.p95_ms
                    .partial_cmp(&a.p95_ms)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then_with(|| b.count.cmp(&a.count))
    });
    Dimension {
        dimension: name.to_owned(),
        buckets,
    }
}

/// The dimensions analyzed when `dimension` is omitted.
const ALL_DIMENSIONS: &[&str] = &[
    "collection",
    "query_type",
    "has_filter",
    "reranked",
    "index_type",
    "result_count",
    "top_k",
];

/// `GET /v1/admin/correlations?collection=&window_secs=&dimension=` —
/// slow-query / error correlation analysis over the recent query log.
/// Tenant-scoped: only collections under the caller's tenant are read.
pub async fn correlations(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Query(q): Query<CorrelationsQuery>,
) -> Json<CorrelationsResponse> {
    let window_secs = q.window_secs.unwrap_or(DEFAULT_WINDOW_SECS).max(1);

    // Resolve (scoped_key, display_name) collections to read — same
    // tenant-scoping logic as `logs::list_slow_queries`.
    let targets: Vec<String> = {
        let schemas = state.inner.schemas.read().await;
        match q.collection.as_deref() {
            Some(name) => {
                let scoped = tenant::scope(&tenant, name);
                if schemas.contains_key(&scoped) {
                    vec![scoped]
                } else {
                    Vec::new()
                }
            }
            None => schemas
                .keys()
                .filter(|key| tenant::strip_prefix(&tenant, key).is_some())
                .cloned()
                .collect(),
        }
    };

    let data_dir = state.inner.data_dir.clone();
    let entries: Vec<QueryLogEntry> = tokio::task::spawn_blocking(move || {
        let mut out: Vec<QueryLogEntry> = Vec::new();
        for scoped in targets {
            out.extend(read_window_entries(&data_dir, &scoped, window_secs));
        }
        out
    })
    .await
    .unwrap_or_default();

    // Which dimensions to compute.
    let wanted: Vec<&str> = match q.dimension.as_deref() {
        Some(d) if ALL_DIMENSIONS.contains(&d) => vec![d],
        _ => ALL_DIMENSIONS.to_vec(),
    };

    let dimensions: Vec<Dimension> = wanted
        .iter()
        .map(|&dim| match dim {
            "collection" => build_dimension(dim, &entries, |e| {
                if e.collection.is_empty() {
                    "<unknown>".to_owned()
                } else {
                    e.collection.clone()
                }
            }),
            "query_type" => build_dimension(dim, &entries, |e| {
                if e.query_type.is_empty() {
                    "<unknown>".to_owned()
                } else {
                    e.query_type.clone()
                }
            }),
            "has_filter" => build_dimension(dim, &entries, |e| {
                format!("filter={}", e.has_filter)
            }),
            "reranked" => {
                build_dimension(dim, &entries, |e| format!("reranked={}", e.reranked))
            }
            "index_type" => build_dimension(dim, &entries, |e| {
                e.index_type.clone().unwrap_or_else(|| "<none>".to_owned())
            }),
            "result_count" => {
                build_dimension(dim, &entries, |e| count_bucket(e.result_count as u64).to_owned())
            }
            "top_k" => {
                build_dimension(dim, &entries, |e| count_bucket(e.top_k as u64).to_owned())
            }
            _ => unreachable!("dimension list is filtered to ALL_DIMENSIONS"),
        })
        .collect();

    // Overall summary across all analyzed entries.
    let mut took: Vec<u64> = entries.iter().map(|e| e.took_ms).collect();
    took.sort_unstable();
    let total_count = took.len() as u64;
    let slow_count = took.iter().filter(|&&t| t > SLOW_QUERY_MS).count() as u64;
    let error_count = entries.iter().filter(|e| e.status == "error").count() as u64;
    let error_rate = if total_count == 0 {
        0.0
    } else {
        error_count as f64 / total_count as f64
    };
    let summary = Summary {
        window_secs,
        total_count,
        p50_ms: percentile(&took, 0.50),
        p95_ms: percentile(&took, 0.95),
        slow_count,
        error_count,
        error_rate,
    };

    Json(CorrelationsResponse {
        summary,
        dimensions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(query_type: &str, took_ms: u64, status: &str) -> QueryLogEntry {
        QueryLogEntry {
            ts_ms: 0,
            top_k: 10,
            doc_ids: Vec::new(),
            took_ms,
            trace_id: None,
            collection: "docs".into(),
            query_type: query_type.into(),
            has_filter: false,
            reranked: false,
            result_count: 0,
            status: status.into(),
            error_kind: None,
            index_type: None,
            shard_ids: Vec::new(),
        }
    }

    #[test]
    fn ranks_slowest_bucket_first_by_p95() {
        let entries = vec![
            entry("dense", 10, "ok"),
            entry("dense", 12, "ok"),
            entry("hybrid", 2000, "ok"),
            entry("hybrid", 2500, "ok"),
        ];
        let dim = build_dimension("query_type", &entries, |e| e.query_type.clone());
        // hybrid has the higher p95 and no errors anywhere → ranked first.
        assert_eq!(dim.buckets[0].key, "hybrid");
        assert!(dim.buckets[0].p95_ms >= dim.buckets[1].p95_ms);
    }

    #[test]
    fn ranks_errored_bucket_first() {
        let entries = vec![
            entry("dense", 5000, "ok"), // slow but no errors
            entry("bm25", 5, "error"),  // fast but errors
            entry("bm25", 5, "ok"),
        ];
        let dim = build_dimension("query_type", &entries, |e| e.query_type.clone());
        // error_rate dominates ranking → bm25 (0.5) before dense (0.0).
        assert_eq!(dim.buckets[0].key, "bm25");
        assert!((dim.buckets[0].error_rate - 0.5).abs() < 1e-9);
        assert_eq!(dim.buckets[0].error_count, 1);
    }

    #[test]
    fn count_bucket_boundaries() {
        assert_eq!(count_bucket(0), "0");
        assert_eq!(count_bucket(1), "1-10");
        assert_eq!(count_bucket(10), "1-10");
        assert_eq!(count_bucket(11), "11-100");
        assert_eq!(count_bucket(1001), "1000+");
    }

    #[test]
    fn percentile_nearest_rank() {
        let sorted = [10u64, 20, 30, 40, 100];
        assert_eq!(percentile(&sorted, 0.95), 100.0);
        assert_eq!(percentile(&sorted, 0.50), 30.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
