//! Shared insert + live-search helpers for the agent's internal vector
//! collections (semantic response cache, vector memory, LLM trace store).
//!
//! These three features each compute an embedding off-lock, then need the
//! resulting record to be **immediately searchable**. The subtlety is that a
//! collection's query executor (`state.inner.query_executors`, built by
//! [`crate::rest::records::rebuild_executors`]) is a *separate* manager
//! rebuilt from disk that only ever loads *sealed* segments. A freshly
//! inserted record lives only in the **live** segment manager's in-memory
//! *growing* segment, which that executor never sees.
//!
//! The original fix sealed the growing segment and rebuilt the executor on
//! every store. That made each write O(collection size) (the rebuild reloads
//! and re-indexes every sealed segment) and produced a tiny sealed segment
//! per write — a severe write-rate problem.
//!
//! This module takes the cheap route instead (no per-write seal/rebuild):
//!   * [`insert_only`] inserts into the routed shard's growing segment and
//!     returns. The growing segment seals itself only on the normal size
//!     threshold (`GrowingSegment::should_seal`), so writes are O(1)
//!     amortized and no segment explosion occurs.
//!   * [`search_live_growing`] brute-force searches the **live** growing
//!     segments (across shards) for the same request the lookup runs against
//!     the executor. The growing segment is bounded by the seal threshold, so
//!     this scan is cheap; it sees uninserted-but-not-sealed records
//!     immediately. The caller merges these with the executor's (sealed)
//!     results and takes the global top-k — so a stored entry is searchable
//!     the instant it is inserted, with zero per-write rebuild.
//!
//! Best-effort and off-lock: the segment-manager write lock is taken only for
//! the insert; any failure is logged at debug and swallowed (fail-open),
//! matching the callers' contract.

use veda_core::record::Record;
use veda_core::schema::CollectionSchema;
use veda_query::search::{ScoredRecord, SearchExecutor, VectorSearchRequest};

use crate::rest::state::AppState;

/// Insert `record` into the `scoped` collection's routed shard growing
/// segment. `what` is a short label used only in debug logs. Best-effort:
/// never panics, never propagates an error. Does NOT seal or rebuild the
/// executor — freshness is provided at read time by [`search_live_growing`].
pub(crate) async fn insert_only(
    state: &AppState,
    scoped: &str,
    schema: &CollectionSchema,
    record: Record,
    what: &str,
) {
    let mut managers = state.inner.segment_managers.write().await;
    let Some(shard_map) = managers.get_mut(scoped) else {
        return;
    };
    let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
        &record.id(),
        schema.shard_count(),
    );
    let Some(sm) = shard_map.get_mut(&shard_id) else {
        return;
    };
    if let Err(e) = sm.insert(record) {
        tracing::debug!(error = %e, what, "agent index: insert failed; skipping store");
    }
}

/// Brute-force search the **live** growing segments of every shard of the
/// `scoped` collection for `request`, returning the scored hits.
///
/// This sees records that have been inserted but not yet sealed — the records
/// the executor (sealed-only) misses — so a just-stored entry is immediately
/// findable. The growing segment is bounded by the seal threshold, so the
/// scan is cheap. Best-effort: returns an empty vec (never errors) when the
/// collection is absent or a per-shard search fails.
pub(crate) async fn search_live_growing(
    state: &AppState,
    scoped: &str,
    request: &VectorSearchRequest,
) -> Vec<ScoredRecord> {
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(scoped) else {
        return Vec::new();
    };
    let mut hits: Vec<ScoredRecord> = Vec::new();
    for sm in shard_map.values() {
        let growing = sm.growing_segment();
        if growing.count() == 0 {
            continue;
        }
        // `SearchExecutor::search` over only the growing segment is a pure
        // brute-force scan with full filter support (no ANN index needed).
        match SearchExecutor::search(request, Some(growing), &[]) {
            Ok((resp, _)) => hits.extend(resp.results),
            Err(e) => {
                tracing::debug!(error = %e, collection = scoped, "agent index: live growing search failed");
            }
        }
    }
    hits
}

/// Merge executor (sealed) results with live-growing results, sort by
/// ascending distance, and truncate to `k`. Pure.
pub(crate) fn merge_top_k(
    mut sealed: Vec<ScoredRecord>,
    growing: Vec<ScoredRecord>,
    k: usize,
) -> Vec<ScoredRecord> {
    sealed.extend(growing);
    sealed.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sealed.truncate(k);
    sealed
}
