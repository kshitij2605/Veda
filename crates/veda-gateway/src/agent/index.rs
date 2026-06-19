//! Shared insert-and-index helper for the agent's internal vector
//! collections (semantic response cache, vector memory, LLM trace store).
//!
//! These three features each compute an embedding off-lock and then need
//! the resulting record to be **immediately searchable**. Inserting into a
//! [`SegmentManager`] alone is not enough: a freshly inserted record lives
//! only in the live manager's in-memory *growing* segment, while the
//! collection's query executor is a *separate* manager rebuilt from disk
//! that only ever loads *sealed* segments. Until the growing segment is
//! sealed to disk and the executor rebuilt, a vector search over the
//! collection misses the new record entirely.
//!
//! The gateway's `embed_insert` HTTP path already does this (insert → seal →
//! [`rebuild_executors`]); the agent's direct-insert paths historically did
//! not, which is why the cache never hit and memory/trace search never
//! returned freshly stored records. This helper centralizes the correct
//! sequence so all three share one mechanism.
//!
//! Best-effort and off-lock: the segment-manager write lock is taken only
//! for the insert + seal and dropped before the executor rebuild; any
//! failure is logged at debug and swallowed (fail-open), matching the
//! callers' contract.

use veda_core::record::Record;
use veda_core::schema::CollectionSchema;

use crate::rest::state::AppState;

/// Insert `record` into `scoped` collection and make it immediately
/// searchable by sealing the growing segment and rebuilding the query
/// executor. `what` is a short label used only in debug logs. Best-effort:
/// never panics, never propagates an error.
pub(crate) async fn insert_and_index(
    state: &AppState,
    scoped: &str,
    schema: &CollectionSchema,
    record: Record,
    what: &str,
) {
    // Insert into the routed shard's growing segment, then seal so the
    // record is flushed to a sealed segment on disk. Hold the write lock
    // only for this synchronous work; never across the executor rebuild.
    {
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
            return;
        }
        if let Err(e) = sm.seal_growing() {
            tracing::debug!(error = %e, what, "agent index: seal failed; record may not be searchable");
            return;
        }
    }

    // Rebuild the collection's executor from the now-sealed on-disk
    // segments so the new record is visible to vector search.
    if let Err((_status, err)) = crate::rest::records::rebuild_executors(state, scoped).await {
        tracing::debug!(error = %err.0.error, what, "agent index: executor rebuild failed");
    }
}
