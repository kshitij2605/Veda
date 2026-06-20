//! Production [`RemediationDispatcher`] backed by live [`AppState`].
//!
//! * `Compact` reuses the **same** segment-manager compaction path the
//!   insert handler spawns (`SegmentManager::compact_if_needed` with the
//!   default [`CompactionConfig`]), honoring the per-collection
//!   `compaction_locks` guard so it never overlaps a background
//!   compaction. It does NOT reimplement compaction.
//! * `Reindex` is intentionally best-effort: the full reindex pipeline
//!   is heavy and its scheduler handler is still stubbed, so we only log
//!   and return a note telling the operator manual follow-up is needed.
//! * `None` is unreachable here — [`run_remediation`](super::run_remediation)
//!   filters it out before dispatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::advisor::event::RemediationAction;
use crate::rest::state::AppState;

use super::RemediationDispatcher;

/// Live dispatcher holding a clone of the shared [`AppState`].
pub struct LiveDispatcher {
    state: AppState,
}

impl LiveDispatcher {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Run compaction on every shard of `collection`, reusing the real
    /// compaction path and the `compaction_locks` guard. Returns a note;
    /// skips with a note if another compaction is already in flight.
    async fn compact(&self, collection: &str) -> Result<String, String> {
        // Acquire (or create) the per-collection compaction guard — the
        // same guard the insert handler uses.
        let lock = {
            let mut locks = self.state.inner.compaction_locks.write().await;
            locks
                .entry(collection.to_owned())
                .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                .clone()
        };
        if lock
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return Ok("compaction already running; skipped".to_owned());
        }
        struct CompactGuard(Arc<AtomicBool>);
        impl Drop for CompactGuard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _guard = CompactGuard(lock.clone());

        let config = veda_storage::compaction::CompactionConfig::default();
        let mut managers = self.state.inner.segment_managers.write().await;
        let shard_map = managers
            .get_mut(collection)
            .ok_or_else(|| format!("collection '{collection}' not found"))?;

        let mut compacted_shards = 0usize;
        let mut errors = 0usize;
        for sm in shard_map.values_mut() {
            match sm.compact_if_needed(&config) {
                Ok(removed) if removed > 0 => compacted_shards += 1,
                Ok(_) => {}
                Err(e) => {
                    errors += 1;
                    tracing::warn!(
                        collection = %collection,
                        error = %e,
                        "auto-remediation: shard compaction failed"
                    );
                }
            }
        }

        if errors > 0 {
            Err(format!(
                "compacted {compacted_shards} shard(s), {errors} shard(s) failed"
            ))
        } else {
            Ok(format!("compacted {compacted_shards} shard(s)"))
        }
    }
}

impl RemediationDispatcher for LiveDispatcher {
    async fn dispatch(&self, action: &RemediationAction, _now_ms: i64) -> Result<String, String> {
        match action {
            RemediationAction::Compact { collection } => self.compact(collection).await,
            RemediationAction::Reindex { collection } => {
                // The full reindex pipeline is heavy and its scheduler
                // handler is still stubbed; do not run it here.
                tracing::info!(
                    collection = %collection,
                    "auto-remediation: reindex requested (manual follow-up required)"
                );
                Ok("reindex requested (manual follow-up required)".to_owned())
            }
            // Filtered out by run_remediation before dispatch.
            RemediationAction::None => Ok("no remediation".to_owned()),
        }
    }
}
