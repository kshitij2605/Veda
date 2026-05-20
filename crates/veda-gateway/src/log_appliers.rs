//! Phase C bootstrap: per-shard `LogApplier`s spawned at gateway
//! startup so follower nodes catch up to the writer's per-shard
//! log without going through the metadata Raft reconcile loop.
//!
//! ## Mode flag
//!
//! Controlled by the env `VEDA_LOG_APPLIER`:
//!   - unset / "0" / "false": no appliers spawned; the writer's
//!     `insert_via_log` continues to apply inline to the local
//!     `SegmentManager` (single-node default — fast feedback,
//!     no double-apply risk).
//!   - "1" / "true": appliers spawned per `(collection,
//!     shard_id)`. The writer's `insert_via_log` skips the
//!     inline apply step (see `applier_owns_apply()` in
//!     records.rs); visibility comes via the applier tick.
//!
//! The single-node deployment can leave the default off. A
//! true multi-node follower deployment turns the flag on; the
//! follower never has the writer-inline path active anyway.
//!
//! ## Locking
//!
//! The applier acquires `state.inner.segment_managers.write()`
//! briefly per tick — same write lock the insert / drop paths
//! contend on. Apply ticks are short (proportional to the
//! batch they read); contention is bounded.

use std::time::Duration;

use bytes::Bytes;
use tokio::task::JoinHandle;
use uuid::Uuid;
use veda_core::record::Record;
use veda_log::format::{LogEntryV2, LogOp};
use veda_log::reader::LogTailer;

use crate::rest::records::ensure_shard_log_for;
use crate::rest::state::AppState;

/// Whether `VEDA_LOG_APPLIER` is on. Cached at first call —
/// changes to the env after process startup have no effect.
pub fn applier_enabled() -> bool {
    std::env::var("VEDA_LOG_APPLIER")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

/// Spawn one applier task per `(collection, shard_id)`
/// currently materialised on this node. Returns the
/// `JoinHandle`s so the caller can shut them down (e.g. tests).
///
/// No-op when [`applier_enabled`] is false.
pub async fn spawn_for_all_known_collections(state: AppState) -> Vec<JoinHandle<()>> {
    if !applier_enabled() {
        return Vec::new();
    }
    let pairs: Vec<(String, u32)> = {
        let managers = state.inner.segment_managers.read().await;
        managers
            .iter()
            .flat_map(|(name, shards)| {
                shards.keys().map(|s| (name.clone(), *s)).collect::<Vec<_>>()
            })
            .collect()
    };
    let mut handles = Vec::with_capacity(pairs.len());
    for (collection, shard_id) in pairs {
        handles.push(spawn_one(state.clone(), collection, shard_id, Duration::from_millis(50)));
    }
    tracing::info!(
        appliers_spawned = handles.len(),
        "Phase C: log appliers running"
    );
    handles
}

/// Spawn one applier for `(collection, shard_id)`. Public so
/// `collections::create_collection_impl` can spawn one after a
/// new collection materialises (without restarting the
/// gateway).
pub fn spawn_one(
    state: AppState,
    collection: String,
    shard_id: u32,
    tick: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let log = match ensure_shard_log_for(&state, &collection, shard_id).await {
            Some(l) => l,
            None => {
                tracing::warn!(
                    collection = %collection,
                    shard = shard_id,
                    "log_applier: failed to open ShardLog; not spawning"
                );
                return;
            }
        };
        let mut tailer = LogTailer::new(log, 1);
        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            match tailer.next_batch() {
                Ok(batch) if batch.is_empty() => {}
                Ok(batch) => {
                    if let Err(e) = apply_batch(&state, &collection, shard_id, batch).await {
                        tracing::warn!(
                            collection = %collection,
                            shard = shard_id,
                            error = %e,
                            "log_applier: apply_batch failed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        collection = %collection,
                        shard = shard_id,
                        error = %e,
                        "log_applier: tailer next_batch failed"
                    );
                }
            }
        }
    })
}

async fn apply_batch(
    state: &AppState,
    collection: &str,
    shard_id: u32,
    batch: Vec<LogEntryV2>,
) -> Result<(), String> {
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers
        .get_mut(collection)
        .ok_or_else(|| format!("collection '{collection}' not in segment_managers"))?;
    let sm = shard_map
        .get_mut(&shard_id)
        .ok_or_else(|| format!("shard {shard_id} not in '{collection}'"))?;
    for entry in batch {
        match entry.op {
            LogOp::Put => {
                let record: Record = serde_json::from_slice(&entry.payload)
                    .map_err(|e| format!("decode record @ seq={}: {e}", entry.seq_id))?;
                if sm.has_mvcc() {
                    let commit_ts =
                        ((entry.hlc_ts.physical_ns as u128) << 32) | (entry.hlc_ts.logical as u128);
                    sm.insert_mvcc(commit_ts, record)
                        .map_err(|e| format!("insert_mvcc: {e}"))?;
                } else {
                    sm.insert(record).map_err(|e| format!("insert: {e}"))?;
                }
            }
            LogOp::Delete => {
                let rid = uuid_from_key(&entry.key)?;
                if sm.has_mvcc() {
                    let commit_ts =
                        ((entry.hlc_ts.physical_ns as u128) << 32) | (entry.hlc_ts.logical as u128);
                    sm.delete_mvcc(commit_ts, &rid)
                        .map_err(|e| format!("delete_mvcc: {e}"))?;
                } else {
                    sm.delete_record(rid)
                        .map_err(|e| format!("delete_record: {e}"))?;
                }
            }
            other => {
                tracing::warn!(
                    op = ?other,
                    seq_id = entry.seq_id,
                    "log_applier: skipping unrecognised op"
                );
            }
        }
    }
    Ok(())
}

fn uuid_from_key(key: &Bytes) -> Result<Uuid, String> {
    if key.len() != 16 {
        return Err(format!("delete key len={}, expected 16", key.len()));
    }
    let mut buf = [0u8; 16];
    buf.copy_from_slice(key);
    Ok(Uuid::from_bytes(buf))
}
