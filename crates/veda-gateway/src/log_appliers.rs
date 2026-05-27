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

use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use tokio::task::JoinHandle;
use uuid::Uuid;
use veda_core::record::Record;
use veda_log::format::{LogEntryV2, LogOp};
use veda_log::reader::LogTailer;
use veda_storage::segment_manager::SegmentManager;

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
        let mut applier = match ShardApplier::open(&state, &collection, shard_id).await {
            Some(a) => a,
            None => {
                tracing::warn!(
                    collection = %collection,
                    shard = shard_id,
                    "log_applier: open failed; not spawning"
                );
                return;
            }
        };
        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            match applier.tick(&state, &collection, shard_id).await {
                Ok(0) => {}
                Ok(n) => tracing::debug!(applied = n, next = applier.next_seq(), "log_applier tick"),
                Err(e) => tracing::warn!(error = %e, "log_applier tick failed"),
            }
        }
    })
}

/// Apply a batch to `sm`. Tracks which entries triggered seals
/// to compute the highest seq that is durably on disk after the batch.
///
/// Returns the highest seq_id now durably in a sealed segment
/// (0 if no seal occurred during this batch).
///
/// **Seal boundary rule:** when `sm.insert(entry_N)` triggers a seal,
/// the records that were in the old growing segment (all entries applied
/// before entry_N) are now durable. Entry_N itself goes into the NEW
/// growing segment (not yet durable). So the durable cursor advances
/// to the seq of the entry immediately before entry_N.
fn apply_entries(sm: &mut SegmentManager, batch: &[LogEntryV2]) -> Result<u64, String> {
    let mut max_durable_seq = 0u64;
    let mut prev_seq = 0u64;

    for entry in batch {
        let sealed_before = sm.sealed_segments().len();

        match &entry.op {
            LogOp::Put => {
                let record: Record = serde_json::from_slice(&entry.payload)
                    .map_err(|e| format!("decode record @ seq={}: {e}", entry.seq_id))?;
                if sm.has_mvcc() {
                    let commit_ts = ((entry.hlc_ts.physical_ns as u128) << 32)
                        | (entry.hlc_ts.logical as u128);
                    sm.insert_mvcc(commit_ts, record)
                        .map_err(|e| format!("insert_mvcc: {e}"))?;
                } else {
                    sm.insert(record).map_err(|e| format!("insert: {e}"))?;
                }
            }
            LogOp::Delete => {
                let rid = uuid_from_key(&entry.key)?;
                if sm.has_mvcc() {
                    let commit_ts = ((entry.hlc_ts.physical_ns as u128) << 32)
                        | (entry.hlc_ts.logical as u128);
                    sm.delete_mvcc(commit_ts, &rid)
                        .map_err(|e| format!("delete_mvcc: {e}"))?;
                } else {
                    sm.delete_record(rid)
                        .map_err(|e| format!("delete_record: {e}"))?;
                }
            }
            other_op => {
                tracing::warn!(
                    op = ?other_op,
                    seq_id = entry.seq_id,
                    "log_applier: skipping unrecognised op"
                );
            }
        }

        // If a seal occurred during this entry's apply, all entries
        // prior to this one (in the now-sealed old growing segment)
        // became durable. `prev_seq` is their highest seq.
        if sm.sealed_segments().len() > sealed_before {
            max_durable_seq = max_durable_seq.max(prev_seq);
        }

        prev_seq = entry.seq_id;
    }

    Ok(max_durable_seq)
}

async fn apply_batch(
    state: &AppState,
    collection: &str,
    shard_id: u32,
    batch: Vec<LogEntryV2>,
) -> Result<u64, String> {
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers
        .get_mut(collection)
        .ok_or_else(|| format!("collection '{collection}' not in segment_managers"))?;
    let sm = shard_map
        .get_mut(&shard_id)
        .ok_or_else(|| format!("shard {shard_id} not in '{collection}'"))?;
    apply_entries(sm, &batch)
}

/// Per-shard applier: tails the shard log and persists a durable
/// cursor at the sealed watermark. One `tick` drains whatever the
/// tailer can give right now.
pub struct ShardApplier {
    tailer: LogTailer,
    cursor_path: PathBuf,
    /// Highest seq_id that is known durable (persisted in a sealed
    /// segment). Written atomically to `applied.cursor` whenever
    /// it advances. Safe to replay entries above this on restart
    /// because apply is idempotent on `(key, commit_ts)`.
    durable_cursor: u64,
}

impl ShardApplier {
    /// Open for `(collection, shard)`: read the persisted cursor
    /// and resume tailing from `cursor + 1`. None if the shard log
    /// cannot be opened.
    pub async fn open(state: &AppState, collection: &str, shard_id: u32) -> Option<Self> {
        let log = ensure_shard_log_for(state, collection, shard_id).await?;
        let cursor_path = state
            .inner
            .shard_logs
            .shard_dir(collection, shard_id)
            .join("applied.cursor");
        let start = veda_log::cursor::read_cursor(&cursor_path);
        Some(Self {
            tailer: LogTailer::new(log, start + 1),
            cursor_path,
            durable_cursor: start,
        })
    }

    /// Next seq_id this applier will attempt to read.
    pub fn next_seq(&self) -> u64 {
        self.durable_cursor + 1
    }

    /// Drain one batch into the SegmentManager; persist the cursor
    /// whenever new entries have been sealed to disk.
    /// Returns the number of entries applied.
    pub async fn tick(
        &mut self,
        state: &AppState,
        collection: &str,
        shard_id: u32,
    ) -> Result<usize, String> {
        let batch = self
            .tailer
            .next_batch()
            .map_err(|e| format!("tailer next_batch: {e}"))?;
        if batch.is_empty() {
            return Ok(0);
        }
        let n = batch.len();
        let max_durable_seq = apply_batch(state, collection, shard_id, batch).await?;
        if max_durable_seq > self.durable_cursor {
            match veda_log::cursor::write_cursor(&self.cursor_path, max_durable_seq) {
                Ok(()) => self.durable_cursor = max_durable_seq,
                Err(e) => tracing::warn!(error = %e, "log_applier: cursor persist failed"),
            }
        }
        Ok(n)
    }
}

fn uuid_from_key(key: &Bytes) -> Result<Uuid, String> {
    if key.len() != 16 {
        return Err(format!("delete key len={}, expected 16", key.len()));
    }
    let mut buf = [0u8; 16];
    buf.copy_from_slice(key);
    Ok(Uuid::from_bytes(buf))
}

#[cfg(test)]
mod tests {
    //! #2 - durable applier cursor: resume-after-seal + replay-unsealed.

    use bytes::Bytes;
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::DistanceMetric;
    use veda_core::vector::DenseVector;
    use veda_log::format::LogOp;
    use veda_time::Hlc;

    use super::ShardApplier;
    use crate::rest::state::AppState;

    fn schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
            .shard_count(1)
            .build()
            .unwrap()
    }

    fn record() -> Record {
        let mut r = Record::new();
        r.set("v", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0])));
        r
    }

    async fn register_sm(state: &AppState, collection: &str, seal_bytes: usize) {
        let sm = veda_storage::segment_manager::SegmentManager::new(
            schema(collection),
            0,
            &state.inner.data_dir.join(collection),
            seal_bytes,
        )
        .unwrap();
        let mut managers = state.inner.segment_managers.write().await;
        managers.entry(collection.to_owned()).or_default().insert(0, sm);
    }

    async fn append_n(state: &AppState, collection: &str, n: u64) {
        let log = state.inner.shard_logs.get_or_open(collection, 0).unwrap();
        for i in 1..=n {
            let r = record();
            let payload = Bytes::from(serde_json::to_vec(&r).unwrap());
            let key = Bytes::copy_from_slice(r.id().as_bytes());
            log.append_one(Hlc::new(i, 0), 1, key, LogOp::Put, payload).unwrap();
        }
    }

    #[tokio::test]
    async fn resumes_from_persisted_cursor_after_seal() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        register_sm(&state, "coll", 256).await; // tiny -> seals quickly
        append_n(&state, "coll", 20).await;

        let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
        for _ in 0..20 {
            applier.tick(&state, "coll", 0).await.unwrap();
        }

        let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
        let persisted = veda_log::cursor::read_cursor(&cursor_path);
        assert!(persisted > 0, "a seal should have advanced the durable cursor");

        let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
        assert_eq!(applier2.next_seq(), persisted + 1, "resume from cursor+1, not 1");
    }

    #[tokio::test]
    async fn replays_unsealed_entries_after_crash() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        register_sm(&state, "coll", 64 * 1024 * 1024).await; // huge -> never seals
        append_n(&state, "coll", 5).await;

        let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
        let applied = applier.tick(&state, "coll", 0).await.unwrap();
        assert_eq!(applied, 5);

        let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
        assert_eq!(veda_log::cursor::read_cursor(&cursor_path), 0, "no seal -> cursor stays 0");

        let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
        assert_eq!(applier2.next_seq(), 1, "fresh applier replays unsealed entries from 1");
    }
}
