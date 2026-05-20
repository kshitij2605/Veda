//! Phase C tailing applier: drains a per-shard `veda-log`
//! `ShardLog` and pushes each new `LogEntryV2` into a local
//! `SegmentManager`.
//!
//! ## Why this exists
//!
//! On the writer node, the gateway's `insert_via_log` path
//! appends to the log AND inserts into the local SegmentManager
//! in the same call (single-node consistency, no replication
//! roundtrip). On every other node the log entry arrives via
//! Phase D / Phase H replication, and the local SegmentManager
//! must be brought up to date by tailing the local log replica
//! — that's what this module does.
//!
//! ## Apply rules
//!
//! - `LogOp::Put`: payload is the bincode-or-JSON encoding of a
//!   [`Record`]; deserialise + `sm.insert` (or `insert_mvcc` if
//!   the SegmentManager has MVCC enabled).
//! - `LogOp::Delete`: key is the record UUID's `as_bytes()`
//!   form; call `sm.delete_record` (or `sm.delete_mvcc` for MVCC
//!   shards).
//! - Unknown ops are skipped with a `tracing::warn` — forward-
//!   compat against future op codes.
//!
//! ## Cursor durability
//!
//! v0 keeps the cursor in memory. The applier exposes
//! [`LogApplier::last_applied`] so the caller can persist it
//! out-of-band (typically alongside the segment metadata so
//! recovery starts from the right seq). Phase F+ ties this into
//! the segment's own checkpoint format.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use uuid::Uuid;
use veda_core::record::Record;
use veda_log::format::{LogEntryV2, LogOp};
use veda_log::reader::LogTailer;
use veda_log::shard_log::ShardLog;

use crate::segment_manager::SegmentManager;

#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    #[error("log read: {0}")]
    LogRead(String),
    #[error("decode record: {0}")]
    Decode(String),
    #[error("segment manager: {0}")]
    Segment(String),
    #[error("invalid delete key (not a UUID, len={0})")]
    InvalidDeleteKey(usize),
}

/// Per-shard applier. One per `(collection, shard_id)` pair.
pub struct LogApplier {
    tailer: Mutex<LogTailer>,
    segment_manager: Arc<Mutex<SegmentManager>>,
    last_applied: std::sync::atomic::AtomicU64,
}

impl LogApplier {
    /// Construct a new applier. `from` is the first seq_id the
    /// applier will attempt to read (defaults to 1 for a fresh
    /// shard; pass `last_persisted_seq + 1` to resume).
    pub fn new(log: Arc<ShardLog>, segment_manager: Arc<Mutex<SegmentManager>>, from: u64) -> Self {
        Self {
            tailer: Mutex::new(LogTailer::new(log, from)),
            segment_manager,
            last_applied: std::sync::atomic::AtomicU64::new(from.saturating_sub(1)),
        }
    }

    /// Highest seq_id this applier has applied to its local
    /// SegmentManager.
    pub fn last_applied(&self) -> u64 {
        self.last_applied
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Drain whatever the tailer can give us right now into the
    /// SegmentManager. Returns the number of entries applied.
    pub async fn apply_once(&self) -> Result<usize, ApplyError> {
        let batch = {
            let mut t = self.tailer.lock().await;
            t.next_batch()
                .map_err(|e| ApplyError::LogRead(format!("{e}")))?
        };
        if batch.is_empty() {
            return Ok(0);
        }
        let n = batch.len();
        let mut sm = self.segment_manager.lock().await;
        for entry in batch {
            self.apply_entry(&mut sm, entry)?;
        }
        Ok(n)
    }

    fn apply_entry(&self, sm: &mut SegmentManager, entry: LogEntryV2) -> Result<(), ApplyError> {
        let commit_ts = hlc_to_u128(&entry.hlc_ts);
        match entry.op {
            LogOp::Put => {
                let record: Record = serde_json::from_slice(&entry.payload)
                    .map_err(|e| ApplyError::Decode(format!("put@{}: {e}", entry.seq_id)))?;
                if sm.has_mvcc() {
                    sm.insert_mvcc(commit_ts, record)
                        .map_err(|e| ApplyError::Segment(format!("insert_mvcc: {e}")))?;
                } else {
                    sm.insert(record)
                        .map_err(|e| ApplyError::Segment(format!("insert: {e}")))?;
                }
            }
            LogOp::Delete => {
                let rid = uuid_from_key(&entry.key)?;
                if sm.has_mvcc() {
                    sm.delete_mvcc(commit_ts, &rid)
                        .map_err(|e| ApplyError::Segment(format!("delete_mvcc: {e}")))?;
                } else {
                    let _ = sm
                        .delete_record(rid)
                        .map_err(|e| ApplyError::Segment(format!("delete_record: {e}")))?;
                }
            }
            other => {
                tracing::warn!(
                    seq_id = entry.seq_id,
                    op = ?other,
                    "log_applier: skipping unrecognised op",
                );
            }
        }
        self.last_applied
            .store(entry.seq_id, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Spawn a background task that periodically calls
    /// `apply_once` every `tick`. Errors are logged and the
    /// loop continues — applier failures don't take the gateway
    /// down. Returns the JoinHandle so callers can shut it down.
    pub fn spawn(self: Arc<Self>, tick: Duration) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                match self.apply_once().await {
                    Ok(0) => {}
                    Ok(n) => {
                        tracing::debug!(applied = n, last_seq = self.last_applied(), "log_applier tick");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "log_applier apply_once failed");
                    }
                }
            }
        })
    }
}

fn hlc_to_u128(hlc: &veda_time::Hlc) -> u128 {
    ((hlc.physical_ns as u128) << 32) | (hlc.logical as u128)
}

fn uuid_from_key(key: &Bytes) -> Result<Uuid, ApplyError> {
    if key.len() != 16 {
        return Err(ApplyError::InvalidDeleteKey(key.len()));
    }
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(key);
    Ok(Uuid::from_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::{DistanceMetric, FieldType};
    use veda_core::vector::DenseVector;
    use veda_time::Hlc;

    fn schema() -> CollectionSchema {
        CollectionSchema::builder("applier_test")
            .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
            .add_field(FieldDefinition::new("label", FieldType::String))
            .shard_count(1)
            .build()
            .unwrap()
    }

    fn record(label: &str) -> Record {
        let mut r = Record::new();
        r.set("v", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0])));
        r.set("label", FieldValue::String(label.to_owned()));
        r
    }

    fn append_put(log: &Arc<ShardLog>, seq_expected: u64, rec: &Record) {
        let payload = Bytes::from(serde_json::to_vec(rec).unwrap());
        let key = Bytes::copy_from_slice(rec.id().as_bytes());
        let actual = log
            .append_one(Hlc::new(seq_expected, 0), 1, key, LogOp::Put, payload)
            .unwrap();
        assert_eq!(actual, seq_expected);
    }

    #[tokio::test]
    async fn applies_put_entries_into_growing_segment() {
        let log_dir = tempfile::tempdir().unwrap();
        let sm_dir = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(log_dir.path()).unwrap());
        let sm = Arc::new(Mutex::new(
            SegmentManager::new(schema(), 0, sm_dir.path(), 512 * 1024).unwrap(),
        ));

        let r1 = record("a");
        let r2 = record("b");
        let r3 = record("c");
        append_put(&log, 1, &r1);
        append_put(&log, 2, &r2);
        append_put(&log, 3, &r3);

        let applier = LogApplier::new(log.clone(), sm.clone(), 1);
        let n = applier.apply_once().await.unwrap();
        assert_eq!(n, 3);
        assert_eq!(applier.last_applied(), 3);
        let g = sm.lock().await;
        assert_eq!(g.growing_segment().count(), 3);
    }

    #[tokio::test]
    async fn apply_once_is_idempotent_when_no_new_entries() {
        let log_dir = tempfile::tempdir().unwrap();
        let sm_dir = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(log_dir.path()).unwrap());
        let sm = Arc::new(Mutex::new(
            SegmentManager::new(schema(), 0, sm_dir.path(), 512 * 1024).unwrap(),
        ));
        append_put(&log, 1, &record("x"));
        let applier = LogApplier::new(log.clone(), sm.clone(), 1);
        assert_eq!(applier.apply_once().await.unwrap(), 1);
        // No more entries — apply_once returns 0 and last_applied
        // stays put.
        assert_eq!(applier.apply_once().await.unwrap(), 0);
        assert_eq!(applier.last_applied(), 1);
    }

    #[tokio::test]
    async fn resume_from_last_persisted_seq_skips_already_applied_entries() {
        let log_dir = tempfile::tempdir().unwrap();
        let sm_dir = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(log_dir.path()).unwrap());
        append_put(&log, 1, &record("a"));
        append_put(&log, 2, &record("b"));
        append_put(&log, 3, &record("c"));

        // Pretend the previous applier persisted seq=2; a fresh
        // applier resumes at seq=3.
        let sm = Arc::new(Mutex::new(
            SegmentManager::new(schema(), 0, sm_dir.path(), 512 * 1024).unwrap(),
        ));
        let applier = LogApplier::new(log.clone(), sm.clone(), 3);
        let n = applier.apply_once().await.unwrap();
        assert_eq!(n, 1);
        assert_eq!(applier.last_applied(), 3);
        let g = sm.lock().await;
        assert_eq!(g.growing_segment().count(), 1);
    }

    #[tokio::test]
    async fn applies_mvcc_when_segment_manager_has_mvcc_on() {
        let log_dir = tempfile::tempdir().unwrap();
        let sm_dir = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(log_dir.path()).unwrap());
        let mut sm_raw = SegmentManager::new(schema(), 0, sm_dir.path(), 512 * 1024).unwrap();
        sm_raw.enable_mvcc();
        let sm = Arc::new(Mutex::new(sm_raw));

        let r = record("mvcc-v1");
        let rid = r.id();
        append_put(&log, 1, &r);

        let applier = LogApplier::new(log.clone(), sm.clone(), 1);
        assert_eq!(applier.apply_once().await.unwrap(), 1);
        // Far-future read sees the row.
        let g = sm.lock().await;
        assert!(g.read_at_ts(&rid, u128::MAX).is_some());
    }

    #[tokio::test]
    async fn delete_op_against_non_mvcc_segment_tombstones_record() {
        let log_dir = tempfile::tempdir().unwrap();
        let sm_dir = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(log_dir.path()).unwrap());
        let sm = Arc::new(Mutex::new(
            SegmentManager::new(schema(), 0, sm_dir.path(), 512 * 1024).unwrap(),
        ));

        let r = record("doomed");
        let rid = r.id();
        append_put(&log, 1, &r);
        // Delete entry: key = rid.as_bytes(), op = Delete.
        log.append_one(
            Hlc::new(2, 0),
            1,
            Bytes::copy_from_slice(rid.as_bytes()),
            LogOp::Delete,
            Bytes::new(),
        )
        .unwrap();

        let applier = LogApplier::new(log.clone(), sm.clone(), 1);
        assert_eq!(applier.apply_once().await.unwrap(), 2);
        let g = sm.lock().await;
        let gs = g.growing_segment();
        // Row was inserted (doc_id 0) then tombstoned.
        assert!(gs.is_deleted(0), "expected tombstone on doc 0 after delete");
    }
}
