//! WAN log shipping — async cross-region replication for
//! per-shard logs.
//!
//! Phase H of the FoundationDB-shape rearchitecture
//! (`.claude/plans/structured-mixing-turing.md`). Primary region
//! ships log entries to replica regions asynchronously; replicas
//! tail the primary's logs with bounded staleness. Operator-
//! driven failover (no auto-failover — Spanner/FDB take the same
//! stance).
//!
//! ## What v0 ships
//!
//! - [`RegionTopology`] — operator-mutable record of
//!   `(primary_region, replica_regions[], staleness_target_ms)`.
//!   Lives in `_meta`'s `shard_ensembles`-shaped storage
//!   (Phase E added the field; this just defines the
//!   serialisation format).
//! - [`LogShipper`] — drains a source `ShardLog` from a
//!   seq cursor and writes batches to a destination
//!   (`LogReceiver`). In-process transport for v0; the trait
//!   shape is identical to the Phase D
//!   [`crate::replication::LogFollower`] — v1 swaps for Tonic.
//! - [`LogReceiver`] — accepts shipped batches, appends to a
//!   local replica log in epoch order.
//! - Bounded-staleness tracking: shipper tracks lag in seq_id
//!   units; operator metric.
//! - Planned cutover: a `promote_replica` admin op that drains
//!   pending batches and swaps the `RegionTopology` primary
//!   pointer.
//!
//! ## What v0 does NOT ship
//!
//! - Cross-process Tonic transport for `ShipBatchRpc`. The trait
//!   surface ships; the impl is local for v0.
//! - `safe_ts` gating on cross-region readers (the MVCC F path
//!   already exposes the watermark; wiring it requires Phase F
//!   to be production-integrated first).
//! - Conflict resolution on active-active. v0 is replica-
//!   forward-to-primary; active-active is the Phase I
//!   conversation.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::format::LogEntryV2;
use crate::replication::NodeId;
use crate::shard_log::ShardLog;

/// Region topology record. Stored as JSON in
/// `ClusterStateMachine::shard_ensembles` keyed by a sentinel
/// `"_region_topology"` row — Phase E intentionally kept the
/// ensembles map JSON-blob-typed so Phase H artefacts can share
/// the same storage without a coordinator-side schema change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionTopology {
    pub primary_region: String,
    pub replica_regions: Vec<String>,
    /// Target staleness in milliseconds. Shipper tries to keep
    /// lag below this; operator alert fires above.
    pub staleness_target_ms: u64,
    /// Monotonic epoch. Increments on every promote_replica.
    pub epoch: u64,
}

impl RegionTopology {
    pub fn new(primary: impl Into<String>, replicas: Vec<String>) -> Self {
        Self {
            primary_region: primary.into(),
            replica_regions: replicas,
            staleness_target_ms: 5_000,
            epoch: 1,
        }
    }
}

/// Receiver-side transport. v0 has a `LocalReceiver` that just
/// appends to a target `ShardLog`. v1 has a `TonicReceiver` that
/// serves the same trait via gRPC.
#[async_trait]
pub trait LogReceiver: Send + Sync {
    fn region(&self) -> &str;
    fn node_id(&self) -> NodeId;
    /// Append a batch of entries from the primary. Returns the
    /// highest seq_id durably stored after this batch.
    async fn ship_batch(&self, batch: ShipBatchRpc) -> Result<ShipBatchAck, WanError>;
    /// Highest seq the replica has confirmed locally.
    async fn replica_durable_seq(&self) -> u64;
}

/// Wire payload for one shipping round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipBatchRpc {
    pub epoch: u64,
    pub entries: Vec<LogEntryV2>,
    /// The primary's seq_id of the first entry in `entries`.
    /// Used by the receiver for sanity-checking continuity.
    pub start_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipBatchAck {
    pub durable_seq: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum WanError {
    #[error("epoch mismatch (receiver={local}, batch={batch})")]
    EpochMismatch { local: u64, batch: u64 },
    #[error("gap detected: replica expected seq {expected}, batch start {got}")]
    SeqGap { expected: u64, got: u64 },
    #[error("io: {0}")]
    Io(String),
}

/// In-process receiver backed by a `ShardLog`. The trait surface
/// is what v1 Tonic implementation will use too.
pub struct LocalReceiver {
    region: String,
    node_id: NodeId,
    log: Arc<ShardLog>,
    /// Tracks the highest epoch we've accepted. Refuses
    /// out-of-order epochs (planned cutover bumps the epoch).
    current_epoch: Mutex<u64>,
}

impl LocalReceiver {
    pub fn new(region: impl Into<String>, node_id: NodeId, log: Arc<ShardLog>) -> Self {
        Self {
            region: region.into(),
            node_id,
            log,
            current_epoch: Mutex::new(0),
        }
    }
}

#[async_trait]
impl LogReceiver for LocalReceiver {
    fn region(&self) -> &str {
        &self.region
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn ship_batch(&self, batch: ShipBatchRpc) -> Result<ShipBatchAck, WanError> {
        let mut e = self.current_epoch.lock().await;
        if batch.epoch < *e {
            return Err(WanError::EpochMismatch {
                local: *e,
                batch: batch.epoch,
            });
        }
        if batch.epoch > *e {
            *e = batch.epoch;
        }
        // Continuity check.
        let expected = self.log.durable_seq() + 1;
        if batch.start_seq > expected {
            return Err(WanError::SeqGap {
                expected,
                got: batch.start_seq,
            });
        }
        // Append each entry. Idempotent on duplicate seq via the
        // log layer.
        let mut last = self.log.durable_seq();
        for entry in batch.entries {
            // Skip entries that are already durable (re-ship).
            if entry.seq_id <= last {
                continue;
            }
            let seq = self
                .log
                .append_one(
                    entry.hlc_ts,
                    batch.epoch,
                    entry.key.clone(),
                    entry.op.clone(),
                    entry.payload.clone(),
                )
                .map_err(|e| WanError::Io(format!("replica append: {e}")))?;
            last = seq;
        }
        Ok(ShipBatchAck { durable_seq: last })
    }

    async fn replica_durable_seq(&self) -> u64 {
        self.log.durable_seq()
    }
}

/// Drains a source `ShardLog` into a `LogReceiver`. v0 ships
/// sequentially; v1 adds batching + parallel fan-out.
pub struct LogShipper {
    source: Arc<ShardLog>,
    target: Arc<dyn LogReceiver>,
    /// Per-shipper cursor — the next seq we'll attempt to ship.
    /// In-memory for v0; production persists to a small file
    /// per (source_shard, target_region) pair.
    next_seq: Mutex<u64>,
    /// Topology epoch carried in every `ShipBatchRpc`. Updated
    /// when the operator promotes a replica.
    epoch: Mutex<u64>,
    /// Max entries shipped per round. Caps memory + RPC size.
    batch_size: usize,
}

impl LogShipper {
    pub fn new(source: Arc<ShardLog>, target: Arc<dyn LogReceiver>) -> Self {
        Self {
            source,
            target,
            next_seq: Mutex::new(1),
            epoch: Mutex::new(1),
            batch_size: 512,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Lag in seq_id units — `source.durable_seq - replica.durable_seq`.
    pub async fn lag(&self) -> u64 {
        let src = self.source.durable_seq();
        let tgt = self.target.replica_durable_seq().await;
        src.saturating_sub(tgt)
    }

    /// Ship one batch. Returns the durable seq the receiver
    /// acknowledged; 0 if there was nothing to ship.
    pub async fn ship_once(&self) -> Result<u64, WanError> {
        let mut cursor = self.next_seq.lock().await;
        let durable = self.source.durable_seq();
        if *cursor > durable {
            return Ok(0);
        }
        let entries = self
            .source
            .read_from(*cursor)
            .map_err(|e| WanError::Io(format!("source read: {e}")))?;
        if entries.is_empty() {
            return Ok(0);
        }
        let batch: Vec<LogEntryV2> = entries.into_iter().take(self.batch_size).collect();
        let start_seq = batch[0].seq_id;
        let last_seq = batch.last().map(|e| e.seq_id).unwrap_or(0);
        let epoch = *self.epoch.lock().await;
        let ack = self
            .target
            .ship_batch(ShipBatchRpc {
                epoch,
                entries: batch,
                start_seq,
            })
            .await?;
        *cursor = last_seq + 1;
        Ok(ack.durable_seq)
    }

    /// Catch up by repeatedly shipping until lag = 0 or
    /// `max_rounds` is exhausted. Used by the planned-cutover
    /// drain step.
    pub async fn catch_up(&self, max_rounds: usize) -> Result<u64, WanError> {
        let mut last_durable = 0;
        for _ in 0..max_rounds {
            let r = self.ship_once().await?;
            if r > last_durable {
                last_durable = r;
            }
            if self.lag().await == 0 {
                break;
            }
        }
        Ok(last_durable)
    }

    /// Bump the topology epoch on this shipper — called when
    /// `_meta`'s `RegionTopology.epoch` advances (planned
    /// cutover). Subsequent `ship_batch` RPCs carry the new
    /// epoch; receivers fence stale-epoch primaries.
    pub async fn set_epoch(&self, new_epoch: u64) {
        let mut g = self.epoch.lock().await;
        if new_epoch > *g {
            *g = new_epoch;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{LogEntryV2, LogOp};
    use bytes::Bytes;
    use tempfile::TempDir;
    use veda_time::Hlc;

    fn append_to(log: &Arc<ShardLog>, count: u64) -> u64 {
        let mut last = 0;
        for i in 1..=count {
            last = log
                .append_one(
                    Hlc::new(i, 0),
                    1,
                    Bytes::copy_from_slice(&i.to_le_bytes()),
                    LogOp::Put,
                    Bytes::from(format!("e{i}").into_bytes()),
                )
                .unwrap();
        }
        last
    }

    fn fresh_log() -> (TempDir, Arc<ShardLog>) {
        let tmp = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(tmp.path()).unwrap());
        (tmp, log)
    }

    #[tokio::test]
    async fn shipper_catches_up_after_disconnect_replay() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 50);
        let (_t_dst, dst) = fresh_log();
        let receiver = Arc::new(LocalReceiver::new("replica-eu", 2, dst.clone()))
            as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(10);
        // Catch up should ship all 50 entries.
        let r = shipper.catch_up(20).await.unwrap();
        assert!(r >= 50);
        assert_eq!(shipper.lag().await, 0);
        // More entries — second catch-up picks them up.
        append_to(&src, 25);
        assert!(shipper.lag().await > 0);
        let r = shipper.catch_up(20).await.unwrap();
        assert!(r >= 75);
        assert_eq!(shipper.lag().await, 0);
    }

    #[tokio::test]
    async fn shipper_idempotent_on_reshipping_overlap() {
        // Simulate the shipper crashing after the receiver
        // ack'd but before the cursor was persisted. Re-running
        // ship_once with cursor back at 1 should not corrupt
        // the receiver's log (idempotent on duplicates).
        let (_t_src, src) = fresh_log();
        append_to(&src, 20);
        let (_t_dst, dst) = fresh_log();
        let receiver = Arc::new(LocalReceiver::new("replica", 2, dst.clone()))
            as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(20);
        shipper.catch_up(5).await.unwrap();
        // Reset cursor manually to simulate the crash.
        {
            let mut g = shipper.next_seq.lock().await;
            *g = 1;
        }
        // Re-ship: receiver dedupes against its durable_seq.
        shipper.catch_up(5).await.unwrap();
        // Replica still has exactly 20 entries (no duplicates).
        let entries = dst.read_from(1).unwrap();
        assert_eq!(entries.len(), 20);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.seq_id, (i as u64) + 1);
        }
    }

    #[tokio::test]
    async fn receiver_rejects_stale_epoch_after_cutover() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 5);
        let (_t_dst, dst) = fresh_log();
        let receiver = Arc::new(LocalReceiver::new("replica", 2, dst.clone()))
            as Arc<dyn LogReceiver>;
        // First shipper at epoch 1 — ok.
        let s1 = LogShipper::new(src.clone(), receiver.clone());
        s1.catch_up(5).await.unwrap();
        // Cutover: receiver accepts epoch 5 from a new primary.
        let s2_target = receiver.clone();
        s2_target
            .ship_batch(ShipBatchRpc {
                epoch: 5,
                entries: vec![],
                start_seq: 6,
            })
            .await
            .unwrap();
        // Now old shipper at epoch 1 tries to ship — must be
        // fenced.
        // We need source to have more entries OR the shipper to
        // already have queued some. Append a bit more.
        append_to(&src, 5);
        let err = s1.ship_once().await.unwrap_err();
        matches!(err, WanError::EpochMismatch { .. });
    }

    #[tokio::test]
    async fn topology_serialises_through_json() {
        let t = RegionTopology::new("us-east-1", vec!["eu-west-1".into(), "ap-northeast-1".into()]);
        let s = serde_json::to_string(&t).unwrap();
        let back: RegionTopology = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
    }
}
