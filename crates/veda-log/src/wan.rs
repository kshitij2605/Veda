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
    /// Highest commit_ts below which all writes are durable at
    /// this replica (HLC, ns units). Used by cross-region readers
    /// to gate snapshot reads at a chosen `read_ts`.
    async fn replica_safe_ts(&self) -> u128;
    /// Returns `Ok(replica_safe_ts)` if `read_ts <= safe_ts`,
    /// else `Err(WanError::ReadTsTooNew)` with the staleness in
    /// HLC units. Cross-region readers call this before serving
    /// a snapshot read.
    async fn check_read_ts(&self, read_ts: u128) -> Result<u128, WanError>;
}

/// Wire payload for one shipping round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipBatchRpc {
    pub epoch: u64,
    pub entries: Vec<LogEntryV2>,
    /// The primary's seq_id of the first entry in `entries`.
    /// Used by the receiver for sanity-checking continuity.
    pub start_seq: u64,
    /// The primary's `safe_ts` (MVCC GC water mark) at the time
    /// the batch was produced. Receivers advance their local
    /// safe_ts to at least this value once the batch is durable;
    /// cross-region readers gate snapshot reads against it.
    #[serde(default)]
    pub primary_safe_ts: u128,
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
    #[error("read_ts {read_ts} ahead of replica safe_ts {safe_ts} by {lag} units")]
    ReadTsTooNew {
        read_ts: u128,
        safe_ts: u128,
        lag: u128,
    },
    #[error("transport: {0}")]
    Transport(String),
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
    /// Replica-side MVCC watermark. Advances when the primary
    /// ships a batch with a higher `primary_safe_ts`, or when
    /// the operator pokes it explicitly (test affordance).
    safe_ts: Mutex<u128>,
}

impl LocalReceiver {
    pub fn new(region: impl Into<String>, node_id: NodeId, log: Arc<ShardLog>) -> Self {
        Self {
            region: region.into(),
            node_id,
            log,
            current_epoch: Mutex::new(0),
            safe_ts: Mutex::new(0),
        }
    }

    /// Manually advance `safe_ts` — primarily for tests / admin.
    /// Production callers set this via `ShipBatchRpc.primary_safe_ts`.
    pub async fn set_safe_ts(&self, ts: u128) -> bool {
        let mut g = self.safe_ts.lock().await;
        if ts > *g {
            *g = ts;
            true
        } else {
            false
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
        // Advance the safe_ts watermark with the primary's value
        // once entries are durable.
        if batch.primary_safe_ts > 0 {
            let mut g = self.safe_ts.lock().await;
            if batch.primary_safe_ts > *g {
                *g = batch.primary_safe_ts;
            }
        }
        Ok(ShipBatchAck { durable_seq: last })
    }

    async fn replica_durable_seq(&self) -> u64 {
        self.log.durable_seq()
    }

    async fn replica_safe_ts(&self) -> u128 {
        *self.safe_ts.lock().await
    }

    async fn check_read_ts(&self, read_ts: u128) -> Result<u128, WanError> {
        let safe = *self.safe_ts.lock().await;
        if read_ts <= safe {
            Ok(safe)
        } else {
            Err(WanError::ReadTsTooNew {
                read_ts,
                safe_ts: safe,
                lag: read_ts - safe,
            })
        }
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
    /// Primary-side `safe_ts` water mark stamped on every
    /// `ShipBatchRpc`. Operator/MVCC GC advances this so the
    /// replica's safe_ts trails the primary's.
    safe_ts: Mutex<u128>,
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
            safe_ts: Mutex::new(0),
            batch_size: 512,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Advance the primary's published safe_ts — subsequent
    /// `ship_batch` RPCs carry this value.
    pub async fn set_safe_ts(&self, ts: u128) {
        let mut g = self.safe_ts.lock().await;
        if ts > *g {
            *g = ts;
        }
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
        let primary_safe_ts = *self.safe_ts.lock().await;
        let ack = self
            .target
            .ship_batch(ShipBatchRpc {
                epoch,
                entries: batch,
                start_seq,
                primary_safe_ts,
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

    /// Inspect the shipper's current topology epoch (read-only).
    pub async fn epoch(&self) -> u64 {
        *self.epoch.lock().await
    }
}

/// Outcome of a [`promote_replica`] call.
#[derive(Debug, Clone)]
pub struct CutoverReport {
    /// `durable_seq` at the replica after the final drain. All
    /// log entries with `seq_id <= drained_seq` are now durable
    /// on the new primary side.
    pub drained_seq: u64,
    /// Replica's `safe_ts` watermark at the end of drain. Used
    /// by cross-region readers to verify snapshot-read freshness.
    pub safe_ts: u128,
    /// The new topology epoch the shipper now stamps on every
    /// outgoing `ship_batch`.
    pub new_epoch: u64,
}

/// Operator-driven planned cutover. Drains all pending batches
/// from `shipper` to its target (the receiver about to become
/// the new primary), then bumps the shipper's epoch so any
/// subsequent appends from the old primary fence.
///
/// Strict ordering of side effects:
///   1. Drain pending — `shipper.catch_up(max_drain_rounds)`.
///   2. Bump epoch — `shipper.set_epoch(new_epoch)`. After this
///      point, the old primary's subsequent ships carry the new
///      epoch (or — if the operator removed the old shipper
///      first — get fenced by the receiver via
///      `EpochMismatch`).
///
/// The operator is responsible for: bumping `RegionTopology.epoch`
/// in `_meta` (so other regions learn the new primary), updating
/// the gateway to route writes to the new primary, and tearing
/// down the old primary's shipper if there is one.
pub async fn promote_replica(
    shipper: &LogShipper,
    new_epoch: u64,
    max_drain_rounds: usize,
) -> Result<CutoverReport, WanError> {
    let drained_seq = shipper.catch_up(max_drain_rounds).await?;
    if shipper.lag().await > 0 {
        return Err(WanError::Io(format!(
            "drain incomplete after {max_drain_rounds} rounds; lag={}",
            shipper.lag().await
        )));
    }
    shipper.set_epoch(new_epoch).await;
    let safe_ts = shipper.target.replica_safe_ts().await;
    Ok(CutoverReport {
        drained_seq,
        safe_ts,
        new_epoch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::LogOp;
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
        let receiver =
            Arc::new(LocalReceiver::new("replica-eu", 2, dst.clone())) as Arc<dyn LogReceiver>;
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
        let receiver =
            Arc::new(LocalReceiver::new("replica", 2, dst.clone())) as Arc<dyn LogReceiver>;
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
        let receiver =
            Arc::new(LocalReceiver::new("replica", 2, dst.clone())) as Arc<dyn LogReceiver>;
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
                primary_safe_ts: 0,
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
        let t = RegionTopology::new(
            "us-east-1",
            vec!["eu-west-1".into(), "ap-northeast-1".into()],
        );
        let s = serde_json::to_string(&t).unwrap();
        let back: RegionTopology = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
    }

    /// Planned promote_replica: every entry that was acked at
    /// the old primary lands on the new primary before the
    /// epoch bump. Zero acked-write loss.
    #[tokio::test]
    async fn promote_replica_drains_then_bumps_epoch() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 100);
        let (_t_dst, dst) = fresh_log();
        let receiver =
            Arc::new(LocalReceiver::new("replica-eu", 2, dst.clone())) as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(16);
        shipper.set_safe_ts(42_000).await;

        // Before cutover the replica trails.
        assert!(shipper.lag().await > 0);

        let report = promote_replica(&shipper, 7, 50).await.unwrap();
        assert!(report.drained_seq >= 100);
        assert_eq!(report.new_epoch, 7);
        assert_eq!(report.safe_ts, 42_000);
        assert_eq!(shipper.lag().await, 0);
        assert_eq!(shipper.epoch().await, 7);

        // Every appended entry made it to the replica side.
        let replica_entries = dst.read_from(1).unwrap();
        assert_eq!(replica_entries.len(), 100);
    }

    /// promote_replica refuses to bump the epoch if the drain
    /// didn't actually finish (lag > 0). Without this check the
    /// caller could promote a stale replica.
    #[tokio::test]
    async fn promote_replica_refuses_when_drain_incomplete() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 500);
        let (_t_dst, dst) = fresh_log();
        let receiver =
            Arc::new(LocalReceiver::new("replica", 2, dst.clone())) as Arc<dyn LogReceiver>;
        // Tiny batch_size + tiny max_rounds: 1 round * 1 entry
        // = 1 entry shipped, leaving 499 behind.
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(1);
        let err = promote_replica(&shipper, 9, 1).await.unwrap_err();
        match err {
            WanError::Io(msg) => assert!(msg.contains("drain incomplete"), "got: {msg}"),
            other => panic!("expected Io(drain incomplete), got {other:?}"),
        }
        // Epoch must NOT have advanced.
        assert_eq!(shipper.epoch().await, 1);
    }

    /// Long-disconnect lag recovery: ship some, "disconnect"
    /// (just don't ship for a while), append many more, then
    /// reconnect — catch_up drains everything with no gaps.
    #[tokio::test]
    async fn lag_recovery_after_long_disconnect_has_no_gaps() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 30);
        let (_t_dst, dst) = fresh_log();
        let receiver =
            Arc::new(LocalReceiver::new("replica", 2, dst.clone())) as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(8);
        shipper.catch_up(10).await.unwrap();
        assert_eq!(shipper.lag().await, 0);

        // "Disconnect": don't ship. Append a lot more to src.
        append_to(&src, 200);
        assert!(shipper.lag().await >= 200);

        // Reconnect — catch_up brings everything across.
        let final_seq = shipper.catch_up(100).await.unwrap();
        assert!(final_seq >= 230);
        assert_eq!(shipper.lag().await, 0);

        // Replica's log is gap-free seq 1..230.
        let entries = dst.read_from(1).unwrap();
        assert_eq!(entries.len(), 230);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.seq_id, (i as u64) + 1, "gap at index {i}");
        }
    }

    /// Phase H planned_failover invariant: a controlled
    /// cutover via `promote_replica` must surface every
    /// acked-on-old-primary write on the new primary's local
    /// log. Zero acked-write loss — the strict cross-region
    /// durability guarantee operators rely on.
    #[tokio::test]
    async fn planned_failover_with_zero_acked_write_loss() {
        // Old primary log + a replica receiver. We'll ship
        // continuously, then run promote_replica, then verify
        // every acked write is present on the replica.
        let (_t_src, src) = fresh_log();
        let (_t_dst, dst) = fresh_log();
        let receiver =
            Arc::new(LocalReceiver::new("replica-eu", 2, dst.clone())) as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone()).with_batch_size(16);

        // Append in batches with interleaved ship_once calls —
        // simulates steady-state log shipping.
        for batch_start in 0..5 {
            let lo = batch_start * 20 + 1;
            let hi = lo + 19;
            for i in lo..=hi {
                src.append_one(
                    Hlc::new(i, 0),
                    1,
                    Bytes::copy_from_slice(&i.to_le_bytes()),
                    LogOp::Put,
                    Bytes::from(format!("acked-{i}").into_bytes()),
                )
                .unwrap();
            }
            // Ship after each batch — these are the "acked" writes.
            shipper.ship_once().await.unwrap();
        }
        let acked_on_old_primary = src.durable_seq();
        assert_eq!(acked_on_old_primary, 100);

        // Run promote_replica — drains + bumps epoch.
        let report = promote_replica(&shipper, 2, 50).await.unwrap();
        assert_eq!(report.drained_seq, 100);
        assert_eq!(report.new_epoch, 2);

        // Zero-loss invariant: the new primary's local log has
        // EVERY entry the old primary acked.
        assert_eq!(
            dst.durable_seq(),
            acked_on_old_primary,
            "new primary durable_seq must match old primary's acked count"
        );
        let entries = dst.read_from(1).unwrap();
        assert_eq!(entries.len(), 100);
        for (i, e) in entries.iter().enumerate() {
            let expected_seq = (i as u64) + 1;
            assert_eq!(e.seq_id, expected_seq);
            assert_eq!(
                e.payload.as_ref(),
                format!("acked-{expected_seq}").as_bytes(),
                "payload mismatch at seq {expected_seq}"
            );
        }
    }

    /// Phase H planned_failover anti-test: an old-epoch
    /// shipper trying to push more data AFTER the cutover
    /// fences via EpochMismatch on the receiver. Without this,
    /// a stale primary could write past the cutover point and
    /// silently corrupt the new primary's log.
    #[tokio::test]
    async fn old_primary_fences_after_planned_failover() {
        let (_t_src, src) = fresh_log();
        append_to(&src, 30);
        let (_t_dst, dst) = fresh_log();
        let receiver =
            Arc::new(LocalReceiver::new("replica", 2, dst.clone())) as Arc<dyn LogReceiver>;
        let shipper = LogShipper::new(src.clone(), receiver.clone());
        shipper.catch_up(10).await.unwrap();
        // Promote bumps the shipper's epoch; an additional
        // ship at the new epoch is what advances the
        // receiver's `current_epoch` to 2 (the LocalReceiver
        // only updates on receipt of a batch with a higher
        // epoch). Operators trigger this in the post-cutover
        // boot of the new primary.
        promote_replica(&shipper, 2, 10).await.unwrap();
        append_to(&src, 1);
        let _ = shipper.ship_once().await.unwrap();
        // The receiver is now at epoch=2.

        // A separate OLD-epoch shipper (still at epoch 1) tries
        // to ship to the same receiver — must be fenced.
        let stale_shipper = LogShipper::new(src.clone(), receiver.clone());
        append_to(&src, 5);
        let stale_result = stale_shipper.ship_once().await;
        match stale_result {
            Err(WanError::EpochMismatch { .. }) => {}
            other => panic!("expected EpochMismatch from stale primary, got {other:?}"),
        }
    }
}
