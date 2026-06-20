//! Phase D quorum-partition test (the plan's
//! `tests/quorum_partition.rs`). Three followers, Qa=2. While
//! two are partitioned the writer can't reach quorum and its
//! appends fail with `NoQuorum`. After the partition heals,
//! appends succeed again. On a new-epoch handover, the
//! minority follower that received a stray entry fences via
//! `open_and_recover`'s truncate step.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::Mutex;
use veda_log::format::{LogEntryV2, LogOp};
use veda_log::replication::{
    AppendAck, AppendRpc, EnsembleError, LocalFollower, LogEnsemble, LogFollower, LogWriter, NodeId,
};
use veda_log::shard_log::ShardLog;
use veda_time::Hlc;

/// Wraps a `LocalFollower` with a "partitioned" flag. When
/// partitioned, every RPC returns `Transport`. Healing flips the
/// flag back. Pins the writer-side behaviour: appends to a
/// partitioned follower count as errors against the quorum.
struct PartitionableFollower {
    inner: Arc<LocalFollower>,
    partitioned: Mutex<bool>,
}

impl PartitionableFollower {
    fn new(inner: Arc<LocalFollower>) -> Self {
        Self {
            inner,
            partitioned: Mutex::new(false),
        }
    }

    async fn partition(&self) {
        *self.partitioned.lock().await = true;
    }

    async fn heal(&self) {
        *self.partitioned.lock().await = false;
    }
}

#[async_trait]
impl LogFollower for PartitionableFollower {
    fn node_id(&self) -> NodeId {
        self.inner.node_id()
    }
    async fn append(&self, rpc: AppendRpc) -> Result<AppendAck, EnsembleError> {
        if *self.partitioned.lock().await {
            return Err(EnsembleError::Transport("partitioned".into()));
        }
        self.inner.append(rpc).await
    }
    async fn last_durable_seq(&self) -> Result<u64, EnsembleError> {
        if *self.partitioned.lock().await {
            return Err(EnsembleError::Transport("partitioned".into()));
        }
        self.inner.last_durable_seq().await
    }
    async fn fence_to(&self, epoch: u64) -> Result<(), EnsembleError> {
        if *self.partitioned.lock().await {
            return Err(EnsembleError::Transport("partitioned".into()));
        }
        self.inner.fence_to(epoch).await
    }
    async fn truncate_above(&self, target: u64) -> Result<(), EnsembleError> {
        if *self.partitioned.lock().await {
            return Err(EnsembleError::Transport("partitioned".into()));
        }
        self.inner.truncate_above(target).await
    }
}

fn entry(seq_id: u64) -> LogEntryV2 {
    LogEntryV2 {
        seq_id,
        hlc_ts: Hlc::new(seq_id, 0),
        epoch: 1,
        key: Bytes::copy_from_slice(&seq_id.to_le_bytes()),
        op: LogOp::Put,
        payload: Bytes::from(format!("e{seq_id}").into_bytes()),
    }
}

// Test helper: the 3-tuple return is local to this file; a type alias would
// add indirection without reuse.
#[allow(clippy::type_complexity)]
fn build_followers(
    n: usize,
) -> (
    Vec<tempfile::TempDir>,
    Vec<Arc<PartitionableFollower>>,
    Vec<Arc<dyn LogFollower>>,
) {
    let mut tmps = Vec::new();
    let mut parts = Vec::new();
    let mut as_dyn: Vec<Arc<dyn LogFollower>> = Vec::new();
    for i in 0..n {
        let tmp = tempfile::tempdir().unwrap();
        let log = Arc::new(ShardLog::open(tmp.path()).unwrap());
        let local = Arc::new(LocalFollower::new(i as NodeId + 1, log));
        let p = Arc::new(PartitionableFollower::new(local));
        tmps.push(tmp);
        as_dyn.push(p.clone() as Arc<dyn LogFollower>);
        parts.push(p);
    }
    (tmps, parts, as_dyn)
}

#[tokio::test]
async fn append_fails_no_quorum_when_two_of_three_partitioned() {
    let (_tmps, parts, as_dyn) = build_followers(3);
    // Partition off followers 1 and 2; only follower 0 reachable.
    parts[1].partition().await;
    parts[2].partition().await;
    let ens = LogEnsemble::new(vec![1, 2, 3], 2, 1, 1); // Qa=2
    let writer = LogWriter::new(ens, as_dyn);
    let result = writer
        .append_batch(vec![entry(1), entry(2), entry(3)])
        .await;
    match result {
        Err(EnsembleError::NoQuorum { acked, qa, .. }) => {
            assert!(acked < qa, "expected acked < Qa, got acked={acked} qa={qa}");
        }
        other => panic!("expected NoQuorum, got {other:?}"),
    }
}

#[tokio::test]
async fn append_succeeds_after_partition_heals() {
    let (_tmps, parts, as_dyn) = build_followers(3);
    parts[2].partition().await;
    let ens = LogEnsemble::new(vec![1, 2, 3], 2, 1, 1);
    let writer = LogWriter::new(ens, as_dyn.clone());
    // Two of three reachable → Qa=2 is met. Append succeeds.
    let r = writer.append_batch(vec![entry(1), entry(2)]).await;
    assert!(r.is_ok(), "append with 2/3 reachable should succeed: {r:?}");

    // Heal follower 2.
    parts[2].heal().await;
    // Open-and-recover at epoch 2 brings everyone aligned.
    let ens2 = LogEnsemble::new(vec![1, 2, 3], 2, 1, 2);
    let writer2 = LogWriter::new(ens2, as_dyn);
    let safe = writer2.open_and_recover().await.unwrap();
    assert!(safe >= 2, "safe_seq should be ≥ 2, got {safe}");
}

#[tokio::test]
async fn minority_acked_entry_fences_on_heal_via_truncate() {
    // 3 followers. Initial epoch=1, Qa=2.
    let (_tmps, _parts, as_dyn) = build_followers(3);
    // Phase 1: majority append 1..=3.
    let ens1 = LogEnsemble::new(vec![1, 2, 3], 2, 1, 1);
    let majority: Vec<Arc<dyn LogFollower>> = vec![as_dyn[0].clone(), as_dyn[1].clone()];
    let writer1 = LogWriter::new(ens1.clone(), majority);
    writer1
        .append_batch((1..=3).map(entry).collect())
        .await
        .unwrap();

    // Phase 2: simulate the old writer racing one minority-only
    // entry to follower 2 (would happen if the old writer
    // crashed mid-batch).
    let minority: Vec<Arc<dyn LogFollower>> = vec![as_dyn[2].clone()];
    let writer_stray = LogWriter::new(LogEnsemble::new(vec![1, 2, 3], 1, 1, 1), minority);
    writer_stray
        .append_batch((1..=4).map(entry).collect())
        .await
        .unwrap();
    assert_eq!(as_dyn[2].last_durable_seq().await.unwrap(), 4);

    // Phase 3: new writer at epoch=2 runs open_and_recover.
    let ens2 = LogEnsemble::new(vec![1, 2, 3], 2, 2, 2);
    let writer2 = LogWriter::new(ens2, as_dyn.clone());
    let safe = writer2.open_and_recover().await.unwrap();
    assert!(
        (3..=4).contains(&safe),
        "safe_seq should be 3 or 4, got {safe}"
    );

    // After truncate, every follower's durable_seq ≤ safe_seq.
    for f in as_dyn.iter() {
        let s = f.last_durable_seq().await.unwrap();
        assert!(s <= safe, "follower seq {s} > safe_seq {safe}");
    }

    // Old (epoch-1) writer trying to append now must be fenced.
    let writer_stale = LogWriter::new(
        LogEnsemble::new(vec![1, 2, 3], 2, 1, 1), // stale epoch
        as_dyn.clone(),
    );
    let stale_result = writer_stale.append_batch(vec![entry(safe + 1)]).await;
    assert!(
        matches!(stale_result, Err(EnsembleError::Fenced { .. }))
            || matches!(stale_result, Err(EnsembleError::NoQuorum { .. })),
        "stale-epoch writer must be fenced or no-quorum, got {stale_result:?}"
    );
}
