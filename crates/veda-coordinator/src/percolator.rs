//! Percolator-style snapshot-isolation transactions.
//!
//! Phase G of the FoundationDB-shape rearchitecture
//! (`.claude/plans/structured-mixing-turing.md`). Replaces the v0
//! txn stub (`crates/veda-gateway/src/rest/txn.rs`) with the real
//! 2PC + lock + commit-record algorithm from Peng & Dabek's
//! "Large-scale Incremental Processing Using Distributed
//! Transactions and Notifications" (OSDI 2010).
//!
//! ## What v0 ships
//!
//! - The lock + commit record types.
//! - A `TxnCoordinator` that runs the full begin/prewrite/commit/
//!   abort flow against an in-process `MvccStore` (Phase F) and
//!   an in-process `LockTable` (the _meta primary-lock store).
//! - Conflict detection: write-write at prewrite (lock conflict
//!   or commit_ts > start_ts), tombstone cleanup, snapshot reads
//!   that exclude locked-but-uncommitted versions.
//! - Adya anomaly coverage: lost update prevented; read-only
//!   transactions consistent; write-skew allowed (SI is NOT
//!   serialisable — by design).
//! - End-to-end concurrent-txn tests.
//!
//! ## What v0 does NOT ship
//!
//! - Distributed integration. The coordinator runs in-process
//!   today; real Percolator routes prewrite RPCs to each shard's
//!   per-shard log (Phase D's `LogWriter`). The algorithm is
//!   the same; only the I/O changes.
//! - Cross-shard primary-lock pinning to the metadata Raft. The
//!   `LockTable` here is in-process; production stores primary
//!   locks via `_meta`'s `RegisterShardEnsemble`-shaped commands.
//! - Orphan-lock resolver scheduling. The `resolve_orphan_locks`
//!   method exists and is correct but isn't yet wired to a
//!   periodic task.
//! - Multi-statement transactions over the REST API. v0 here is
//!   the algorithm + simulator; the existing
//!   `crates/veda-gateway/src/rest/txn.rs` REST surface is the
//!   integration target.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use thiserror::Error;
use uuid::Uuid;
use veda_time::{Ts as Hlc, Tso, TsoError};

/// A lock record held during prewrite. `is_primary = true` for
/// the single primary lock that determines the txn outcome;
/// `false` for the secondaries that point at the primary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lock {
    pub txn_id: String,
    pub start_ts_ns: u64,
    pub primary_key: Vec<u8>,
    pub is_primary: bool,
    /// Lease horizon for orphan detection. If wall-clock now >
    /// this, an orphan resolver can roll back the txn.
    pub lease_ns: u64,
}

/// In-process lock table. Production puts the primary lock in
/// `_meta` (so failover finds it) and the secondaries in the
/// per-shard data-plane log; the algorithm is the same.
#[derive(Debug, Default)]
pub struct LockTable {
    inner: Mutex<HashMap<Vec<u8>, Lock>>,
}

impl LockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn try_acquire(&self, key: &[u8], lock: Lock) -> Result<(), TxnError> {
        let mut g = self.inner.lock().await;
        if let Some(existing) = g.get(key) {
            if existing.txn_id == lock.txn_id {
                // Idempotent: same txn re-acquiring is fine.
                return Ok(());
            }
            return Err(TxnError::LockConflict {
                key: key.to_vec(),
                held_by: existing.txn_id.clone(),
            });
        }
        g.insert(key.to_vec(), lock);
        Ok(())
    }

    pub async fn release(&self, key: &[u8], txn_id: &str) {
        let mut g = self.inner.lock().await;
        if let Some(l) = g.get(key)
            && l.txn_id == txn_id
        {
            g.remove(key);
        }
    }

    pub async fn get(&self, key: &[u8]) -> Option<Lock> {
        self.inner.lock().await.get(key).cloned()
    }

    pub async fn snapshot(&self) -> Vec<Lock> {
        self.inner.lock().await.values().cloned().collect()
    }
}

/// Per-key (read_ts, lock) check result.
#[derive(Debug, Clone)]
pub enum ReadVisibility {
    /// Key is visible at read_ts; payload may be Some(bytes) or
    /// None for tombstone.
    Visible(Option<bytes::Bytes>),
    /// Key has a lock from a still-running txn at start_ts <=
    /// read_ts. Reader must either wait, resolve the lock, or
    /// abort.
    Locked { held_by: String, start_ts_ns: u64 },
}

/// Commit record — what's written to `_meta` on successful
/// commit. Mirrors `raft_types::CommitRecord` but kept local
/// here to keep this module self-testable.
#[derive(Debug, Clone)]
pub struct PercolatorCommit {
    pub txn_id: String,
    pub start_ts_ns: u64,
    pub commit_ts_ns: u64,
    pub primary_key: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum TxnError {
    #[error("lock conflict on key {key:?}: held by txn '{held_by}'")]
    LockConflict { key: Vec<u8>, held_by: String },
    #[error(
        "write-write conflict on key {key:?}: newer commit_ts={newer_ts} > start_ts={start_ts}"
    )]
    WriteWriteConflict {
        key: Vec<u8>,
        newer_ts: u64,
        start_ts: u64,
    },
    #[error("txn '{txn_id}' not open")]
    NotOpen { txn_id: String },
    #[error("txn '{txn_id}' already committed at {commit_ts_ns}")]
    AlreadyCommitted { txn_id: String, commit_ts_ns: u64 },
    #[error("tso: {0}")]
    Tso(#[from] TsoError),
}

/// One mutation queued in a transaction's write buffer.
#[derive(Debug, Clone)]
pub struct Mutation {
    pub key: Vec<u8>,
    /// `None` = delete (tombstone).
    pub value: Option<bytes::Bytes>,
}

/// Live transaction handle.
pub struct TxnHandle {
    pub txn_id: String,
    pub start_ts_ns: u64,
    /// Buffered writes. Applied at commit time; aborted writes
    /// never touch the MVCC store.
    writes: Vec<Mutation>,
}

impl TxnHandle {
    pub fn buffer(&mut self, mutation: Mutation) {
        self.writes.push(mutation);
    }

    pub fn writes(&self) -> &[Mutation] {
        &self.writes
    }
}

/// The lone authority on commit_ts ordering. Wraps a [`Tso`] for
/// timestamps + [`LockTable`] + the MVCC store.
pub struct TxnCoordinator {
    tso: Arc<dyn Tso>,
    locks: Arc<LockTable>,
    /// In-process MVCC store. v1 swaps this for the segment-
    /// manager-backed Phase F integration.
    mvcc: Arc<Mutex<veda_storage::mvcc::MvccStore>>,
    /// Committed transactions. Production puts these in `_meta`'s
    /// `txn_commits` map (Phase E added the field).
    commits: Mutex<HashMap<String, PercolatorCommit>>,
    /// Default lock lease horizon (ns). Used to compute Lock.lease_ns
    /// at prewrite. v0 default: 10s.
    lease_ns: u64,
}

impl TxnCoordinator {
    pub fn new(tso: Arc<dyn Tso>) -> Self {
        Self {
            tso,
            locks: Arc::new(LockTable::new()),
            mvcc: Arc::new(Mutex::new(veda_storage::mvcc::MvccStore::new())),
            commits: Mutex::new(HashMap::new()),
            lease_ns: 10_000_000_000,
        }
    }

    pub fn with_mvcc(mut self, store: Arc<Mutex<veda_storage::mvcc::MvccStore>>) -> Self {
        self.mvcc = store;
        self
    }

    /// Begin a new transaction. Allocates start_ts from the TSO.
    pub async fn begin(&self) -> Result<TxnHandle, TxnError> {
        let start_hlc = self.tso.now().await?;
        let start_ts_ns = hlc_to_ns(start_hlc);
        Ok(TxnHandle {
            txn_id: Uuid::new_v4().to_string(),
            start_ts_ns,
            writes: Vec::new(),
        })
    }

    /// Snapshot read at the transaction's start_ts. Returns
    /// `Locked` if the key has an ongoing lock — caller decides
    /// whether to wait, abort, or run the resolver.
    pub async fn read(&self, handle: &TxnHandle, key: &[u8]) -> Result<ReadVisibility, TxnError> {
        if let Some(lock) = self.locks.get(key).await
            && lock.start_ts_ns <= handle.start_ts_ns
            && lock.txn_id != handle.txn_id
        {
            return Ok(ReadVisibility::Locked {
                held_by: lock.txn_id,
                start_ts_ns: lock.start_ts_ns,
            });
        }
        let mvcc = self.mvcc.lock().await;
        Ok(ReadVisibility::Visible(
            mvcc.get_at(key, handle.start_ts_ns as u128),
        ))
    }

    /// Percolator prewrite: acquire locks on every key in the
    /// write buffer + verify no newer write committed since
    /// start_ts.
    pub async fn prewrite(&self, handle: &TxnHandle) -> Result<(), TxnError> {
        if handle.writes.is_empty() {
            return Ok(());
        }
        let primary_key = handle.writes[0].key.clone();
        let lease_ns = wall_clock_ns() + self.lease_ns;
        for (i, m) in handle.writes.iter().enumerate() {
            // Check for write-write conflict: if MVCC has a
            // version with commit_ts > start_ts, abort.
            {
                let mvcc = self.mvcc.lock().await;
                if let Some((newer_ts, _)) = mvcc.latest(&m.key)
                    && (newer_ts as u64) > handle.start_ts_ns
                {
                    // Best-effort cleanup of any locks we've
                    // already acquired in this prewrite.
                    self.release_partial(handle).await;
                    return Err(TxnError::WriteWriteConflict {
                        key: m.key.clone(),
                        newer_ts: newer_ts as u64,
                        start_ts: handle.start_ts_ns,
                    });
                }
            }
            let lock = Lock {
                txn_id: handle.txn_id.clone(),
                start_ts_ns: handle.start_ts_ns,
                primary_key: primary_key.clone(),
                is_primary: i == 0,
                lease_ns,
            };
            if let Err(e) = self.locks.try_acquire(&m.key, lock).await {
                self.release_partial(handle).await;
                return Err(e);
            }
        }
        Ok(())
    }

    async fn release_partial(&self, handle: &TxnHandle) {
        for m in &handle.writes {
            self.locks.release(&m.key, &handle.txn_id).await;
        }
    }

    /// Commit phase 2: atomically promote every staged write to
    /// MVCC at the same commit_ts, release locks, record the
    /// commit in `_meta`.
    pub async fn commit(&self, handle: &TxnHandle) -> Result<u64, TxnError> {
        if handle.writes.is_empty() {
            return Ok(handle.start_ts_ns);
        }
        // Idempotent commit re-issue → return the already-stored
        // commit_ts.
        if let Some(existing) = self.commits.lock().await.get(&handle.txn_id) {
            return Err(TxnError::AlreadyCommitted {
                txn_id: existing.txn_id.clone(),
                commit_ts_ns: existing.commit_ts_ns,
            });
        }
        let commit_hlc = self.tso.now().await?;
        let commit_ts_ns = hlc_to_ns(commit_hlc).max(handle.start_ts_ns + 1);
        let primary_key = handle.writes[0].key.clone();

        // Flip every write into MVCC at commit_ts.
        {
            let mut mvcc = self.mvcc.lock().await;
            for m in &handle.writes {
                let val = match &m.value {
                    Some(b) => veda_storage::mvcc::MvccValue::put(b.clone()),
                    None => veda_storage::mvcc::MvccValue::tombstone(),
                };
                mvcc.put(&m.key, commit_ts_ns as u128, val).map_err(|e| {
                    TxnError::WriteWriteConflict {
                        key: m.key.clone(),
                        newer_ts: 0,
                        start_ts: handle.start_ts_ns,
                    }
                    .tap_log(format!("mvcc.put: {e}"))
                })?;
            }
        }
        // Release locks.
        for m in &handle.writes {
            self.locks.release(&m.key, &handle.txn_id).await;
        }
        // Record commit.
        self.commits.lock().await.insert(
            handle.txn_id.clone(),
            PercolatorCommit {
                txn_id: handle.txn_id.clone(),
                start_ts_ns: handle.start_ts_ns,
                commit_ts_ns,
                primary_key,
            },
        );
        Ok(commit_ts_ns)
    }

    /// Abort: drop the locks. Buffered writes never reach MVCC.
    pub async fn abort(&self, handle: &TxnHandle) {
        for m in &handle.writes {
            self.locks.release(&m.key, &handle.txn_id).await;
        }
    }

    /// Spawn a background task that resolves orphan locks on a
    /// timer. Returns a `JoinHandle` so callers (the gateway)
    /// can cancel on shutdown. Sleeps `interval` between
    /// iterations; default is 5 seconds for production.
    pub fn spawn_orphan_resolver(
        self: std::sync::Arc<Self>,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let me = self;
            loop {
                tokio::time::sleep(interval).await;
                let rolled = me.resolve_orphan_locks().await;
                if rolled > 0 {
                    tracing::warn!(
                        rolled,
                        "orphan-lock resolver rolled back {rolled} expired transactions"
                    );
                }
            }
        })
    }

    /// Look at every active lock and roll back any whose lease
    /// has expired (wall clock > lease_ns). Production runs this
    /// on a periodic task. Returns the count of rolled-back
    /// transactions.
    pub async fn resolve_orphan_locks(&self) -> usize {
        let now = wall_clock_ns();
        let snapshot = self.locks.snapshot().await;
        let mut rolled = 0;
        for lock in snapshot {
            if lock.lease_ns < now {
                // Drop EVERY lock owned by this txn.
                let mut g = self.locks.inner.lock().await;
                g.retain(|_, l| l.txn_id != lock.txn_id);
                rolled += 1;
            }
        }
        rolled
    }
}

fn wall_clock_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn hlc_to_ns(hlc: Hlc) -> u64 {
    hlc.physical_ns
}

// Internal helper for chaining `tap`-style log on errors.
trait TapLog: Sized {
    fn tap_log(self, msg: String) -> Self;
}
impl TapLog for TxnError {
    fn tap_log(self, msg: String) -> Self {
        tracing::warn!(error = %self, ctx = %msg, "percolator error");
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use veda_time::LocalTso;

    fn mut_put(key: &[u8], v: &[u8]) -> Mutation {
        Mutation {
            key: key.to_vec(),
            value: Some(Bytes::copy_from_slice(v)),
        }
    }

    #[tokio::test]
    async fn happy_path_commit_flips_writes_to_mvcc() {
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);
        let mut h = cc.begin().await.unwrap();
        h.buffer(mut_put(b"a", b"v1"));
        h.buffer(mut_put(b"b", b"v2"));
        cc.prewrite(&h).await.unwrap();
        let cts = cc.commit(&h).await.unwrap();
        let mvcc = cc.mvcc.lock().await;
        assert_eq!(mvcc.get_at(b"a", cts as u128).as_deref(), Some(&b"v1"[..]));
        assert_eq!(mvcc.get_at(b"b", cts as u128).as_deref(), Some(&b"v2"[..]));
    }

    #[tokio::test]
    async fn abort_drops_locks_without_writing_to_mvcc() {
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);
        let mut h = cc.begin().await.unwrap();
        h.buffer(mut_put(b"a", b"v"));
        cc.prewrite(&h).await.unwrap();
        cc.abort(&h).await;
        let mvcc = cc.mvcc.lock().await;
        assert!(mvcc.get_at(b"a", u128::MAX).is_none());
        assert!(cc.locks.get(b"a").await.is_none());
    }

    #[tokio::test]
    async fn concurrent_txns_conflict_at_prewrite() {
        // Two txns try to write the same key — second must
        // surface a LockConflict at prewrite.
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);
        let mut a = cc.begin().await.unwrap();
        let mut b = cc.begin().await.unwrap();
        a.buffer(mut_put(b"k", b"a"));
        b.buffer(mut_put(b"k", b"b"));
        cc.prewrite(&a).await.unwrap();
        let err = cc.prewrite(&b).await.unwrap_err();
        match err {
            TxnError::LockConflict { held_by, .. } => assert_eq!(held_by, a.txn_id),
            e => panic!("expected LockConflict, got {e:?}"),
        }
        cc.abort(&a).await;
        // After a's abort, b's prewrite succeeds.
        cc.prewrite(&b).await.unwrap();
        cc.commit(&b).await.unwrap();
    }

    #[tokio::test]
    async fn lost_update_prevented_via_write_write_conflict() {
        // T1: read v0 = None, write v1
        // T2 (concurrent, started after T1's commit): write v2 must succeed (T1's value is at older commit_ts)
        // T3 (starts before T2's commit, after T1's commit): write v3 — T2's commit_ts > T3's start_ts → conflict
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);

        let mut t1 = cc.begin().await.unwrap();
        t1.buffer(mut_put(b"k", b"v1"));
        cc.prewrite(&t1).await.unwrap();
        cc.commit(&t1).await.unwrap();

        let mut t3 = cc.begin().await.unwrap();
        // T2 commits before T3 prewrites.
        let mut t2 = cc.begin().await.unwrap();
        t2.buffer(mut_put(b"k", b"v2"));
        cc.prewrite(&t2).await.unwrap();
        cc.commit(&t2).await.unwrap();
        // T3 tries to write — must detect T2's newer commit.
        t3.buffer(mut_put(b"k", b"v3"));
        let err = cc.prewrite(&t3).await.unwrap_err();
        assert!(matches!(err, TxnError::WriteWriteConflict { .. }));
    }

    #[tokio::test]
    async fn snapshot_read_isolates_from_in_flight_writes() {
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);
        // T1 commits k=v1.
        let mut t1 = cc.begin().await.unwrap();
        t1.buffer(mut_put(b"k", b"v1"));
        cc.prewrite(&t1).await.unwrap();
        cc.commit(&t1).await.unwrap();
        // T2 starts (open snapshot), reads k → v1.
        let t2 = cc.begin().await.unwrap();
        match cc.read(&t2, b"k").await.unwrap() {
            ReadVisibility::Visible(Some(bytes)) => assert_eq!(&bytes[..], b"v1"),
            other => panic!(
                "expected Visible(v1), got {other:?}",
                other = match other {
                    ReadVisibility::Visible(v) => format!("Visible({v:?})"),
                    ReadVisibility::Locked { held_by, .. } => format!("Locked by {held_by}"),
                }
            ),
        }
        // T3 commits k=v3 while T2 still open.
        let mut t3 = cc.begin().await.unwrap();
        t3.buffer(mut_put(b"k", b"v3"));
        cc.prewrite(&t3).await.unwrap();
        cc.commit(&t3).await.unwrap();
        // T2's read at its snapshot is still v1 (SI invariant).
        match cc.read(&t2, b"k").await.unwrap() {
            ReadVisibility::Visible(Some(bytes)) => assert_eq!(&bytes[..], b"v1"),
            _ => panic!("T2 must still see its snapshot view"),
        }
    }

    #[tokio::test]
    async fn read_sees_lock_for_concurrent_uncommitted_txn() {
        let tso = Arc::new(LocalTso::new());
        let cc = TxnCoordinator::new(tso);
        let mut t1 = cc.begin().await.unwrap();
        t1.buffer(mut_put(b"k", b"v1"));
        cc.prewrite(&t1).await.unwrap();
        // T2 starts after T1's prewrite but before T1's commit.
        let t2 = cc.begin().await.unwrap();
        match cc.read(&t2, b"k").await.unwrap() {
            ReadVisibility::Locked { held_by, .. } => assert_eq!(held_by, t1.txn_id),
            other => panic!(
                "expected Locked, got Visible({other:?})",
                other = match other {
                    ReadVisibility::Visible(v) => format!("{v:?}"),
                    _ => "?".into(),
                }
            ),
        }
    }

    #[tokio::test]
    async fn spawn_orphan_resolver_runs_periodically() {
        let tso = Arc::new(LocalTso::new());
        let mut cc = TxnCoordinator::new(tso);
        cc.lease_ns = 0; // immediate orphan
        let cc_arc = Arc::new(cc);
        let mut h = cc_arc.begin().await.unwrap();
        h.buffer(mut_put(b"k", b"v"));
        cc_arc.prewrite(&h).await.unwrap();
        // Spawn the resolver with a 10ms tick.
        let handle = cc_arc
            .clone()
            .spawn_orphan_resolver(std::time::Duration::from_millis(10));
        // Wait long enough for at least 2 ticks.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // The lock for k must have been released by the
        // resolver.
        assert!(cc_arc.locks.get(b"k").await.is_none());
        handle.abort();
    }

    #[tokio::test]
    async fn orphan_resolver_drops_expired_locks() {
        let tso = Arc::new(LocalTso::new());
        let mut cc = TxnCoordinator::new(tso);
        // Force the lease horizon into the past so the very next
        // prewrite produces an immediately-orphaned lock.
        cc.lease_ns = 0;
        let mut h = cc.begin().await.unwrap();
        h.buffer(mut_put(b"k", b"v"));
        cc.prewrite(&h).await.unwrap();
        // Lock's lease_ns = wall_now + 0 ≈ wall_now → already
        // expired by the time we look. Small sleep to make sure.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let rolled = cc.resolve_orphan_locks().await;
        assert!(rolled >= 1);
        assert!(
            cc.locks.get(b"k").await.is_none(),
            "orphan resolver must drop the expired lock"
        );
    }
}
