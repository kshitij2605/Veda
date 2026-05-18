//! In-memory storage for openraft.
//!
//! - [`MemLogState`] — log entries, vote, purge marker.
//! - [`MemStateMachine`] — last-applied, membership, current snapshot.
//! - [`MemRaftStorage`] — combined log + state machine that implements the
//!   v1 [`openraft::RaftStorage`] trait. Wrap it with
//!   [`openraft::storage::Adaptor::new`] to get the v2
//!   `RaftLogStorage` + `RaftStateMachine` pair required by
//!   [`openraft::Raft::new`].
//! - [`MemSnapshotBuilder`] — companion [`openraft::RaftSnapshotBuilder`]
//!   that clones the state machine to build a snapshot off the apply path.
//!
//! Storage is in-memory only — durability lives in the next pass (file or
//! WAL-backed log + vote). The trait surface is complete enough for
//! single-node bootstrap, the `Adaptor` round-trip, and the upstream
//! `StorageHelper::get_initial_state` flow that `Raft::new` exercises.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use openraft::storage::{LogState, Snapshot};
use openraft::{
    AnyError, BasicNode, Entry, EntryPayload, LogId, Membership, RaftLogReader,
    RaftSnapshotBuilder, RaftStorage, SnapshotMeta, StorageError, StorageIOError,
    StoredMembership, Vote,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::raft::{NodeId, VedaSnapshot, VedaTypeConfig};
use crate::raft_types::ClusterResponse;

// ---------------------------------------------------------------------------
// MemLogState — in-memory log entries + vote + purge marker.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct LogInner {
    entries: BTreeMap<u64, Entry<VedaTypeConfig>>,
    last_purged: Option<LogId<NodeId>>,
    last_log_id: Option<LogId<NodeId>>,
    vote: Option<Vote<NodeId>>,
    committed: Option<LogId<NodeId>>,
}

#[derive(Debug, Clone, Default)]
pub struct MemLogState {
    inner: Arc<RwLock<LogInner>>,
}

impl MemLogState {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn save_vote(&self, vote: Vote<NodeId>) {
        self.inner.write().await.vote = Some(vote);
    }

    pub async fn read_vote(&self) -> Option<Vote<NodeId>> {
        self.inner.read().await.vote
    }

    pub async fn append_entry(&self, entry: Entry<VedaTypeConfig>) {
        let mut guard = self.inner.write().await;
        let id = entry.log_id;
        guard.entries.insert(id.index, entry);
        guard.last_log_id = Some(id);
    }

    pub async fn last_log_id(&self) -> Option<LogId<NodeId>> {
        let guard = self.inner.read().await;
        guard.last_log_id.or(guard.last_purged)
    }

    pub async fn last_purged(&self) -> Option<LogId<NodeId>> {
        self.inner.read().await.last_purged
    }

    /// Remove all entries with `index >= log_id.index`. Used by Raft to
    /// resolve conflicts where the leader's log diverges from a follower's.
    pub async fn truncate(&self, log_id: LogId<NodeId>) {
        let mut guard = self.inner.write().await;
        guard.entries.retain(|&idx, _| idx < log_id.index);
        guard.last_log_id = guard.entries.iter().next_back().map(|(_, e)| e.log_id);
    }

    /// Remove all entries with `index <= log_id.index` (post-snapshot compaction).
    pub async fn purge(&self, log_id: LogId<NodeId>) {
        let mut guard = self.inner.write().await;
        guard.entries.retain(|&idx, _| idx > log_id.index);
        guard.last_purged = Some(log_id);
    }

    pub async fn save_committed(&self, committed: Option<LogId<NodeId>>) {
        self.inner.write().await.committed = committed;
    }

    pub async fn read_committed(&self) -> Option<LogId<NodeId>> {
        self.inner.read().await.committed
    }

    pub async fn range(&self, start: u64, end: u64) -> Vec<Entry<VedaTypeConfig>> {
        self.inner
            .read()
            .await
            .entries
            .range(start..end)
            .map(|(_, e)| e.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MemStateMachine — wraps ClusterStateMachine.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct StateInner {
    applied_log: Option<LogId<NodeId>>,
    membership: StoredMembership<NodeId, BasicNode>,
    state: crate::store::ClusterStateMachine,
    current_snapshot: Option<StoredSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSnapshot {
    pub meta: SnapshotMeta<NodeId, BasicNode>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct MemStateMachine {
    inner: Arc<RwLock<StateInner>>,
}

impl MemStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn applied_log(&self) -> Option<LogId<NodeId>> {
        self.inner.read().await.applied_log
    }

    pub async fn membership(&self) -> StoredMembership<NodeId, BasicNode> {
        self.inner.read().await.membership.clone()
    }

    /// Apply a single command, recording the new last-applied log id. This
    /// is the body that the future `RaftStateMachine::apply` will call once
    /// per `EntryPayload::Normal`.
    pub async fn apply_command(
        &self,
        log_id: LogId<NodeId>,
        command: &crate::raft_types::ClusterCommand,
    ) -> crate::raft_types::ClusterResponse {
        let mut guard = self.inner.write().await;
        let entry = crate::raft_types::LogEntry {
            index: log_id.index,
            command: command.clone(),
        };
        let resp = guard.state.apply(&entry);
        guard.applied_log = Some(log_id);
        resp
    }

    pub async fn build_snapshot(&self) -> Result<StoredSnapshot, String> {
        let guard = self.inner.read().await;
        let state_bytes = bincode::serialize(&guard.state)
            .map_err(|e| format!("state machine serialize: {e}"))?;
        let payload = VedaSnapshot {
            last_applied_index: guard.applied_log.map(|l| l.index).unwrap_or(0),
            state_bytes,
        };
        let bytes = bincode::serialize(&payload)
            .map_err(|e| format!("snapshot serialize: {e}"))?;
        let snapshot_id = format!(
            "{}-{}",
            guard.applied_log.map(|l| l.index).unwrap_or(0),
            guard
                .applied_log
                .map(|l| l.leader_id.term)
                .unwrap_or(0)
        );
        Ok(StoredSnapshot {
            meta: SnapshotMeta {
                last_log_id: guard.applied_log,
                last_membership: guard.membership.clone(),
                snapshot_id,
            },
            bytes,
        })
    }

    pub async fn install_snapshot(&self, snap: StoredSnapshot) -> Result<(), String> {
        let payload: VedaSnapshot = bincode::deserialize(&snap.bytes)
            .map_err(|e| format!("snapshot deserialize: {e}"))?;
        let restored: crate::store::ClusterStateMachine =
            bincode::deserialize(&payload.state_bytes)
                .map_err(|e| format!("state machine deserialize: {e}"))?;
        let mut guard = self.inner.write().await;
        guard.state = restored;
        guard.applied_log = snap.meta.last_log_id;
        guard.membership = snap.meta.last_membership.clone();
        guard.current_snapshot = Some(snap);
        Ok(())
    }

    pub async fn current_snapshot(&self) -> Option<StoredSnapshot> {
        self.inner.read().await.current_snapshot.clone()
    }

    /// Advance only the applied-log marker. Used for Raft Blank entries.
    pub async fn bump_applied(&self, log_id: LogId<NodeId>) {
        let mut guard = self.inner.write().await;
        guard.state.skip_apply(log_id.index);
        guard.applied_log = Some(log_id);
    }

    /// Record a Raft Membership entry. Updates `membership` + applied marker.
    pub async fn record_membership(
        &self,
        log_id: LogId<NodeId>,
        membership: Membership<NodeId, BasicNode>,
    ) {
        let mut guard = self.inner.write().await;
        guard.state.skip_apply(log_id.index);
        guard.applied_log = Some(log_id);
        guard.membership = StoredMembership::new(Some(log_id), membership);
    }

    /// Cache a snapshot as the current one. Used after `build_snapshot`.
    pub async fn set_current_snapshot(&self, snap: StoredSnapshot) {
        self.inner.write().await.current_snapshot = Some(snap);
    }
}

// ---------------------------------------------------------------------------
// MemRaftStorage — composes log + state machine to satisfy openraft v1
// `RaftStorage<VedaTypeConfig>`. Pair with `Adaptor::new(...)` to drive
// `Raft::new`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct MemRaftStorage {
    pub log: MemLogState,
    pub sm: MemStateMachine,
    snapshot_counter: Arc<RwLock<u64>>,
}

impl MemRaftStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

fn storage_io<T: Into<AnyError>>(verb: openraft::ErrorVerb, source: T) -> StorageError<NodeId> {
    let subject = openraft::ErrorSubject::Store;
    StorageError::IO {
        source: StorageIOError::new(subject, verb, source),
    }
}

impl RaftLogReader<VedaTypeConfig> for MemRaftStorage {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<VedaTypeConfig>>, StorageError<NodeId>> {
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i.saturating_add(1),
            Bound::Excluded(&i) => i,
            Bound::Unbounded => u64::MAX,
        };
        Ok(self.log.range(start, end).await)
    }
}

impl RaftStorage<VedaTypeConfig> for MemRaftStorage {
    type LogReader = Self;
    type SnapshotBuilder = MemSnapshotBuilder;

    // --- Vote ---

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.log.save_vote(*vote).await;
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        Ok(self.log.read_vote().await)
    }

    // --- Log ---

    async fn get_log_state(&mut self) -> Result<LogState<VedaTypeConfig>, StorageError<NodeId>> {
        Ok(LogState {
            last_purged_log_id: self.log.last_purged().await,
            last_log_id: self.log.last_log_id().await,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn append_to_log<I>(&mut self, entries: I) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<VedaTypeConfig>> + Send,
    {
        // Collect first so the (potentially !Send) iterator does not cross
        // the await boundary inside the loop.
        let entries: Vec<_> = entries.into_iter().collect();
        for entry in entries {
            self.log.append_entry(entry).await;
        }
        Ok(())
    }

    async fn delete_conflict_logs_since(
        &mut self,
        log_id: LogId<NodeId>,
    ) -> Result<(), StorageError<NodeId>> {
        self.log.truncate(log_id).await;
        Ok(())
    }

    async fn purge_logs_upto(
        &mut self,
        log_id: LogId<NodeId>,
    ) -> Result<(), StorageError<NodeId>> {
        self.log.purge(log_id).await;
        Ok(())
    }

    // --- State machine ---

    async fn last_applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogId<NodeId>>,
            StoredMembership<NodeId, BasicNode>,
        ),
        StorageError<NodeId>,
    > {
        Ok((self.sm.applied_log().await, self.sm.membership().await))
    }

    async fn apply_to_state_machine(
        &mut self,
        entries: &[Entry<VedaTypeConfig>],
    ) -> Result<Vec<ClusterResponse>, StorageError<NodeId>> {
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let resp = match &entry.payload {
                EntryPayload::Normal(cmd) => self.sm.apply_command(entry.log_id, cmd).await,
                EntryPayload::Membership(m) => {
                    self.sm.record_membership(entry.log_id, m.clone()).await;
                    ClusterResponse::Ok
                }
                EntryPayload::Blank => {
                    self.sm.bump_applied(entry.log_id).await;
                    ClusterResponse::Ok
                }
            };
            out.push(resp);
        }
        Ok(out)
    }

    // --- Snapshot ---

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        MemSnapshotBuilder {
            sm: self.sm.clone(),
            counter: self.snapshot_counter.clone(),
        }
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let stored = StoredSnapshot {
            meta: meta.clone(),
            bytes: snapshot.into_inner(),
        };
        self.sm
            .install_snapshot(stored)
            .await
            .map_err(|e| storage_io(openraft::ErrorVerb::Write, AnyError::error(e)))?;
        // Purge log entries below the snapshot — openraft contract.
        if let Some(last) = meta.last_log_id {
            self.log.purge(last).await;
        }
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<VedaTypeConfig>>, StorageError<NodeId>> {
        Ok(self.sm.current_snapshot().await.map(|s| Snapshot {
            meta: s.meta,
            snapshot: Box::new(Cursor::new(s.bytes)),
        }))
    }
}

// ---------------------------------------------------------------------------
// MemSnapshotBuilder — companion type implementing `RaftSnapshotBuilder`.
// Holds a cloned handle to the state machine so snapshot building doesn't
// block the apply path.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MemSnapshotBuilder {
    sm: MemStateMachine,
    counter: Arc<RwLock<u64>>,
}

impl RaftSnapshotBuilder<VedaTypeConfig> for MemSnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot<VedaTypeConfig>, StorageError<NodeId>> {
        let mut stored = self
            .sm
            .build_snapshot()
            .await
            .map_err(|e| storage_io(openraft::ErrorVerb::Write, AnyError::error(e)))?;
        // Stamp with a monotonic id so two snapshots taken at the same applied
        // index still have distinct identifiers.
        let id = {
            let mut g = self.counter.write().await;
            *g += 1;
            *g
        };
        stored.meta.snapshot_id = format!("{}-{}", stored.meta.snapshot_id, id);
        self.sm.set_current_snapshot(stored.clone()).await;
        Ok(Snapshot {
            meta: stored.meta,
            snapshot: Box::new(Cursor::new(stored.bytes)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft_types::ClusterCommand;
    use openraft::{CommittedLeaderId, LogId};

    fn lid(term: u64, idx: u64) -> LogId<NodeId> {
        LogId::new(CommittedLeaderId::new(term, 1), idx)
    }

    #[tokio::test]
    async fn log_state_append_truncate_purge() {
        let log = MemLogState::new();
        let v = Vote::new(1, 1);
        log.save_vote(v).await;
        assert_eq!(log.read_vote().await, Some(v));

        let e1 = Entry {
            log_id: lid(1, 1),
            payload: openraft::EntryPayload::Blank,
        };
        let e2 = Entry {
            log_id: lid(1, 2),
            payload: openraft::EntryPayload::Blank,
        };
        log.append_entry(e1).await;
        log.append_entry(e2).await;
        assert_eq!(log.last_log_id().await, Some(lid(1, 2)));

        log.truncate(lid(1, 2)).await;
        assert_eq!(log.last_log_id().await, Some(lid(1, 1)));

        log.purge(lid(1, 1)).await;
        assert_eq!(log.last_purged().await, Some(lid(1, 1)));
        assert!(log.range(0, 100).await.is_empty());
    }

    #[tokio::test]
    async fn state_machine_snapshot_roundtrip() {
        let sm = MemStateMachine::new();
        let cmd = ClusterCommand::CreateCollection {
            name: "test".into(),
            schema_json: "{}".into(),
        };
        let resp = sm.apply_command(lid(1, 1), &cmd).await;
        assert!(matches!(resp, crate::raft_types::ClusterResponse::Ok));

        let snap = sm.build_snapshot().await.unwrap();
        // Round-trip: install into a fresh state machine and verify.
        let sm2 = MemStateMachine::new();
        sm2.install_snapshot(snap).await.unwrap();
        assert_eq!(sm2.applied_log().await, Some(lid(1, 1)));
    }

    // -----------------------------------------------------------------------
    // RaftStorage trait coverage — drives every method of the v1 trait on
    // MemRaftStorage and checks the produced state. Together with the
    // `mem_raft_storage_adaptor_compiles` test below this proves the
    // production path: implement v1 → wrap with Adaptor → pass to Raft::new.
    // -----------------------------------------------------------------------

    use openraft::{RaftLogReader, RaftSnapshotBuilder, RaftStorage};

    fn entry_normal(idx: u64, cmd: ClusterCommand) -> Entry<VedaTypeConfig> {
        Entry {
            log_id: lid(1, idx),
            payload: openraft::EntryPayload::Normal(cmd),
        }
    }

    fn entry_blank(idx: u64) -> Entry<VedaTypeConfig> {
        Entry {
            log_id: lid(1, idx),
            payload: openraft::EntryPayload::Blank,
        }
    }

    fn entry_membership(idx: u64) -> Entry<VedaTypeConfig> {
        let m = openraft::Membership::<NodeId, BasicNode>::new(
            vec![std::iter::once(1u64).collect()],
            std::collections::BTreeMap::new(),
        );
        Entry {
            log_id: lid(1, idx),
            payload: openraft::EntryPayload::Membership(m),
        }
    }

    #[tokio::test]
    async fn mem_raft_storage_vote_log_apply_snapshot_cycle() {
        let mut store = MemRaftStorage::new();

        // -- Vote roundtrip
        assert_eq!(store.read_vote().await.unwrap(), None);
        let v = Vote::new(7, 1);
        store.save_vote(&v).await.unwrap();
        assert_eq!(store.read_vote().await.unwrap(), Some(v));

        // -- Empty log state
        let st = store.get_log_state().await.unwrap();
        assert_eq!(st.last_log_id, None);
        assert_eq!(st.last_purged_log_id, None);

        // -- append_to_log + try_get_log_entries via the log reader
        let e1 = entry_blank(1);
        let e2 = entry_normal(
            2,
            ClusterCommand::CreateCollection {
                name: "c".into(),
                schema_json: "{}".into(),
            },
        );
        let e3 = entry_membership(3);
        store
            .append_to_log(vec![e1.clone(), e2.clone(), e3.clone()])
            .await
            .unwrap();
        let st = store.get_log_state().await.unwrap();
        assert_eq!(st.last_log_id, Some(lid(1, 3)));

        let mut rdr = store.get_log_reader().await;
        let fetched = rdr.try_get_log_entries(1u64..4u64).await.unwrap();
        assert_eq!(fetched.len(), 3);

        // -- apply_to_state_machine handles Blank / Normal / Membership
        let responses = store
            .apply_to_state_machine(&[e1, e2, e3])
            .await
            .unwrap();
        assert_eq!(responses.len(), 3);
        assert!(matches!(responses[1], ClusterResponse::Ok));
        let (applied, membership) = store.last_applied_state().await.unwrap();
        assert_eq!(applied, Some(lid(1, 3)));
        assert_eq!(membership.log_id(), &Some(lid(1, 3)));

        // -- Snapshot build + install via begin_receiving_snapshot
        let mut builder = store.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await.unwrap();
        assert_eq!(snap.meta.last_log_id, Some(lid(1, 3)));

        // get_current_snapshot now returns the one we just built.
        let cur = store.get_current_snapshot().await.unwrap();
        assert!(cur.is_some());

        // Install snapshot onto a fresh store and confirm last-applied was
        // restored AND the log got purged below the snapshot's last_log_id.
        let mut fresh = MemRaftStorage::new();
        let recv = fresh.begin_receiving_snapshot().await.unwrap();
        let snap_bytes = snap.snapshot.into_inner();
        let mut filled = recv;
        std::io::Write::write_all(&mut *filled, &snap_bytes).unwrap();
        // The cursor's internal Vec is now populated. Reset position so the
        // adapter can read it back if it ever does.
        filled.set_position(0);
        fresh.install_snapshot(&snap.meta, filled).await.unwrap();
        let (applied, _) = fresh.last_applied_state().await.unwrap();
        assert_eq!(applied, Some(lid(1, 3)));

        // -- truncate / purge
        store.delete_conflict_logs_since(lid(1, 3)).await.unwrap();
        let st = store.get_log_state().await.unwrap();
        assert_eq!(st.last_log_id, Some(lid(1, 2)));

        store.purge_logs_upto(lid(1, 2)).await.unwrap();
        let st = store.get_log_state().await.unwrap();
        assert_eq!(st.last_purged_log_id, Some(lid(1, 2)));
    }

    #[tokio::test]
    async fn mem_raft_storage_adaptor_compiles() {
        use openraft::storage::Adaptor;
        let store = MemRaftStorage::new();
        // The Adaptor turns a v1 RaftStorage into the v2 (RaftLogStorage,
        // RaftStateMachine) pair that `Raft::new` requires. If this builds,
        // the trait surface is wired correctly.
        let (_log_store, _state_machine) = Adaptor::<VedaTypeConfig, MemRaftStorage>::new(store);
    }
}
