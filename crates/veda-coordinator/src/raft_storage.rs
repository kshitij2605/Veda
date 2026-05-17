//! In-memory storage state for openraft.
//!
//! ## Status
//!
//! This file holds the structural state that an openraft node needs: log
//! entries, vote, last-applied tracking, membership, current snapshot. The
//! `RaftLogStorage` / `RaftStateMachine` *trait* implementations are
//! intentionally deferred.
//!
//! ## Why deferred
//!
//! openraft 0.9 split the older `RaftStorage` trait into a v2 pair
//! (`RaftLogStorage` + `RaftStateMachine`) that are **sealed** — only
//! implementable via the `Adaptor<C, S>` wrapper around the v1
//! `RaftStorage` trait. A correct implementation therefore needs to:
//!
//! 1. Implement v1 `RaftStorage<VedaTypeConfig>` end-to-end (12+ methods,
//!    each with non-trivial semantics around log truncation, snapshot
//!    streaming, and persistent vote).
//! 2. Wrap with `openraft::storage::Adaptor` to get the v2 traits.
//! 3. Pass through to `openraft::Raft::new(...)`.
//!
//! Doing this correctly — especially for snapshots and the
//! truncate-vs-purge distinction — is a focused multi-day task that needs
//! quiet review against the upstream `examples/raft-kv-memstore` reference
//! implementation. Rushing it produces silent log-corruption bugs that are
//! the worst class of distributed-systems bug to debug.
//!
//! What lives here right now:
//!
//! - [`MemLogState`] — the in-memory log + vote storage that the future
//!   `RaftStorage` impl will read/write.
//! - [`MemStateMachine`] — wraps our existing [`crate::store::ClusterStateMachine`]
//!   and tracks `applied_log` + `membership` + a current snapshot blob.
//!
//! Both expose plain methods so the rest of the coordinator can use them in
//! the meantime (e.g. to replay a snapshot loaded from disk), and so the
//! `RaftStorage` impl is a thin trait-method-to-plain-method translation.

use std::collections::BTreeMap;
use std::sync::Arc;

use openraft::{BasicNode, Entry, LogId, SnapshotMeta, StoredMembership, Vote};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::raft::{NodeId, VedaSnapshot, VedaTypeConfig};

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
}
