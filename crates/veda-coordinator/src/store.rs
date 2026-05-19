//! In-memory log store and state machine for the Veda coordinator.
//!
//! This module provides:
//! - [`LogStore`] — an append-only log of [`ClusterCommand`]s.
//! - [`ClusterStateMachine`] — the state machine that applies commands
//!   and maintains cluster metadata (collections, nodes, shard assignments).
//!
//! Both are designed behind a `tokio::sync::RwLock` so they can later be
//! swapped for a Raft-backed implementation without changing callers.

use std::collections::HashMap;

use crate::raft_types::{
    ClusterCommand, ClusterResponse, ClusterState, LogEntry, NodeInfo, NodeStatus,
};

// ---------------------------------------------------------------------------
// LogStore
// ---------------------------------------------------------------------------

/// Append-only in-memory log of cluster commands.
///
/// Each entry is assigned a monotonically increasing index starting at 1.
/// Old entries can be purged up to a given index (inclusive) to bound memory.
#[derive(Debug, Default)]
pub struct LogStore {
    entries: Vec<LogEntry>,
    /// Index of the last purged entry (entries up to and including this have
    /// been removed). 0 means nothing has been purged.
    last_purged_index: u64,
}

impl LogStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a command and return the index it was assigned.
    pub fn append(&mut self, command: ClusterCommand) -> u64 {
        let index = self.next_index();
        self.entries.push(LogEntry {
            index,
            command,
        });
        index
    }

    /// The index that the next appended entry will receive.
    pub fn next_index(&self) -> u64 {
        match self.entries.last() {
            Some(e) => e.index + 1,
            None => self.last_purged_index + 1,
        }
    }

    /// Return the last entry's index, or 0 if the log is empty.
    pub fn last_index(&self) -> u64 {
        self.entries.last().map_or(self.last_purged_index, |e| e.index)
    }

    /// Get entries in the range `[start, end)` (both are 1-based indices).
    pub fn get_range(&self, start: u64, end: u64) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.index >= start && e.index < end)
            .collect()
    }

    /// Get all entries from `start` (inclusive) to the end.
    pub fn get_from(&self, start: u64) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.index >= start).collect()
    }

    /// Purge entries with index <= `up_to_inclusive`.
    pub fn purge(&mut self, up_to_inclusive: u64) {
        self.entries.retain(|e| e.index > up_to_inclusive);
        if up_to_inclusive > self.last_purged_index {
            self.last_purged_index = up_to_inclusive;
        }
    }

    /// Total number of entries currently held (after purges).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ClusterStateMachine
// ---------------------------------------------------------------------------

/// The cluster state machine.
///
/// Deterministically applies [`ClusterCommand`]s to produce a consistent view
/// of the cluster state. In a full Raft deployment every node runs the same
/// sequence of commands through the same state machine logic, so all nodes
/// converge to identical state.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterStateMachine {
    /// Index of the last applied log entry.
    last_applied: u64,
    /// collection name -> schema JSON
    collections: HashMap<String, String>,
    /// node_id -> node info
    nodes: HashMap<u64, NodeInfo>,
    /// collection -> (shard_id -> [node_ids])
    shard_assignments: HashMap<String, HashMap<u32, Vec<u64>>>,
    /// Replicated record store: (collection, record_id) -> record_json.
    /// Populated by `ClusterCommand::InsertRecord` applies. Each
    /// node maintains its own copy; the gateway-side reconcile
    /// path picks up new entries and pushes them into the local
    /// SegmentManager.
    ///
    /// **Memory**: this map holds every inserted record in
    /// process memory on every node. For small-to-medium states
    /// this is fine (the same state machine is bincode-serialised
    /// for snapshot install anyway). For very large datasets,
    /// per-shard Raft groups are the right architectural answer —
    /// see documentation/raft-migration.md.
    #[serde(default)]
    records: HashMap<String, HashMap<String, String>>,
    /// Open transactions, keyed by txn_id. Each holds the staged
    /// writes — moved into `records` on `CommitTxn`, discarded on
    /// `AbortTxn`. v0 scaffolding (see ClusterCommand::BeginTxn
    /// doc). Last-writer-wins on commit collisions.
    #[serde(default)]
    open_txns: HashMap<String, PendingTxn>,
}

/// Staged writes for an open transaction. v0 holds them in memory
/// on every node — production Percolator would persist lock rows
/// per key in a dedicated CF instead.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingTxn {
    /// (collection, record_id) -> record_json
    pub writes: Vec<(String, String, String)>,
}

impl ClusterStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a single log entry and return the response.
    ///
    /// The entry's index **must** equal `last_applied + 1`; otherwise the
    /// function returns an error response indicating an out-of-order apply.
    pub fn apply(&mut self, entry: &LogEntry) -> ClusterResponse {
        // Enforce sequential application.
        if entry.index != self.last_applied + 1 {
            return ClusterResponse::Error(format!(
                "out-of-order apply: expected index {}, got {}",
                self.last_applied + 1,
                entry.index
            ));
        }

        let resp = self.apply_command(&entry.command);
        self.last_applied = entry.index;
        resp
    }

    /// Advance `last_applied` without running any business logic. Used by
    /// the openraft adapter for `EntryPayload::Blank` / `Membership`
    /// entries that don't touch domain state but still occupy a log index.
    pub fn skip_apply(&mut self, idx: u64) {
        if idx > self.last_applied {
            self.last_applied = idx;
        }
    }

    /// Apply a command (without index bookkeeping) — pure business logic.
    fn apply_command(&mut self, cmd: &ClusterCommand) -> ClusterResponse {
        match cmd {
            ClusterCommand::CreateCollection { name, schema_json } => {
                if self.collections.contains_key(name) {
                    return ClusterResponse::Error(format!("collection '{name}' already exists"));
                }
                self.collections.insert(name.clone(), schema_json.clone());
                ClusterResponse::Ok
            }

            ClusterCommand::DropCollection { name } => {
                if self.collections.remove(name).is_none() {
                    return ClusterResponse::Error(format!("collection '{name}' not found"));
                }
                self.shard_assignments.remove(name);
                self.records.remove(name);
                ClusterResponse::Ok
            }

            ClusterCommand::RegisterNode {
                node_id,
                address,
                roles,
            } => {
                if self.nodes.contains_key(node_id) {
                    return ClusterResponse::Error(format!("node {node_id} already registered"));
                }
                self.nodes.insert(
                    *node_id,
                    NodeInfo {
                        address: address.clone(),
                        roles: roles.clone(),
                        status: NodeStatus::Active,
                    },
                );
                ClusterResponse::Ok
            }

            ClusterCommand::DeregisterNode { node_id } => {
                if self.nodes.remove(node_id).is_none() {
                    return ClusterResponse::Error(format!("node {node_id} not found"));
                }
                ClusterResponse::Ok
            }

            // Linearisable-read barrier — applies as a no-op but
            // forces the apply log to advance, which guarantees a
            // subsequent state read includes every prior commit.
            ClusterCommand::Noop => ClusterResponse::Ok,

            ClusterCommand::InsertRecord {
                collection,
                record_id,
                record_json,
            } => {
                if !self.collections.contains_key(collection) {
                    return ClusterResponse::Error(format!(
                        "collection '{collection}' does not exist"
                    ));
                }
                let entry = self
                    .records
                    .entry(collection.clone())
                    .or_default();
                entry.insert(record_id.clone(), record_json.clone());
                ClusterResponse::Ok
            }

            ClusterCommand::AssignShard {
                collection,
                shard_id,
                node_id,
            } => {
                if !self.collections.contains_key(collection) {
                    return ClusterResponse::Error(format!(
                        "collection '{collection}' does not exist"
                    ));
                }
                if !self.nodes.contains_key(node_id) {
                    return ClusterResponse::Error(format!("node {node_id} not registered"));
                }
                let shards = self
                    .shard_assignments
                    .entry(collection.clone())
                    .or_default();
                let nodes = shards.entry(*shard_id).or_default();
                if !nodes.contains(node_id) {
                    nodes.push(*node_id);
                }
                ClusterResponse::Ok
            }

            ClusterCommand::UnassignShard {
                collection,
                shard_id,
                node_id,
            } => {
                let removed = self
                    .shard_assignments
                    .get_mut(collection)
                    .and_then(|shards| shards.get_mut(shard_id))
                    .map(|nodes| {
                        let before = nodes.len();
                        nodes.retain(|n| n != node_id);
                        nodes.len() < before
                    })
                    .unwrap_or(false);

                if !removed {
                    return ClusterResponse::Error(format!(
                        "shard {shard_id} of '{collection}' is not assigned to node {node_id}"
                    ));
                }
                ClusterResponse::Ok
            }

            ClusterCommand::BeginTxn { txn_id } => {
                // Idempotent — re-issuing the same txn_id is a no-op
                // so a retried client (after a network blip) doesn't
                // get a spurious error.
                self.open_txns.entry(txn_id.clone()).or_default();
                ClusterResponse::Ok
            }

            ClusterCommand::TxnWrite {
                txn_id,
                collection,
                record_id,
                record_json,
            } => {
                if !self.collections.contains_key(collection) {
                    return ClusterResponse::Error(format!(
                        "collection '{collection}' does not exist"
                    ));
                }
                let Some(txn) = self.open_txns.get_mut(txn_id) else {
                    return ClusterResponse::Error(format!(
                        "transaction '{txn_id}' is not open"
                    ));
                };
                txn.writes.push((
                    collection.clone(),
                    record_id.clone(),
                    record_json.clone(),
                ));
                ClusterResponse::Ok
            }

            ClusterCommand::CommitTxn { txn_id } => {
                let Some(txn) = self.open_txns.remove(txn_id) else {
                    // Idempotent on re-commit (already moved).
                    return ClusterResponse::Ok;
                };
                for (collection, record_id, record_json) in txn.writes {
                    self.records
                        .entry(collection)
                        .or_default()
                        .insert(record_id, record_json);
                }
                ClusterResponse::Ok
            }

            ClusterCommand::AbortTxn { txn_id } => {
                // Idempotent — no error if the txn isn't open.
                self.open_txns.remove(txn_id);
                ClusterResponse::Ok
            }
        }
    }

    // -- Read accessors --

    /// Return a snapshot of the full cluster state (cloned).
    pub fn snapshot(&self) -> ClusterState {
        ClusterState {
            collections: self.collections.clone(),
            nodes: self.nodes.clone(),
            shard_assignments: self.shard_assignments.clone(),
            records: self.records.clone(),
            open_txns: self.open_txns.keys().cloned().collect(),
            last_applied_index: self.last_applied,
        }
    }

    /// Index of the last applied log entry.
    pub fn last_applied_index(&self) -> u64 {
        self.last_applied
    }

    /// Borrow the collections map.
    pub fn collections(&self) -> &HashMap<String, String> {
        &self.collections
    }

    /// Borrow the nodes map.
    pub fn nodes(&self) -> &HashMap<u64, NodeInfo> {
        &self.nodes
    }

    /// Borrow the shard assignments.
    pub fn shard_assignments(&self) -> &HashMap<String, HashMap<u32, Vec<u64>>> {
        &self.shard_assignments
    }

    /// Look up which nodes host a given shard of a collection.
    pub fn nodes_for_shard(&self, collection: &str, shard_id: u32) -> Vec<u64> {
        self.shard_assignments
            .get(collection)
            .and_then(|shards| shards.get(&shard_id))
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_store_basic_operations() {
        let mut log = LogStore::new();
        assert!(log.is_empty());
        assert_eq!(log.last_index(), 0);

        let idx1 = log.append(ClusterCommand::CreateCollection {
            name: "c1".into(),
            schema_json: "{}".into(),
        });
        assert_eq!(idx1, 1);

        let idx2 = log.append(ClusterCommand::RegisterNode {
            node_id: 1,
            address: "127.0.0.1:9000".into(),
            roles: vec!["data".into()],
        });
        assert_eq!(idx2, 2);
        assert_eq!(log.len(), 2);
        assert_eq!(log.last_index(), 2);

        // Range query
        let range = log.get_range(1, 3);
        assert_eq!(range.len(), 2);

        // Purge first entry
        log.purge(1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.last_index(), 2);
        // Next index should still be 3
        assert_eq!(log.next_index(), 3);
    }

    #[test]
    fn state_machine_create_and_drop_collection() {
        let mut sm = ClusterStateMachine::new();
        let mut log = LogStore::new();

        let idx = log.append(ClusterCommand::CreateCollection {
            name: "vectors".into(),
            schema_json: r#"{"dim": 128}"#.into(),
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        assert!(sm.collections().contains_key("vectors"));

        // Duplicate create should fail.
        let idx = log.append(ClusterCommand::CreateCollection {
            name: "vectors".into(),
            schema_json: "{}".into(),
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert!(matches!(sm.apply(entry), ClusterResponse::Error(_)));

        // Drop
        let idx = log.append(ClusterCommand::DropCollection {
            name: "vectors".into(),
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        assert!(!sm.collections().contains_key("vectors"));
    }

    #[test]
    fn state_machine_register_and_deregister_node() {
        let mut sm = ClusterStateMachine::new();
        let mut log = LogStore::new();

        let idx = log.append(ClusterCommand::RegisterNode {
            node_id: 42,
            address: "10.0.0.1:5000".into(),
            roles: vec!["data".into(), "query".into()],
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);

        let node = sm.nodes().get(&42).unwrap();
        assert_eq!(node.address, "10.0.0.1:5000");
        assert_eq!(node.status, NodeStatus::Active);

        let idx = log.append(ClusterCommand::DeregisterNode { node_id: 42 });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        assert!(!sm.nodes().contains_key(&42));
    }

    #[test]
    fn state_machine_shard_assignment() {
        let mut sm = ClusterStateMachine::new();
        let mut log = LogStore::new();

        // Create collection and register a node first.
        for cmd in [
            ClusterCommand::CreateCollection {
                name: "images".into(),
                schema_json: "{}".into(),
            },
            ClusterCommand::RegisterNode {
                node_id: 1,
                address: "n1:9000".into(),
                roles: vec!["data".into()],
            },
        ] {
            let idx = log.append(cmd);
            let entry = &log.get_range(idx, idx + 1)[0];
            assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        }

        // Assign shard
        let idx = log.append(ClusterCommand::AssignShard {
            collection: "images".into(),
            shard_id: 0,
            node_id: 1,
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        assert_eq!(sm.nodes_for_shard("images", 0), vec![1]);

        // Unassign
        let idx = log.append(ClusterCommand::UnassignShard {
            collection: "images".into(),
            shard_id: 0,
            node_id: 1,
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        assert_eq!(sm.apply(entry), ClusterResponse::Ok);
        assert!(sm.nodes_for_shard("images", 0).is_empty());
    }

    #[test]
    fn state_machine_rejects_out_of_order_apply() {
        let mut sm = ClusterStateMachine::new();
        // Entry at index 2 while last_applied is 0 (expects 1).
        let entry = LogEntry {
            index: 2,
            command: ClusterCommand::DropCollection {
                name: "x".into(),
            },
        };
        let resp = sm.apply(&entry);
        assert!(matches!(resp, ClusterResponse::Error(_)));
    }

    /// Apply a sequence of commands directly to the SM, building
    /// entries with the next expected index each time. Avoids
    /// fresh-LogStore index reuse across helper calls.
    fn drive(sm: &mut ClusterStateMachine, cmds: impl IntoIterator<Item = ClusterCommand>) {
        for cmd in cmds {
            let entry = LogEntry {
                index: sm.last_applied + 1,
                command: cmd,
            };
            let resp = sm.apply(&entry);
            assert_eq!(resp, ClusterResponse::Ok, "expected Ok, got {resp:?}");
        }
    }

    #[test]
    fn txn_commit_flips_staged_writes_into_records() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::CreateCollection {
                    name: "docs".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::BeginTxn { txn_id: "T1".into() },
                ClusterCommand::TxnWrite {
                    txn_id: "T1".into(),
                    collection: "docs".into(),
                    record_id: "a".into(),
                    record_json: "{\"v\":1}".into(),
                },
                ClusterCommand::TxnWrite {
                    txn_id: "T1".into(),
                    collection: "docs".into(),
                    record_id: "b".into(),
                    record_json: "{\"v\":2}".into(),
                },
            ],
        );

        // Pre-commit: records map is empty, txn is in open set.
        assert!(sm.snapshot().records.get("docs").is_none());
        assert_eq!(sm.snapshot().open_txns, vec!["T1".to_owned()]);

        drive(&mut sm, [ClusterCommand::CommitTxn { txn_id: "T1".into() }]);

        let snap = sm.snapshot();
        let docs = snap.records.get("docs").expect("records[\"docs\"]");
        assert_eq!(docs.get("a"), Some(&"{\"v\":1}".to_owned()));
        assert_eq!(docs.get("b"), Some(&"{\"v\":2}".to_owned()));
        assert!(snap.open_txns.is_empty(), "txn cleared on commit");
    }

    #[test]
    fn txn_abort_drops_staged_writes() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::CreateCollection {
                    name: "docs".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::BeginTxn { txn_id: "T1".into() },
                ClusterCommand::TxnWrite {
                    txn_id: "T1".into(),
                    collection: "docs".into(),
                    record_id: "a".into(),
                    record_json: "{\"v\":1}".into(),
                },
                ClusterCommand::AbortTxn { txn_id: "T1".into() },
            ],
        );

        let snap = sm.snapshot();
        assert!(snap.records.get("docs").is_none());
        assert!(snap.open_txns.is_empty(), "txn cleared on abort");
    }

    #[test]
    fn txn_write_against_unknown_txn_is_error() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::CreateCollection {
                name: "docs".into(),
                schema_json: "{}".into(),
            }],
        );
        let mut log = LogStore::new();
        let idx = log.append(ClusterCommand::TxnWrite {
            txn_id: "NEVER_OPENED".into(),
            collection: "docs".into(),
            record_id: "a".into(),
            record_json: "{}".into(),
        });
        let entry = &log.get_range(idx, idx + 1)[0];
        let resp = sm.apply(entry);
        assert!(matches!(resp, ClusterResponse::Error(_)));
    }

    #[test]
    fn txn_begin_is_idempotent() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::BeginTxn { txn_id: "T1".into() },
                ClusterCommand::BeginTxn { txn_id: "T1".into() },
            ],
        );
        assert_eq!(sm.snapshot().open_txns, vec!["T1".to_owned()]);
    }

    #[test]
    fn last_writer_wins_on_concurrent_commit_to_same_key() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::CreateCollection {
                    name: "docs".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::BeginTxn { txn_id: "T1".into() },
                ClusterCommand::TxnWrite {
                    txn_id: "T1".into(),
                    collection: "docs".into(),
                    record_id: "a".into(),
                    record_json: "{\"by\":\"T1\"}".into(),
                },
                ClusterCommand::BeginTxn { txn_id: "T2".into() },
                ClusterCommand::TxnWrite {
                    txn_id: "T2".into(),
                    collection: "docs".into(),
                    record_id: "a".into(),
                    record_json: "{\"by\":\"T2\"}".into(),
                },
                ClusterCommand::CommitTxn { txn_id: "T1".into() },
                ClusterCommand::CommitTxn { txn_id: "T2".into() },
            ],
        );

        let snap = sm.snapshot();
        let docs = snap.records.get("docs").unwrap();
        // T2 commits second — its write wins. v0 has no conflict
        // detection; this is the documented behaviour.
        assert_eq!(docs.get("a"), Some(&"{\"by\":\"T2\"}".to_owned()));
    }
}
