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

/// Phase H sentinel key for the cross-region `RegionTopology`
/// JSON blob, stored inside `shard_ensembles`. Chosen to be
/// distinct from any valid `{collection}:{shard_id}` form (no
/// real collection name contains a leading underscore in the
/// schema validator).
pub const REGION_TOPOLOGY_KEY: &str = "_region_topology";

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
        self.entries.push(LogEntry { index, command });
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
        self.entries
            .last()
            .map_or(self.last_purged_index, |e| e.index)
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
    // Phase E hard-remove: `records` and `open_txns` are gone.
    // Record state lives in per-shard `veda-log`; transaction
    // state lives in the Percolator `TxnCoordinator`. Old
    // snapshots round-trip through `veda-admin migrate-metadata`,
    // which reads the v0 layout via a shadow `LegacyV0` struct
    // and rewrites without these fields.
    /// Current TSO lease, if granted. Phase A scaffolding — populated
    /// by `GrantTsLease` / `RenewTsLease`. Phase C wires
    /// `MetaLeasedTso` to read this. See
    /// `.claude/plans/structured-mixing-turing.md` and
    /// `crates/veda-time/src/lease.rs`.
    #[serde(default)]
    ts_lease: Option<veda_time::TsLease>,
    /// Phase E: per-shard BookKeeper-style ensembles for the
    /// data-plane log. Keyed by `{collection}:{shard_id}`. Values
    /// are JSON-encoded `veda_log::LogEnsemble`s.
    #[serde(default)]
    shard_ensembles: HashMap<String, String>,
    /// Phase E: committed-transaction registry. One small row
    /// per txn; the per-write log entries live in per-shard logs.
    #[serde(default)]
    txn_commits: HashMap<String, crate::raft_types::CommitRecord>,
    /// Phase G alias table. `alias_name -> target_collection`.
    /// Populated by `SetCollectionAlias`. Used by the atomic
    /// re-chunk swap so `docs` resolves to `docs@v2` after a
    /// successful reindex without invalidating any URLs.
    #[serde(default)]
    aliases: HashMap<String, String>,
}

// Phase E hard-remove: `PendingTxn` is gone. Transaction state
// lives in the Percolator `TxnCoordinator` (per-shard log
// locks + a `RecordTxnCommit` entry in `_meta::txn_commits`).

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

    /// Phase H: read the persisted `RegionTopology` JSON, if
    /// any. Stored as a sentinel-keyed row in `shard_ensembles`
    /// so we don't need a new top-level field on the state
    /// machine (matches the wan.rs RegionTopology doc note).
    pub fn region_topology_json(&self) -> Option<&str> {
        self.shard_ensembles
            .get(REGION_TOPOLOGY_KEY)
            .map(|s| s.as_str())
    }

    /// Phase G alias resolver. Returns the target collection
    /// name for `name`, following at most one alias hop. If
    /// `name` is not aliased, returns `name` unchanged.
    pub fn resolve_collection<'a>(&'a self, name: &'a str) -> &'a str {
        self.aliases.get(name).map(|s| s.as_str()).unwrap_or(name)
    }

    /// Phase G: read access to the alias map. Used by the
    /// gateway to surface alias state in admin endpoints.
    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
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

            ClusterCommand::GrantTsLease {
                epoch,
                holder_node_id,
                ceiling_ns,
            } => {
                // Reject any grant whose epoch isn't strictly newer
                // than the current lease. This is the fencing
                // invariant for Phase A: one holder per epoch.
                if let Some(current) = &self.ts_lease
                    && current.epoch >= *epoch
                {
                    return ClusterResponse::Error(format!(
                        "GrantTsLease epoch {epoch} not strictly newer than current {}",
                        current.epoch
                    ));
                }
                self.ts_lease = Some(veda_time::TsLease::new(
                    *epoch,
                    *holder_node_id,
                    *ceiling_ns,
                ));
                ClusterResponse::Ok
            }

            ClusterCommand::RenewTsLease { epoch, ceiling_ns } => {
                // Renew requires the epoch to match the current
                // lease. Older epoch => the renewer was fenced.
                // Newer epoch => caller should issue GrantTsLease.
                match &mut self.ts_lease {
                    Some(current) if current.epoch == *epoch => {
                        // Ceiling must only move forward.
                        if *ceiling_ns >= current.ceiling_ns {
                            current.ceiling_ns = *ceiling_ns;
                            ClusterResponse::Ok
                        } else {
                            ClusterResponse::Error(format!(
                                "RenewTsLease ceiling {ceiling_ns} below current {}",
                                current.ceiling_ns
                            ))
                        }
                    }
                    Some(current) => ClusterResponse::Error(format!(
                        "RenewTsLease epoch {epoch} != current {}",
                        current.epoch
                    )),
                    None => {
                        ClusterResponse::Error("RenewTsLease has no current lease to renew".into())
                    }
                }
            }

            ClusterCommand::RegisterShardEnsemble {
                shard_id,
                ensemble_json,
            } => {
                self.shard_ensembles
                    .insert(shard_id.clone(), ensemble_json.clone());
                ClusterResponse::Ok
            }

            ClusterCommand::ChangeShardWriter {
                shard_id,
                new_writer,
                new_epoch,
            } => {
                // Read current ensemble (if any), enforce strict
                // epoch monotonicity, write new writer + epoch.
                let new_ensemble_json = match self.shard_ensembles.get(shard_id) {
                    Some(existing) => match serde_json::from_str::<serde_json::Value>(existing) {
                        Ok(mut v) => {
                            let cur_epoch = v.get("epoch").and_then(|e| e.as_u64()).unwrap_or(0);
                            if *new_epoch <= cur_epoch {
                                return ClusterResponse::Error(format!(
                                    "ChangeShardWriter new_epoch {new_epoch} not strictly > {cur_epoch}"
                                ));
                            }
                            if let Some(obj) = v.as_object_mut() {
                                obj.insert(
                                    "writer_leader".to_owned(),
                                    serde_json::Value::Number((*new_writer).into()),
                                );
                                obj.insert(
                                    "epoch".to_owned(),
                                    serde_json::Value::Number((*new_epoch).into()),
                                );
                            }
                            serde_json::to_string(&v).unwrap_or_else(|_| existing.clone())
                        }
                        Err(_) => {
                            return ClusterResponse::Error(format!(
                                "shard_ensembles[{shard_id}] is not valid JSON"
                            ));
                        }
                    },
                    None => {
                        return ClusterResponse::Error(format!(
                            "ChangeShardWriter: no ensemble registered for {shard_id}"
                        ));
                    }
                };
                self.shard_ensembles
                    .insert(shard_id.clone(), new_ensemble_json);
                ClusterResponse::Ok
            }

            ClusterCommand::RecordTxnCommit {
                txn_id,
                commit_ts_ns,
                primary_shard,
                shards_touched,
            } => {
                self.txn_commits.insert(
                    txn_id.clone(),
                    crate::raft_types::CommitRecord {
                        commit_ts_ns: *commit_ts_ns,
                        primary_shard: primary_shard.clone(),
                        shards_touched: shards_touched.clone(),
                    },
                );
                ClusterResponse::Ok
            }

            ClusterCommand::SetCollectionAlias { alias, target } => {
                // The alias may point at the same name as itself
                // (rollback). The target must be a real
                // collection — otherwise reads via the alias
                // would surface a phantom NotFound, which is
                // worse than rejecting the swap at apply time.
                if !self.collections.contains_key(target) {
                    return ClusterResponse::Error(format!(
                        "SetCollectionAlias: target '{target}' does not exist"
                    ));
                }
                if alias == target {
                    // Pointing an alias at itself = clear it.
                    self.aliases.remove(alias);
                } else {
                    self.aliases.insert(alias.clone(), target.clone());
                }
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
            ts_lease: self.ts_lease.clone(),
            shard_ensembles: self.shard_ensembles.clone(),
            txn_commits: self.txn_commits.clone(),
            aliases: self.aliases.clone(),
            last_applied_index: self.last_applied,
        }
    }

    /// Borrow the current TSO lease, if any.
    pub fn ts_lease(&self) -> Option<&veda_time::TsLease> {
        self.ts_lease.as_ref()
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
            command: ClusterCommand::DropCollection { name: "x".into() },
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

    // Phase E hard-remove: the four txn tests
    // (txn_commit_flips_staged_writes_into_records,
    // txn_abort_drops_staged_writes,
    // txn_write_against_unknown_txn_is_error,
    // txn_begin_is_idempotent) covered the v0
    // BeginTxn/TxnWrite/CommitTxn/AbortTxn variants. With those
    // variants gone, the corresponding behaviour now lives in
    // veda-coordinator::percolator::TxnCoordinator (Prod-5) and
    // is exercised by veda-gateway/tests/txn_test.rs.

    #[test]
    fn grant_ts_lease_then_renew() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::GrantTsLease {
                    epoch: 1,
                    holder_node_id: 7,
                    ceiling_ns: 1_000,
                },
                ClusterCommand::RenewTsLease {
                    epoch: 1,
                    ceiling_ns: 2_000,
                },
            ],
        );
        let lease = sm.ts_lease().expect("lease present");
        assert_eq!(lease.epoch, 1);
        assert_eq!(lease.holder_node_id, 7);
        assert_eq!(lease.ceiling_ns, 2_000);
    }

    #[test]
    fn grant_with_lower_or_equal_epoch_is_error() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::GrantTsLease {
                epoch: 5,
                holder_node_id: 1,
                ceiling_ns: 100,
            }],
        );
        // Try same epoch -> error
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::GrantTsLease {
                epoch: 5,
                holder_node_id: 2,
                ceiling_ns: 200,
            },
        };
        assert!(matches!(sm.apply(&entry), ClusterResponse::Error(_)));
        // Try lower epoch -> error
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::GrantTsLease {
                epoch: 1,
                holder_node_id: 2,
                ceiling_ns: 200,
            },
        };
        assert!(matches!(sm.apply(&entry), ClusterResponse::Error(_)));
        // Lease unchanged
        assert_eq!(sm.ts_lease().unwrap().holder_node_id, 1);
    }

    #[test]
    fn renew_wrong_epoch_is_error() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::GrantTsLease {
                epoch: 5,
                holder_node_id: 1,
                ceiling_ns: 100,
            }],
        );
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::RenewTsLease {
                epoch: 4,
                ceiling_ns: 200,
            },
        };
        assert!(matches!(sm.apply(&entry), ClusterResponse::Error(_)));
    }

    #[test]
    fn renew_ceiling_cannot_go_backwards() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::GrantTsLease {
                epoch: 1,
                holder_node_id: 1,
                ceiling_ns: 1_000,
            }],
        );
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::RenewTsLease {
                epoch: 1,
                ceiling_ns: 500,
            },
        };
        assert!(matches!(sm.apply(&entry), ClusterResponse::Error(_)));
        assert_eq!(sm.ts_lease().unwrap().ceiling_ns, 1_000);
    }

    #[test]
    fn set_collection_alias_swap_and_rollback() {
        let mut sm = ClusterStateMachine::new();
        // Create both source + sibling.
        drive(
            &mut sm,
            [
                ClusterCommand::CreateCollection {
                    name: "docs".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::CreateCollection {
                    name: "docs@v2".into(),
                    schema_json: "{}".into(),
                },
            ],
        );
        assert_eq!(sm.resolve_collection("docs"), "docs");

        // Swap alias docs -> docs@v2.
        drive(
            &mut sm,
            [ClusterCommand::SetCollectionAlias {
                alias: "docs".into(),
                target: "docs@v2".into(),
            }],
        );
        assert_eq!(sm.resolve_collection("docs"), "docs@v2");
        assert_eq!(sm.aliases().get("docs"), Some(&"docs@v2".to_owned()));

        // Rollback: SetCollectionAlias { alias: docs, target: docs }
        // clears the alias.
        drive(
            &mut sm,
            [ClusterCommand::SetCollectionAlias {
                alias: "docs".into(),
                target: "docs".into(),
            }],
        );
        assert_eq!(sm.resolve_collection("docs"), "docs");
        assert!(sm.aliases().is_empty());
    }

    #[test]
    fn set_collection_alias_rejects_unknown_target() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::CreateCollection {
                name: "docs".into(),
                schema_json: "{}".into(),
            }],
        );
        // Try to alias to a target that doesn't exist.
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::SetCollectionAlias {
                alias: "docs".into(),
                target: "ghost".into(),
            },
        };
        match sm.apply(&entry) {
            ClusterResponse::Error(msg) => assert!(msg.contains("does not exist"), "got {msg}"),
            other => panic!("expected Error, got {other:?}"),
        }
        // Alias map is unchanged.
        assert!(sm.aliases().is_empty());
    }

    #[test]
    fn aliases_round_trip_through_snapshot() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::CreateCollection {
                    name: "a".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::CreateCollection {
                    name: "b".into(),
                    schema_json: "{}".into(),
                },
                ClusterCommand::SetCollectionAlias {
                    alias: "a".into(),
                    target: "b".into(),
                },
            ],
        );
        let snap = sm.snapshot();
        assert_eq!(snap.aliases.get("a"), Some(&"b".to_owned()));
    }

    #[test]
    fn region_topology_round_trips_through_register_shard_ensemble() {
        let mut sm = ClusterStateMachine::new();
        // Before set: not present.
        assert!(sm.region_topology_json().is_none());
        // The operator writes the topology via the same
        // RegisterShardEnsemble path, with the sentinel key.
        let topology = r#"{"primary_region":"us-east-1","replica_regions":["eu-west-1"],"staleness_target_ms":5000,"epoch":1}"#;
        drive(
            &mut sm,
            [ClusterCommand::RegisterShardEnsemble {
                shard_id: REGION_TOPOLOGY_KEY.into(),
                ensemble_json: topology.to_owned(),
            }],
        );
        assert_eq!(sm.region_topology_json(), Some(topology));
        // Operator advances the topology epoch on a cutover.
        let topology_v2 = r#"{"primary_region":"eu-west-1","replica_regions":["us-east-1"],"staleness_target_ms":5000,"epoch":2}"#;
        drive(
            &mut sm,
            [ClusterCommand::RegisterShardEnsemble {
                shard_id: REGION_TOPOLOGY_KEY.into(),
                ensemble_json: topology_v2.to_owned(),
            }],
        );
        assert_eq!(sm.region_topology_json(), Some(topology_v2));
        // Round-trips through the snapshot.
        let snap = sm.snapshot();
        assert_eq!(
            snap.shard_ensembles
                .get(REGION_TOPOLOGY_KEY)
                .map(|s| s.as_str()),
            Some(topology_v2)
        );
    }

    #[test]
    fn register_shard_ensemble_stores_json_blob() {
        let mut sm = ClusterStateMachine::new();
        let ensemble_json =
            r#"{"writers":[1,2,3],"write_quorum":3,"ack_quorum":2,"epoch":1,"writer_leader":1}"#;
        drive(
            &mut sm,
            [ClusterCommand::RegisterShardEnsemble {
                shard_id: "docs:0".into(),
                ensemble_json: ensemble_json.to_owned(),
            }],
        );
        let snap = sm.snapshot();
        assert_eq!(
            snap.shard_ensembles.get("docs:0"),
            Some(&ensemble_json.to_owned())
        );
    }

    #[test]
    fn change_shard_writer_requires_strictly_newer_epoch() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [
                ClusterCommand::RegisterShardEnsemble {
                    shard_id: "docs:0".into(),
                    ensemble_json: r#"{"writers":[1,2,3],"write_quorum":3,"ack_quorum":2,"epoch":3,"writer_leader":1}"#.to_owned(),
                },
                ClusterCommand::ChangeShardWriter {
                    shard_id: "docs:0".into(),
                    new_writer: 2,
                    new_epoch: 4,
                },
            ],
        );
        let snap = sm.snapshot();
        let stored = snap.shard_ensembles.get("docs:0").unwrap();
        let v: serde_json::Value = serde_json::from_str(stored).unwrap();
        assert_eq!(v["writer_leader"].as_u64(), Some(2));
        assert_eq!(v["epoch"].as_u64(), Some(4));

        // Same-epoch ChangeShardWriter must be rejected.
        let entry = LogEntry {
            index: sm.last_applied + 1,
            command: ClusterCommand::ChangeShardWriter {
                shard_id: "docs:0".into(),
                new_writer: 3,
                new_epoch: 4,
            },
        };
        assert!(matches!(sm.apply(&entry), ClusterResponse::Error(_)));
    }

    #[test]
    fn record_txn_commit_stores_metadata() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::RecordTxnCommit {
                txn_id: "T1".into(),
                commit_ts_ns: 1_700_000_000_000_000_000,
                primary_shard: "docs:0".into(),
                shards_touched: vec!["docs:0".into(), "docs:1".into()],
            }],
        );
        let snap = sm.snapshot();
        let rec = snap.txn_commits.get("T1").unwrap();
        assert_eq!(rec.commit_ts_ns, 1_700_000_000_000_000_000);
        assert_eq!(rec.shards_touched.len(), 2);
    }

    #[test]
    fn ts_lease_round_trips_through_snapshot() {
        let mut sm = ClusterStateMachine::new();
        drive(
            &mut sm,
            [ClusterCommand::GrantTsLease {
                epoch: 9,
                holder_node_id: 42,
                ceiling_ns: 12_345,
            }],
        );
        let snap = sm.snapshot();
        let lease = snap.ts_lease.unwrap();
        assert_eq!(lease.epoch, 9);
        assert_eq!(lease.holder_node_id, 42);
        assert_eq!(lease.ceiling_ns, 12_345);
    }

    // Phase E hard-remove: the last-writer-wins-on-concurrent-
    // commit test exercised the v0 TxnWrite/CommitTxn variants
    // which are now gone. Equivalent semantics under the
    // Percolator coordinator are exercised by
    // veda-coordinator/src/percolator.rs unit tests.
}
