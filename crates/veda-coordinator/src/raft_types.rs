//! Type definitions for the Veda cluster consensus layer.
//!
//! Defines the command and response types used by the cluster state machine,
//! as well as the `Consensus` trait that abstracts over different consensus
//! implementations (single-node, Raft, etc.).

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use veda_time::TsLease;

// ---------------------------------------------------------------------------
// ClusterCommand — commands that modify cluster state
// ---------------------------------------------------------------------------

/// A command that modifies cluster state. These are the "log entries" that
/// the consensus layer replicates and then applies to the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterCommand {
    // -- Schema / collection operations --
    /// Create a new collection with the given schema (stored as JSON).
    CreateCollection {
        name: String,
        schema_json: String,
    },
    /// Drop an existing collection.
    DropCollection {
        name: String,
    },

    // -- Node lifecycle --
    /// Register a node in the cluster.
    RegisterNode {
        node_id: u64,
        address: String,
        roles: Vec<String>,
    },
    /// Remove a node from the cluster.
    DeregisterNode {
        node_id: u64,
    },

    /// No-op log entry. Used as a linearisable-read barrier:
    /// a caller that needs a read reflecting every committed entry
    /// up to "now" issues `Noop` first; the state machine apply
    /// pathway treats it as a no-op (just advances last_applied),
    /// and once it commits the read happens against state that's
    /// guaranteed to include every prior write.
    Noop,

    /// Insert one record into a collection. The cluster state
    /// machine doesn't materialise the data itself (record blobs
    /// can be large); on apply, each node uses the
    /// `(collection, record_id, record_json)` tuple to drive its
    /// local SegmentManager. The leader assigns `record_id` before
    /// proposing so all nodes agree on identity.
    ///
    /// **Performance trade-off**: routing every insert through
    /// one cluster-wide Raft caps write throughput at the leader's
    /// single-node speed. Real horizontal scaling needs per-shard
    /// Raft groups (CockroachDB / TiDB pattern) — a fundamental
    /// architectural refactor of veda-coordinator that's out of
    /// scope here. The current implementation is correct (every
    /// node converges to the same data) but not yet fast under
    /// high write loads. See documentation/raft-migration.md.
    InsertRecord {
        collection: String,
        record_id: String,
        record_json: String,
    },

    // -- Shard assignment --
    /// Assign a shard of a collection to a node.
    AssignShard {
        collection: String,
        shard_id: u32,
        node_id: u64,
    },
    /// Remove a shard assignment.
    UnassignShard {
        collection: String,
        shard_id: u32,
        node_id: u64,
    },

    // -- Transactions (v0 scaffolding — NOT production Percolator) --
    //
    // The v0 transaction model is a strict 2PC-style stage-and-flip:
    // BeginTxn opens a holder, TxnWrite stages each record write
    // under it, CommitTxn moves every staged write into the main
    // record map atomically, AbortTxn drops the holder + its writes.
    // Replicated through Raft, so every node sees the same outcome.
    //
    // What this v0 deliberately does NOT do:
    //   * Optimistic / pessimistic concurrency control. Two
    //     concurrent txns writing the same key both succeed; the
    //     last commit wins.
    //   * MVCC / snapshot isolation. Reads bypass the transaction
    //     overlay completely — read-your-writes within a txn is
    //     NOT supported.
    //   * Distributed coordination across shards. The implementation
    //     funnels every staged write through the cluster-wide group,
    //     same as InsertRecord.
    //   * Deadlock detection, retry on conflict, lock timeouts.
    //   * Large transactions — every staged write sits in memory on
    //     every node until commit / abort.
    //
    // Roadmap to production: vendor TiKV's `components/concurrency_manager`
    // (Apache-2.0, ~4k LoC) + `txn_types` (~3k LoC) for real Percolator
    // semantics. See research/2026-05-19-multi-week-survey/01-cross-shard-transactions.md
    /// Open a new transaction holder. Idempotent: re-issuing the
    /// same txn_id returns Ok without mutating state.
    BeginTxn {
        txn_id: String,
    },
    /// Stage a record write under an open transaction. Same shape
    /// as InsertRecord, plus a `txn_id`. Apply must check that the
    /// transaction is open and the collection exists.
    TxnWrite {
        txn_id: String,
        collection: String,
        record_id: String,
        record_json: String,
    },
    /// Commit a transaction: move every staged write into the main
    /// record map and drop the holder. No-op if the txn isn't open.
    CommitTxn {
        txn_id: String,
    },
    /// Abort a transaction: drop the holder and every staged write.
    /// No-op if the txn isn't open.
    AbortTxn {
        txn_id: String,
    },

    // -- TSO lease (Phase A scaffolding — see
    //    `.claude/plans/structured-mixing-turing.md`)
    //
    // The metadata Raft is the source of truth for which node holds
    // the TSO right-to-vend. A lease is `(epoch, holder, ceiling)`.
    // Phase A applies these commands on the SM but no consumer reads
    // the lease state yet — Phase C wires `MetaLeasedTso` to it.
    /// Grant the TSO lease to `holder` at `epoch`, valid until
    /// `ceiling_ns`. Rejected if `epoch` does not strictly exceed
    /// the current lease's epoch.
    GrantTsLease {
        epoch: u64,
        holder_node_id: u64,
        ceiling_ns: u64,
    },
    /// Extend the current lease's ceiling. Rejected unless `epoch`
    /// equals the current lease's epoch.
    RenewTsLease {
        epoch: u64,
        ceiling_ns: u64,
    },

    // -- Phase E: slim metadata Raft ----------------------------
    //
    // The cluster-wide records HashMap stays present in v0 for
    // backwards-compat; new deployments use these commands to
    // register per-shard ensembles + record txn commits without
    // routing record bodies through the metadata Raft.
    /// Register / update the BookKeeper-style ensemble for a
    /// shard's data-plane log. Stored in `ClusterStateMachine::
    /// ensembles[shard_id]`. Carried as JSON to keep the
    /// metadata SM independent of `veda-log` types (avoids a
    /// dep cycle).
    RegisterShardEnsemble {
        shard_id: String,
        ensemble_json: String,
    },
    /// Transfer writer leadership for a shard's data-plane log
    /// to `new_writer` at `new_epoch`. The state machine apply
    /// path rejects any non-strictly-newer epoch.
    ChangeShardWriter {
        shard_id: String,
        new_writer: u64,
        new_epoch: u64,
    },
    /// Record that a transaction has committed. Replaces the
    /// per-write `TxnWrite` + `CommitTxn` chatter for production
    /// transactions; carries one small record per committed
    /// txn, not per write. Phase G consumes this; Phase E just
    /// stores it.
    RecordTxnCommit {
        txn_id: String,
        commit_ts_ns: u64,
        primary_shard: String,
        shards_touched: Vec<String>,
    },
}

impl fmt::Display for ClusterCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateCollection { name, .. } => write!(f, "CreateCollection({name})"),
            Self::DropCollection { name } => write!(f, "DropCollection({name})"),
            Self::RegisterNode { node_id, .. } => write!(f, "RegisterNode({node_id})"),
            Self::DeregisterNode { node_id } => write!(f, "DeregisterNode({node_id})"),
            Self::Noop => write!(f, "Noop"),
            Self::InsertRecord { collection, record_id, .. } => {
                write!(f, "InsertRecord({collection}/{record_id})")
            }
            Self::AssignShard {
                collection,
                shard_id,
                node_id,
            } => write!(f, "AssignShard({collection}/{shard_id}->{node_id})"),
            Self::UnassignShard {
                collection,
                shard_id,
                node_id,
            } => write!(f, "UnassignShard({collection}/{shard_id}->{node_id})"),
            Self::BeginTxn { txn_id } => write!(f, "BeginTxn({txn_id})"),
            Self::TxnWrite { txn_id, collection, record_id, .. } => {
                write!(f, "TxnWrite({txn_id}: {collection}/{record_id})")
            }
            Self::CommitTxn { txn_id } => write!(f, "CommitTxn({txn_id})"),
            Self::AbortTxn { txn_id } => write!(f, "AbortTxn({txn_id})"),
            Self::GrantTsLease { epoch, holder_node_id, .. } => {
                write!(f, "GrantTsLease(epoch={epoch}, holder={holder_node_id})")
            }
            Self::RenewTsLease { epoch, ceiling_ns } => {
                write!(f, "RenewTsLease(epoch={epoch}, ceiling_ns={ceiling_ns})")
            }
            Self::RegisterShardEnsemble { shard_id, .. } => {
                write!(f, "RegisterShardEnsemble({shard_id})")
            }
            Self::ChangeShardWriter { shard_id, new_writer, new_epoch } => {
                write!(
                    f,
                    "ChangeShardWriter({shard_id} -> {new_writer} @ epoch {new_epoch})"
                )
            }
            Self::RecordTxnCommit { txn_id, commit_ts_ns, .. } => {
                write!(f, "RecordTxnCommit({txn_id} @ {commit_ts_ns})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ClusterResponse — result of applying a command
// ---------------------------------------------------------------------------

/// The response returned after a command has been applied to the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterResponse {
    /// Command applied successfully.
    Ok,
    /// Command failed with the given reason.
    Error(String),
}

// ---------------------------------------------------------------------------
// NodeInfo — metadata about a cluster node
// ---------------------------------------------------------------------------

/// Status of a node in the cluster.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    #[default]
    Active,
    Draining,
    Inactive,
}

/// Information about a single node in the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub address: String,
    pub roles: Vec<String>,
    pub status: NodeStatus,
}

// ---------------------------------------------------------------------------
// LogEntry — a log entry wrapping a command with an index
// ---------------------------------------------------------------------------

/// A single entry in the replicated log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Monotonically increasing index (1-based).
    pub index: u64,
    /// The command stored in this entry.
    pub command: ClusterCommand,
}

// ---------------------------------------------------------------------------
// ClusterState — read-only snapshot of the state machine (for queries)
// ---------------------------------------------------------------------------

/// A read-only view of the cluster state, returned by queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterState {
    pub collections: HashMap<String, String>,
    pub nodes: HashMap<u64, NodeInfo>,
    /// collection -> shard_id -> [node_ids]
    pub shard_assignments: HashMap<String, HashMap<u32, Vec<u64>>>,
    /// collection -> record_id -> record_json (replicated record data)
    #[serde(default)]
    pub records: HashMap<String, HashMap<String, String>>,
    /// Open transaction ids — surfaces to the gateway so the REST
    /// layer can reject writes against unknown txns without a
    /// round-trip.
    #[serde(default)]
    pub open_txns: Vec<String>,
    /// Current TSO lease, if granted. Phase A populates this via
    /// `GrantTsLease` / `RenewTsLease`; Phase C consumers read it
    /// to decide whether to honour requests from a given holder.
    #[serde(default)]
    pub ts_lease: Option<TsLease>,
    /// Per-shard BookKeeper-style ensembles. Keyed by
    /// `{collection}:{shard_id}`. Values are JSON-encoded
    /// `veda_log::LogEnsemble`s — JSON rather than the typed
    /// shape so `veda-coordinator` doesn't depend on `veda-log`
    /// (would be a cycle once Phase D wires the data plane in).
    /// Phase E ships the field; Phase F+ consumers will read it.
    #[serde(default)]
    pub shard_ensembles: HashMap<String, String>,
    /// Committed transactions, keyed by txn_id. One row per
    /// txn — NOT one row per write (those live in the per-shard
    /// logs). Populated by `RecordTxnCommit`. Phase G consumes
    /// during snapshot reads.
    #[serde(default)]
    pub txn_commits: HashMap<String, CommitRecord>,
    /// Index of the last applied log entry (0 = none applied).
    pub last_applied_index: u64,
}

/// One committed transaction, as recorded in the metadata Raft.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRecord {
    pub commit_ts_ns: u64,
    pub primary_shard: String,
    pub shards_touched: Vec<String>,
}
