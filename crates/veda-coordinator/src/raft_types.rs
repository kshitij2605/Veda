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
    // Phase E hard-remove: `InsertRecord`, `BeginTxn`,
    // `TxnWrite`, `CommitTxn`, and `AbortTxn` have been deleted.
    // Record writes go through per-shard `veda-log` (Phase C);
    // transactions go through the Percolator `TxnCoordinator`
    // (Phase G). Existing pre-v1 `raft_state.bin` snapshots
    // migrate via the `veda-admin migrate-metadata` tool, which
    // uses a `LegacyV0` shadow type to read the old layout.

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

    /// Phase G atomic re-chunk: swap an alias so subsequent
    /// reads of `alias` resolve to `target`. `target` must
    /// already exist in `collections` at apply time (the
    /// command rejects otherwise). Setting an alias to a
    /// non-existent target is the rollback handle —
    /// `SetCollectionAlias { alias: "docs", target: "docs" }`
    /// makes `docs` resolve to itself again, undoing a prior
    /// reindex.
    SetCollectionAlias {
        alias: String,
        target: String,
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
            Self::SetCollectionAlias { alias, target } => {
                write!(f, "SetCollectionAlias({alias} -> {target})")
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
    // Phase E hard-remove: `records` and `open_txns` are gone.
    // Record state lives in per-shard `veda-log`; txn state
    // lives in the Percolator `TxnCoordinator`.
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
    /// Phase G alias map: `alias_name -> target_collection`.
    /// Reads via `alias_name` resolve to `target_collection`.
    /// Populated by `SetCollectionAlias`. Used by the atomic
    /// re-chunk swap: original `docs` becomes an alias pointing
    /// at the newly-built `docs@v2`.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
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
