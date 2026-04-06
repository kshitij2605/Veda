//! Type definitions for the Veda cluster consensus layer.
//!
//! Defines the command and response types used by the cluster state machine,
//! as well as the `Consensus` trait that abstracts over different consensus
//! implementations (single-node, Raft, etc.).

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

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
}

impl fmt::Display for ClusterCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateCollection { name, .. } => write!(f, "CreateCollection({name})"),
            Self::DropCollection { name } => write!(f, "DropCollection({name})"),
            Self::RegisterNode { node_id, .. } => write!(f, "RegisterNode({node_id})"),
            Self::DeregisterNode { node_id } => write!(f, "DeregisterNode({node_id})"),
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Draining,
    Inactive,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self::Active
    }
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
    /// Index of the last applied log entry (0 = none applied).
    pub last_applied_index: u64,
}
