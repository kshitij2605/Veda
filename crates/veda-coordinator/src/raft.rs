//! openraft scaffolding for distributed cluster consensus.
//!
//! ## Status
//!
//! The types and trait skeleton here are the integration surface — they
//! exist so the rest of the codebase can refer to a real Raft type, and so
//! that the migration from the current channel-based coordinator (see
//! `network.rs`) to a fully distributed Raft is a localised change.
//!
//! What works today:
//!   * Type aliases for the openraft type-config.
//!   * `VedaTypeConfig` — wires our existing `ClusterCommand`/`ClusterResponse`
//!     types into openraft's request/response surface.
//!   * A `MemRaftLogStorage` skeleton that delegates to our in-memory
//!     [`crate::store::LogStore`].
//!
//! What does NOT yet work, and is intentionally left as `unimplemented!()`
//! with a `// TODO(raft):` marker:
//!   * `RaftLogStorage` trait methods other than `get_log_entries` and
//!     `append`. Requires implementing log truncation, snapshotting, and
//!     persistent vote storage.
//!   * `RaftStateMachine` adapter around `ClusterStateMachine` — needs
//!     snapshot install / build plus the new applied-state hooks.
//!   * `RaftNetwork` — the existing `ChannelNetwork` is a placeholder for
//!     in-process tests; a real implementation needs a Tonic/gRPC transport
//!     keyed by `NodeId` -> `address`.
//!   * Cluster bootstrap / membership change orchestration.
//!
//! See `documentation/raft-migration.md` for the staged plan from current
//! single-leader state to a 3-node Raft quorum.

use openraft::{BasicNode, declare_raft_types};
use serde::{Deserialize, Serialize};

use crate::raft_types::{ClusterCommand, ClusterResponse};

/// Node ID type used across openraft.
pub type NodeId = u64;

/// A snapshot of the in-memory log + state machine, suitable for `InstallSnapshot`.
/// Only the minimum needed to drive openraft — actual snapshot content lives
/// inside the state machine.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VedaSnapshot {
    /// Index of the last applied log entry at snapshot time.
    pub last_applied_index: u64,
    /// Serialised cluster state machine bytes.
    pub state_bytes: Vec<u8>,
}

// Wire our existing types into openraft's type config.
declare_raft_types!(
    pub VedaTypeConfig:
        D = ClusterCommand,
        R = ClusterResponse,
        NodeId = NodeId,
        Node = BasicNode,
        Entry = openraft::Entry<VedaTypeConfig>,
        SnapshotData = std::io::Cursor<Vec<u8>>,
        AsyncRuntime = openraft::TokioRuntime,
);

/// Alias for `Raft<VedaTypeConfig>` so callers don't have to spell the config.
pub type VedaRaft = openraft::Raft<VedaTypeConfig>;
