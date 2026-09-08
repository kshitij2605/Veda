//! openraft type configuration for the `_meta` consensus group.
//!
//! ## Status
//!
//! This module is the type surface only: the `VedaTypeConfig` that wires
//! [`ClusterCommand`] / [`ClusterResponse`] into openraft, plus the snapshot
//! payload shape. The implementations live next door —
//! [`crate::raft_storage`] (log + state machine, durable via
//! [`crate::raft_wal`]), [`crate::raft_tonic`] (gRPC transport),
//! [`crate::raft_coordinator`] (the `execute` / `query_state` handle), and
//! [`crate::peer_health`] (cross-node liveness).
//!
//! **What this module's header used to say was wrong for three months.** It
//! described `RaftLogStorage` methods, the state-machine adapter, the Tonic
//! transport and cluster bootstrap as unimplemented stubs. All four shipped
//! on 2026-05-19/20 (`6ef5d46`, `7374a3e`, `0641a1f`, `9095454`); no such
//! stub remains anywhere in this crate. That stale text was
//! copied into the project's task list and cost a later session real time,
//! which is why this note is here rather than a silent deletion.
//!
//! ## Scope: `_meta` only
//!
//! Per ADR-013 this group carries **metadata** — schema, shard plan, node
//! membership, TSO leases, transaction commit records, aliases, region
//! topology. It does **not** carry record writes. Those go through per-shard
//! `veda-log` (BookKeeper-style `(E, Qw, Qa)` quorum with epoch fencing).
//!
//! Routing every record write through one Raft group was the original design
//! and was abandoned deliberately: it caps write throughput at a single
//! leader's speed and grows the metadata snapshot linearly with the dataset.
//! The `ClusterCommand::InsertRecord` / `TxnWrite` / `BeginTxn` / `CommitTxn`
//! / `AbortTxn` variants that design needed were hard-removed in `c8baf29`.
//!
//! See `documentation/raft-migration.md` for the historical staging (its
//! Phase 4-6 sections describe the abandoned record-write path) and
//! `documentation/rearchitecture-overview.md` for where it landed.
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
