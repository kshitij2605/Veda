//! veda-coordinator: Cluster coordination, consensus, and shard routing.
//!
//! This crate implements the coordination layer for the Veda distributed
//! vector database. It manages cluster state (collections, nodes, shard
//! assignments) through a replicated state machine.
//!
//! ## Architecture
//!
//! - **`raft_types`** — Command, response, and state types shared across
//!   consensus implementations.
//! - **`store`** — In-memory log store and deterministic state machine.
//! - **`network`** — Channel-based coordinator and in-process network
//!   transport for testing.
//! - **`raft`** — openraft type-config scaffolding for distributed
//!   consensus (work in progress; see `documentation/raft-migration.md`).
//!
//! The current production path uses a single-leader, channel-based approach.
//! The types and state machine are designed so that the openraft layer can
//! take over without changing the command/response interface.

pub mod linearisability_checker;
pub mod meta_leased_tso;
pub mod multi_group;
pub mod network;
pub mod percolator;
pub mod peer_health;
pub mod raft;
pub mod raft_coordinator;
pub mod raft_network;
pub mod raft_storage;
pub mod raft_tonic;
pub mod raft_types;
pub mod raft_wal;
pub mod shard_router;
pub mod store;

// Re-exports so downstream crates (the gateway) don't have to declare
// their own openraft dep just to mention a couple of types.
pub use openraft::ServerState as RaftServerState;
