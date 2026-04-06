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
//!
//! The current implementation uses a single-leader, channel-based approach.
//! The types and state machine are designed so that a full Raft consensus
//! layer (via `openraft`) can be plugged in later without changing the
//! command/response interface.

pub mod network;
pub mod raft_types;
pub mod store;
