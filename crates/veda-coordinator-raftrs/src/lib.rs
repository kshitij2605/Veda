//! Surface-area-only stub for a hypothetical tikv/raft-rs swap.
//!
//! **This crate is a design artifact, not a working implementation.**
//! Every function below returns `unimplemented!()`. It exists so that
//! anyone evaluating "should we swap openraft for raft-rs?" can see
//! exactly which traits, methods, and types they would need to fill
//! in. Currently the answer is no — see
//! [`documentation/consensus-swap-decision.md`](../../documentation/consensus-swap-decision.md)
//! for the reasoning.
//!
//! ## Why the stub exists
//!
//! openraft has publicly declined to add Raft pre-vote (see
//! Discussion #15 upstream; verified against `raft_core.rs` in
//! openraft 0.10.0-alpha.20 — no `pre_vote` references). That means
//! the disruptive-election-on-partition-heal failure mode is
//! narrowed by openraft's `Vote` design but never fully eliminated.
//! Today the decision is to live with that and mitigate at the
//! application layer (CheckQuorum, jitter, single-term-leader).
//! This crate captures the surface area of the *only* alternative
//! that delivers true pre-vote — a swap to tikv/raft-rs 0.7 — so
//! the 3-5 engineer-week cost stays visible if the operational
//! calculus ever changes.
//!
//! ## Surface area
//!
//! A raft-rs swap touches four traits and one driver loop:
//!
//! - [`Storage`]: log read/append, state save, snapshot install /
//!   create. raft-rs uses a synchronous "pull" model: the core
//!   reads `Storage` directly. Our current `MemRaftStorage` is
//!   async — that's the first impedance mismatch.
//! - [`Transport`]: send/recv raft messages. raft-rs delivers
//!   messages by returning a `Ready` value; we'd own the recv side
//!   ourselves.
//! - [`StateMachine`]: apply committed entries. raft-rs doesn't
//!   own this — we already have `ClusterStateMachine` and would
//!   re-use it.
//! - [`Driver`]: the Ready loop. raft-rs requires a step → ready →
//!   handle_ready → advance cycle, all synchronous. Today the
//!   openraft adapter is async + actor-style.
//!
//! Filling all four in is the ~3-5 engineer-week cost the swap
//! research estimates.

use raft::Storage as RaftStorage;
use raft::{
    eraftpb::{ConfState, Entry, HardState, Message, Snapshot},
    storage::MemStorage,
    Config, RaftState, RawNode,
};

/// Storage abstraction that mirrors our `MemRaftStorage` but
/// wraps `raft::Storage` instead of openraft's
/// `RaftLogStorage`+`RaftStateMachine`+`RaftSnapshotBuilder`.
///
/// In the real implementation, every method body becomes a call
/// through to the underlying durable store (per-entry WAL +
/// snapshot file). For the stub all are `unimplemented!()`.
pub struct VedaRaftRsStorage {
    inner: MemStorage,
}

impl VedaRaftRsStorage {
    pub fn new() -> Self {
        Self {
            inner: MemStorage::new(),
        }
    }
}

impl Default for VedaRaftRsStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftStorage for VedaRaftRsStorage {
    fn initial_state(&self) -> raft::Result<RaftState> {
        unimplemented!("VedaRaftRsStorage::initial_state — stub")
    }

    fn entries(
        &self,
        _low: u64,
        _high: u64,
        _max_size: impl Into<Option<u64>>,
        _context: raft::GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        unimplemented!("VedaRaftRsStorage::entries — stub")
    }

    fn term(&self, _idx: u64) -> raft::Result<u64> {
        unimplemented!("VedaRaftRsStorage::term — stub")
    }

    fn first_index(&self) -> raft::Result<u64> {
        unimplemented!("VedaRaftRsStorage::first_index — stub")
    }

    fn last_index(&self) -> raft::Result<u64> {
        unimplemented!("VedaRaftRsStorage::last_index — stub")
    }

    fn snapshot(&self, _request_index: u64, _to: u64) -> raft::Result<Snapshot> {
        unimplemented!("VedaRaftRsStorage::snapshot — stub")
    }
}

/// Surface for what `MemRaftStorage`'s write side becomes in the
/// raft-rs world. raft-rs delivers writes via `Ready` — the
/// implementer reads them out of `Ready` and persists them here.
pub trait WriteableStorage {
    /// Persist a new hard state (current term + vote + commit).
    fn save_hard_state(&mut self, hs: &HardState) -> Result<(), StubError>;
    /// Append log entries (durably).
    fn append(&mut self, entries: &[Entry]) -> Result<(), StubError>;
    /// Install a snapshot, truncating the log to its index.
    fn apply_snapshot(&mut self, snapshot: Snapshot) -> Result<(), StubError>;
    /// Persist the latest conf state (membership).
    fn save_conf_state(&mut self, cs: &ConfState) -> Result<(), StubError>;
}

impl WriteableStorage for VedaRaftRsStorage {
    fn save_hard_state(&mut self, _hs: &HardState) -> Result<(), StubError> {
        unimplemented!("WriteableStorage::save_hard_state — stub")
    }
    fn append(&mut self, _entries: &[Entry]) -> Result<(), StubError> {
        unimplemented!("WriteableStorage::append — stub")
    }
    fn apply_snapshot(&mut self, _snapshot: Snapshot) -> Result<(), StubError> {
        unimplemented!("WriteableStorage::apply_snapshot — stub")
    }
    fn save_conf_state(&mut self, _cs: &ConfState) -> Result<(), StubError> {
        unimplemented!("WriteableStorage::save_conf_state — stub")
    }
}

/// raft-rs transport — recv (push from network into core via
/// `RawNode::step`) and send (drain `Ready::messages` to the wire).
pub trait Transport {
    fn send(&self, msg: Message) -> Result<(), StubError>;
}

/// Stand-in for the `Driver` actor we'd need: a loop that calls
/// `RawNode::tick` on a timer, drains `Ready` after each step,
/// persists log entries via [`WriteableStorage`], dispatches
/// messages via [`Transport`], and applies committed entries
/// to our existing `ClusterStateMachine`.
pub struct Driver {
    pub node: RawNode<VedaRaftRsStorage>,
}

impl Driver {
    pub fn new(_config: Config) -> Result<Self, StubError> {
        unimplemented!("Driver::new — stub. Build RawNode against VedaRaftRsStorage here.")
    }

    /// One iteration of the Ready loop. The implementer's job:
    ///   1. `node.tick()` if elapsed >= tick interval.
    ///   2. Read messages from the wire, `node.step(msg)`.
    ///   3. If `node.has_ready()`, take Ready and:
    ///        a. persist hard_state if changed
    ///        b. apply snapshot if present
    ///        c. append entries to log
    ///        d. send Ready::messages over Transport
    ///        e. apply committed entries to state machine
    ///        f. `node.advance(ready)`
    pub fn step_once(&mut self) -> Result<(), StubError> {
        unimplemented!("Driver::step_once — stub")
    }
}

/// Stub error type for the surface methods. The real impl would
/// wrap raft-rs errors + storage IO errors + transport errors.
#[derive(Debug, thiserror::Error)]
pub enum StubError {
    #[error("stub: not implemented")]
    NotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mostly a compile fence — proves the crate at least
    /// type-checks against the raft-rs API surface we'd import.
    /// Calling into methods will trigger `unimplemented!()`.
    #[test]
    fn types_resolve() {
        let _ = VedaRaftRsStorage::new();
    }
}
