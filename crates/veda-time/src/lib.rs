//! Hybrid Logical Clock + Timestamp Oracle.
//!
//! Phase A of the FoundationDB-shape rearchitecture
//! (`.claude/plans/structured-mixing-turing.md`). This crate ships
//! the time abstraction; downstream phases wire it through writes,
//! reads, and the metadata Raft.
//!
//! ## What's here
//!
//! - [`Hlc`] — a 64-bit physical ns + 32-bit logical pair with
//!   lexicographic ordering. Two Hlcs from anywhere in the
//!   cluster can be compared.
//! - [`HlcClock`] — a stateful clock that produces monotonic Hlcs
//!   via [`HlcClock::now`] and absorbs remote timestamps via
//!   [`HlcClock::update`]. Follows the CockroachDB
//!   `pkg/util/hlc/hlc.go` algorithm.
//! - [`Tso`] — async trait for vending timestamps. Two impls:
//!   - [`LocalTso`] — single-node, monotonic on local HlcClock.
//!     Default for tests and pre-Phase-E single-node deployments.
//!   - `MetaLeasedTso` — talks to `_meta` via a lease-grant
//!     command. Lives in `veda-coordinator` (where the lease
//!     state machine sits); not in this crate.
//! - [`TsLease`] — the replicated lease record. Defined here so
//!   both `veda-time` and `veda-coordinator` can serde it without
//!   either pulling the other as a dependency.
//!
//! ## What's *not* here (yet)
//!
//! - The `MetaLeasedTso` impl (Phase C wires it).
//! - Lease-handoff failover semantics (Phase A follow-up).
//! - Batched allocation (will land when the gateway needs >1
//!   ts per RPC).

pub mod hlc;
pub mod lease;
pub mod tso;

pub use hlc::{Hlc, HlcClock};
pub use lease::TsLease;
pub use tso::{LocalTso, Tso, TsoError};

/// A timestamp = an `Hlc`. Re-exported as `Ts` because that's
/// what downstream call sites read more naturally.
pub type Ts = Hlc;
