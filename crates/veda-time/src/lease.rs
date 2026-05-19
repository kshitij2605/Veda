//! Replicated lease record granted by `_meta` to a TSO holder.
//!
//! The lease binds a `(epoch, holder_node_id)` pair to a `ceiling`
//! — a wall-clock instant past which the holder MUST NOT vend
//! timestamps. The metadata Raft is the source of truth; the TSO
//! holder writes `RenewTsLease` before the ceiling, and any other
//! node observing the lease can refuse stale reads.
//!
//! Phase A ships only the struct + `_meta` apply path. The
//! `MetaLeasedTso` impl that uses it lives in `veda-coordinator`
//! and lands in a Phase A follow-up.

use serde::{Deserialize, Serialize};

/// A grant of TSO authority from `_meta` to a node.
///
/// Invariants:
/// - `epoch` is strictly monotonic per cluster (enforced by `_meta`
///   apply). Newer epoch supersedes older.
/// - `ceiling_ns` is the wall-clock UNIX-epoch nanosecond past which
///   the holder must renew or stop vending.
/// - A holder MAY vend any number of timestamps from any logical
///   counter, as long as the physical part of every vended ts is
///   `<= ceiling_ns` AND the lease is still the current epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TsLease {
    pub epoch: u64,
    pub holder_node_id: u64,
    pub ceiling_ns: u64,
}

impl TsLease {
    /// Construct a lease. No validation here — `_meta` apply does
    /// the epoch monotonicity check.
    pub fn new(epoch: u64, holder_node_id: u64, ceiling_ns: u64) -> Self {
        Self { epoch, holder_node_id, ceiling_ns }
    }

    /// Check if `wall_ns` is within the lease's validity window.
    /// Callers compose this with their own epoch check.
    pub fn within_ceiling(&self, wall_ns: u64) -> bool {
        wall_ns < self.ceiling_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_ceiling_is_strict() {
        let lease = TsLease::new(1, 7, 1_000);
        assert!(lease.within_ceiling(999));
        assert!(!lease.within_ceiling(1_000));
        assert!(!lease.within_ceiling(1_001));
    }

    #[test]
    fn serde_roundtrip() {
        let lease = TsLease::new(42, 7, 1_700_000_000_000_000_000);
        let s = serde_json::to_string(&lease).unwrap();
        let back: TsLease = serde_json::from_str(&s).unwrap();
        assert_eq!(lease, back);
    }
}
