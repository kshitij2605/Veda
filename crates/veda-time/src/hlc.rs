//! Hybrid Logical Clock.
//!
//! Algorithm: CockroachDB's `pkg/util/hlc/hlc.go` (Kulkarni et al.
//! 2014, "Logical Physical Clocks and Consistent Snapshots in
//! Globally Distributed Databases"). Compact summary:
//!
//! - `now()` returns the max of (wall clock, last_physical), with
//!   logical bumped if wall didn't advance.
//! - `update(remote)` returns the max of (wall, last, remote.physical)
//!   with the right logical disambiguation.
//!
//! Two Hlcs are ordered lexicographically on `(physical_ns,
//! logical)`. Across the cluster this is a partial order that
//! happens to be total when wall clocks are synced to within the
//! configured uncertainty bound — exactly what CockroachDB / TiDB
//! / Yugabyte rely on.
//!
//! We use `SystemTime::now()` as the physical source. The
//! `HlcClock` ratchets internally so a backwards wall-clock jump
//! never causes the HLC to go backwards.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// A logical timestamp.
///
/// `physical_ns` is nanoseconds since Unix epoch; `logical` is the
/// tiebreaker that increments when multiple HLCs are generated in
/// the same nanosecond.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Hlc {
    pub physical_ns: u64,
    pub logical: u32,
}

impl Hlc {
    pub const ZERO: Hlc = Hlc {
        physical_ns: 0,
        logical: 0,
    };

    pub fn new(physical_ns: u64, logical: u32) -> Self {
        Self {
            physical_ns,
            logical,
        }
    }

    /// Pack into u128 for storage where a single integer is more
    /// convenient. Layout: `physical_ns << 32 | logical`.
    pub fn as_u128(self) -> u128 {
        (u128::from(self.physical_ns) << 32) | u128::from(self.logical)
    }

    /// Inverse of `as_u128`.
    pub fn from_u128(v: u128) -> Self {
        Self {
            physical_ns: (v >> 32) as u64,
            logical: (v & 0xFFFF_FFFF) as u32,
        }
    }
}

/// Stateful HLC. Cloneable handle around a `Mutex<Hlc>`.
///
/// Cheap to share — every `now()` / `update()` takes the mutex
/// briefly. For a high-throughput single-node case, the mutex is
/// the only contention point; profiling can move it to an atomic
/// `u128` later. Not the bottleneck at any realistic write rate.
#[derive(Debug)]
pub struct HlcClock {
    state: Mutex<Hlc>,
}

impl HlcClock {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Hlc::ZERO),
        }
    }

    /// Build a clock seeded with a starting Hlc. Used by tests
    /// and snapshot-restore paths.
    pub fn from_state(seed: Hlc) -> Self {
        Self {
            state: Mutex::new(seed),
        }
    }

    /// Snapshot the current state without advancing.
    pub fn peek(&self) -> Hlc {
        *self.state.lock().expect("HlcClock mutex poisoned")
    }

    /// Generate a fresh HLC strictly greater than the previous one.
    /// Combines the local wall clock with the in-memory ratchet.
    pub fn now(&self) -> Hlc {
        let wall = wall_clock_ns();
        let mut guard = self.state.lock().expect("HlcClock mutex poisoned");
        let next = if wall > guard.physical_ns {
            Hlc {
                physical_ns: wall,
                logical: 0,
            }
        } else {
            // Wall clock didn't advance (or went backwards) — bump
            // logical. Saturating add so a pathological caller
            // can't wrap and produce a non-monotonic Hlc.
            Hlc {
                physical_ns: guard.physical_ns,
                logical: guard.logical.saturating_add(1),
            }
        };
        *guard = next;
        next
    }

    /// Absorb a remote HLC and return one strictly greater than
    /// both the remote and the previous local state. Standard
    /// HLC merge rule.
    pub fn update(&self, remote: Hlc) -> Hlc {
        let wall = wall_clock_ns();
        let mut guard = self.state.lock().expect("HlcClock mutex poisoned");

        let max_phys = wall.max(guard.physical_ns).max(remote.physical_ns);

        let next_logical =
            if max_phys == wall && max_phys > guard.physical_ns && max_phys > remote.physical_ns {
                0
            } else if max_phys == guard.physical_ns && max_phys == remote.physical_ns {
                guard.logical.max(remote.logical).saturating_add(1)
            } else if max_phys == guard.physical_ns {
                guard.logical.saturating_add(1)
            } else {
                // max_phys == remote.physical_ns and not equal to local
                remote.logical.saturating_add(1)
            };

        let next = Hlc {
            physical_ns: max_phys,
            logical: next_logical,
        };
        *guard = next;
        next
    }
}

impl Default for HlcClock {
    fn default() -> Self {
        Self::new()
    }
}

fn wall_clock_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_monotonic_on_single_clock() {
        let clock = HlcClock::new();
        let mut prev = clock.now();
        for _ in 0..10_000 {
            let next = clock.now();
            assert!(
                next > prev,
                "expected monotonic HLC: prev={prev:?} next={next:?}"
            );
            prev = next;
        }
    }

    #[test]
    fn update_with_future_remote_advances_local() {
        let clock = HlcClock::new();
        let local0 = clock.now();
        let remote = Hlc {
            physical_ns: local0.physical_ns + 1_000_000_000, // 1s ahead
            logical: 5,
        };
        let after = clock.update(remote);
        assert!(after > remote, "update should produce ts > remote");
        // Next now() should still be greater.
        let later = clock.now();
        assert!(later > after, "subsequent now() must be > absorbed remote");
    }

    #[test]
    fn update_with_past_remote_does_not_rewind() {
        let clock = HlcClock::new();
        let _ = clock.now();
        let local = clock.peek();
        let remote = Hlc {
            physical_ns: local.physical_ns.saturating_sub(1_000_000),
            logical: 0,
        };
        let after = clock.update(remote);
        assert!(after >= local, "absorbing a past remote must not rewind");
    }

    #[test]
    fn interleaved_now_and_update_are_monotonic() {
        // Property: any sequence of now()/update() calls produces
        // strictly increasing HLCs from each call's POV.
        let clock = HlcClock::new();
        let mut prev = clock.now();
        let mut remotes = [
            Hlc {
                physical_ns: prev.physical_ns + 500,
                logical: 0,
            },
            Hlc {
                physical_ns: prev.physical_ns + 100,
                logical: 7,
            },
            Hlc {
                physical_ns: prev.physical_ns + 2_000,
                logical: 1,
            },
        ]
        .into_iter()
        .cycle();
        for _ in 0..1_000 {
            let local = clock.now();
            assert!(
                local > prev,
                "now() not monotonic at step prev={prev:?} local={local:?}"
            );
            prev = local;

            let remote = remotes.next().unwrap();
            let absorbed = clock.update(remote);
            assert!(absorbed > prev, "update() not monotonic");
            assert!(absorbed > remote, "absorbed not > remote");
            prev = absorbed;
        }
    }

    #[test]
    fn u128_roundtrip() {
        let cases = [
            Hlc::ZERO,
            Hlc::new(1, 0),
            Hlc::new(0, 1),
            Hlc::new(u64::MAX, u32::MAX),
            Hlc::new(1_700_000_000_000_000_000, 42),
        ];
        for h in cases {
            assert_eq!(Hlc::from_u128(h.as_u128()), h);
        }
    }

    #[test]
    fn lexicographic_order_on_pair() {
        assert!(Hlc::new(1, 0) < Hlc::new(1, 1));
        assert!(Hlc::new(1, u32::MAX) < Hlc::new(2, 0));
        assert!(Hlc::new(0, 0) < Hlc::new(1, 0));
        assert_eq!(Hlc::new(5, 7), Hlc::new(5, 7));
    }
}
