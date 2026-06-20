//! Timestamp Oracle (TSO) trait + the local single-node impl.
//!
//! The TSO vends monotonic timestamps to writers and readers. A
//! single allocator across the cluster is what gives Phase G
//! snapshot isolation across shards — per-shard HLCs alone aren't
//! enough.
//!
//! Two impls live in the tree:
//!
//! - [`LocalTso`] (this file): wraps an [`HlcClock`]. Correct for
//!   tests and single-node deployments. The default in `AppState`.
//! - `MetaLeasedTso` (in `veda-coordinator` once it ships): the
//!   real cross-node allocator. Holds a lease granted by `_meta`,
//!   vends timestamps from a pre-allocated batch under the lease,
//!   and refreshes the lease before its `ceiling` expires.
//!
//! Both impls satisfy:
//!   `tso.now() >= every commit_ts that any caller has observed
//!   via tso.allocate(...) ack`.

use async_trait::async_trait;
use std::sync::Arc;
use thiserror::Error;

use crate::Ts;
use crate::hlc::{Hlc, HlcClock};

#[derive(Debug, Error)]
pub enum TsoError {
    #[error("TSO unavailable: {0}")]
    Unavailable(String),
    #[error("lease expired (epoch={epoch}, ceiling={ceiling_ns})")]
    LeaseExpired { epoch: u64, ceiling_ns: u64 },
}

/// A contiguous range of allocated timestamps. The caller is
/// guaranteed `count` distinct monotonic timestamps starting at
/// `start` (each next allocation from the TSO will be `> start +
/// count` in HLC order).
#[derive(Debug, Clone)]
pub struct TsRange {
    pub start: Ts,
    pub count: u32,
}

impl TsRange {
    /// Iterator over the allocated timestamps. Each one bumps the
    /// logical counter from `start`, keeping physical fixed.
    pub fn iter(&self) -> impl Iterator<Item = Ts> + use<'_> {
        (0..self.count).map(move |i| Hlc {
            physical_ns: self.start.physical_ns,
            logical: self.start.logical.saturating_add(i),
        })
    }
}

#[async_trait]
pub trait Tso: Send + Sync + std::fmt::Debug {
    /// Allocate `n` consecutive timestamps. Subsequent `now()`
    /// calls (locally or elsewhere) will return Ts > every
    /// timestamp in the returned range.
    async fn allocate(&self, n: u32) -> Result<TsRange, TsoError>;

    /// Read the latest "now" timestamp without reserving a range.
    /// Useful for read_ts.
    async fn now(&self) -> Result<Ts, TsoError>;
}

/// Single-node TSO backed by an in-process `HlcClock`. Correct
/// when the process owns the entire timestamp space — which is
/// the situation for tests, single-node servers, and the
/// pre-Phase-C migration window.
#[derive(Debug)]
pub struct LocalTso {
    clock: Arc<HlcClock>,
}

impl LocalTso {
    pub fn new() -> Self {
        Self {
            clock: Arc::new(HlcClock::new()),
        }
    }

    pub fn with_clock(clock: Arc<HlcClock>) -> Self {
        Self { clock }
    }

    pub fn clock(&self) -> &Arc<HlcClock> {
        &self.clock
    }
}

impl Default for LocalTso {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tso for LocalTso {
    async fn allocate(&self, n: u32) -> Result<TsRange, TsoError> {
        // Single now() call captures the physical; downstream
        // iter() bumps the logical. We then call now() once more
        // to ensure the clock's internal `last` is >= range.end,
        // so the next allocate() can't overlap.
        if n == 0 {
            return Ok(TsRange {
                start: self.clock.now(),
                count: 0,
            });
        }
        let start = self.clock.now();
        // Walk the clock forward n-1 times so the next now() is
        // strictly greater than the highest ts returned.
        for _ in 1..n {
            let _ = self.clock.now();
        }
        Ok(TsRange { start, count: n })
    }

    async fn now(&self) -> Result<Ts, TsoError> {
        Ok(self.clock.now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_allocate_returns_monotonic_range() {
        let tso = LocalTso::new();
        let r1 = tso.allocate(5).await.unwrap();
        let r2 = tso.allocate(5).await.unwrap();
        let last_in_r1 = r1.iter().last().unwrap();
        let first_in_r2 = r2.start;
        assert!(first_in_r2 > last_in_r1, "ranges must not overlap");
    }

    #[tokio::test]
    async fn iter_yields_count_distinct_monotonic_timestamps() {
        let tso = LocalTso::new();
        let range = tso.allocate(10).await.unwrap();
        let ts: Vec<_> = range.iter().collect();
        assert_eq!(ts.len(), 10);
        for w in ts.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[tokio::test]
    async fn now_after_allocate_is_greater_than_range_max() {
        let tso = LocalTso::new();
        let range = tso.allocate(3).await.unwrap();
        let after = tso.now().await.unwrap();
        let last = range.iter().last().unwrap();
        assert!(after > last, "now() must exceed allocated range");
    }

    #[tokio::test]
    async fn allocate_zero_is_a_peek() {
        let tso = LocalTso::new();
        let r = tso.allocate(0).await.unwrap();
        assert_eq!(r.count, 0);
        let next = tso.allocate(1).await.unwrap();
        assert!(next.start > r.start);
    }
}
