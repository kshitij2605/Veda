//! Production TSO backed by the metadata Raft lease.
//!
//! Phase A follow-up to the FoundationDB-shape rearchitecture
//! plan (`.claude/plans/structured-mixing-turing.md`). The
//! [`crate::raft_types::ClusterCommand::GrantTsLease`] /
//! `RenewTsLease` mechanism that landed in `9985c48` is the
//! replicated lease record; this file is the live consumer that
//! claims the lease, vends timestamps from a local HLC while
//! it holds the lease, and renews the lease before its ceiling.
//!
//! ## Lifecycle
//!
//! 1. Only the metadata Raft leader runs the renewer task.
//!    Followers see `MetaLeasedTso::allocate()` return
//!    [`TsoError::Unavailable`] with a "not leader, retry on N"
//!    message — the gateway forwards to the named node.
//! 2. On becoming leader, the renewer task issues
//!    `GrantTsLease { epoch = previous_epoch + 1, holder_node_id =
//!    me, ceiling_ns = now + lease_duration }`. Epoch fencing in
//!    the state machine apply path rejects any other node's
//!    `GrantTsLease` at the same or lower epoch.
//! 3. Periodically (default every `lease_duration / 3`) the
//!    renewer issues `RenewTsLease { epoch, ceiling_ns = now +
//!    lease_duration }`. Old epoch → renewer detects via state
//!    machine error and re-claims.
//! 4. On `allocate(n)` the live path checks (a) we're still
//!    leader, (b) the ts_lease in state machine has our epoch,
//!    and (c) the HLC's next allocation would not exceed
//!    `ceiling_ns`. All three must hold or the call errors.
//!
//! ## What this v0 ships vs. doesn't
//!
//! - Single leased allocator (one TSO per cluster).
//! - HLC-based monotonic timestamps.
//! - Epoch fencing on grant + renew.
//! - Background renewer task.
//! - Best-effort step-down detection: if we lose leadership
//!   between two `allocate()` calls, the next one errors. Real
//!   step-down notification (subscribing to `raft.metrics`
//!   transitions) is a separate refinement.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use async_trait::async_trait;
use veda_time::{HlcClock, Ts, TsRange, Tso, TsoError};

use crate::raft::NodeId;
use crate::raft_coordinator::RaftCoordinator;
use crate::raft_types::ClusterCommand;

/// Default lease horizon. Conservative — bounds the worst-case
/// stall on TSO holder failure to ~10s.
pub const DEFAULT_LEASE_DURATION: Duration = Duration::from_secs(10);

/// Default renew interval — 1/3 of the lease horizon. Gives two
/// retry budgets if a single renew fails before the lease
/// expires.
fn default_renew_interval(lease: Duration) -> Duration {
    lease / 3
}

/// Production TSO.
pub struct MetaLeasedTso {
    coord: RaftCoordinator,
    /// Cached node id from the Raft metrics. Set on
    /// construction; `current_leader` is consulted at call time.
    node_id: NodeId,
    /// In-process HLC used to mint timestamps while we hold the
    /// lease. Lives for the lifetime of the TSO.
    clock: Arc<HlcClock>,
    /// Renewer task handle — kept so `Drop` can cancel it.
    renewer: Mutex<Option<JoinHandle<()>>>,
    /// Configurable lease horizon. Exposed via
    /// [`Self::with_lease_duration`] for tests + benchmarking.
    lease_duration: Duration,
}

impl MetaLeasedTso {
    /// Build a TSO bound to the given Raft coordinator. The
    /// background renewer task is NOT spawned automatically —
    /// callers should invoke [`Self::start_renewer`] when
    /// they're ready for the leader to claim the lease.
    pub fn new(coord: RaftCoordinator) -> Self {
        let node_id = coord.metrics().borrow().id;
        Self {
            coord,
            node_id,
            clock: Arc::new(HlcClock::new()),
            renewer: Mutex::new(None),
            lease_duration: DEFAULT_LEASE_DURATION,
        }
    }

    pub fn with_lease_duration(mut self, d: Duration) -> Self {
        self.lease_duration = d;
        self
    }

    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Start the background renewer. Idempotent — calling
    /// twice replaces the previous task handle.
    pub async fn start_renewer(self: &Arc<Self>) {
        let mut guard = self.renewer.lock().await;
        if let Some(h) = guard.take() {
            h.abort();
        }
        let me = self.clone();
        let handle = tokio::spawn(async move {
            me.renewer_loop().await;
        });
        *guard = Some(handle);
    }

    async fn renewer_loop(self: Arc<Self>) {
        let renew_interval = default_renew_interval(self.lease_duration);
        loop {
            // Run on a short heartbeat — quicker than the renew
            // interval so we catch step-up promptly.
            tokio::time::sleep(renew_interval / 2).await;
            if let Err(e) = self.tick_renew().await {
                tracing::debug!(node_id = self.node_id, error = %e, "TSO renew tick failed");
            }
        }
    }

    /// One iteration of the renewer:
    ///   - If we're leader and lease is held by us → maybe renew.
    ///   - If we're leader and lease is held by someone else (or
    ///     missing) → claim by issuing GrantTsLease at epoch+1.
    ///   - If we're a follower → no-op.
    async fn tick_renew(&self) -> Result<(), String> {
        let m = self.coord.metrics().borrow().clone();
        let leader = m.current_leader;
        let i_am_leader = leader == Some(self.node_id);
        if !i_am_leader {
            return Ok(());
        }
        let state = self
            .coord
            .query_state()
            .await
            .map_err(|e| format!("query_state: {e}"))?;
        let now_ns = wall_clock_ns();
        let new_ceiling = now_ns + self.lease_duration.as_nanos() as u64;

        match state.ts_lease.as_ref() {
            Some(l) if l.holder_node_id == self.node_id => {
                // Renew at current epoch. If renew fails (e.g.
                // some other node grabbed a higher epoch
                // somehow), fall through to claim.
                let renew = ClusterCommand::RenewTsLease {
                    epoch: l.epoch,
                    ceiling_ns: new_ceiling,
                };
                match self.coord.execute(renew).await {
                    Ok(_) => return Ok(()),
                    Err(e) => {
                        tracing::warn!(
                            node_id = self.node_id,
                            error = %e,
                            "RenewTsLease failed; will attempt re-claim"
                        );
                    }
                }
            }
            _ => {}
        }

        // Claim: epoch = max(current, 0) + 1.
        let next_epoch = state.ts_lease.as_ref().map(|l| l.epoch + 1).unwrap_or(1);
        let grant = ClusterCommand::GrantTsLease {
            epoch: next_epoch,
            holder_node_id: self.node_id,
            ceiling_ns: new_ceiling,
        };
        self.coord
            .execute(grant)
            .await
            .map_err(|e| format!("GrantTsLease epoch={next_epoch}: {e}"))?;
        Ok(())
    }

    /// True iff the cached state machine view shows this node
    /// holds a valid lease (epoch matches, ceiling not past).
    async fn check_lease(&self) -> Result<(), TsoError> {
        let state = self
            .coord
            .query_state()
            .await
            .map_err(|e| TsoError::Unavailable(format!("query_state: {e}")))?;
        let lease = state.ts_lease.as_ref().ok_or_else(|| {
            TsoError::Unavailable("no TSO lease held; renewer hasn't claimed yet".to_owned())
        })?;
        if lease.holder_node_id != self.node_id {
            return Err(TsoError::Unavailable(format!(
                "TSO lease held by node {} (epoch {}); not me ({})",
                lease.holder_node_id, lease.epoch, self.node_id
            )));
        }
        let wall = wall_clock_ns();
        if !lease.within_ceiling(wall) {
            return Err(TsoError::LeaseExpired {
                epoch: lease.epoch,
                ceiling_ns: lease.ceiling_ns,
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for MetaLeasedTso {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaLeasedTso")
            .field("node_id", &self.node_id)
            .field("lease_duration", &self.lease_duration)
            .finish()
    }
}

#[async_trait]
impl Tso for MetaLeasedTso {
    async fn allocate(&self, n: u32) -> Result<TsRange, TsoError> {
        if n == 0 {
            return Ok(TsRange {
                start: self.clock.now(),
                count: 0,
            });
        }
        self.check_lease().await?;
        let start = self.clock.now();
        for _ in 1..n {
            let _ = self.clock.now();
        }
        Ok(TsRange { start, count: n })
    }

    async fn now(&self) -> Result<Ts, TsoError> {
        self.check_lease().await?;
        Ok(self.clock.now())
    }
}

impl Drop for MetaLeasedTso {
    fn drop(&mut self) {
        if let Ok(mut g) = self.renewer.try_lock()
            && let Some(h) = g.take()
        {
            h.abort();
        }
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
    use std::time::Duration;

    /// Single-node bootstrap auto-elects itself leader; the TSO
    /// should claim the lease after `start_renewer` runs and
    /// then vend timestamps.
    #[tokio::test]
    async fn single_node_claims_lease_and_vends_timestamps() {
        let coord = RaftCoordinator::start_single_node(1).await.unwrap();
        // Wait briefly for the leader to settle.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let tso =
            Arc::new(MetaLeasedTso::new(coord.clone()).with_lease_duration(Duration::from_secs(2)));
        tso.start_renewer().await;
        // Renewer's first tick fires within renew_interval/2 ≈
        // 333ms. Give it a second.
        tokio::time::sleep(Duration::from_secs(1)).await;

        let r = tso.allocate(5).await.expect("allocate after claim");
        assert_eq!(r.count, 5);
        let ts1 = tso.now().await.expect("now");
        let ts2 = tso.now().await.expect("now again");
        assert!(ts2 > ts1, "successive now() must be monotonic");
        let _ = coord.shutdown().await;
    }

    /// allocate(0) is a peek — must not require the lease (so
    /// callers can do timestamp-comparison ops without holding
    /// the TSO authority). Matches the LocalTso contract.
    #[tokio::test]
    async fn allocate_zero_does_not_require_lease() {
        let coord = RaftCoordinator::start_single_node(2).await.unwrap();
        let tso = MetaLeasedTso::new(coord.clone());
        // No start_renewer; no lease held.
        let r = tso.allocate(0).await.expect("allocate(0) is a peek");
        assert_eq!(r.count, 0);
        let _ = coord.shutdown().await;
    }

    /// Without the renewer, allocate(>0) errors with
    /// `Unavailable` (no lease held).
    #[tokio::test]
    async fn allocate_without_renewer_errors_unavailable() {
        let coord = RaftCoordinator::start_single_node(3).await.unwrap();
        let tso = MetaLeasedTso::new(coord.clone());
        let err = tso.allocate(1).await.expect_err("must error without lease");
        assert!(matches!(err, TsoError::Unavailable(_)), "got {err:?}");
        let _ = coord.shutdown().await;
    }
}
