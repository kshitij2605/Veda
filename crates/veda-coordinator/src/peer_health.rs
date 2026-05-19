//! Cross-node health probes.
//!
//! Each node runs a single background task that periodically GETs
//! `/v1/admin/health` on every peer in [`crate::raft_tonic::PeerAddressBook`]
//! and caches the result. The dashboard's `/admin/nodes` table reads
//! the cache instead of reporting `"unknown"` for peers.
//!
//! Probe cadence is configurable via `VEDA_PEER_HEALTH_INTERVAL_SECS`
//! (default 5s). Each probe has a 2-second timeout — short enough that
//! a dead peer doesn't pile up probe tasks, long enough that a slow
//! network hop doesn't flap status.
//!
//! ## What this is and isn't
//!
//! IS: a coarse "is this peer reachable + does it claim to be
//! healthy" signal. Useful for the dashboard, useful for ops triage.
//!
//! ISN'T: a Raft-level liveness signal. openraft has its own
//! heartbeat path through AppendEntries — this probe is purely for
//! UI. A peer might be healthy by this probe but unreachable via
//! Raft RPC (e.g. gRPC port firewalled) or vice versa.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::raft::NodeId;
use crate::raft_tonic::PeerAddressBook;

/// Status of one peer at the last probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerHealth {
    /// HTTP 200 from /admin/health with status:"healthy".
    Healthy,
    /// HTTP 200 but status is "degraded" or "unhealthy".
    Degraded(String),
    /// No HTTP response within the timeout, or connection refused.
    Unreachable,
}

impl PeerHealth {
    pub fn label(&self) -> &'static str {
        match self {
            PeerHealth::Healthy => "healthy",
            PeerHealth::Degraded(_) => "degraded",
            PeerHealth::Unreachable => "unreachable",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PeerHealthCache {
    inner: Arc<RwLock<HashMap<NodeId, PeerHealth>>>,
}

impl PeerHealthCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get(&self, node_id: NodeId) -> Option<PeerHealth> {
        self.inner.read().await.get(&node_id).cloned()
    }

    pub async fn snapshot(&self) -> HashMap<NodeId, PeerHealth> {
        self.inner.read().await.clone()
    }

    async fn set(&self, node_id: NodeId, status: PeerHealth) {
        self.inner.write().await.insert(node_id, status);
    }
}

/// Spawn the probe loop. Returns the cache so callers can read
/// statuses. The task runs for the lifetime of the runtime.
///
/// The `self_id` is excluded from probes (we know our own health
/// directly). The book is re-snapshotted on every tick so membership
/// changes are picked up automatically.
pub fn spawn_probe(
    self_id: NodeId,
    peers: PeerAddressBook,
) -> PeerHealthCache {
    let cache = PeerHealthCache::new();
    let cache_clone = cache.clone();
    let interval = std::env::var("VEDA_PEER_HEALTH_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5)
        .max(1);
    let probe_timeout = Duration::from_secs(2);

    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(probe_timeout)
            .no_proxy()
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "peer health probe client build failed");
                return;
            }
        };
        let mut ticker = tokio::time::interval(Duration::from_secs(interval));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            // Snapshot the address book each tick so membership
            // changes propagate. Peers store host:6341 (Raft gRPC
            // port); /admin/health is on the REST port — by
            // convention 6340. We swap the port.
            let book = peers.snapshot().await;
            for (peer_id, addr) in book {
                if peer_id == self_id {
                    continue;
                }
                let rest_addr = swap_to_rest_port(&addr);
                let url = format!("http://{rest_addr}/v1/admin/health");
                let status = probe(&client, &url).await;
                cache_clone.set(peer_id, status).await;
            }
        }
    });

    cache
}

/// Heuristic: Raft addresses end in `:6341`; REST is on `:6340` by
/// convention. When the addr already uses a different port (operator
/// customized), we leave it alone — they can override via
/// `VEDA_PEER_REST_PORT_OFFSET` later if needed.
fn swap_to_rest_port(addr: &str) -> String {
    if let Some((host, port)) = addr.rsplit_once(':') {
        if port == "6341" {
            return format!("{host}:6340");
        }
    }
    addr.to_owned()
}

async fn probe(client: &reqwest::Client, url: &str) -> PeerHealth {
    let resp = match client.get(url).send().await {
        Ok(r) => r,
        Err(_) => return PeerHealth::Unreachable,
    };
    if !resp.status().is_success() {
        return PeerHealth::Degraded(format!("HTTP {}", resp.status()));
    }
    let Ok(body) = resp.text().await else {
        return PeerHealth::Degraded("bad body".into());
    };
    // /admin/health returns {"status":"healthy"|"degraded"|"unhealthy", ...}
    match serde_json::from_str::<serde_json::Value>(&body) {
        Ok(v) => match v.get("status").and_then(|s| s.as_str()) {
            Some("healthy") => PeerHealth::Healthy,
            Some(other) => PeerHealth::Degraded(other.to_owned()),
            None => PeerHealth::Degraded("no status field".into()),
        },
        Err(_) => PeerHealth::Degraded("not JSON".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_port_swap() {
        assert_eq!(swap_to_rest_port("veda-2:6341"), "veda-2:6340");
        assert_eq!(swap_to_rest_port("10.0.0.5:6341"), "10.0.0.5:6340");
        // Non-6341 left alone — operator customized.
        assert_eq!(swap_to_rest_port("veda-9:9999"), "veda-9:9999");
    }

    #[tokio::test]
    async fn cache_get_set() {
        let c = PeerHealthCache::new();
        assert!(c.get(1).await.is_none());
        c.set(1, PeerHealth::Healthy).await;
        assert_eq!(c.get(1).await, Some(PeerHealth::Healthy));
        let snap = c.snapshot().await;
        assert_eq!(snap.len(), 1);
    }
}
