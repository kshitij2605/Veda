//! `RaftCoordinator` — Phase 4 of the Raft migration.
//!
//! Wraps `openraft::Raft<VedaTypeConfig>` behind the same
//! `CoordinatorHandle`-shaped API the channel-based coordinator exposes,
//! so `veda-server` can flip its consensus backend with an env var
//! (`VEDA_COORDINATOR=raft`) without changes to every call site.
//!
//! ## Scope of this implementation
//!
//! - **Single-node**: produces a working 1-node Raft loop with
//!   `MemRaftStorage` (in-memory log + state machine) and the
//!   in-process [`crate::raft_network::InProcessNetwork`] (which is
//!   effectively a no-op because the leader is the only voter).
//! - **Multi-node**: the trait surface is correct, but actual
//!   replication is gated on the Tonic transport landing — see
//!   `documentation/raft-migration.md` Phase 3.
//! - **Durability**: the log + state machine are in-memory only. The
//!   process restart story is "you lose cluster state" — fine for a
//!   1-node deployment that still owns segment data on disk, not fine
//!   for production multi-node.
//!
//! What this file is good for *today*: proving end-to-end that an
//! openraft instance constructed from our type config can be
//! initialised, accept a `ClusterCommand`, apply it through our
//! existing `ClusterStateMachine`, and surface the result back to a
//! caller using the `CoordinatorHandle` API.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use openraft::storage::Adaptor;
use openraft::{BasicNode, Config, Raft};

use crate::network::CoordinatorError;
use crate::peer_health::{spawn_probe, PeerHealthCache};
use crate::raft::{NodeId, VedaTypeConfig};
use crate::raft_network::InProcessNetwork;
use crate::raft_storage::MemRaftStorage;
use crate::raft_tonic::{PeerAddressBook, RaftService, TonicRaftNetwork};
use crate::raft_types::{ClusterCommand, ClusterResponse, ClusterState};

/// The `RaftCoordinator` itself. Cheap to clone (the `Raft` handle from
/// openraft is internally `Arc`-wrapped, as is the state-machine ref).
#[derive(Clone)]
pub struct RaftCoordinator {
    /// The raw Raft handle. Not cloneable, but we never need to —
    /// openraft's `Raft` is `Send + Sync` and we hold it behind `Arc`.
    raft: Arc<Raft<VedaTypeConfig>>,
    /// Direct reference to the state machine so `query_state()` can
    /// produce a `ClusterState` snapshot without round-tripping
    /// through Raft. (Reads are local; this matches how the
    /// channel-based coordinator answers queries.)
    storage: MemRaftStorage,
    /// Address book for cross-process peers. Always present, even
    /// in in-process mode (empty book then). Membership-change RPCs
    /// register peers here.
    peers: PeerAddressBook,
    /// Cross-node health cache populated by the probe task spawned
    /// alongside the Tonic transport. None in single-node mode.
    peer_health: Option<PeerHealthCache>,
}

impl RaftCoordinator {
    /// Build a Raft instance for `node_id`, then `initialize` it as a
    /// 1-node cluster (membership = {node_id}). After this returns the
    /// instance is a leader and ready to accept commands.
    ///
    /// Storage is in-memory only.
    pub async fn start_single_node(node_id: NodeId) -> Result<Self, BootstrapError> {
        Self::start_with_storage(node_id, MemRaftStorage::new(), true).await
    }

    /// Variant that persists log + state machine under `data_dir`.
    /// Recovers existing state on construction so a restart preserves
    /// applied commands. Skips `initialize()` when the on-disk state
    /// already reflects an initialized cluster (last_log_id is set).
    pub async fn start_single_node_with_dir(
        node_id: NodeId,
        data_dir: &std::path::Path,
    ) -> Result<Self, BootstrapError> {
        let storage = MemRaftStorage::with_dir(data_dir)
            .await
            .map_err(|e| BootstrapError::RaftConstruct(format!("storage init: {e}")))?;
        let needs_init = storage.log.last_log_id().await.is_none();
        Self::start_with_storage(node_id, storage, needs_init).await
    }

    /// Spawn a WAL compaction task that watches both the
    /// periodic timer AND the Raft snapshot lifecycle. The
    /// timer fires every `VEDA_RAFT_WAL_COMPACT_INTERVAL_SECS`
    /// (default 60s) and compacts when WAL size exceeds
    /// `VEDA_RAFT_WAL_COMPACT_THRESHOLD_BYTES` (default 16 MB).
    /// Independently, whenever the metrics stream reports a new
    /// `snapshot.last_log_id`, every log entry up to that
    /// position can be safely dropped — we trigger an immediate
    /// compaction without waiting for the timer.
    fn spawn_compaction_with_metrics(
        storage: MemRaftStorage,
        raft: Arc<openraft::Raft<VedaTypeConfig>>,
    ) {
        let interval_secs = std::env::var("VEDA_RAFT_WAL_COMPACT_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60)
            .max(5);
        let threshold = std::env::var("VEDA_RAFT_WAL_COMPACT_THRESHOLD_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16 * 1024 * 1024);
        let storage_a = storage.clone();
        let storage_b = storage;
        // Timer-driven compaction (size threshold).
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let size = storage_a.log.wal_bytes().await;
                if size < threshold {
                    continue;
                }
                tracing::info!(wal_bytes = size, threshold, "WAL exceeded threshold; compacting");
                if let Err(e) = storage_a.log.compact().await {
                    tracing::error!(error = %e, "WAL compaction failed");
                }
            }
        });
        // Snapshot-driven compaction. Watch the metrics stream
        // for advances in snapshot.last_log_id — every time
        // openraft builds a new snapshot, entries up to that
        // log index are durable in the state machine and can
        // be purged from the WAL.
        let mut rx = raft.metrics();
        tokio::spawn(async move {
            let mut last_snap_index: u64 = 0;
            loop {
                let m = rx.borrow_and_update().clone();
                let snap_idx = m.snapshot.map(|l| l.index).unwrap_or(0);
                if snap_idx > last_snap_index {
                    tracing::info!(
                        snapshot_log_index = snap_idx,
                        previous = last_snap_index,
                        "raft snapshot advanced; triggering WAL compaction",
                    );
                    if let Err(e) = storage_b.log.compact().await {
                        tracing::error!(error = %e, "snapshot-triggered WAL compaction failed");
                    }
                    last_snap_index = snap_idx;
                }
                if rx.changed().await.is_err() {
                    break;
                }
            }
            tracing::info!("snapshot-driven WAL compaction watcher exiting");
        });
    }

    /// Legacy entry point — keep around so single-node bootstrap
    /// doesn't need the raft handle just for compaction. Delegates
    /// to the timer-only variant.
    fn spawn_compaction(storage: MemRaftStorage) {
        let interval_secs = std::env::var("VEDA_RAFT_WAL_COMPACT_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60)
            .max(5);
        let threshold = std::env::var("VEDA_RAFT_WAL_COMPACT_THRESHOLD_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16 * 1024 * 1024);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let size = storage.log.wal_bytes().await;
                if size < threshold {
                    continue;
                }
                tracing::info!(
                    wal_bytes = size,
                    threshold,
                    "raft WAL exceeded compaction threshold; compacting",
                );
                if let Err(e) = storage.log.compact().await {
                    tracing::error!(error = %e, "raft WAL compaction failed");
                }
            }
        });
    }

    async fn start_with_storage(
        node_id: NodeId,
        storage: MemRaftStorage,
        bootstrap: bool,
    ) -> Result<Self, BootstrapError> {
        let config = Config {
            heartbeat_interval: 250,
            election_timeout_min: 1500,
            election_timeout_max: 3000,
            ..Default::default()
        };
        let config = Arc::new(config.validate().map_err(BootstrapError::Config)?);
        let network = Arc::new(InProcessNetwork::new());

        let (log_store, state_machine) =
            Adaptor::<VedaTypeConfig, MemRaftStorage>::new(storage.clone());

        let raft = Raft::new(node_id, config, network, log_store, state_machine)
            .await
            .map_err(|e| BootstrapError::RaftConstruct(e.to_string()))?;

        if bootstrap {
            let mut members = BTreeMap::new();
            members.insert(node_id, BasicNode::default());
            raft.initialize(members)
                .await
                .map_err(|e| BootstrapError::Initialize(e.to_string()))?;
        }

        Self::spawn_compaction(storage.clone());
        Ok(Self {
            raft: Arc::new(raft),
            storage,
            peers: PeerAddressBook::new(),
            peer_health: None,
        })
    }

    /// Multi-node bootstrap path — the production deployment shape.
    ///
    /// Builds the Raft instance with the Tonic cross-process
    /// transport, spawns a gRPC server on `listen_addr`, populates
    /// the peer address book from `peers`, and (when `bootstrap` is
    /// true) calls `raft.initialize(...)` with the full peer set.
    /// `bootstrap` should be true on first boot of the cluster; on
    /// subsequent boots openraft picks up state from disk and the
    /// caller should pass `false`.
    pub async fn start_multi_node(
        node_id: NodeId,
        data_dir: &std::path::Path,
        listen_addr: SocketAddr,
        peers: Vec<(NodeId, String)>,
        bootstrap: bool,
    ) -> Result<Self, BootstrapError> {
        let storage = MemRaftStorage::with_dir(data_dir)
            .await
            .map_err(|e| BootstrapError::RaftConstruct(format!("storage init: {e}")))?;
        // If on-disk state already shows initialised cluster, skip
        // the bootstrap regardless of the caller's intent.
        let needs_init = bootstrap && storage.log.last_log_id().await.is_none();

        let config = Config {
            heartbeat_interval: 250,
            election_timeout_min: 1500,
            election_timeout_max: 3000,
            ..Default::default()
        };
        let config = Arc::new(config.validate().map_err(BootstrapError::Config)?);

        // Build address book + Tonic network factory.
        let address_book = PeerAddressBook::new();
        for (id, addr) in &peers {
            address_book.register(*id, addr.clone()).await;
        }
        let network = TonicRaftNetwork::new(address_book.clone());

        let (log_store, state_machine) =
            Adaptor::<VedaTypeConfig, MemRaftStorage>::new(storage.clone());

        let raft = Raft::new(node_id, config, network, log_store, state_machine)
            .await
            .map_err(|e| BootstrapError::RaftConstruct(e.to_string()))?;
        let raft = Arc::new(raft);

        // Spawn the Tonic gRPC server. Dropping the JoinHandle is
        // intentional — the server runs for the lifetime of the
        // process, and Tonic handles its own graceful shutdown via
        // the underlying tokio runtime.
        let service = RaftService::new(raft.clone()).into_grpc_service();
        let server_addr = listen_addr;
        tokio::spawn(async move {
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(service)
                .serve(server_addr)
                .await
            {
                tracing::error!(error = %e, addr = %server_addr, "raft gRPC server exited");
            }
        });

        // Bootstrap the cluster once all peers are in the book.
        if needs_init {
            let mut members = BTreeMap::new();
            members.insert(node_id, BasicNode::default());
            for (id, _addr) in &peers {
                if *id != node_id {
                    members.insert(*id, BasicNode::default());
                }
            }
            // First boot: try to initialize. If another peer
            // beat us to it, openraft returns AlreadyInitialized
            // — that's fine, we just join the existing cluster.
            match raft.initialize(members).await {
                Ok(()) => tracing::info!(node_id, "raft cluster initialised"),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "raft.initialize() failed (already initialised on another node?)",
                    );
                }
            }
        }

        // Spawn the peer health probe loop. Background task lives
        // for the runtime; reads back through `peer_health` accessor.
        let peer_health = spawn_probe(node_id, address_book.clone());
        // Use the metrics-aware compaction so a snapshot install
        // triggers compaction immediately rather than waiting
        // for the 60s timer.
        Self::spawn_compaction_with_metrics(storage.clone(), raft.clone());

        Ok(Self {
            raft,
            storage,
            peers: address_book,
            peer_health: Some(peer_health),
        })
    }

    /// Access the peer address book — used by the membership API
    /// to register new peers after an add_voter RPC.
    pub fn peers(&self) -> &PeerAddressBook {
        &self.peers
    }

    /// Cross-node health cache, present in multi-node mode. None in
    /// single-node mode (no peers to probe).
    pub fn peer_health(&self) -> Option<&PeerHealthCache> {
        self.peer_health.as_ref()
    }

    /// Add a voter to the cluster. Membership change goes through
    /// the Raft log; once committed the new node is a full voter.
    /// Also registers the address in the local peer book so the
    /// Tonic transport can dial it.
    pub async fn add_voter(
        &self,
        node_id: NodeId,
        address: String,
    ) -> Result<(), String> {
        self.peers.register(node_id, address).await;
        self.raft
            .add_learner(node_id, BasicNode::default(), true)
            .await
            .map_err(|e| format!("add_learner: {e}"))?;
        // Compute the full membership set including the new voter.
        let metrics = self.raft.metrics().borrow().clone();
        let mut members: std::collections::BTreeSet<NodeId> = metrics
            .membership_config
            .membership()
            .voter_ids()
            .collect();
        members.insert(node_id);
        self.raft
            .change_membership(members, false)
            .await
            .map_err(|e| format!("change_membership add: {e}"))?;
        Ok(())
    }

    /// Remove a voter. Replicates the membership-change log entry,
    /// then deregisters from the local peer book.
    pub async fn remove_voter(&self, node_id: NodeId) -> Result<(), String> {
        let metrics = self.raft.metrics().borrow().clone();
        let mut members: std::collections::BTreeSet<NodeId> = metrics
            .membership_config
            .membership()
            .voter_ids()
            .collect();
        members.remove(&node_id);
        self.raft
            .change_membership(members, false)
            .await
            .map_err(|e| format!("change_membership remove: {e}"))?;
        self.peers.deregister(node_id).await;
        Ok(())
    }

    /// Subscribe to the Raft metrics stream. Used by the gateway's
    /// Prometheus task to publish term/leader/last_applied gauges.
    pub fn metrics(
        &self,
    ) -> tokio::sync::watch::Receiver<openraft::RaftMetrics<NodeId, BasicNode>> {
        self.raft.metrics()
    }

    /// Snapshot of the cluster membership configuration: who is a
    /// voter, who is the current leader. Drives the dashboard's
    /// /admin/nodes table when Raft is the source of truth (the
    /// ClusterState.nodes map only sees RegisterNode commands).
    pub fn membership_snapshot(&self) -> RaftMembershipSnapshot {
        let m = self.raft.metrics().borrow().clone();
        let voters: Vec<NodeId> = m
            .membership_config
            .membership()
            .voter_ids()
            .collect();
        RaftMembershipSnapshot {
            current_term: m.current_term,
            current_leader: m.current_leader,
            voters,
            last_applied_index: m.last_applied.map(|l| l.index).unwrap_or(0),
            last_log_index: m.last_log_index.unwrap_or(0),
        }
    }

    /// Submit a `ClusterCommand` to the Raft leader and wait for it to
    /// be committed and applied. Returns the `ClusterResponse` from
    /// the state machine.
    pub async fn execute(
        &self,
        command: ClusterCommand,
    ) -> Result<ClusterResponse, CoordinatorError> {
        match self.raft.client_write(command).await {
            Ok(resp) => Ok(resp.data),
            Err(e) => {
                tracing::warn!(error = %e, "raft client_write failed");
                Err(CoordinatorError::Shutdown)
            }
        }
    }

    /// Read the current cluster state. This is a local read — we go
    /// straight to the state machine snapshot without a Raft round
    /// trip. Acceptable for follower-safe reads. If a caller needs
    /// linearisable semantics (read reflects every committed write
    /// up to "now"), use [`Self::linearisable_query_state`] instead.
    pub async fn query_state(&self) -> Result<ClusterState, CoordinatorError> {
        Ok(self.storage.sm.snapshot().await)
    }

    /// Linearisable read of the cluster state. Issues a `Noop`
    /// barrier write first; once the leader confirms commit + apply,
    /// the subsequent local read is guaranteed to include every
    /// write that returned to a client before this call started.
    ///
    /// Callers should reach for this only when they need it —
    /// barrier writes go through the full Raft replication path,
    /// so it's slower than the plain `query_state`.
    pub async fn linearisable_query_state(
        &self,
    ) -> Result<ClusterState, CoordinatorError> {
        // Best-effort barrier — if the cluster has no leader the
        // barrier write fails; treat that as the caller not getting
        // linearisability rather than a panic. The state machine
        // read still happens locally.
        if let Err(e) = self.raft.client_write(ClusterCommand::Noop).await {
            tracing::warn!(error = %e, "linearisable_query_state: barrier write failed; returning local read");
        }
        Ok(self.storage.sm.snapshot().await)
    }

    /// Best-effort graceful shutdown — stops the Raft loop and drops
    /// background tasks. Errors are logged but never re-raised; the
    /// channel coordinator's `shutdown()` has the same shape.
    pub async fn shutdown(&self) -> Result<(), CoordinatorError> {
        // openraft 0.9 exposes shutdown via Raft::shutdown(). It can
        // fail if the instance is already gone, which we treat as
        // a no-op.
        if let Err(e) = self.raft.shutdown().await {
            tracing::debug!(error = %e, "raft shutdown returned error (likely already gone)");
        }
        Ok(())
    }

    /// Direct accessor for tests / Admin Nodes view: the underlying
    /// `Raft` handle. Production callers should use `execute` /
    /// `query_state` instead.
    pub fn raft(&self) -> &Raft<VedaTypeConfig> {
        &self.raft
    }
}

/// Lightweight cluster-membership view returned by
/// `RaftCoordinator::membership_snapshot()`. Decoupled from
/// `openraft::RaftMetrics` so callers (the gateway) don't have to
/// depend on openraft directly.
#[derive(Debug, Clone)]
pub struct RaftMembershipSnapshot {
    pub current_term: u64,
    pub current_leader: Option<NodeId>,
    pub voters: Vec<NodeId>,
    pub last_applied_index: u64,
    pub last_log_index: u64,
}

/// Errors that can stop us from constructing a `RaftCoordinator`.
#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("invalid raft config: {0}")]
    Config(openraft::ConfigError),
    #[error("Raft::new failed: {0}")]
    RaftConstruct(String),
    #[error("Raft::initialize failed: {0}")]
    Initialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft_types::ClusterCommand;

    /// End-to-end: stand up a 1-node Raft, submit a CreateCollection
    /// command, read the state machine back and verify the command
    /// landed in `ClusterState.collections`. This is the smallest
    /// honest test that "Raft works" — we drive a real
    /// `client_write`, the entry goes through the log, gets
    /// committed, gets applied to our state machine, and the
    /// response comes back.
    #[tokio::test(flavor = "current_thread")]
    async fn raft_coordinator_single_node_round_trip() {
        let coord = RaftCoordinator::start_single_node(1)
            .await
            .expect("bootstrap should succeed");

        // Wait briefly for openraft to settle as leader after
        // initialize() — election timeout is short but not zero.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let resp = coord
            .execute(ClusterCommand::CreateCollection {
                name: "raft_smoke".into(),
                schema_json: "{\"name\":\"raft_smoke\"}".into(),
            })
            .await
            .expect("execute should succeed");
        assert_eq!(resp, ClusterResponse::Ok);

        let state = coord.query_state().await.expect("query_state");
        assert!(
            state.collections.contains_key("raft_smoke"),
            "state machine should contain the new collection after apply: {:?}",
            state.collections,
        );
        assert!(state.last_applied_index >= 1);

        coord.shutdown().await.expect("shutdown");
    }

    /// Linearisable read of a fresh 1-node cluster: barrier commits
    /// (term-1 noop entry through the log), state read returns
    /// empty cluster. This isn't proof of linearisability under
    /// concurrent writes — that needs a multi-node setup + a real
    /// Jepsen-style checker — but it does prove the barrier write
    /// + apply pathway are intact.
    #[tokio::test(flavor = "current_thread")]
    async fn raft_coordinator_linearisable_read_works() {
        let coord = RaftCoordinator::start_single_node(11).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let state = coord.linearisable_query_state().await.expect("read");
        assert!(state.collections.is_empty());
        // The barrier write should have advanced last_applied_index.
        // For a fresh cluster initialize() bumps it to 1; the noop
        // adds another.
        assert!(
            state.last_applied_index >= 2,
            "barrier write should bump last_applied_index, got {}",
            state.last_applied_index,
        );
        coord.shutdown().await.ok();
    }

    #[tokio::test]
    async fn raft_coordinator_query_state_starts_empty() {
        let coord = RaftCoordinator::start_single_node(2).await.unwrap();
        let state = coord.query_state().await.unwrap();
        assert!(state.collections.is_empty());
        assert!(state.nodes.is_empty());
        coord.shutdown().await.ok();
    }

    /// Durable storage survives a coordinator restart: bootstrap with
    /// a tempdir, write a command, drop the coordinator, build a
    /// fresh coordinator pointed at the same tempdir, verify the
    /// command is still applied.
    #[tokio::test(flavor = "current_thread")]
    async fn raft_coordinator_disk_persistence_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_owned();

        // First boot — bootstrap, write a command, shut down.
        {
            let coord = RaftCoordinator::start_single_node_with_dir(7, &dir)
                .await
                .expect("first bootstrap");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let resp = coord
                .execute(ClusterCommand::CreateCollection {
                    name: "durable_smoke".into(),
                    schema_json: "{\"name\":\"durable_smoke\"}".into(),
                })
                .await
                .expect("execute");
            assert_eq!(resp, ClusterResponse::Ok);
            // Make sure persist tasks have a chance to land before
            // we shutdown. The persist call is awaited inside the
            // mutator, but RAS the openraft state machine apply may
            // schedule extra work — give it a beat.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            coord.shutdown().await.expect("shutdown");
        }

        // Second boot — recover from the same dir, expect the
        // collection still present without any new client_write.
        {
            let coord = RaftCoordinator::start_single_node_with_dir(7, &dir)
                .await
                .expect("recovery bootstrap");
            // Allow openraft to settle (it may need to re-elect since
            // the vote/log are loaded but the runtime is fresh).
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let state = coord.query_state().await.expect("query_state");
            assert!(
                state.collections.contains_key("durable_smoke"),
                "durable_smoke should survive coordinator restart: {:?}",
                state.collections,
            );
            assert!(
                state.last_applied_index >= 1,
                "last_applied_index should be >= 1 after recovery, got {}",
                state.last_applied_index,
            );
            coord.shutdown().await.ok();
        }
    }
}
