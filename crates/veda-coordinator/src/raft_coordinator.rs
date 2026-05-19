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
use std::sync::Arc;

use openraft::storage::Adaptor;
use openraft::{BasicNode, Config, Raft};

use crate::network::CoordinatorError;
use crate::raft::{NodeId, VedaTypeConfig};
use crate::raft_network::InProcessNetwork;
use crate::raft_storage::MemRaftStorage;
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

        Ok(Self {
            raft: Arc::new(raft),
            storage,
        })
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
    /// trip. Acceptable for single-node and for follower-safe reads
    /// in a multi-node cluster; if a caller needs linearisable reads
    /// it should issue a `client_write` of a no-op instead, which is
    /// a follow-up.
    pub async fn query_state(&self) -> Result<ClusterState, CoordinatorError> {
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
