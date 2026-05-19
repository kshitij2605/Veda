//! Per-shard Raft groups — multi-group coordinator.
//!
//! ## Why
//!
//! The cluster-wide Raft (`RaftCoordinator::start_multi_node`)
//! serialises every write through one leader. That caps cluster
//! write throughput at single-node speed and is the explicit
//! horizontal-scaling block called out in
//! `documentation/raft-migration.md`.
//!
//! CockroachDB / TiDB / Spanner solve this by running MANY Raft
//! groups per process — one per shard (CockroachDB calls these
//! "ranges"). Each group has its own log, state machine, and
//! leader. Writes to different shards proceed in parallel; the
//! cluster throughput scales with the shard count.
//!
//! This module is the v1 of that architecture for Veda.
//!
//! ## What this v1 ships
//!
//! - A [`MultiGroupCoordinator`] that hosts an arbitrary
//!   number of named [`RaftCoordinator`] instances keyed by a
//!   string [`GroupId`]. Each instance has its own data
//!   directory under `{base_dir}/{group_id}/raft/`, its own
//!   Tonic gRPC port, and its own openraft loop.
//! - One "control plane" group (default name `"_meta"`) hosts
//!   the cluster-wide schema + node-membership state machine
//!   we already use. Existing code paths that reach for the
//!   single Raft instance get redirected at this group
//!   transparently.
//! - One or more "data plane" groups can be added; each
//!   represents a (collection, shard_id) bucket. Inserts
//!   targeting that bucket submit to that group's Raft and
//!   replicate in parallel with inserts to other buckets.
//!
//! ## What this v1 does NOT ship
//!
//! - **Automatic group lifecycle from CreateCollection.** The
//!   control plane carries a list of "expected data groups"
//!   each node should be hosting; spinning them up on
//!   schema-create / spinning them down on drop is a
//!   follow-up. Today an operator pre-declares groups via env
//!   var `VEDA_RAFT_DATA_GROUPS=col1:0,col1:1,col2:0`.
//! - **Inserts routed by ShardRouter at the REST handler.**
//!   The existing insert path still goes through the
//!   `_meta` group's ClusterCommand::InsertRecord. A future
//!   commit teaches `records::insert` to compute
//!   `(collection, shard_id)` and submit to the right group.
//! - **Cross-group transactions.** Each group is independent;
//!   there is no two-phase commit yet. CockroachDB's parallel
//!   commits / atomic multi-range writes are a separate
//!   chapter.
//! - **Snapshot-based group recovery from the control plane.**
//!   On node restart we just re-spawn the groups we knew
//!   about; there is no "ask the leader for the latest group
//!   list" call.
//!
//! ## Status
//!
//! Smoke-tested: two groups spawned in the same process,
//! independent leaders, parallel client_writes against each,
//! both groups' state machines converge separately.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::raft::NodeId;
use crate::raft_coordinator::{BootstrapError, RaftCoordinator};
use crate::raft_types::{ClusterCommand, ClusterResponse, ClusterState};

/// Identifier for one Raft group. By convention: `"_meta"` for
/// the cluster-wide control plane; `"<collection>:<shard>"` for
/// per-shard data planes (e.g. `"rag_anything:0"`).
pub type GroupId = String;

/// The control-plane group id used throughout Veda when no
/// explicit group is specified.
pub const META_GROUP: &str = "_meta";

/// Multi-group coordinator. Cheap to clone (all inner state is
/// in `Arc<RwLock>`-style shares).
#[derive(Clone)]
pub struct MultiGroupCoordinator {
    inner: Arc<MultiGroupInner>,
}

struct MultiGroupInner {
    groups: RwLock<HashMap<GroupId, RaftCoordinator>>,
    /// Base directory under which `{base_dir}/{group_id}/raft/`
    /// data lives.
    base_dir: std::path::PathBuf,
}

impl MultiGroupCoordinator {
    pub fn new(base_dir: &std::path::Path) -> Self {
        Self {
            inner: Arc::new(MultiGroupInner {
                groups: RwLock::new(HashMap::new()),
                base_dir: base_dir.to_owned(),
            }),
        }
    }

    /// Spawn a new Raft group identified by `group_id`. Uses the
    /// same multi-node bootstrap as the cluster-wide path
    /// (`RaftCoordinator::start_multi_node`) but with a
    /// group-specific data directory and Tonic listen port.
    ///
    /// `listen_addr` and `peers` are exactly the parameters the
    /// existing start_multi_node takes — each group needs its
    /// own ones because Tonic services don't share a listener
    /// in v1.
    pub async fn start_group(
        &self,
        group_id: &str,
        node_id: NodeId,
        listen_addr: std::net::SocketAddr,
        peers: Vec<(NodeId, String)>,
        bootstrap: bool,
    ) -> Result<(), BootstrapError> {
        let dir = self.inner.base_dir.join(group_id);
        let coord = RaftCoordinator::start_multi_node(
            node_id,
            &dir,
            listen_addr,
            peers,
            bootstrap,
        )
        .await?;
        self.inner
            .groups
            .write()
            .await
            .insert(group_id.to_owned(), coord);
        tracing::info!(
            group_id = %group_id,
            node_id,
            listen = %listen_addr,
            "raft group started",
        );
        Ok(())
    }

    /// Single-node variant — used in tests and for the simplest
    /// deployment. Does not start a gRPC service.
    pub async fn start_single_group(
        &self,
        group_id: &str,
        node_id: NodeId,
    ) -> Result<(), BootstrapError> {
        let dir = self.inner.base_dir.join(group_id);
        let coord = RaftCoordinator::start_single_node_with_dir(node_id, &dir).await?;
        self.inner
            .groups
            .write()
            .await
            .insert(group_id.to_owned(), coord);
        tracing::info!(group_id = %group_id, node_id, "raft single-node group started");
        Ok(())
    }

    /// Submit a `ClusterCommand` to the given group's leader.
    /// Errors when the group does not exist.
    pub async fn execute(
        &self,
        group_id: &str,
        command: ClusterCommand,
    ) -> Result<ClusterResponse, MultiGroupError> {
        let coord = {
            let guard = self.inner.groups.read().await;
            guard
                .get(group_id)
                .cloned()
                .ok_or_else(|| MultiGroupError::UnknownGroup(group_id.to_owned()))?
        };
        coord
            .execute(command)
            .await
            .map_err(|_| MultiGroupError::Coordinator)
    }

    /// Read the local state machine snapshot for the given
    /// group. Returns the empty default `ClusterState` when the
    /// group does not exist.
    pub async fn query_state(&self, group_id: &str) -> ClusterState {
        let coord = {
            let guard = self.inner.groups.read().await;
            guard.get(group_id).cloned()
        };
        match coord {
            Some(c) => c.query_state().await.unwrap_or_default(),
            None => ClusterState::default(),
        }
    }

    pub async fn group_ids(&self) -> Vec<String> {
        self.inner.groups.read().await.keys().cloned().collect()
    }

    /// Borrow a coordinator handle for the named group. Returns
    /// None if the group hasn't been started here.
    pub async fn get(&self, group_id: &str) -> Option<RaftCoordinator> {
        self.inner.groups.read().await.get(group_id).cloned()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MultiGroupError {
    #[error("unknown raft group: {0}")]
    UnknownGroup(String),
    #[error("coordinator error")]
    Coordinator,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two groups in one process, single-node mode. Verifies
    /// they're truly independent: a CreateCollection in
    /// group_a does not show up in group_b's state machine.
    #[tokio::test(flavor = "current_thread")]
    async fn two_independent_single_node_groups() {
        let tmp = tempfile::tempdir().unwrap();
        let mg = MultiGroupCoordinator::new(tmp.path());

        mg.start_single_group("group_a", 1).await.unwrap();
        mg.start_single_group("group_b", 1).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Insert a "collection" into each.
        mg.execute(
            "group_a",
            ClusterCommand::CreateCollection {
                name: "col_a".into(),
                schema_json: "{}".into(),
            },
        )
        .await
        .unwrap();
        mg.execute(
            "group_b",
            ClusterCommand::CreateCollection {
                name: "col_b".into(),
                schema_json: "{}".into(),
            },
        )
        .await
        .unwrap();

        // Verify isolation.
        let state_a = mg.query_state("group_a").await;
        let state_b = mg.query_state("group_b").await;
        assert!(state_a.collections.contains_key("col_a"));
        assert!(!state_a.collections.contains_key("col_b"));
        assert!(state_b.collections.contains_key("col_b"));
        assert!(!state_b.collections.contains_key("col_a"));

        let ids = mg.group_ids().await;
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"group_a".to_owned()));
        assert!(ids.contains(&"group_b".to_owned()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unknown_group_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let mg = MultiGroupCoordinator::new(tmp.path());
        mg.start_single_group("only_one", 1).await.unwrap();
        let res = mg
            .execute("doesnt_exist", ClusterCommand::Noop)
            .await;
        assert!(matches!(res, Err(MultiGroupError::UnknownGroup(_))));
    }
}
