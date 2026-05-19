//! Network transport for openraft — in-process only (single-node + tests).
//!
//! ## Status
//!
//! Implements [`openraft::RaftNetworkFactory`] and [`openraft::RaftNetwork`]
//! on top of [`InProcessNetwork`]. This is sufficient for:
//!   - 1-node Raft clusters (the production path until cross-process
//!     transport lands).
//!   - Multi-node unit tests where every "peer" lives in the same Tokio
//!     runtime and we route bytes through `tokio::sync::mpsc`.
//!
//! A cross-process Tonic/gRPC transport is still deferred — see
//! `documentation/raft-migration.md` for the Phase 3-onwards roadmap.
//! In single-node mode the RaftNetwork methods are unreachable in practice
//! (openraft never asks a leader to talk to itself), but they must
//! compile and return reasonable errors when called.

use std::collections::HashMap;
use std::sync::Arc;

use openraft::error::{InstallSnapshotError, RPCError, RaftError, Unreachable};
use openraft::network::{RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
    InstallSnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::BasicNode;
use tokio::sync::{Mutex, RwLock, mpsc};

use crate::raft::{NodeId, VedaTypeConfig};

/// `NodeId -> address` mapping used by the network factory to resolve a
/// peer's address. Populated by the coordinator when nodes register via
/// `ClusterCommand::RegisterNode`.
#[derive(Debug, Default, Clone)]
pub struct NodeAddressBook {
    inner: Arc<RwLock<HashMap<NodeId, String>>>,
}

impl NodeAddressBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register(&self, node_id: NodeId, address: impl Into<String>) {
        self.inner.write().await.insert(node_id, address.into());
    }

    pub async fn lookup(&self, node_id: NodeId) -> Option<String> {
        self.inner.read().await.get(&node_id).cloned()
    }

    pub async fn deregister(&self, node_id: NodeId) {
        self.inner.write().await.remove(&node_id);
    }

    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }
}

/// In-process channel-based "network" for unit tests. Sends bincode bytes
/// over mpsc; a real `RaftNetwork` would replace this with Tonic/gRPC.
#[derive(Debug)]
pub struct InProcessNetwork {
    routes: Mutex<HashMap<NodeId, mpsc::Sender<Vec<u8>>>>,
}

impl Default for InProcessNetwork {
    fn default() -> Self {
        Self {
            routes: Mutex::new(HashMap::new()),
        }
    }
}

impl InProcessNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register(&self, node_id: NodeId, sender: mpsc::Sender<Vec<u8>>) {
        self.routes.lock().await.insert(node_id, sender);
    }

    pub async fn deregister(&self, node_id: NodeId) {
        self.routes.lock().await.remove(&node_id);
    }

    pub async fn send(
        &self,
        target: NodeId,
        payload: Vec<u8>,
    ) -> Result<(), &'static str> {
        let sender = {
            let guard = self.routes.lock().await;
            guard.get(&target).cloned()
        };
        match sender {
            Some(s) => s.send(payload).await.map_err(|_| "receiver dropped"),
            None => Err("no route to target node"),
        }
    }

    pub async fn node_count(&self) -> usize {
        self.routes.lock().await.len()
    }
}

/// Per-target RaftNetwork client. Owned by openraft and reused across
/// many RPC calls to the same peer.
///
/// In the single-node path this never actually transmits anything — the
/// leader is the only voter, so openraft never invokes these methods
/// after the initial `initialize()`. The methods are there because the
/// trait requires them.
pub struct InProcessConn {
    target: NodeId,
    network: Arc<InProcessNetwork>,
}

impl RaftNetworkFactory<VedaTypeConfig> for Arc<InProcessNetwork> {
    type Network = InProcessConn;

    async fn new_client(
        &mut self,
        target: NodeId,
        _node: &BasicNode,
    ) -> Self::Network {
        InProcessConn {
            target,
            network: self.clone(),
        }
    }
}

#[derive(Debug)]
struct NoRoute {
    node: NodeId,
    msg: &'static str,
}

impl std::fmt::Display for NoRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node {}: {}", self.node, self.msg)
    }
}
impl std::error::Error for NoRoute {}

fn unreachable<E: std::error::Error>(
    node: NodeId,
    msg: &'static str,
) -> RPCError<NodeId, BasicNode, E> {
    RPCError::Unreachable(Unreachable::new(&NoRoute { node, msg }))
}

impl RaftNetwork<VedaTypeConfig> for InProcessConn {
    async fn append_entries(
        &mut self,
        _rpc: AppendEntriesRequest<VedaTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        AppendEntriesResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId>>,
    > {
        // The in-process transport never delivers responses across
        // tasks today — it is a fire-and-forget byte router. In
        // single-node mode openraft never asks the leader to talk to
        // itself so this method is unreachable; we surface
        // `Unreachable` rather than panicking so a future
        // misconfiguration (e.g. accidentally booting a 2-node
        // cluster with this transport) gets a recoverable error
        // instead of a crash.
        let _ = &self.network;
        Err(unreachable(
            self.target,
            "in-process transport: cross-process reply routing is the \
             deferred Tonic layer",
        ))
    }

    async fn install_snapshot(
        &mut self,
        _rpc: InstallSnapshotRequest<VedaTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId, InstallSnapshotError>>,
    > {
        Err(unreachable(
            self.target,
            "InstallSnapshot via in-process transport not routed",
        ))
    }

    async fn vote(
        &mut self,
        _rpc: VoteRequest<NodeId>,
        _option: RPCOption,
    ) -> Result<
        VoteResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId>>,
    > {
        Err(unreachable(
            self.target,
            "Vote via in-process transport not routed",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn address_book_register_lookup() {
        let book = NodeAddressBook::new();
        book.register(7, "10.0.0.7:6340").await;
        assert_eq!(book.lookup(7).await.as_deref(), Some("10.0.0.7:6340"));
        book.deregister(7).await;
        assert_eq!(book.lookup(7).await, None);
    }

    #[tokio::test]
    async fn in_process_network_routes() {
        let net = InProcessNetwork::new();
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
        net.register(2, tx).await;
        net.send(2, b"hi".to_vec()).await.unwrap();
        assert_eq!(rx.recv().await.as_deref(), Some(&b"hi"[..]));
        assert!(net.send(99, b"x".to_vec()).await.is_err());
    }
}
