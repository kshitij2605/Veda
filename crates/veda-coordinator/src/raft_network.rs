//! Network transport scaffolding for openraft.
//!
//! ## Status
//!
//! This file holds the data-plumbing pieces a real `RaftNetwork` /
//! `RaftNetworkFactory` would build on:
//!
//! - [`NodeAddressBook`] — `NodeId -> address` lookup, populated from the
//!   existing [`crate::raft_types::ClusterState::nodes`] view.
//! - [`InProcessNetwork`] — channel-based RPC for unit tests, mirrored
//!   from the existing [`crate::network::ChannelNetwork`].
//!
//! The actual `RaftNetwork` trait impl is deferred for the same reason as
//! the storage traits (see `raft_storage.rs`): producing a correct
//! AppendEntries/InstallSnapshot/Vote implementation needs careful
//! review against openraft's expected error semantics, and a real
//! production deployment wants Tonic/gRPC rather than mpsc channels.
//!
//! See `documentation/raft-migration.md` Phase 3.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock, mpsc};

use crate::raft::NodeId;

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
