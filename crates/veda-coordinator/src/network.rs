//! Channel-based in-process coordination layer.
//!
//! Provides [`CoordinatorHandle`], a cloneable async interface that submits
//! [`ClusterCommand`]s to a single leader node running the [`LogStore`] and
//! [`ClusterStateMachine`]. Communication happens via `tokio::sync::mpsc`
//! channels, making it fully testable without TCP.
//!
//! When Raft consensus is integrated later, [`CoordinatorHandle`] can be backed
//! by a real Raft client instead of a direct channel, without changing the
//! caller-facing API.

use std::collections::HashMap;

use tokio::sync::{mpsc, oneshot, RwLock};

use crate::raft_types::{ClusterCommand, ClusterResponse, ClusterState};
use crate::store::{ClusterStateMachine, LogStore};

// ---------------------------------------------------------------------------
// Internal message envelope
// ---------------------------------------------------------------------------

/// A request sent through the channel to the coordinator loop.
struct CommandEnvelope {
    command: ClusterCommand,
    reply: oneshot::Sender<ClusterResponse>,
}

/// A read-query sent through the channel.
struct QueryEnvelope {
    reply: oneshot::Sender<ClusterState>,
}

enum CoordinatorMsg {
    Command(CommandEnvelope),
    Query(QueryEnvelope),
    Shutdown,
}

// ---------------------------------------------------------------------------
// Coordinator (the "leader" task)
// ---------------------------------------------------------------------------

/// Runs the single-leader coordination loop.
///
/// Receives commands via channel, appends them to the log, applies them to
/// the state machine, and responds to the caller.
async fn coordinator_loop(mut rx: mpsc::Receiver<CoordinatorMsg>) {
    let mut log = LogStore::new();
    let mut sm = ClusterStateMachine::new();

    while let Some(msg) = rx.recv().await {
        match msg {
            CoordinatorMsg::Command(CommandEnvelope { command, reply }) => {
                let idx = log.append(command);
                let entries = log.get_range(idx, idx + 1);
                let entry = entries[0];
                let resp = sm.apply(entry);
                // Ignore send error — caller may have dropped the receiver.
                let _ = reply.send(resp);
            }
            CoordinatorMsg::Query(QueryEnvelope { reply }) => {
                let state = sm.snapshot();
                let _ = reply.send(state);
            }
            CoordinatorMsg::Shutdown => break,
        }
    }
}

// ---------------------------------------------------------------------------
// CoordinatorHandle — the public, cloneable client
// ---------------------------------------------------------------------------

/// A cloneable handle that submits commands to the coordinator loop.
///
/// Create one via [`Coordinator::start`]. The handle is `Send + Sync` and
/// can be shared across tasks.
#[derive(Clone)]
pub struct CoordinatorHandle {
    tx: mpsc::Sender<CoordinatorMsg>,
}

impl CoordinatorHandle {
    /// Submit a command and wait for the response.
    pub async fn execute(&self, command: ClusterCommand) -> Result<ClusterResponse, CoordinatorError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CoordinatorMsg::Command(CommandEnvelope {
                command,
                reply: reply_tx,
            }))
            .await
            .map_err(|_| CoordinatorError::Shutdown)?;

        reply_rx.await.map_err(|_| CoordinatorError::Shutdown)
    }

    /// Read the current cluster state snapshot.
    pub async fn query_state(&self) -> Result<ClusterState, CoordinatorError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CoordinatorMsg::Query(QueryEnvelope { reply: reply_tx }))
            .await
            .map_err(|_| CoordinatorError::Shutdown)?;

        reply_rx.await.map_err(|_| CoordinatorError::Shutdown)
    }

    /// Ask the coordinator loop to shut down gracefully.
    pub async fn shutdown(&self) -> Result<(), CoordinatorError> {
        self.tx
            .send(CoordinatorMsg::Shutdown)
            .await
            .map_err(|_| CoordinatorError::Shutdown)
    }
}

// ---------------------------------------------------------------------------
// Coordinator — builder / entry point
// ---------------------------------------------------------------------------

/// Builder that spawns the coordinator background task and returns a handle.
pub struct Coordinator;

impl Coordinator {
    /// Start the coordinator loop and return a cloneable handle.
    ///
    /// The background task runs until the handle's [`shutdown`](CoordinatorHandle::shutdown)
    /// method is called or all handles are dropped.
    pub fn start() -> CoordinatorHandle {
        let (tx, rx) = mpsc::channel(256);
        tokio::spawn(coordinator_loop(rx));
        CoordinatorHandle { tx }
    }
}

// ---------------------------------------------------------------------------
// ChannelNetwork — multi-node in-process transport (for future Raft testing)
// ---------------------------------------------------------------------------

/// In-process network for testing Raft consensus between multiple nodes.
///
/// Maps node IDs to mpsc senders so that Raft RPCs can be delivered without
/// real TCP connections. This is a placeholder that will be fleshed out when
/// full Raft integration is added.
pub struct ChannelNetwork {
    senders: RwLock<HashMap<u64, mpsc::Sender<Vec<u8>>>>,
}

impl ChannelNetwork {
    pub fn new() -> Self {
        Self {
            senders: RwLock::new(HashMap::new()),
        }
    }

    /// Register a node's inbound channel.
    pub async fn register(&self, node_id: u64, tx: mpsc::Sender<Vec<u8>>) {
        self.senders.write().await.insert(node_id, tx);
    }

    /// Remove a node from the network.
    pub async fn deregister(&self, node_id: u64) {
        self.senders.write().await.remove(&node_id);
    }

    /// Send a raw message to a node. Returns `false` if the node is unknown
    /// or the channel is closed.
    pub async fn send(&self, node_id: u64, data: Vec<u8>) -> bool {
        let senders = self.senders.read().await;
        match senders.get(&node_id) {
            Some(tx) => tx.send(data).await.is_ok(),
            None => false,
        }
    }

    /// Number of currently registered nodes.
    pub async fn node_count(&self) -> usize {
        self.senders.read().await.len()
    }
}

impl Default for ChannelNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the coordinator handle.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoordinatorError {
    /// The coordinator loop has shut down.
    #[error("coordinator has shut down")]
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_network_register_send() {
        let net = ChannelNetwork::new();
        let (tx, mut rx) = mpsc::channel(8);

        net.register(1, tx).await;
        assert_eq!(net.node_count().await, 1);

        assert!(net.send(1, b"hello".to_vec()).await);
        assert!(!net.send(99, b"nope".to_vec()).await);

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg, b"hello");

        net.deregister(1).await;
        assert_eq!(net.node_count().await, 0);
    }
}
