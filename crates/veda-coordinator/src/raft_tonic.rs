//! Tonic gRPC transport for openraft RPCs.
//!
//! This is the production-shaped network layer that replaces the
//! `InProcessNetwork` placeholder for multi-node deployments. Three
//! pieces live here:
//!
//! - [`raft_proto`] — the tonic-generated `RawRpc` message and
//!   `VedaRaft` service from `proto/raft.proto`.
//! - [`RaftService`] — server-side implementation. Holds an
//!   `Arc<VedaRaft>` and routes each gRPC call into
//!   `raft.append_entries() / raft.vote() / raft.install_snapshot()`.
//! - [`TonicRaftNetwork`] / [`TonicRaftNetworkConn`] — client-side
//!   implementations of `RaftNetworkFactory` / `RaftNetwork`. Each
//!   peer is keyed by `NodeId` and resolved via [`PeerAddressBook`]
//!   to a `host:port` string. Tonic channels are cached per peer.
//!
//! ## Wire format
//!
//! Bincode-encoded openraft request/response types over a single
//! `bytes payload` field. See `proto/raft.proto`. This deliberately
//! sidesteps re-defining every openraft type in protobuf — openraft
//! already has serde derives via its `serde` feature, so a bincode
//! round-trip is correct and complete. The trade-off is that the
//! wire is opaque to non-Veda tooling; Raft RPCs are internal so
//! that's a fair price for the simplicity.

use std::collections::HashMap;
use std::sync::Arc;

use openraft::error::{
    InstallSnapshotError, NetworkError, RPCError, RaftError, Unreachable,
};
use openraft::network::{RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
    InstallSnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::{AnyError, BasicNode, Raft};
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};

use crate::raft::{NodeId, VedaTypeConfig};

pub mod raft_proto {
    tonic::include_proto!("veda.raft");
}

use raft_proto::veda_raft_client::VedaRaftClient;
use raft_proto::veda_raft_server::{VedaRaft, VedaRaftServer};
use raft_proto::RawRpc;

// ---------------------------------------------------------------------------
// PeerAddressBook — NodeId -> "host:port"
// ---------------------------------------------------------------------------

/// Map of peer `NodeId` to gRPC address. Populated from
/// `VEDA_RAFT_PEERS` at startup or via `register` for runtime
/// membership changes.
#[derive(Debug, Default, Clone)]
pub struct PeerAddressBook {
    inner: Arc<RwLock<HashMap<NodeId, String>>>,
}

impl PeerAddressBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register(&self, node_id: NodeId, address: impl Into<String>) {
        self.inner.write().await.insert(node_id, address.into());
    }

    pub async fn deregister(&self, node_id: NodeId) {
        self.inner.write().await.remove(&node_id);
    }

    pub async fn lookup(&self, node_id: NodeId) -> Option<String> {
        self.inner.read().await.get(&node_id).cloned()
    }

    pub async fn snapshot(&self) -> HashMap<NodeId, String> {
        self.inner.read().await.clone()
    }
}

/// Parse `VEDA_RAFT_PEERS` env value: `"1@host1:6341,2@host2:6341"`
/// → `[(1, "host1:6341"), (2, "host2:6341")]`. Returns the (possibly
/// empty) list. Silently skips malformed entries with a warn log so a
/// typo in one peer doesn't take the whole node down.
pub fn parse_peers_env(s: &str) -> Vec<(NodeId, String)> {
    let mut out = Vec::new();
    for raw in s.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((id_str, addr)) = trimmed.split_once('@') else {
            tracing::warn!(peer = %trimmed, "VEDA_RAFT_PEERS: missing '@', skipping");
            continue;
        };
        let id: NodeId = match id_str.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(peer = %trimmed, "VEDA_RAFT_PEERS: bad node id, skipping");
                continue;
            }
        };
        out.push((id, addr.trim().to_owned()));
    }
    out
}

// ---------------------------------------------------------------------------
// RaftService — server side. The Tonic server dispatches each incoming
// gRPC into the local Raft instance.
// ---------------------------------------------------------------------------

pub struct RaftService {
    raft: Arc<Raft<VedaTypeConfig>>,
}

impl RaftService {
    pub fn new(raft: Arc<Raft<VedaTypeConfig>>) -> Self {
        Self { raft }
    }

    /// Helper for the server crate to wrap this service into a Tonic
    /// `Server` layer.
    pub fn into_grpc_service(self) -> VedaRaftServer<Self> {
        VedaRaftServer::new(self)
    }
}

fn bincode_status<E: std::fmt::Display>(verb: &str, err: E) -> Status {
    Status::invalid_argument(format!("bincode {verb}: {err}"))
}

#[tonic::async_trait]
impl VedaRaft for RaftService {
    async fn append_entries(
        &self,
        req: Request<RawRpc>,
    ) -> Result<Response<RawRpc>, Status> {
        let inbound: AppendEntriesRequest<VedaTypeConfig> =
            bincode::deserialize(&req.into_inner().payload)
                .map_err(|e| bincode_status("decode AppendEntriesRequest", e))?;
        let resp = self
            .raft
            .append_entries(inbound)
            .await
            .map_err(|e| Status::internal(format!("raft append_entries: {e}")))?;
        let payload = bincode::serialize(&resp)
            .map_err(|e| bincode_status("encode AppendEntriesResponse", e))?;
        Ok(Response::new(RawRpc { payload }))
    }

    async fn vote(
        &self,
        req: Request<RawRpc>,
    ) -> Result<Response<RawRpc>, Status> {
        let inbound: VoteRequest<NodeId> =
            bincode::deserialize(&req.into_inner().payload)
                .map_err(|e| bincode_status("decode VoteRequest", e))?;
        let resp = self
            .raft
            .vote(inbound)
            .await
            .map_err(|e| Status::internal(format!("raft vote: {e}")))?;
        let payload = bincode::serialize(&resp)
            .map_err(|e| bincode_status("encode VoteResponse", e))?;
        Ok(Response::new(RawRpc { payload }))
    }

    async fn install_snapshot(
        &self,
        req: Request<RawRpc>,
    ) -> Result<Response<RawRpc>, Status> {
        let inbound: InstallSnapshotRequest<VedaTypeConfig> =
            bincode::deserialize(&req.into_inner().payload)
                .map_err(|e| bincode_status("decode InstallSnapshotRequest", e))?;
        let resp = self
            .raft
            .install_snapshot(inbound)
            .await
            .map_err(|e| Status::internal(format!("raft install_snapshot: {e}")))?;
        let payload = bincode::serialize(&resp)
            .map_err(|e| bincode_status("encode InstallSnapshotResponse", e))?;
        Ok(Response::new(RawRpc { payload }))
    }
}

// ---------------------------------------------------------------------------
// TonicRaftNetwork — client side. RaftNetworkFactory hands out
// per-target Connections; each Connection dials lazily on first use
// and caches the resulting Channel.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TonicRaftNetwork {
    pub peers: PeerAddressBook,
    /// Cached gRPC channels keyed by NodeId. Channels are
    /// reference-counted internally; cloning is cheap.
    channels: Arc<RwLock<HashMap<NodeId, Channel>>>,
}

impl TonicRaftNetwork {
    pub fn new(peers: PeerAddressBook) -> Self {
        Self {
            peers,
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn channel_for(&self, target: NodeId) -> Option<Channel> {
        if let Some(ch) = self.channels.read().await.get(&target).cloned() {
            return Some(ch);
        }
        let addr = self.peers.lookup(target).await?;
        let uri = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.clone()
        } else {
            format!("http://{addr}")
        };
        let endpoint = Endpoint::from_shared(uri).ok()?
            .timeout(std::time::Duration::from_secs(5))
            .connect_timeout(std::time::Duration::from_secs(2));
        let channel = endpoint.connect().await.ok()?;
        self.channels.write().await.insert(target, channel.clone());
        Some(channel)
    }

    /// Evict a cached channel — called after an RPC failure so the
    /// next call dials a fresh connection. Without this, a peer
    /// that recovered from a partition can stay "stuck" because
    /// the cached channel points at a TCP socket that's been
    /// closed on the other end.
    async fn drop_channel(&self, target: NodeId) {
        self.channels.write().await.remove(&target);
    }
}

#[derive(Clone)]
pub struct TonicRaftNetworkConn {
    target: NodeId,
    factory: TonicRaftNetwork,
}

impl RaftNetworkFactory<VedaTypeConfig> for TonicRaftNetwork {
    type Network = TonicRaftNetworkConn;

    async fn new_client(
        &mut self,
        target: NodeId,
        _node: &BasicNode,
    ) -> Self::Network {
        TonicRaftNetworkConn {
            target,
            factory: self.clone(),
        }
    }
}

#[derive(Debug)]
struct NoRouteErr {
    node: NodeId,
    msg: String,
}
impl std::fmt::Display for NoRouteErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node {}: {}", self.node, self.msg)
    }
}
impl std::error::Error for NoRouteErr {}

fn unreachable<E: std::error::Error>(
    node: NodeId,
    msg: impl Into<String>,
) -> RPCError<NodeId, BasicNode, E> {
    RPCError::Unreachable(Unreachable::new(&NoRouteErr { node, msg: msg.into() }))
}

fn net_err<E: std::error::Error>(
    node: NodeId,
    msg: impl Into<String>,
) -> RPCError<NodeId, BasicNode, E> {
    let _ = node;
    RPCError::Network(NetworkError::new(&AnyError::new(&NoRouteErr {
        node,
        msg: msg.into(),
    })))
}

impl TonicRaftNetworkConn {
    async fn invoke<Req, Resp, E>(
        &self,
        verb: &'static str,
        req: Req,
        send: impl FnOnce(VedaRaftClient<Channel>, RawRpc) -> SendFut,
    ) -> Result<Resp, RPCError<NodeId, BasicNode, E>>
    where
        Req: serde::Serialize,
        Resp: for<'de> serde::Deserialize<'de>,
        E: std::error::Error,
    {
        let bytes = bincode::serialize(&req)
            .map_err(|e| net_err(self.target, format!("encode {verb}: {e}")))?;
        let channel = self
            .factory
            .channel_for(self.target)
            .await
            .ok_or_else(|| unreachable(self.target, format!("no route or dial failed ({verb})")))?;
        let client = VedaRaftClient::new(channel);
        let resp = match send(client, RawRpc { payload: bytes }).await {
            Ok(r) => r,
            Err(s) => {
                // Drop the cached channel so the next call dials
                // fresh — handles partition heal where the
                // long-lived Channel never recovers.
                self.factory.drop_channel(self.target).await;
                return Err(net_err(self.target, format!("rpc {verb}: {s}")));
            }
        };
        let resp_bytes = resp.into_inner().payload;
        let decoded: Resp = bincode::deserialize(&resp_bytes)
            .map_err(|e| net_err(self.target, format!("decode {verb} response: {e}")))?;
        Ok(decoded)
    }
}

type SendFut = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Response<RawRpc>, Status>> + Send>,
>;

impl RaftNetwork<VedaTypeConfig> for TonicRaftNetworkConn {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<VedaTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        AppendEntriesResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId>>,
    > {
        self.invoke::<_, AppendEntriesResponse<NodeId>, RaftError<NodeId>>(
            "AppendEntries",
            rpc,
            |mut client, msg| {
                Box::pin(async move { client.append_entries(msg).await })
            },
        )
        .await
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        _option: RPCOption,
    ) -> Result<
        VoteResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId>>,
    > {
        self.invoke::<_, VoteResponse<NodeId>, RaftError<NodeId>>(
            "Vote",
            rpc,
            |mut client, msg| {
                Box::pin(async move { client.vote(msg).await })
            },
        )
        .await
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<VedaTypeConfig>,
        _option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId, InstallSnapshotError>>,
    > {
        self.invoke::<_, InstallSnapshotResponse<NodeId>, RaftError<NodeId, InstallSnapshotError>>(
            "InstallSnapshot",
            rpc,
            |mut client, msg| {
                Box::pin(async move { client.install_snapshot(msg).await })
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_peers_env_basic() {
        let peers = parse_peers_env("1@host1:6341,2@host2:6341");
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[0], (1, "host1:6341".to_string()));
        assert_eq!(peers[1], (2, "host2:6341".to_string()));
    }

    #[test]
    fn parse_peers_env_skips_malformed() {
        let peers = parse_peers_env("1@host1:6341, no-at-sign , 3@host3:6341, abc@nope");
        // First three valid: 1, and 3 (no-at-sign and abc both fail).
        // abc@nope fails because abc is not a u64.
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[0].0, 1);
        assert_eq!(peers[1].0, 3);
    }

    #[test]
    fn parse_peers_env_empty() {
        assert!(parse_peers_env("").is_empty());
        assert!(parse_peers_env("  ,  ,").is_empty());
    }

    #[tokio::test]
    async fn address_book_register_lookup() {
        let book = PeerAddressBook::new();
        book.register(5, "10.0.0.5:6341").await;
        assert_eq!(book.lookup(5).await.as_deref(), Some("10.0.0.5:6341"));
        book.deregister(5).await;
        assert_eq!(book.lookup(5).await, None);
    }
}
