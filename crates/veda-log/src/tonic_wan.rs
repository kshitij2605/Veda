//! Tonic transport for Phase H WAN log shipping. The in-process
//! `LocalReceiver` from `wan.rs` appends to a `ShardLog`
//! directly; this module ships:
//!
//! - [`TonicWanService`] — a tonic service that hosts one or
//!   more replica `LocalReceiver`s and exposes them over gRPC.
//! - [`TonicReceiver`] — a `LogReceiver` impl that talks to a
//!   remote `TonicWanService` via gRPC. `LogShipper` does not
//!   care whether its target is local or remote — same trait.
//!
//! Wire format: each RPC carries a `bytes payload` that's the
//! bincode encoding of the matching Rust struct. Same pattern
//! as `tonic_log.rs` (see `proto/wan_service.proto`).
//!
//! ## safe_ts gating
//!
//! Cross-region readers call [`TonicReceiver::check_read_ts`]
//! before serving a snapshot read. The receiver compares the
//! caller's `read_ts` against its local `safe_ts` (advanced by
//! `ShipBatchRpc.primary_safe_ts` on every accepted batch) and
//! rejects reads ahead of the watermark.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::replication::NodeId;
use crate::wan::{LocalReceiver, LogReceiver, ShipBatchAck, ShipBatchRpc, WanError};

pub mod proto {
    tonic::include_proto!("veda.wan");
}

use proto::RawRpc;
pub use proto::veda_wan_client::VedaWanClient;
pub use proto::veda_wan_server::{VedaWan, VedaWanServer};

/// Wire payload for `ReplicaDurableSeq` (no inputs).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplicaDurableSeqRpc {}

/// Wire payload for `ReplicaSafeTs` (no inputs).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ReplicaSafeTsRpc {}

/// Wire payload for `CheckReadTs` — the caller's intended
/// snapshot timestamp.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckReadTsRpc {
    pub read_ts: u128,
}

/// Wire-friendly mirror of `Result<u128, WanError>` for
/// `CheckReadTs`. `WanError` is `thiserror`-derived, not
/// serde-derived, so we convert at the boundary.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum CheckReadTsResult {
    Ok { safe_ts: u128 },
    TooNew { safe_ts: u128, lag: u128 },
}

/// Wire-friendly mirror of `Result<ShipBatchAck, WanError>`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum ShipBatchResult {
    Ok(ShipBatchAck),
    EpochMismatch { local: u64, batch: u64 },
    SeqGap { expected: u64, got: u64 },
    Io(String),
}

/// `(shard_id, payload)` envelope — same shape as
/// `tonic_log::ShardEnvelope` so multi-shard receivers can route.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WanEnvelope<T> {
    pub shard_id: String,
    pub payload: T,
}

/// Server-side: maps `(shard_id) → LocalReceiver`. One service
/// instance can host any number of replica shards.
pub struct TonicWanService {
    receivers: Arc<RwLock<HashMap<String, Arc<LocalReceiver>>>>,
}

impl Default for TonicWanService {
    fn default() -> Self {
        Self::new()
    }
}

impl TonicWanService {
    pub fn new() -> Self {
        Self {
            receivers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, shard_id: impl Into<String>, recv: Arc<LocalReceiver>) {
        self.receivers.write().await.insert(shard_id.into(), recv);
    }

    async fn route(&self, shard_id: &str) -> Result<Arc<LocalReceiver>, Status> {
        self.receivers
            .read()
            .await
            .get(shard_id)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("shard_id {shard_id} not hosted")))
    }
}

#[async_trait]
impl VedaWan for TonicWanService {
    async fn ship_batch(&self, req: Request<RawRpc>) -> Result<Response<RawRpc>, Status> {
        let env: WanEnvelope<ShipBatchRpc> = bincode::deserialize(&req.into_inner().payload)
            .map_err(|e| Status::invalid_argument(format!("decode ShipBatchRpc: {e}")))?;
        let recv = self.route(&env.shard_id).await?;
        let result = match recv.ship_batch(env.payload).await {
            Ok(ack) => ShipBatchResult::Ok(ack),
            Err(WanError::EpochMismatch { local, batch }) => {
                ShipBatchResult::EpochMismatch { local, batch }
            }
            Err(WanError::SeqGap { expected, got }) => ShipBatchResult::SeqGap { expected, got },
            Err(WanError::Io(s)) => ShipBatchResult::Io(s),
            Err(other) => ShipBatchResult::Io(format!("{other}")),
        };
        let payload = bincode::serialize(&result)
            .map_err(|e| Status::internal(format!("encode ShipBatchResult: {e}")))?;
        Ok(Response::new(RawRpc { payload }))
    }

    async fn replica_durable_seq(&self, req: Request<RawRpc>) -> Result<Response<RawRpc>, Status> {
        let env: WanEnvelope<ReplicaDurableSeqRpc> =
            bincode::deserialize(&req.into_inner().payload)
                .map_err(|e| Status::invalid_argument(format!("decode: {e}")))?;
        let recv = self.route(&env.shard_id).await?;
        let seq = recv.replica_durable_seq().await;
        let payload =
            bincode::serialize(&seq).map_err(|e| Status::internal(format!("encode u64: {e}")))?;
        Ok(Response::new(RawRpc { payload }))
    }

    async fn replica_safe_ts(&self, req: Request<RawRpc>) -> Result<Response<RawRpc>, Status> {
        let env: WanEnvelope<ReplicaSafeTsRpc> = bincode::deserialize(&req.into_inner().payload)
            .map_err(|e| Status::invalid_argument(format!("decode: {e}")))?;
        let recv = self.route(&env.shard_id).await?;
        let ts = recv.replica_safe_ts().await;
        let payload =
            bincode::serialize(&ts).map_err(|e| Status::internal(format!("encode u128: {e}")))?;
        Ok(Response::new(RawRpc { payload }))
    }

    async fn check_read_ts(&self, req: Request<RawRpc>) -> Result<Response<RawRpc>, Status> {
        let env: WanEnvelope<CheckReadTsRpc> = bincode::deserialize(&req.into_inner().payload)
            .map_err(|e| Status::invalid_argument(format!("decode: {e}")))?;
        let recv = self.route(&env.shard_id).await?;
        let result = match recv.check_read_ts(env.payload.read_ts).await {
            Ok(safe_ts) => CheckReadTsResult::Ok { safe_ts },
            Err(WanError::ReadTsTooNew { safe_ts, lag, .. }) => {
                CheckReadTsResult::TooNew { safe_ts, lag }
            }
            Err(other) => return Err(Status::internal(format!("check_read_ts: {other}"))),
        };
        let payload = bincode::serialize(&result)
            .map_err(|e| Status::internal(format!("encode CheckReadTsResult: {e}")))?;
        Ok(Response::new(RawRpc { payload }))
    }
}

/// Client-side `LogReceiver` that proxies to a remote
/// `TonicWanService`.
pub struct TonicReceiver {
    region: String,
    node_id: NodeId,
    shard_id: String,
    client: tokio::sync::Mutex<VedaWanClient<tonic::transport::Channel>>,
}

impl TonicReceiver {
    pub async fn connect(
        region: impl Into<String>,
        node_id: NodeId,
        shard_id: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Result<Self, WanError> {
        let ep = endpoint.into();
        let channel = tonic::transport::Endpoint::from_shared(ep.clone())
            .map_err(|e| WanError::Transport(format!("endpoint: {e}")))?
            .connect()
            .await
            .map_err(|e| WanError::Transport(format!("dial {ep}: {e}")))?;
        Ok(Self {
            region: region.into(),
            node_id,
            shard_id: shard_id.into(),
            client: tokio::sync::Mutex::new(VedaWanClient::new(channel)),
        })
    }

    pub fn from_channel(
        region: impl Into<String>,
        node_id: NodeId,
        shard_id: impl Into<String>,
        channel: tonic::transport::Channel,
    ) -> Self {
        Self {
            region: region.into(),
            node_id,
            shard_id: shard_id.into(),
            client: tokio::sync::Mutex::new(VedaWanClient::new(channel)),
        }
    }

    async fn rpc<T: serde::Serialize, R: for<'de> serde::Deserialize<'de>>(
        &self,
        which: RpcKind,
        payload: T,
    ) -> Result<R, WanError> {
        let env = WanEnvelope {
            shard_id: self.shard_id.clone(),
            payload,
        };
        let bytes =
            bincode::serialize(&env).map_err(|e| WanError::Transport(format!("encode: {e}")))?;
        let mut client = self.client.lock().await;
        let req = Request::new(RawRpc { payload: bytes });
        let resp = match which {
            RpcKind::ShipBatch => client.ship_batch(req).await,
            RpcKind::ReplicaDurableSeq => client.replica_durable_seq(req).await,
            RpcKind::ReplicaSafeTs => client.replica_safe_ts(req).await,
            RpcKind::CheckReadTs => client.check_read_ts(req).await,
        }
        .map_err(|e| WanError::Transport(format!("rpc: {e}")))?;
        let resp_bytes = resp.into_inner().payload;
        bincode::deserialize(&resp_bytes).map_err(|e| WanError::Transport(format!("decode: {e}")))
    }
}

enum RpcKind {
    ShipBatch,
    ReplicaDurableSeq,
    ReplicaSafeTs,
    CheckReadTs,
}

#[async_trait]
impl LogReceiver for TonicReceiver {
    fn region(&self) -> &str {
        &self.region
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn ship_batch(&self, batch: ShipBatchRpc) -> Result<ShipBatchAck, WanError> {
        let result: ShipBatchResult = self.rpc(RpcKind::ShipBatch, batch).await?;
        match result {
            ShipBatchResult::Ok(ack) => Ok(ack),
            ShipBatchResult::EpochMismatch { local, batch } => {
                Err(WanError::EpochMismatch { local, batch })
            }
            ShipBatchResult::SeqGap { expected, got } => Err(WanError::SeqGap { expected, got }),
            ShipBatchResult::Io(s) => Err(WanError::Io(s)),
        }
    }

    async fn replica_durable_seq(&self) -> u64 {
        self.rpc::<_, u64>(RpcKind::ReplicaDurableSeq, ReplicaDurableSeqRpc {})
            .await
            .unwrap_or(0)
    }

    async fn replica_safe_ts(&self) -> u128 {
        self.rpc::<_, u128>(RpcKind::ReplicaSafeTs, ReplicaSafeTsRpc {})
            .await
            .unwrap_or(0)
    }

    async fn check_read_ts(&self, read_ts: u128) -> Result<u128, WanError> {
        let result: CheckReadTsResult = self
            .rpc(RpcKind::CheckReadTs, CheckReadTsRpc { read_ts })
            .await?;
        match result {
            CheckReadTsResult::Ok { safe_ts } => Ok(safe_ts),
            CheckReadTsResult::TooNew { safe_ts, lag } => Err(WanError::ReadTsTooNew {
                read_ts,
                safe_ts,
                lag,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{LogEntryV2, LogOp};
    use crate::shard_log::ShardLog;
    use crate::wan::LogShipper;
    use bytes::Bytes;
    use std::time::Duration;
    use veda_time::Hlc;

    fn append_to(log: &Arc<ShardLog>, count: u64) {
        for i in 1..=count {
            log.append_one(
                Hlc::new(i, 0),
                1,
                Bytes::copy_from_slice(&i.to_le_bytes()),
                LogOp::Put,
                Bytes::from(format!("e{i}").into_bytes()),
            )
            .unwrap();
        }
    }

    /// Round-trip the LogReceiver contract against a real
    /// TonicWanService over a TCP loopback.
    #[tokio::test]
    async fn tonic_receiver_round_trips_against_real_server() {
        let tmp_dst = tempfile::tempdir().unwrap();
        let dst = Arc::new(ShardLog::open(tmp_dst.path()).unwrap());
        let local = Arc::new(LocalReceiver::new("replica-eu", 2, dst.clone()));
        let svc = TonicWanService::new();
        svc.register("docs:0", local.clone()).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let server = tonic::transport::Server::builder()
            .add_service(VedaWanServer::new(svc))
            .serve_with_incoming(incoming);
        let _handle = tokio::spawn(server);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let receiver: Arc<dyn LogReceiver> = Arc::new(
            TonicReceiver::connect("replica-eu", 2, "docs:0", format!("http://{addr}"))
                .await
                .unwrap(),
        );
        assert_eq!(receiver.replica_durable_seq().await, 0);
        assert_eq!(receiver.replica_safe_ts().await, 0);

        // Ship a batch.
        let entries: Vec<LogEntryV2> = (1..=3)
            .map(|i| LogEntryV2 {
                seq_id: i,
                hlc_ts: Hlc::new(i, 0),
                epoch: 1,
                key: Bytes::copy_from_slice(&i.to_le_bytes()),
                op: LogOp::Put,
                payload: Bytes::from(format!("v{i}").into_bytes()),
            })
            .collect();
        let ack = receiver
            .ship_batch(ShipBatchRpc {
                epoch: 1,
                entries,
                start_seq: 1,
                primary_safe_ts: 1_000,
            })
            .await
            .unwrap();
        assert!(ack.durable_seq >= 3);
        assert!(receiver.replica_durable_seq().await >= 3);
        // safe_ts advanced with the batch.
        assert_eq!(receiver.replica_safe_ts().await, 1_000);

        // check_read_ts gating.
        assert_eq!(receiver.check_read_ts(500).await.unwrap(), 1_000);
        assert_eq!(receiver.check_read_ts(1_000).await.unwrap(), 1_000);
        let err = receiver.check_read_ts(2_000).await.unwrap_err();
        match err {
            WanError::ReadTsTooNew {
                read_ts,
                safe_ts,
                lag,
            } => {
                assert_eq!(read_ts, 2_000);
                assert_eq!(safe_ts, 1_000);
                assert_eq!(lag, 1_000);
            }
            other => panic!("expected ReadTsTooNew, got {other:?}"),
        }
    }

    /// Shipper drains a source ShardLog into a TonicReceiver
    /// over the wire; safe_ts is published through the batch.
    #[tokio::test]
    async fn shipper_drains_into_tonic_receiver() {
        let tmp_src = tempfile::tempdir().unwrap();
        let src = Arc::new(ShardLog::open(tmp_src.path()).unwrap());
        append_to(&src, 25);

        let tmp_dst = tempfile::tempdir().unwrap();
        let dst = Arc::new(ShardLog::open(tmp_dst.path()).unwrap());
        let local = Arc::new(LocalReceiver::new("replica", 2, dst.clone()));
        let svc = TonicWanService::new();
        svc.register("col:0", local.clone()).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let server = tonic::transport::Server::builder()
            .add_service(VedaWanServer::new(svc))
            .serve_with_incoming(incoming);
        let _handle = tokio::spawn(server);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let remote: Arc<dyn LogReceiver> = Arc::new(
            TonicReceiver::connect("replica", 2, "col:0", format!("http://{addr}"))
                .await
                .unwrap(),
        );
        let shipper = LogShipper::new(src.clone(), remote.clone()).with_batch_size(10);
        shipper.set_safe_ts(7_500).await;
        shipper.catch_up(10).await.unwrap();
        assert_eq!(shipper.lag().await, 0);
        // Replica picked up the primary's safe_ts.
        assert!(remote.replica_safe_ts().await >= 7_500);
    }
}
