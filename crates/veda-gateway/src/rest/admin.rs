use std::collections::HashMap;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use crate::auth;

use super::state::AppState;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub checks: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct ListNodesResponse {
    /// Cluster mode: "single" (no Raft) or "raft". Distinguished
    /// because in single mode there is no quorum, no leader
    /// election, and the only entry is always this process.
    pub mode: String,
    pub nodes: Vec<NodeInfo>,
}

#[derive(Debug, Serialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    /// "leader" | "follower" | "candidate" | "unknown". In single
    /// mode this is always "leader" — the lone process owns
    /// everything.
    pub role: String,
    /// Derived from /admin/health checks for this process.
    /// "healthy" | "degraded" | "unhealthy" | "unreachable".
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
    /// Sum of shards across all owned collections.
    pub shard_count: u32,
    /// Number of collections owned by this node.
    pub collection_count: u32,
    /// Whether this node is the one answering the current request.
    pub is_self: bool,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
}

#[derive(Debug, Serialize)]
pub struct LoginError {
    pub error: String,
}

pub async fn health(
    State(_state): State<AppState>,
) -> Json<HealthResponse> {
    let checks = run_health_checks().await;
    let status = compute_status(&checks);
    Json(HealthResponse {
        status,
        version: "0.1.0".into(),
        checks,
    })
}

pub async fn readiness(
    State(_state): State<AppState>,
) -> Result<Json<HealthResponse>, (StatusCode, Json<HealthResponse>)> {
    let checks = run_health_checks().await;
    let status = compute_status(&checks);
    let resp = HealthResponse {
        status: status.clone(),
        version: "0.1.0".into(),
        checks,
    };
    if status == "unhealthy" {
        Err((StatusCode::SERVICE_UNAVAILABLE, Json(resp)))
    } else {
        Ok(Json(resp))
    }
}

/// Phase 6.4 — Admin Nodes view.
///
/// In Raft mode (when `VEDA_COORDINATOR=raft`), reports peers from
/// `ClusterState.nodes` and flips `mode` to `"raft"`. Otherwise
/// returns a single entry describing this process — same shape, so
/// the dashboard table doesn't move when Raft turns on.
pub async fn list_nodes(
    State(state): State<AppState>,
) -> Json<ListNodesResponse> {
    let self_id_env = std::env::var("VEDA_NODE_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node-0".to_owned());
    let self_addr_env = std::env::var("VEDA_NODE_ADDRESS")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "0.0.0.0:6340".to_owned());
    let checks = run_health_checks().await;
    let self_status = compute_status(&checks);
    let uptime_secs = state.inner.started_at.elapsed().as_secs();

    let (collection_count, shard_count) = {
        let segs = state.inner.segment_managers.read().await;
        let cols = segs.len() as u32;
        let shards: u32 = segs.values().map(|m| m.len() as u32).sum();
        (cols, shards)
    };

    // Raft path: read the membership view (voters, current leader)
    // from openraft directly. The peer address book has the
    // host:port we dial each peer at; for the local node we use
    // VEDA_NODE_ADDRESS. Peer status is "unknown" until we wire a
    // cross-node /admin/health probe — deferred.
    let raft_view = {
        let guard = state.inner.raft.read().await;
        match guard.as_ref() {
            Some(c) => {
                let peers_book = c.peers().snapshot().await;
                let health = match c.peer_health() {
                    Some(cache) => cache.snapshot().await,
                    None => std::collections::HashMap::new(),
                };
                Some((c.membership_snapshot(), peers_book, health))
            }
            None => None,
        }
    };
    if let Some((mem, peers_book, peer_health)) = raft_view {
        let self_nid: u64 = std::env::var("VEDA_NODE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let nodes: Vec<NodeInfo> = mem
            .voters
            .iter()
            .copied()
            .map(|nid| {
                let is_self = nid == self_nid;
                let address = if is_self {
                    self_addr_env.clone()
                } else {
                    peers_book.get(&nid).cloned().unwrap_or_default()
                };
                let is_leader = mem.current_leader == Some(nid);
                let role = if is_leader { "leader" } else { "follower" };
                let peer_status = peer_health
                    .get(&nid)
                    .map(|h| h.label().to_owned())
                    .unwrap_or_else(|| "unknown".to_owned());
                NodeInfo {
                    id: format!("node-{nid}"),
                    address,
                    role: role.into(),
                    status: if is_self { self_status.clone() } else { peer_status },
                    version: "0.1.0".into(),
                    uptime_secs: if is_self { uptime_secs } else { 0 },
                    shard_count: if is_self { shard_count } else { 0 },
                    collection_count: if is_self { collection_count } else { 0 },
                    is_self,
                }
            })
            .collect();
        return Json(ListNodesResponse {
            mode: "raft".into(),
            nodes,
        });
    }

    Json(ListNodesResponse {
        mode: "single".into(),
        nodes: vec![NodeInfo {
            id: self_id_env,
            address: self_addr_env,
            role: "leader".into(),
            status: self_status,
            version: "0.1.0".into(),
            uptime_secs,
            shard_count,
            collection_count,
            is_self: true,
        }],
    })
}

// ── Raft membership API (Phase 6 of the migration) ─────────────────────────

#[derive(Debug, Deserialize)]
pub struct JoinRequest {
    pub node_id: u64,
    pub address: String,
}

#[derive(Debug, Serialize)]
pub struct ClusterMembershipError {
    pub error: String,
}

/// `POST /v1/admin/cluster/join` — add a voter to the Raft cluster.
/// Body: `{node_id, address}`. Returns 503 when the gateway is not
/// running in Raft mode.
pub async fn cluster_join(
    State(state): State<AppState>,
    Json(req): Json<JoinRequest>,
) -> Result<StatusCode, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterMembershipError {
                error: "raft coordinator not configured (set VEDA_COORDINATOR=raft)".into(),
            }),
        ));
    };
    coord.add_voter(req.node_id, req.address).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError { error: e }),
        )
    })?;
    Ok(StatusCode::CREATED)
}

/// `POST /v1/admin/cluster/noop` — submit a `ClusterCommand::Noop`
/// through the Raft log. Used by the snapshot-soak smoke test and
/// the partition-recovery harness to drive log replication without
/// needing a Raft-routed collection-creation API. Returns 503 when
/// Raft is not configured.
pub async fn cluster_noop(
    State(state): State<AppState>,
) -> Result<StatusCode, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterMembershipError {
                error: "raft coordinator not configured".into(),
            }),
        ));
    };
    coord
        .execute(veda_coordinator::raft_types::ClusterCommand::Noop)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ClusterMembershipError {
                    error: format!("{e}"),
                }),
            )
        })?;
    Ok(StatusCode::CREATED)
}

/// `DELETE /v1/admin/cluster/nodes/{id}` — remove a voter from the
/// Raft cluster.
pub async fn cluster_remove(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Result<StatusCode, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterMembershipError {
                error: "raft coordinator not configured".into(),
            }),
        ));
    };
    coord.remove_voter(id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError { error: e }),
        )
    })?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, Json<LoginError>)> {
    // SEC-1 (2026-05-21): the previous implementation did
    // `user.password != req.password` — direct string compare on
    // stored cleartext. Both halves are now fixed:
    //   1. New users are stored as Argon2id PHC hashes via
    //      `UserRecord::with_password`.
    //   2. `verify_credentials` verifies via constant-time hash
    //      check, with a legacy-plaintext fall-back that logs a
    //      one-shot warn so operators migrate.
    //
    // We also do a constant-work lookup miss: when the username
    // isn't found we still call `verify_credentials` against a
    // throwaway hash so login timing doesn't reveal whether a
    // username exists.
    use crate::rest::state::verify_credentials;

    static DUMMY_HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let dummy = DUMMY_HASH.get_or_init(|| crate::rest::state::hash_password("not-a-real-password"));

    let (stored, role) = match state.inner.users.get(&req.username) {
        Some(u) => (u.password.as_str(), Some(u.role)),
        None => (dummy.as_str(), None),
    };

    let ok = verify_credentials(stored, &req.password, &req.username) && role.is_some();
    if !ok {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(LoginError {
                error: "invalid credentials".into(),
            }),
        ));
    }

    let token = auth::generate_token(&req.username, role.unwrap(), &state.inner.jwt_secret)
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoginError {
                    error: "failed to generate token".into(),
                }),
            )
        })?;

    Ok(Json(LoginResponse { token }))
}

pub async fn metrics(
    State(state): State<AppState>,
) -> (StatusCode, [(axum::http::header::HeaderName, &'static str); 1], String) {
    let body = match &state.inner.prometheus_handle {
        Some(handle) => handle.render(),
        None => String::from("# metrics recorder not installed\n"),
    };
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

fn compute_status(checks: &HashMap<String, String>) -> String {
    let any_critical_down = checks.get("minio").is_some_and(|s| s == "unavailable")
        || checks.get("nats").is_some_and(|s| s == "unavailable");
    if any_critical_down {
        "unhealthy".into()
    } else if checks.values().any(|s| s == "unavailable") {
        "degraded".into()
    } else {
        "healthy".into()
    }
}

pub(crate) async fn run_health_checks() -> HashMap<String, String> {
    let mut checks = HashMap::new();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .no_proxy()
        .build()
        .unwrap_or_default();

    // MinIO
    let blob_endpoint = std::env::var("VEDA_BLOB_ENDPOINT").unwrap_or_default();
    if blob_endpoint.is_empty() {
        checks.insert("minio".into(), "unconfigured".into());
    } else {
        let url = format!("{}/minio/health/live", blob_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("minio".into(), status.into());
    }

    // NATS
    let nats_url = std::env::var("VEDA_MQ_ENDPOINT").unwrap_or_default();
    if nats_url.is_empty() {
        checks.insert("nats".into(), "unconfigured".into());
    } else {
        let addr = nats_url.trim_start_matches("nats://");
        let status = match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::net::TcpStream::connect(addr),
        ).await {
            Ok(Ok(_)) => "healthy",
            _ => "unavailable",
        };
        checks.insert("nats".into(), status.into());
    }

    // Embedding
    let embed_endpoint = std::env::var("EMBEDDING_ENDPOINT").unwrap_or_default();
    if embed_endpoint.is_empty() {
        checks.insert("embedding".into(), "unconfigured".into());
    } else {
        let url = format!("{}/health", embed_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("embedding".into(), status.into());
    }

    // Multimodal embedding
    let mm_endpoint = std::env::var("MULTIMODAL_EMBEDDING_ENDPOINT").unwrap_or_default();
    if mm_endpoint.is_empty() {
        checks.insert("multimodal_embedding".into(), "unconfigured".into());
    } else {
        let url = format!("{}/health", mm_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("multimodal_embedding".into(), status.into());
    }

    checks
}

// ── Phase H: RegionTopology admin endpoints ────────────────────────────────

/// Wire shape we accept on PUT and return on GET. Mirrors
/// `veda_log::wan::RegionTopology` exactly but defined locally
/// so this module doesn't need to import veda-log just for the
/// type (avoiding a dep cycle direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionTopologyBody {
    pub primary_region: String,
    pub replica_regions: Vec<String>,
    /// Target staleness in milliseconds. Operator alert fires
    /// above this; shipper tries to keep lag below.
    #[serde(default = "default_staleness")]
    pub staleness_target_ms: u64,
    /// Monotonic epoch. Bumped on every promote_replica.
    #[serde(default = "default_epoch")]
    pub epoch: u64,
}

fn default_staleness() -> u64 { 5_000 }
fn default_epoch() -> u64 { 1 }

#[derive(Debug, Serialize)]
pub struct RegionTopologyResponse {
    pub topology: Option<RegionTopologyBody>,
}

/// `GET /v1/admin/region_topology` — read the current cross-
/// region topology. Returns `topology: null` when none has been
/// registered yet or when the gateway is in non-Raft mode (the
/// topology lives in `_meta::shard_ensembles` under the sentinel
/// key `_region_topology`).
pub async fn get_region_topology(
    State(state): State<AppState>,
) -> Result<Json<RegionTopologyResponse>, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Ok(Json(RegionTopologyResponse { topology: None }));
    };
    let snap = coord.linearisable_query_state().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError {
                error: format!("query state: {e}"),
            }),
        )
    })?;
    let topology = snap
        .shard_ensembles
        .get(veda_coordinator::store::REGION_TOPOLOGY_KEY)
        .and_then(|json| serde_json::from_str::<RegionTopologyBody>(json).ok());
    Ok(Json(RegionTopologyResponse { topology }))
}

/// `PUT /v1/admin/region_topology` — overwrite the current
/// topology. Use this to seed the initial topology on the
/// primary region. For epoch bumps after a cutover the
/// `/promote` endpoint is the right tool (it drains pending
/// before bumping).
pub async fn put_region_topology(
    State(state): State<AppState>,
    Json(body): Json<RegionTopologyBody>,
) -> Result<Json<RegionTopologyResponse>, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterMembershipError {
                error: "raft coordinator not configured; region topology requires _meta".into(),
            }),
        ));
    };
    let json = serde_json::to_string(&body).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError {
                error: format!("serialize topology: {e}"),
            }),
        )
    })?;
    coord
        .execute(veda_coordinator::raft_types::ClusterCommand::RegisterShardEnsemble {
            shard_id: veda_coordinator::store::REGION_TOPOLOGY_KEY.to_string(),
            ensemble_json: json,
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ClusterMembershipError {
                    error: format!("raft submit: {e}"),
                }),
            )
        })?;
    Ok(Json(RegionTopologyResponse {
        topology: Some(body),
    }))
}

#[derive(Debug, Deserialize)]
pub struct PromoteRegionRequest {
    /// New topology after the cutover. `epoch` MUST be strictly
    /// greater than the current epoch. The new `primary_region`
    /// is the operator's promotion target.
    pub new_topology: RegionTopologyBody,
}

#[derive(Debug, Serialize)]
pub struct PromoteRegionResponse {
    pub status: String,
    pub old_epoch: u64,
    pub new_epoch: u64,
}

/// `POST /v1/admin/region_topology/promote` — operator-driven
/// regional cutover. Refuses if the new epoch is not strictly
/// newer than the current one. The actual drain-and-fence is
/// done by callers holding a `LogShipper` handle — this REST
/// endpoint persists the topology change to `_meta`, which is
/// what makes the cutover globally visible to other regions.
///
/// Typical operator flow:
///   1. Drain the primary's shipper to the target replica via
///      `LogShipper::catch_up` (operator script or future
///      gateway-driven RPC).
///   2. Call POST /promote here with `new_topology.epoch =
///      current + 1`, `primary_region = <new primary>`.
///   3. Old-primary shippers fence on next attempt because
///      `_meta::aliases::shard_ensembles[_region_topology].epoch`
///      now exceeds theirs.
pub async fn promote_region(
    State(state): State<AppState>,
    Json(req): Json<PromoteRegionRequest>,
) -> Result<Json<PromoteRegionResponse>, (StatusCode, Json<ClusterMembershipError>)> {
    let guard = state.inner.raft.read().await;
    let Some(coord) = guard.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ClusterMembershipError {
                error: "raft coordinator not configured; promote requires _meta".into(),
            }),
        ));
    };

    // Read current topology to verify the epoch bump.
    let snap = coord.linearisable_query_state().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError {
                error: format!("query state: {e}"),
            }),
        )
    })?;
    let current = snap
        .shard_ensembles
        .get(veda_coordinator::store::REGION_TOPOLOGY_KEY)
        .and_then(|json| serde_json::from_str::<RegionTopologyBody>(json).ok());
    let old_epoch = current.as_ref().map(|t| t.epoch).unwrap_or(0);
    if req.new_topology.epoch <= old_epoch {
        return Err((
            StatusCode::CONFLICT,
            Json(ClusterMembershipError {
                error: format!(
                    "promote epoch {} must be strictly greater than current {}",
                    req.new_topology.epoch, old_epoch
                ),
            }),
        ));
    }

    let json = serde_json::to_string(&req.new_topology).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ClusterMembershipError {
                error: format!("serialize topology: {e}"),
            }),
        )
    })?;
    coord
        .execute(veda_coordinator::raft_types::ClusterCommand::RegisterShardEnsemble {
            shard_id: veda_coordinator::store::REGION_TOPOLOGY_KEY.to_string(),
            ensemble_json: json,
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ClusterMembershipError {
                    error: format!("raft submit: {e}"),
                }),
            )
        })?;
    Ok(Json(PromoteRegionResponse {
        status: "promoted".into(),
        old_epoch,
        new_epoch: req.new_topology.epoch,
    }))
}

#[derive(Debug, Deserialize)]
pub struct ReplicaSafeTsBody {
    pub ts: String, // u128 decimal — JSON numbers can't hold u128
}

#[derive(Debug, Serialize)]
pub struct ReplicaSafeTsResponse {
    pub replica_safe_ts: String,
}

/// `GET /v1/admin/replica/safe_ts` — current cross-region
/// reader-gating watermark on this gateway. `u128::MAX` means
/// no gating (default).
pub async fn get_replica_safe_ts(
    State(state): State<AppState>,
) -> Json<ReplicaSafeTsResponse> {
    let ts = *state.inner.replica_safe_ts.read().await;
    Json(ReplicaSafeTsResponse {
        replica_safe_ts: ts.to_string(),
    })
}

/// `POST /v1/admin/replica/safe_ts` — advance the replica
/// reader-gating watermark. Snapshot reads at `?at=<ts>` are
/// rejected with 503 when `ts > replica_safe_ts`. Monotonic:
/// requests to move the watermark backwards are no-ops.
pub async fn set_replica_safe_ts(
    State(state): State<AppState>,
    Json(body): Json<ReplicaSafeTsBody>,
) -> Result<Json<ReplicaSafeTsResponse>, (StatusCode, Json<ClusterMembershipError>)> {
    let new_ts: u128 = body.ts.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ClusterMembershipError {
                error: format!("invalid ts '{}': expected decimal u128", body.ts),
            }),
        )
    })?;
    let mut guard = state.inner.replica_safe_ts.write().await;
    if new_ts > *guard {
        *guard = new_ts;
    }
    Ok(Json(ReplicaSafeTsResponse {
        replica_safe_ts: guard.to_string(),
    }))
}

// ── GAP-3 (2026-05-21): MVCC WAL checkpoint admin endpoint ────────────

#[derive(Debug, Serialize)]
pub struct MvccCheckpointResponse {
    pub collection: String,
    /// Total shards considered.
    pub shards: usize,
    /// Shards that ran a checkpoint (MVCC durable was enabled there).
    pub shards_checkpointed: usize,
    /// Sum of entries snapshotted across shards.
    pub entries: usize,
}

/// `POST /v1/admin/collections/{name}/mvcc_checkpoint` — force an
/// atomic snapshot of the MVCC store across all shards of a
/// collection AND truncate each shard's MVCC WAL. Operator hook
/// for bounded WAL growth (the unbounded WAL is fine for short-
/// lived test runs but scales poorly under sustained churn).
///
/// Per shard:
///   - returns `Ok(Some(N))` when MVCC durable is enabled —
///     snapshot file is written atomically (`.tmp` → fsync →
///     rename) and the WAL is truncated to zero.
///   - returns `Ok(None)` when MVCC durable is NOT enabled
///     (counted as `shards - shards_checkpointed`).
///
/// 404 when the collection is unknown. 500 on any per-shard
/// checkpoint failure — partial progress is preserved (no
/// rollback) so subsequent calls retry only the remaining shards.
pub async fn mvcc_checkpoint(
    State(state): State<AppState>,
    crate::tenant::Tenant(tenant): crate::tenant::Tenant,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<MvccCheckpointResponse>, (StatusCode, Json<ClusterMembershipError>)> {
    let scoped = crate::tenant::scope(&tenant, &name);
    let mut guard = state.inner.segment_managers.write().await;
    let shard_map = guard.get_mut(&scoped).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ClusterMembershipError {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    let mut shards_checkpointed = 0usize;
    let mut entries = 0usize;
    for sm in shard_map.values_mut() {
        match sm.mvcc_checkpoint() {
            Ok(Some(n)) => {
                shards_checkpointed += 1;
                entries += n;
            }
            Ok(None) => {}
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ClusterMembershipError {
                        error: format!("mvcc_checkpoint failed on shard: {e}"),
                    }),
                ));
            }
        }
    }

    Ok(Json(MvccCheckpointResponse {
        collection: name.clone(),
        shards: shard_map.len(),
        shards_checkpointed,
        entries,
    }))
}
