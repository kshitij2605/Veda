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

    // Raft path: fan out over the cluster state from openraft, then
    // overlay the "this node" details (status/uptime/shards) from
    // local-process knowledge. Peer status remains "unknown" until
    // the Tonic transport lands — we know membership but not health.
    let raft_state = {
        let guard = state.inner.raft.read().await;
        match guard.as_ref() {
            Some(c) => c.query_state().await.ok(),
            None => None,
        }
    };
    if let Some(cs) = raft_state {
        let self_nid: u64 = std::env::var("VEDA_NODE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let mut nodes: Vec<NodeInfo> = cs
            .nodes
            .into_iter()
            .map(|(nid, info)| {
                let is_self = nid == self_nid;
                NodeInfo {
                    id: format!("node-{nid}"),
                    address: info.address,
                    role: if is_self { "leader".into() } else { "follower".into() },
                    status: if is_self { self_status.clone() } else { "unknown".into() },
                    version: "0.1.0".into(),
                    uptime_secs: if is_self { uptime_secs } else { 0 },
                    shard_count: if is_self { shard_count } else { 0 },
                    collection_count: if is_self { collection_count } else { 0 },
                    is_self,
                }
            })
            .collect();
        // ClusterState.nodes is populated by RegisterNode commands;
        // if nothing has registered yet (the common single-node
        // bootstrap case), still surface ourselves so the table is
        // never empty.
        if !nodes.iter().any(|n| n.is_self) {
            nodes.push(NodeInfo {
                id: format!("node-{self_nid}"),
                address: self_addr_env.clone(),
                role: "leader".into(),
                status: self_status.clone(),
                version: "0.1.0".into(),
                uptime_secs,
                shard_count,
                collection_count,
                is_self: true,
            });
        }
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
    let user = state.inner.users.get(&req.username).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(LoginError {
                error: "invalid credentials".into(),
            }),
        )
    })?;

    if user.password != req.password {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(LoginError {
                error: "invalid credentials".into(),
            }),
        ));
    }

    let token = auth::generate_token(&req.username, user.role, &state.inner.jwt_secret)
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

async fn run_health_checks() -> HashMap<String, String> {
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
