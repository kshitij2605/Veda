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
/// Single-node mode: returns one entry describing this process, with
/// role = "leader" and status derived from /admin/health. The
/// response shape (and the `mode` field) is intentionally what a
/// future N-node Raft cluster would return — when veda-coordinator's
/// Raft layer wires up, this handler grows a fan-out and the
/// frontend reads the same JSON.
pub async fn list_nodes(
    State(state): State<AppState>,
) -> Json<ListNodesResponse> {
    let id = std::env::var("VEDA_NODE_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node-0".to_owned());
    let address = std::env::var("VEDA_NODE_ADDRESS")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "0.0.0.0:6340".to_owned());
    let checks = run_health_checks().await;
    let status = compute_status(&checks);
    let uptime_secs = state.inner.started_at.elapsed().as_secs();

    let (collection_count, shard_count) = {
        let segs = state.inner.segment_managers.read().await;
        let cols = segs.len() as u32;
        let shards: u32 = segs.values().map(|m| m.len() as u32).sum();
        (cols, shards)
    };

    Json(ListNodesResponse {
        mode: "single".into(),
        nodes: vec![NodeInfo {
            id,
            address,
            role: "leader".into(),
            status,
            version: "0.1.0".into(),
            uptime_secs,
            shard_count,
            collection_count,
            is_self: true,
        }],
    })
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
