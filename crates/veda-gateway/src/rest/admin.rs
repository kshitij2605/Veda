use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use veda_core::metrics::Metrics;

use crate::auth;

use super::state::AppState;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct ListNodesResponse {
    pub nodes: Vec<NodeInfo>,
}

#[derive(Debug, Serialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
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

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".into(),
        version: "0.1.0".into(),
    })
}

pub async fn list_nodes() -> Json<ListNodesResponse> {
    Json(ListNodesResponse { nodes: vec![] })
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

pub async fn metrics() -> (StatusCode, [(axum::http::header::HeaderName, &'static str); 1], String) {
    let body = Metrics::global().export_prometheus();
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}
