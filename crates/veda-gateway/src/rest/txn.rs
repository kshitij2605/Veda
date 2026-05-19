//! Transaction lifecycle REST endpoints (v0 — NOT production
//! Percolator).
//!
//! Surface:
//!   POST /v1/txn                  open a new transaction
//!     → 201 { "txn_id": "<uuid>" }
//!   POST /v1/txn/{txn_id}/commit  commit, flipping staged writes
//!     → 200 { "txn_id": "...", "status": "committed" }
//!   POST /v1/txn/{txn_id}/abort   abort, dropping staged writes
//!     → 200 { "txn_id": "...", "status": "aborted" }
//!
//! Writes participate in a transaction when the caller sends
//! `X-Veda-Txn-Id: <txn_id>` on POST /v1/collections/{name}/insert.
//! The insert handler will then submit `TxnWrite` instead of
//! `InsertRecord` and the row stays invisible until commit.
//!
//! Limits in v0 — see [`veda_coordinator::raft_types::ClusterCommand::BeginTxn`]
//! doc for the full list. Headline: no concurrency control, no
//! read-your-writes inside a txn, no cross-shard coordination.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use uuid::Uuid;
use veda_coordinator::raft_types::ClusterCommand;

use super::collections::{raft_redirect, ErrorResponse};
use super::state::AppState;

#[derive(Debug, Serialize)]
pub struct BeginTxnResponse {
    pub txn_id: String,
}

#[derive(Debug, Serialize)]
pub struct TxnStatusResponse {
    pub txn_id: String,
    pub status: &'static str,
}

/// Header callers attach to writes to enroll them in a txn.
pub const TXN_HEADER: &str = "x-veda-txn-id";

pub async fn begin(
    State(state): State<AppState>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return raft_redirect(&leader_url, "/v1/txn");
    }
    use axum::response::IntoResponse;
    let txn_id = Uuid::new_v4().to_string();
    if let Err((status, body)) = submit(&state, ClusterCommand::BeginTxn {
        txn_id: txn_id.clone(),
    })
    .await
    {
        return (status, body).into_response();
    }
    (StatusCode::CREATED, Json(BeginTxnResponse { txn_id })).into_response()
}

pub async fn commit(
    State(state): State<AppState>,
    Path(txn_id): Path<String>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return raft_redirect(&leader_url, &format!("/v1/txn/{txn_id}/commit"));
    }
    use axum::response::IntoResponse;
    if let Err((status, body)) = submit(&state, ClusterCommand::CommitTxn {
        txn_id: txn_id.clone(),
    })
    .await
    {
        return (status, body).into_response();
    }
    Json(TxnStatusResponse {
        txn_id,
        status: "committed",
    })
    .into_response()
}

pub async fn abort(
    State(state): State<AppState>,
    Path(txn_id): Path<String>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return raft_redirect(&leader_url, &format!("/v1/txn/{txn_id}/abort"));
    }
    use axum::response::IntoResponse;
    if let Err((status, body)) = submit(&state, ClusterCommand::AbortTxn {
        txn_id: txn_id.clone(),
    })
    .await
    {
        return (status, body).into_response();
    }
    Json(TxnStatusResponse {
        txn_id,
        status: "aborted",
    })
    .into_response()
}

/// Push the txn command through Raft. v0 requires Raft mode —
/// without consensus, transactional staging has nowhere to live.
async fn submit(
    state: &AppState,
    cmd: ClusterCommand,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let raft_guard = state.inner.raft.read().await;
    let coord = raft_guard.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "transactions require VEDA_COORDINATOR=raft".to_owned(),
            }),
        )
    })?;
    coord.execute(cmd).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("raft submit: {e}"),
            }),
        )
    })?;
    Ok(())
}
