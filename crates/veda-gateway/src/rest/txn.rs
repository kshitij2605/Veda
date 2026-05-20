//! Transaction lifecycle REST endpoints — production Phase G
//! integration. Routes through `TxnCoordinator` (Percolator SI)
//! instead of the v0 ClusterCommand-based stub.
//!
//! Surface (unchanged from v0, the wire shape is stable):
//!   POST /v1/txn                  open a new transaction
//!     → 201 { "txn_id": "<uuid>" }
//!   POST /v1/txn/{txn_id}/commit  commit (prewrite + commit)
//!     → 200 { "txn_id": "...", "status": "committed", "commit_ts": "..." }
//!   POST /v1/txn/{txn_id}/abort   abort, dropping staged writes
//!     → 200 { "txn_id": "...", "status": "aborted" }
//!
//! Writes participate via `X-Veda-Txn-Id: <txn_id>` on
//! `POST /v1/collections/{name}/insert`. The insert handler
//! buffers the mutation into the matching TxnHandle (in
//! `AppState::open_txn_handles`); commit triggers prewrite +
//! commit on the coordinator.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use veda_coordinator::percolator::TxnError;

use super::collections::ErrorResponse;
use super::state::AppState;

#[derive(Debug, Serialize)]
pub struct BeginTxnResponse {
    pub txn_id: String,
}

#[derive(Debug, Serialize)]
pub struct CommitTxnResponse {
    pub txn_id: String,
    pub status: &'static str,
    pub commit_ts_ns: u64,
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
) -> Result<(StatusCode, Json<BeginTxnResponse>), (StatusCode, Json<ErrorResponse>)> {
    let handle = state
        .inner
        .txn_coordinator
        .begin()
        .await
        .map_err(map_txn_err)?;
    let txn_id = handle.txn_id.clone();
    state
        .inner
        .open_txn_handles
        .write()
        .await
        .insert(txn_id.clone(), handle);
    Ok((StatusCode::CREATED, Json(BeginTxnResponse { txn_id })))
}

pub async fn commit(
    State(state): State<AppState>,
    Path(txn_id): Path<String>,
) -> Result<Json<CommitTxnResponse>, (StatusCode, Json<ErrorResponse>)> {
    let handle = state
        .inner
        .open_txn_handles
        .write()
        .await
        .remove(&txn_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("txn '{txn_id}' not open"),
                }),
            )
        })?;
    state
        .inner
        .txn_coordinator
        .prewrite(&handle)
        .await
        .map_err(map_txn_err)?;
    let commit_ts_ns = state
        .inner
        .txn_coordinator
        .commit(&handle)
        .await
        .map_err(map_txn_err)?;
    Ok(Json(CommitTxnResponse {
        txn_id,
        status: "committed",
        commit_ts_ns,
    }))
}

pub async fn abort(
    State(state): State<AppState>,
    Path(txn_id): Path<String>,
) -> Result<Json<TxnStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let handle = state
        .inner
        .open_txn_handles
        .write()
        .await
        .remove(&txn_id);
    if let Some(h) = handle {
        state.inner.txn_coordinator.abort(&h).await;
    }
    // Idempotent: aborting an unknown / already-aborted txn
    // returns OK rather than 404 (caller may retry on a flaky
    // network).
    Ok(Json(TxnStatusResponse {
        txn_id,
        status: "aborted",
    }))
}

fn map_txn_err(e: TxnError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &e {
        TxnError::NotOpen { .. } => StatusCode::NOT_FOUND,
        TxnError::AlreadyCommitted { .. } => StatusCode::CONFLICT,
        TxnError::LockConflict { .. } | TxnError::WriteWriteConflict { .. } => {
            StatusCode::CONFLICT
        }
        TxnError::Tso(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    (
        status,
        Json(ErrorResponse {
            error: format!("{e}"),
        }),
    )
}
