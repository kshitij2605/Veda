//! Admin REST API for synthetic self-check probes
//! (`/v1/admin/synthetics*`).
//!
//! Admin-only, mirroring [`crate::rest::slos`]: registered under the
//! protected router with an extra `rbac::require_admin` route layer. CRUD
//! over the persisted [`SyntheticRegistry`] on `AdvisorState`, plus a
//! `{id}/results` view (recent [`ProbeResult`]s + the derived
//! [`SyntheticSummary`]) read from the [`SyntheticResults`] rings.
//!
//!   GET    /v1/admin/synthetics              — list probe definitions
//!   POST   /v1/admin/synthetics              — create (server-assigns id if absent)
//!   GET    /v1/admin/synthetics/{id}         — get one definition
//!   PUT    /v1/admin/synthetics/{id}         — replace one definition
//!   DELETE /v1/admin/synthetics/{id}         — delete one definition
//!   GET    /v1/admin/synthetics/{id}/results — recent results + summary

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

use crate::advisor::synthetic::{ProbeResult, SyntheticProbe, SyntheticSummary};
use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;

/// Synthetic registry file name under `data_dir` (kept in sync with
/// [`crate::advisor::engine`]).
const SYNTHETICS_FILE: &str = "_synthetics.json";

#[derive(Debug, Serialize)]
pub struct SyntheticsResponse {
    pub synthetics: Vec<SyntheticProbe>,
}

#[derive(Debug, Serialize)]
pub struct SyntheticResultsResponse {
    pub summary: SyntheticSummary,
    pub results: Vec<ProbeResult>,
}

fn err(code: StatusCode, msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (code, Json(ErrorResponse { error: msg }))
}

/// `GET /v1/admin/synthetics` — all probe definitions.
pub async fn list_synthetics(State(state): State<AppState>) -> Json<SyntheticsResponse> {
    let synthetics = state.advisor().synthetics.read().await.list().to_vec();
    Json(SyntheticsResponse { synthetics })
}

/// `POST /v1/admin/synthetics` — create a probe. The server assigns a
/// random id when the body's `id` is empty; a supplied id that already
/// exists is a 409. Persists the registry. Returns the created probe (201).
pub async fn create_synthetic(
    State(state): State<AppState>,
    Json(mut probe): Json<SyntheticProbe>,
) -> Result<(StatusCode, Json<SyntheticProbe>), (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut reg = advisor.synthetics.write().await;
    if probe.id.trim().is_empty() {
        probe.id = format!("synthetic-{}", uuid::Uuid::new_v4());
    } else if reg.get(&probe.id).is_some() {
        return Err(err(
            StatusCode::CONFLICT,
            format!("synthetic probe '{}' already exists", probe.id),
        ));
    }
    reg.upsert(probe.clone());
    reg.save(&state.inner.data_dir.join(SYNTHETICS_FILE));
    Ok((StatusCode::CREATED, Json(probe)))
}

/// `GET /v1/admin/synthetics/{id}` — one probe definition (404 if absent).
pub async fn get_synthetic(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SyntheticProbe>, (StatusCode, Json<ErrorResponse>)> {
    match state.advisor().synthetics.read().await.get(&id) {
        Some(p) => Ok(Json(p.clone())),
        None => Err(err(
            StatusCode::NOT_FOUND,
            format!("no synthetic probe with id '{id}'"),
        )),
    }
}

/// `PUT /v1/admin/synthetics/{id}` — replace the probe with `id`. The
/// path id wins over any `id` in the body. 404 if it does not exist.
pub async fn put_synthetic(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut probe): Json<SyntheticProbe>,
) -> Result<Json<SyntheticProbe>, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut reg = advisor.synthetics.write().await;
    if reg.get(&id).is_none() {
        return Err(err(
            StatusCode::NOT_FOUND,
            format!("no synthetic probe with id '{id}'"),
        ));
    }
    probe.id = id;
    reg.upsert(probe.clone());
    reg.save(&state.inner.data_dir.join(SYNTHETICS_FILE));
    Ok(Json(probe))
}

/// `DELETE /v1/admin/synthetics/{id}` — remove the probe (204). 404 if absent.
pub async fn delete_synthetic(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut reg = advisor.synthetics.write().await;
    if reg.remove(&id) {
        reg.save(&state.inner.data_dir.join(SYNTHETICS_FILE));
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(err(
            StatusCode::NOT_FOUND,
            format!("no synthetic probe with id '{id}'"),
        ))
    }
}

/// `GET /v1/admin/synthetics/{id}/results` — recent [`ProbeResult`]s plus
/// the derived [`SyntheticSummary`]. 404 if no such probe definition.
pub async fn get_synthetic_results(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SyntheticResultsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let name = match advisor.synthetics.read().await.get(&id) {
        Some(p) => p.name.clone(),
        None => {
            return Err(err(
                StatusCode::NOT_FOUND,
                format!("no synthetic probe with id '{id}'"),
            ))
        }
    };
    let results = advisor.synthetic_results.lock().await;
    let summary = results.summary(&id, &name);
    let recent = results.recent(&id);
    Ok(Json(SyntheticResultsResponse {
        summary,
        results: recent,
    }))
}
