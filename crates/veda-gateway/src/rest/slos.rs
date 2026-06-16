//! Admin REST API for service-level objectives (`/v1/admin/slos*`).
//!
//! Admin-only, mirroring [`crate::rest::insights`]: registered under the
//! protected router with an extra `rbac::require_admin` route layer. CRUD
//! over the persisted [`SloRegistry`] on `AdvisorState`, plus a live
//! `{id}/status` view computed from the rolling [`SloState`] counters.
//!
//!   GET    /v1/admin/slos              — list SLO definitions
//!   POST   /v1/admin/slos              — create (server-assigns id if absent)
//!   GET    /v1/admin/slos/{id}         — get one definition
//!   PUT    /v1/admin/slos/{id}         — replace one definition
//!   DELETE /v1/admin/slos/{id}         — delete one definition
//!   GET    /v1/admin/slos/{id}/status  — live SLI / budget / burn status

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

use crate::advisor::engine::now_ms;
use crate::advisor::slo::{SloDef, SloStatus};
use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;

/// SLO registry file name under `data_dir` (kept in sync with
/// [`crate::advisor::engine`]).
const SLOS_FILE: &str = "_slos.json";

#[derive(Debug, Serialize)]
pub struct SlosResponse {
    pub slos: Vec<SloDef>,
}

fn err(code: StatusCode, msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (code, Json(ErrorResponse { error: msg }))
}

/// `GET /v1/admin/slos` — all SLO definitions.
pub async fn list_slos(State(state): State<AppState>) -> Json<SlosResponse> {
    let slos = state.advisor().slos.read().await.list().to_vec();
    Json(SlosResponse { slos })
}

/// `POST /v1/admin/slos` — create an SLO. The server assigns a random id
/// when the body's `id` is empty; a supplied id that already exists is a
/// 409. Persists the registry. Returns the created definition (201).
pub async fn create_slo(
    State(state): State<AppState>,
    Json(mut def): Json<SloDef>,
) -> Result<(StatusCode, Json<SloDef>), (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut slos = advisor.slos.write().await;
    if def.id.trim().is_empty() {
        def.id = format!("slo-{}", uuid::Uuid::new_v4());
    } else if slos.get(&def.id).is_some() {
        return Err(err(
            StatusCode::CONFLICT,
            format!("SLO '{}' already exists", def.id),
        ));
    }
    slos.upsert(def.clone());
    slos.save(&state.inner.data_dir.join(SLOS_FILE));
    Ok((StatusCode::CREATED, Json(def)))
}

/// `GET /v1/admin/slos/{id}` — one SLO definition (404 if absent).
pub async fn get_slo(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SloDef>, (StatusCode, Json<ErrorResponse>)> {
    match state.advisor().slos.read().await.get(&id) {
        Some(def) => Ok(Json(def.clone())),
        None => Err(err(StatusCode::NOT_FOUND, format!("no SLO with id '{id}'"))),
    }
}

/// `PUT /v1/admin/slos/{id}` — replace the definition with `id`. The
/// path id wins over any `id` in the body. 404 if it does not exist.
pub async fn put_slo(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut def): Json<SloDef>,
) -> Result<Json<SloDef>, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut slos = advisor.slos.write().await;
    if slos.get(&id).is_none() {
        return Err(err(StatusCode::NOT_FOUND, format!("no SLO with id '{id}'")));
    }
    def.id = id;
    slos.upsert(def.clone());
    slos.save(&state.inner.data_dir.join(SLOS_FILE));
    Ok(Json(def))
}

/// `DELETE /v1/admin/slos/{id}` — remove the definition (204). 404 if
/// absent.
pub async fn delete_slo(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let mut slos = advisor.slos.write().await;
    if slos.remove(&id) {
        slos.save(&state.inner.data_dir.join(SLOS_FILE));
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(err(StatusCode::NOT_FOUND, format!("no SLO with id '{id}'")))
    }
}

/// `GET /v1/admin/slos/{id}/status` — live [`SloStatus`] computed from
/// the rolling counters as of now. 404 if no such SLO definition.
pub async fn get_slo_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SloStatus>, (StatusCode, Json<ErrorResponse>)> {
    let advisor = state.advisor();
    let def = match advisor.slos.read().await.get(&id) {
        Some(d) => d.clone(),
        None => return Err(err(StatusCode::NOT_FOUND, format!("no SLO with id '{id}'"))),
    };
    let now = now_ms().max(0) as u64;
    let status = advisor.slo_state.lock().await.status(&def, now);
    Ok(Json(status))
}
