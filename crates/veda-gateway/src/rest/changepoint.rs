//! Change-point admin API (`GET /v1/admin/change-points`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::forecast`].
//!
//! Surfaces the sustained level/trend shifts the advisor attaches to its
//! latest snapshot (`OpsSnapshot::change_points`): per metric the kind
//! (level/trend), the before→after values, the shift time (`at_ms`), and a
//! normalized magnitude + confidence.
//!
//! Change points are split-test detections over the snapshot history
//! (warm-up + min-magnitude guarded so they never fire on a flat/noisy
//! series) — advisory only; nothing here mutates state.

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::advisor::changepoint::ChangePoint;
use crate::rest::state::AppState;

#[derive(Debug, Serialize)]
pub struct ChangePointResponse {
    /// Latest change points, newest snapshot. Empty when the advisor hasn't
    /// ticked yet or no series cleared the warm-up / min-magnitude guards.
    pub change_points: Vec<ChangePoint>,
}

/// `GET /v1/admin/change-points` — the latest detected level/trend shifts.
pub async fn change_points(State(state): State<AppState>) -> Json<ChangePointResponse> {
    let advisor = state.advisor();
    let change_points = {
        let history = advisor.history.lock().await;
        history
            .latest()
            .map(|snap| snap.change_points.clone())
            .unwrap_or_default()
    };
    Json(ChangePointResponse { change_points })
}
