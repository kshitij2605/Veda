//! Capacity-forecast admin API (`GET /v1/admin/forecast`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::index_advisor`].
//!
//! Surfaces the predictive ETAs the advisor attaches to its latest snapshot
//! (`OpsSnapshot::forecasts`): per metric the current value, the fitted
//! trend (slope + R² confidence), the threshold being projected against,
//! and the projected epoch-ms crossing (`eta_ms`, `None` when the trend is
//! flat or moving away). This is the data the Niriksha forecast panel reads.
//!
//! Forecasts are directional least-squares projections (warm-up + min-R²
//! guarded so they never fire on a flat/noisy series) — advisory only;
//! nothing here mutates state.

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::advisor::forecast::Forecast;
use crate::rest::state::AppState;

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    /// Latest forecasts, newest snapshot. Empty when the advisor hasn't
    /// ticked yet or no series cleared the warm-up / min-R² guards.
    pub forecasts: Vec<Forecast>,
}

/// `GET /v1/admin/forecast` — the latest capacity forecasts (disk-full /
/// memory-exhaustion / per-collection recall-decay ETAs).
pub async fn forecast(State(state): State<AppState>) -> Json<ForecastResponse> {
    let advisor = state.advisor();
    let forecasts = {
        let history = advisor.history.lock().await;
        history
            .latest()
            .map(|snap| snap.forecasts.clone())
            .unwrap_or_default()
    };
    Json(ForecastResponse { forecasts })
}
