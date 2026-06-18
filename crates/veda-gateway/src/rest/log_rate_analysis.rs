//! Semantic spike root-cause attribution admin API
//! (`GET /v1/admin/log-rate-analysis`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::changepoint`]
//! and [`crate::rest::correlations`].
//!
//! When the WARN/ERROR log rate spikes, this attributes the spike to the
//! semantic **categories** driving it: it pulls WARN+ERROR records for the
//! spike window `[now-window, now]` and the baseline window immediately
//! before it, categorizes their union (embedding path when a text model is
//! configured, normalized-template fallback otherwise), and returns the
//! categories ranked by how much they grew. Advisory only — nothing here
//! mutates state.

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;

use crate::advisor::spike_attribution::{analyze_log_rate, SpikeAttribution};
use crate::rest::state::AppState;

/// Default spike window (seconds) — matches the advisor snapshot window.
const DEFAULT_WINDOW_SECS: u64 = 300;
/// Default baseline window (seconds) — the span just before the spike.
const DEFAULT_BASELINE_SECS: u64 = 900;

#[derive(Debug, Deserialize)]
pub struct LogRateQuery {
    /// Spike window in seconds (default 300).
    #[serde(default)]
    pub window_secs: Option<u64>,
    /// Baseline window in seconds (default 900).
    #[serde(default)]
    pub baseline_secs: Option<u64>,
    /// Minimum log level to consider (default WARN). One of WARN | ERROR.
    #[serde(default)]
    pub min_level: Option<String>,
}

/// `GET /v1/admin/log-rate-analysis?window_secs=&baseline_secs=&min_level=`
/// — attribute a WARN/ERROR log-rate spike to the semantic categories
/// driving it.
pub async fn log_rate_analysis(
    State(state): State<AppState>,
    Query(q): Query<LogRateQuery>,
) -> Json<SpikeAttribution> {
    let window_secs = q.window_secs.unwrap_or(DEFAULT_WINDOW_SECS).max(1);
    let baseline_secs = q.baseline_secs.unwrap_or(DEFAULT_BASELINE_SECS).max(1);
    let min_level = match q.min_level.as_deref() {
        Some(l) if l.eq_ignore_ascii_case("error") => "ERROR",
        _ => "WARN",
    };

    let attribution = analyze_log_rate(&state, window_secs, baseline_secs, min_level).await;
    Json(attribution)
}
