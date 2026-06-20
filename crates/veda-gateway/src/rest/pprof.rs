//! On-demand CPU profiling REST API (`GET /v1/admin/pprof`).
//!
//! Compiled only under the `profiling` cargo feature — the `pprof`
//! dependency (and the signal-based sampling profiler it installs) is
//! not free, so the default build excludes this module entirely and the
//! route is never registered.
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::resources`]
//! and [`crate::rest::correlations`].
//!
//! On request the handler starts a [`pprof::ProfilerGuard`], samples for
//! `seconds` (1..=60) at `frequency` Hz, then renders either a flamegraph
//! SVG (`format=flamegraph`, default) or a pprof protobuf
//! (`format=pb`/`proto`).
//!
//! The CPU sampling profiler is **process-global**: only one capture can
//! run at a time. A module-level `AtomicBool` guards it; a second
//! overlapping request gets `409 Conflict` instead of corrupting the
//! in-flight profile.

use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::Query;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use protobuf::Message;

/// Process-global guard: `true` while a profile is being captured.
/// The signal-based sampling profiler can only have one guard live at a
/// time, so a second concurrent request is rejected with 409 rather than
/// fighting over the global profiler state.
static PROFILING: AtomicBool = AtomicBool::new(false);

const DEFAULT_SECONDS: u64 = 5;
const MIN_SECONDS: u64 = 1;
const MAX_SECONDS: u64 = 60;
const DEFAULT_FREQUENCY: i32 = 100;

#[derive(Debug, Deserialize)]
pub struct PprofQuery {
    /// Sampling duration in seconds (default 5, clamped to 1..=60).
    #[serde(default)]
    pub seconds: Option<u64>,
    /// Sampling frequency in Hz (default 100).
    #[serde(default)]
    pub frequency: Option<i32>,
    /// Output format: `flamegraph` (SVG, default) or `pb`/`proto`
    /// (pprof protobuf).
    #[serde(default)]
    pub format: Option<String>,
}

/// RAII reset of the process-global profiling flag so a panic or early
/// return always releases the guard.
struct ProfilingGuard;

impl Drop for ProfilingGuard {
    fn drop(&mut self) {
        PROFILING.store(false, Ordering::SeqCst);
    }
}

/// `GET /v1/admin/pprof?seconds=&frequency=&format=` — capture an
/// on-demand CPU profile and return it as a flamegraph SVG or a pprof
/// protobuf. Admin-only (extra `require_admin` route layer). Returns 409
/// if a profile is already running.
pub async fn pprof(Query(q): Query<PprofQuery>) -> Response {
    // Acquire the process-global profiler guard (one capture at a time).
    if PROFILING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return (
            StatusCode::CONFLICT,
            "a CPU profile is already running; only one capture may run at a time",
        )
            .into_response();
    }
    let _release = ProfilingGuard;

    let seconds = q
        .seconds
        .unwrap_or(DEFAULT_SECONDS)
        .clamp(MIN_SECONDS, MAX_SECONDS);
    let frequency = q.frequency.unwrap_or(DEFAULT_FREQUENCY).max(1);
    let format = q.format.as_deref().unwrap_or("flamegraph");

    let guard = match pprof::ProfilerGuardBuilder::default()
        .frequency(frequency)
        .blocklist(&["libc", "libpthread", "vdso"])
        .build()
    {
        Ok(g) => g,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to start profiler: {e}"),
            )
                .into_response();
        }
    };

    tokio::time::sleep(std::time::Duration::from_secs(seconds)).await;

    let report = match guard.report().build() {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build profile report: {e}"),
            )
                .into_response();
        }
    };

    match format {
        "pb" | "proto" | "pprof" => {
            let profile = match report.pprof() {
                Ok(p) => p,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to build pprof profile: {e}"),
                    )
                        .into_response();
                }
            };
            let body = match profile.write_to_bytes() {
                Ok(b) => b,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to encode pprof profile: {e}"),
                    )
                        .into_response();
                }
            };
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/octet-stream"),
                    (
                        header::CONTENT_DISPOSITION,
                        "attachment; filename=\"profile.pb\"",
                    ),
                ],
                body,
            )
                .into_response()
        }
        // "flamegraph" and anything else default to SVG.
        _ => {
            let mut buf: Vec<u8> = Vec::new();
            if let Err(e) = report.flamegraph(&mut buf) {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to render flamegraph: {e}"),
                )
                    .into_response();
            }
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/svg+xml")],
                buf,
            )
                .into_response()
        }
    }
}
