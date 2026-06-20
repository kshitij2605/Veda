//! Admin REST API for the local node's resource snapshot.
//!
//!   GET /v1/admin/resources — disk / cpu% / load / fds / memory / RSS
//!
//! Admin-only — registered under the same protected router as the
//! insights endpoints, with the `rbac::require_admin` route layer.
//!
//! Values come from the ops-advisor's live [`SnapshotHistory`]: the
//! latest entry supplies the gauges, and `cpu_pct(previous, latest)`
//! supplies the busy-CPU percentage (null until there are two
//! snapshots). When the history is empty (the advisor loop hasn't ticked
//! yet) we collect a fresh snapshot with the same args the engine uses,
//! and `cpu_pct` is null. All signal fields are `Option` → serialized as
//! `null` on non-Linux; the route always returns 200 with this shape.

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::advisor::engine::now_ms;
use crate::advisor::snapshot::{OpsSnapshot, collect, cpu_pct};
use crate::rest::state::AppState;

#[derive(Debug, Serialize)]
pub struct DiskView {
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub used_pct: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct LoadView {
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub load15: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct FdView {
    pub open: Option<u64>,
    pub limit: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct MemoryView {
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ResourcesView {
    pub disk: DiskView,
    /// Busy-CPU percentage between the advisor's previous and latest
    /// snapshots; `null` with fewer than two snapshots.
    pub cpu_pct: Option<f64>,
    pub load: LoadView,
    pub num_cpus: usize,
    pub fds: FdView,
    pub memory: MemoryView,
    pub rss_bytes: Option<u64>,
}

/// Project an [`OpsSnapshot`] (+ already-computed `cpu_pct`) into the
/// REST view shape.
fn view(snap: &OpsSnapshot, cpu_pct: Option<f64>) -> ResourcesView {
    ResourcesView {
        disk: DiskView {
            total_bytes: snap.disk_total_bytes,
            free_bytes: snap.disk_free_bytes,
            used_pct: snap.disk_used_pct,
        },
        cpu_pct,
        load: LoadView {
            load1: snap.load1,
            load5: snap.load5,
            load15: snap.load15,
        },
        num_cpus: snap.num_cpus,
        fds: FdView {
            open: snap.open_fds,
            limit: snap.fd_limit,
        },
        memory: MemoryView {
            total_bytes: snap.mem_total_bytes,
            available_bytes: snap.mem_available_bytes,
        },
        rss_bytes: snap.rss_bytes,
    }
}

/// `GET /v1/admin/resources` — the local node's latest resource
/// snapshot. Sourced from the advisor's history when available
/// (with `cpu_pct` over previous→latest); otherwise a fresh
/// `collect` with `cpu_pct` null. Always 200.
pub async fn get_resources(State(state): State<AppState>) -> Json<ResourcesView> {
    {
        let history = state.advisor().history.lock().await;
        if let Some(latest) = history.latest() {
            let pct = history.previous().and_then(|prev| cpu_pct(prev, latest));
            return Json(view(latest, pct));
        }
    }
    // Empty history (advisor hasn't ticked yet): collect a fresh
    // snapshot with the engine's args. No previous snapshot → cpu_pct null.
    let snap = collect(&state, now_ms()).await;
    Json(view(&snap, None))
}
