//! Integration tests for the change-point REST API
//! (`GET /v1/admin/change-points`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::advisor::snapshot::{
    GlobalSnapshot, HealthSnapshot, OpsSnapshot, QuerySnapshot,
};
use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server. Returns the base URL, the live `AppState` clone
/// (so the test can seed advisor history), and the TempDir guard.
async fn start_server() -> (String, AppState, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let state_handle = app_state.clone();
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, state_handle, tmp)
}

async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert("admin".to_owned(), UserRecord::with_password("admin", Role::Admin));
    users.insert("viewer".to_owned(), UserRecord::with_password("viewer", Role::Viewer));
    let app_state = AppState::with_auth_and_users(tmp.path(), "test-jwt-secret", users);
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
}

async fn login(base: &str, user: &str, pass: &str) -> String {
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": user, "password": pass }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "login failed for {user}");
    let body: Value = r.json().await.unwrap();
    body["token"].as_str().unwrap().to_owned()
}

/// A minimal snapshot at `taken_at_ms` carrying a `disk_used_pct` reading.
fn disk_snapshot(taken_at_ms: i64, disk_used_pct: f64) -> OpsSnapshot {
    OpsSnapshot {
        taken_at_ms,
        global: GlobalSnapshot { active_collections: 0, total_records: 0 },
        collections: Vec::new(),
        nodes: Vec::new(),
        replica_safe_ts: None,
        health: HealthSnapshot {
            minio: "healthy".into(),
            nats: "healthy".into(),
            embedding: "healthy".into(),
            multimodal_embedding: "healthy".into(),
        },
        query: QuerySnapshot {
            window_secs: 300,
            count: 0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            slow_count: 0,
            recent_latencies_ms: Vec::new(),
        },
        rss_bytes: None,
        disk_total_bytes: None,
        disk_free_bytes: None,
        disk_used_pct: Some(disk_used_pct),
        cpu_total_jiffies: None,
        cpu_busy_jiffies: None,
        num_cpus: 1,
        load1: None,
        load5: None,
        load15: None,
        open_fds: None,
        fd_limit: None,
        mem_total_bytes: None,
        mem_available_bytes: None,
        counters: HashMap::new(),
        error_log_count: 0,
        warn_log_count: 0,
        slo_statuses: Vec::new(),
        synthetic_summaries: Vec::new(),
        index_health: Vec::new(),
        forecasts: Vec::new(),
        change_points: Vec::new(),
        faithfulness_recent: Vec::new(),
    }
}

#[tokio::test]
async fn change_points_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/change-points")).send().await.unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["change_points"].is_array());
    assert_eq!(body["change_points"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn change_points_returns_level_shift_for_step_series() {
    let (base, state, _tmp) = start_server().await;

    // Seed a clean disk step (40% → 80% at the midpoint) into the advisor,
    // then attach the computed change points onto the latest snapshot
    // (exactly as run_tick does).
    {
        let advisor = state.advisor();
        let mut history = advisor.history.lock().await;
        for i in 0..40usize {
            let ts = i as i64 * 10_000;
            let pct = if i < 20 { 40.0 } else { 80.0 };
            history.push(disk_snapshot(ts, pct));
        }
        let cps = veda_gateway::advisor::changepoint::compute_change_points(
            &history,
            veda_gateway::advisor::changepoint::DEFAULT_MIN_POINTS,
            veda_gateway::advisor::changepoint::DEFAULT_MIN_MAGNITUDE,
        );
        history.set_latest_change_points(cps);
    }

    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/change-points")).send().await.unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    let cps = body["change_points"].as_array().expect("change_points array");

    let disk = cps
        .iter()
        .find(|c| c["metric"] == json!("disk_used_pct"))
        .expect("disk_used_pct change point present");
    // Shape: metric / kind / at_ms / before / after / magnitude / confidence.
    assert_eq!(disk["kind"], json!("level"));
    assert!((disk["before"].as_f64().unwrap() - 40.0).abs() < 1.0);
    assert!((disk["after"].as_f64().unwrap() - 80.0).abs() < 1.0);
    assert!(disk["magnitude"].as_f64().unwrap() > 0.0);
    assert!(disk["at_ms"].as_u64().is_some());
}

#[tokio::test]
async fn change_points_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/change-points"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/change-points"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
