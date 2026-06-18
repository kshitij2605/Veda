//! Integration tests for the capacity-forecast REST API
//! (`GET /v1/admin/forecast`).

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

/// A minimal snapshot at `taken_at_ms` carrying a rising `disk_used_pct`.
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
    }
}

#[tokio::test]
async fn forecast_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/forecast")).send().await.unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["forecasts"].is_array());
    assert_eq!(body["forecasts"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn forecast_returns_disk_eta_for_rising_series() {
    let (base, state, _tmp) = start_server().await;

    // Seed a clean rising-disk history into the advisor, then attach the
    // computed forecasts onto the latest snapshot (exactly as run_tick does).
    let now_ms = 1_700_000_000_000i64;
    {
        let advisor = state.advisor();
        let mut history = advisor.history.lock().await;
        // 30 points, 1000s apart, rising 0.5%/step from 50% → ~64.5% now.
        for i in 0..30 {
            let ts = now_ms - (29 - i) as i64 * 1000 * 1000;
            history.push(disk_snapshot(ts, 50.0 + 0.5 * i as f64));
        }
        let forecasts = veda_gateway::advisor::forecast::compute_forecasts(
            &history,
            now_ms,
            veda_gateway::advisor::forecast::DEFAULT_MIN_POINTS,
            veda_gateway::advisor::forecast::DEFAULT_MIN_R2,
        );
        history.set_latest_forecasts(forecasts);
    }

    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/forecast")).send().await.unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    let forecasts = body["forecasts"].as_array().expect("forecasts array");

    let disk = forecasts
        .iter()
        .find(|f| f["metric"] == json!("disk_used_pct"))
        .expect("disk_used_pct forecast present");
    // Shape: current / slope_per_sec / r2 / threshold / eta_ms.
    assert!(disk["current"].as_f64().unwrap() > 60.0);
    assert!(disk["r2"].as_f64().unwrap() > 0.9, "high-confidence fit");
    assert_eq!(disk["threshold"], json!(95.0));
    let eta = disk["eta_ms"].as_i64().expect("eta present");
    assert!(eta > now_ms, "eta projected into the future");
}

#[tokio::test]
async fn forecast_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/forecast"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/forecast"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
