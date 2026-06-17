//! Integration tests for the on-demand CPU-profiling REST API
//! (`GET /v1/admin/pprof`) — admin-gated, returns a flamegraph SVG or a
//! pprof protobuf. Compiled only under the `profiling` cargo feature.
#![cfg(feature = "profiling")]

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// The CPU profiler is process-global, but cargo runs tests in parallel
/// threads within one process — so the profiling tests must serialize
/// against each other or they collide on the single global guard (409).
static PROFILE_SERIAL: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Auth-disabled server over a fresh temp dir (mirrors the resources test).
async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
}

/// Auth-enabled server with an admin + a viewer user.
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

/// Spawn a CPU-busy task for `secs` so the profiler captures some
/// samples while the request is in flight.
fn spin_for(secs: u64) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        let mut x: u64 = 0;
        while std::time::Instant::now() < deadline {
            // busy work the profiler can sample
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            std::hint::black_box(x);
        }
    })
}

#[tokio::test]
async fn flamegraph_returns_svg() {
    let _serial = PROFILE_SERIAL.lock().await;
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let _spin = spin_for(2);
    let r = c
        .get(format!("{base}/v1/admin/pprof?seconds=1&frequency=200&format=flamegraph"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let ct = r
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert_eq!(ct, "image/svg+xml", "unexpected content-type");
    let body = r.text().await.unwrap();
    assert!(!body.is_empty(), "flamegraph body is empty");
    // Tolerate a small/near-empty flamegraph — just assert it's a valid SVG.
    assert!(body.contains("<svg"), "body is not an SVG document: {}", &body[..body.len().min(200)]);
}

#[tokio::test]
async fn pprof_protobuf_returns_octet_stream() {
    let _serial = PROFILE_SERIAL.lock().await;
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let _spin = spin_for(2);
    let r = c
        .get(format!("{base}/v1/admin/pprof?seconds=1&format=pb"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let ct = r
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert_eq!(ct, "application/octet-stream", "unexpected content-type");
    let body = r.bytes().await.unwrap();
    assert!(!body.is_empty(), "pprof protobuf body is empty");
}

#[tokio::test]
async fn pprof_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer token is rejected by the require_admin route layer.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/pprof?seconds=1"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");
}

#[tokio::test]
async fn concurrent_profiles_conflict() {
    let _serial = PROFILE_SERIAL.lock().await;
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let _spin = spin_for(3);

    // Fire two overlapping profile requests; the process-global profiler
    // guard must let exactly one through and 409 the other.
    let b1 = base.clone();
    let c1 = c.clone();
    let h1 = tokio::spawn(async move {
        c1.get(format!("{b1}/v1/admin/pprof?seconds=2&format=pb"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16()
    });
    // Tiny stagger so the first request grabs the guard first.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let s2 = c
        .get(format!("{base}/v1/admin/pprof?seconds=2&format=pb"))
        .send()
        .await
        .unwrap()
        .status()
        .as_u16();
    let s1 = h1.await.unwrap();

    let mut statuses = [s1, s2];
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 409], "expected one 200 and one 409, got {statuses:?}");
}
