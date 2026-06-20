//! Integration tests for the node-resources REST API
//! (`GET /v1/admin/resources`) — admin-gated, returns the local node's
//! latest resource snapshot as a stable JSON shape (all signal fields
//! may be null on non-Linux / CI).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server over a fresh temp dir (mirrors the insights test).
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
    users.insert(
        "admin".to_owned(),
        UserRecord::with_password("admin", Role::Admin),
    );
    users.insert(
        "viewer".to_owned(),
        UserRecord::with_password("viewer", Role::Viewer),
    );
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

#[tokio::test]
async fn resources_returns_stable_shape() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/resources"))
        .send()
        .await
        .unwrap();
    // 200 with the full shape even when every signal is null (non-Linux).
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();

    // Top-level keys are always present.
    for key in [
        "disk",
        "cpu_pct",
        "load",
        "num_cpus",
        "fds",
        "memory",
        "rss_bytes",
    ] {
        assert!(body.get(key).is_some(), "missing key '{key}' in {body}");
    }
    // Nested object shapes.
    let disk = &body["disk"];
    for k in ["total_bytes", "free_bytes", "used_pct"] {
        assert!(disk.get(k).is_some(), "disk.{k} missing in {disk}");
    }
    let load = &body["load"];
    for k in ["load1", "load5", "load15"] {
        assert!(load.get(k).is_some(), "load.{k} missing in {load}");
    }
    let fds = &body["fds"];
    for k in ["open", "limit"] {
        assert!(fds.get(k).is_some(), "fds.{k} missing in {fds}");
    }
    let mem = &body["memory"];
    for k in ["total_bytes", "available_bytes"] {
        assert!(mem.get(k).is_some(), "memory.{k} missing in {mem}");
    }
    // num_cpus is always a positive integer.
    assert!(body["num_cpus"].as_u64().unwrap() >= 1, "num_cpus < 1");
    // Empty advisor history -> no previous snapshot -> cpu_pct is null.
    assert!(
        body["cpu_pct"].is_null(),
        "cpu_pct must be null on empty history"
    );
}

#[tokio::test]
async fn resources_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer token is rejected by the require_admin route layer.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/resources"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin token passes.
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/resources"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "admin must be allowed: {:?}",
        r.text().await
    );
}
