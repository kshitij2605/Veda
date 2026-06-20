//! Integration tests for the log-rate-analysis REST API
//! (`GET /v1/admin/log-rate-analysis`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server. Returns the base URL and the TempDir guard.
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
async fn log_rate_analysis_empty_on_fresh_server() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/log-rate-analysis"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    // Shape: window_secs / baseline_secs / spike_count / baseline_count /
    // total_delta / drivers / method.
    assert_eq!(body["window_secs"], json!(300));
    assert_eq!(body["baseline_secs"], json!(900));
    assert_eq!(body["spike_count"], json!(0));
    assert_eq!(body["baseline_count"], json!(0));
    assert_eq!(body["total_delta"], json!(0));
    assert!(body["drivers"].is_array());
    assert_eq!(body["drivers"].as_array().unwrap().len(), 0);
    // No embedder in tests → template fallback.
    assert_eq!(body["method"], json!("template"));
}

#[tokio::test]
async fn log_rate_analysis_respects_query_params() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!(
            "{base}/v1/admin/log-rate-analysis?window_secs=60&baseline_secs=120&min_level=error"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["window_secs"], json!(60));
    assert_eq!(body["baseline_secs"], json!(120));
}

#[tokio::test]
async fn log_rate_analysis_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/log-rate-analysis"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/log-rate-analysis"))
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
