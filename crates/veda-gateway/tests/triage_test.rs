//! Integration tests for AI-assisted ops triage
//! (`POST /v1/admin/insights/{event_id}/triage`).
//!
//! Deterministic HTTP paths (no live LLM): unknown event → 404, and the
//! admin gate (viewer → 403, admin passes the gate). The success path
//! (200 with explanation + steps, via a fake `ChatModel`) and the no-LLM
//! → 503 path are covered as in-crate unit tests in
//! `crate::rest::insights` and `crate::rest::agents` respectively, since
//! both require either injecting a fake model or seeding an open event
//! that has no public mutation endpoint.

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

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
async fn triage_unknown_event_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/admin/insights/bogus:global/triage"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404, "{:?}", r.text().await);
}

#[tokio::test]
async fn triage_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer is rejected by the require_admin route layer (403), before
    // any event lookup or LLM check.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .post(format!("{base}/v1/admin/insights/any:global/triage"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin passes the gate; with no such event it is 404 (not 403).
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .post(format!("{base}/v1/admin/insights/any:global/triage"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        404,
        "admin passes gate; missing event → 404: {:?}",
        r.text().await
    );
}
