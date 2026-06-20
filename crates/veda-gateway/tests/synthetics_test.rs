//! Integration tests for the synthetic-probe REST API
//! (`/v1/admin/synthetics*`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server over a fresh temp dir.
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

fn health_probe_body(id: &str) -> Value {
    json!({
        "id": id,
        "name": "dependency health",
        "kind": { "kind": "health" },
        "interval_secs": 60,
        "timeout_ms": 2000
    })
}

#[tokio::test]
async fn create_list_get_results_delete_synthetic() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Create (201).
    let r = c
        .post(format!("{base}/v1/admin/synthetics"))
        .json(&health_probe_body("dep"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["id"], json!("dep"));
    assert_eq!(body["kind"]["kind"], json!("health"));

    // List shows it.
    let r = c
        .get(format!("{base}/v1/admin/synthetics"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let ids: Vec<String> = body["synthetics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(ids.contains(&"dep".to_owned()), "{ids:?}");

    // Get one (200).
    let r = c
        .get(format!("{base}/v1/admin/synthetics/dep"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // Results (200) — no runs yet, so the summary is empty-shaped.
    let r = c
        .get(format!("{base}/v1/admin/synthetics/dep/results"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["summary"]["probe_id"], json!("dep"));
    assert_eq!(body["summary"]["consecutive_failures"], json!(0));
    assert!(body["summary"]["last_ok"].is_null());
    assert_eq!(body["results"].as_array().unwrap().len(), 0);

    // Delete (204).
    let r = c
        .delete(format!("{base}/v1/admin/synthetics/dep"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204);

    // Now gone (404).
    let r = c
        .get(format!("{base}/v1/admin/synthetics/dep"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn create_assigns_id_when_absent() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let mut body = health_probe_body("");
    // Truly OMIT the id key (the real client case), not just send "".
    body.as_object_mut().unwrap().remove("id");
    let r = c
        .post(format!("{base}/v1/admin/synthetics"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    let id = body["id"].as_str().unwrap();
    assert!(
        id.starts_with("synthetic-"),
        "server should assign an id, got {id}"
    );
}

#[tokio::test]
async fn results_for_unknown_probe_is_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/synthetics/nope/results"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn synthetics_require_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer is forbidden.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/synthetics"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin is allowed.
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/synthetics"))
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
