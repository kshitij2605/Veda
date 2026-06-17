//! Integration tests for the SLO REST API (`/v1/admin/slos*`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
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

fn latency_slo_body(id: &str) -> Value {
    json!({
        "id": id,
        "name": "API latency",
        "kind": { "kind": "latency", "threshold_ms": 200.0 },
        "target_pct": 99.0,
        "window_secs": 2592000
    })
}

#[tokio::test]
async fn create_list_get_status_delete_slo() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Create (201).
    let r = c
        .post(format!("{base}/v1/admin/slos"))
        .json(&latency_slo_body("api"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["id"], json!("api"));
    assert_eq!(body["kind"]["kind"], json!("latency"));

    // List shows it.
    let r = c.get(format!("{base}/v1/admin/slos")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let ids: Vec<String> = body["slos"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(ids.contains(&"api".to_owned()), "{ids:?}");

    // Get one (200).
    let r = c
        .get(format!("{base}/v1/admin/slos/api"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // Status (200) — no data yet, so sli_pct is null but the shape is present.
    let r = c
        .get(format!("{base}/v1/admin/slos/api/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["slo_id"], json!("api"));
    assert_eq!(body["target_pct"], json!(99.0));
    assert_eq!(body["breaching_fast"], json!(false));
    assert_eq!(body["budget_exhausted"], json!(false));
    assert_eq!(body["compliant"], json!(true)); // no data → compliant

    // Delete (204).
    let r = c
        .delete(format!("{base}/v1/admin/slos/api"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204);

    // Now gone (404).
    let r = c
        .get(format!("{base}/v1/admin/slos/api"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn create_assigns_id_when_absent() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let mut body = latency_slo_body("");
    // Truly OMIT the id key (the real client case), not just send "".
    body.as_object_mut().unwrap().remove("id");
    let r = c
        .post(format!("{base}/v1/admin/slos"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    let id = body["id"].as_str().unwrap();
    assert!(id.starts_with("slo-"), "server should assign an id, got {id}");
}

#[tokio::test]
async fn status_for_unknown_slo_is_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/slos/nope/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn slos_require_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer is forbidden.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/slos"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin is allowed.
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/slos"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
