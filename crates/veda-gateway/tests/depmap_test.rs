//! Integration tests for the dependency-map REST API
//! (`GET /v1/admin/depmap`).

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
async fn depmap_returns_expected_nodes_and_edges() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/depmap"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();

    let nodes = body["nodes"].as_array().expect("nodes array");
    let ids: Vec<&str> = nodes.iter().filter_map(|n| n["id"].as_str()).collect();
    for must in [
        "gateway",
        "query",
        "shards",
        "index",
        "embedding",
        "llm",
        "rerank",
        "blob",
        "nats",
    ] {
        assert!(ids.contains(&must), "node `{must}` present; got {ids:?}");
    }

    let edges = body["edges"].as_array().expect("edges array");
    // gateway → query (search) and shards → index (ann) must exist.
    let has = |from: &str, to: &str| {
        edges
            .iter()
            .any(|e| e["from"] == json!(from) && e["to"] == json!(to))
    };
    assert!(has("gateway", "query"), "gateway → query edge");
    assert!(has("shards", "index"), "shards → index edge");

    // Every edge endpoint references a declared node id.
    for e in edges {
        let from = e["from"].as_str().unwrap();
        let to = e["to"].as_str().unwrap();
        assert!(ids.contains(&from), "edge.from `{from}` is a node");
        assert!(ids.contains(&to), "edge.to `{to}` is a node");
    }
}

#[tokio::test]
async fn depmap_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/depmap"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/depmap"))
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
