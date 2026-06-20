//! Integration tests for the data-source connectors framework
//! (`/v1/admin/connectors*`).
//!
//! Covers:
//!   - CRUD round-trip: create without id → server-assigned `conn-…` id;
//!     list / get / update / delete.
//!   - Tenant isolation: tenant A's connectors are invisible to tenant B.
//!   - Admin-gated: a viewer is 403, an admin is 200.
//!   - `run` error paths: non-Web connector → 400, unknown id → 404, and
//!     (with `VEDA_CRAWL4AI_ENDPOINT` unset) a Web connector → 503.
//!   - Bonus: `run` against a mock crawl4ai actually crawls + inserts.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Mutex;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Serializes the env-var-driven crawl4ai tests so they don't race on the
/// process-global `VEDA_CRAWL4AI_ENDPOINT`.
static CRAWL_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Auth-disabled server over a fresh temp dir (mirrors the other admin
/// endpoint tests). Registers a mock text embedding model so the ingest
/// path has somewhere to embed into.
async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    {
        let mut registry = state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "mock-text".into(),
            endpoint: "http://mock:9999/embed".into(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 1024 * 1024,
            timeout_seconds: 10,
            batch_size: 32,
        });
    }
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

/// Auth-enabled server with an admin + a viewer user.
async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert("admin".to_owned(), UserRecord::with_password("admin", Role::Admin));
    users.insert("viewer".to_owned(), UserRecord::with_password("viewer", Role::Viewer));
    let state = AppState::with_auth_and_users(tmp.path(), "test-jwt-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
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

/// A mock crawl4ai server: `POST /md` returns the real per-URL shape
/// `{ "url", "markdown": "<string>", "success": true }` (same endpoint the
/// client uses).
async fn start_mock_crawl4ai() -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    use axum::Router;

    async fn md() -> axum::Json<Value> {
        axum::Json(json!({
            "url": "https://example.com/doc",
            "markdown": "# Veda\n\nVeda is a multimodal vector database. It supports many modalities.\n\n## Indexing\n\nVeda offers HNSW, IVF-PQ, and DiskANN indexes for approximate nearest neighbour search.",
            "title": "Veda Docs",
            "success": true
        }))
    }

    let router = Router::new().route("/md", post(md));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, handle)
}

async fn create_web_collection(base: &str, name: &str) {
    let resp = Client::new()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "content", "type": "string", "filterable": true },
                { "name": "source_url", "type": "string", "filterable": true },
                { "name": "title", "type": "string" },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create collection failed: {:?}", resp.text().await);
}

#[tokio::test]
async fn connector_crud_round_trip() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Create without an id → server assigns `conn-…`.
    let r = c
        .post(format!("{base}/v1/admin/connectors"))
        .json(&json!({
            "kind": "web",
            "name": "Docs crawler",
            "config": { "urls": ["https://example.com/doc"], "collection": "web_col" },
            "enabled": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create failed: {:?}", r.text().await);
    let created: Value = r.json().await.unwrap();
    let id = created["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("conn-"), "server should assign conn- id, got {id}");
    assert_eq!(created["kind"], "web");
    assert_eq!(created["name"], "Docs crawler");
    assert_eq!(created["enabled"], true);
    assert!(created["created_at"].as_i64().unwrap() > 0);

    // List → contains the new connector.
    let r = c.get(format!("{base}/v1/admin/connectors")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let list = body["connectors"].as_array().unwrap();
    assert_eq!(list.len(), 1, "expected one connector: {body}");
    assert_eq!(list[0]["id"], id);

    // Get by id.
    let r = c.get(format!("{base}/v1/admin/connectors/{id}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let got: Value = r.json().await.unwrap();
    assert_eq!(got["name"], "Docs crawler");

    // Update (PUT) — disable + rename.
    let r = c
        .put(format!("{base}/v1/admin/connectors/{id}"))
        .json(&json!({
            "kind": "web",
            "name": "Docs crawler v2",
            "config": { "urls": ["https://example.com/doc"], "collection": "web_col" },
            "enabled": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "put failed: {:?}", r.text().await);
    let updated: Value = r.json().await.unwrap();
    assert_eq!(updated["id"], id, "id must be stable across update");
    assert_eq!(updated["name"], "Docs crawler v2");
    assert_eq!(updated["enabled"], false);

    // Delete.
    let r = c.delete(format!("{base}/v1/admin/connectors/{id}")).send().await.unwrap();
    assert_eq!(r.status(), 204);

    // Gone → 404 on get.
    let r = c.get(format!("{base}/v1/admin/connectors/{id}")).send().await.unwrap();
    assert_eq!(r.status(), 404);

    // List empty again.
    let r = c.get(format!("{base}/v1/admin/connectors")).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    assert!(body["connectors"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn connector_tenant_isolation() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Tenant A creates a connector.
    let r = c
        .post(format!("{base}/v1/admin/connectors"))
        .header("X-Veda-Tenant", "tenant-a")
        .json(&json!({
            "kind": "websearch",
            "name": "A search",
            "config": {},
            "enabled": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_owned();

    // Tenant B does not see it in its list.
    let r = c
        .get(format!("{base}/v1/admin/connectors"))
        .header("X-Veda-Tenant", "tenant-b")
        .send()
        .await
        .unwrap();
    let body: Value = r.json().await.unwrap();
    assert!(
        body["connectors"].as_array().unwrap().is_empty(),
        "tenant B must not see tenant A's connectors: {body}"
    );

    // Tenant B cannot GET it by id either (404).
    let r = c
        .get(format!("{base}/v1/admin/connectors/{id}"))
        .header("X-Veda-Tenant", "tenant-b")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    // Tenant A still sees exactly one.
    let r = c
        .get(format!("{base}/v1/admin/connectors"))
        .header("X-Veda-Tenant", "tenant-a")
        .send()
        .await
        .unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["connectors"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn connectors_require_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/connectors"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/connectors"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}

#[tokio::test]
async fn run_non_web_connector_returns_400() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let r = c
        .post(format!("{base}/v1/admin/connectors"))
        .json(&json!({ "kind": "websearch", "name": "search marker", "config": {} }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_owned();

    let r = c
        .post(format!("{base}/v1/admin/connectors/{id}/run"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "run on a non-Web connector must be 400");
}

#[tokio::test]
async fn run_unknown_connector_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/admin/connectors/conn-nope/run"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn run_web_connector_unconfigured_returns_503() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    create_web_collection(&base, "web_col").await;

    let r = c
        .post(format!("{base}/v1/admin/connectors"))
        .json(&json!({
            "kind": "web",
            "name": "Docs crawler",
            "config": { "urls": ["https://example.com/doc"], "collection": "web_col" }
        }))
        .send()
        .await
        .unwrap();
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_owned();

    let _guard = CRAWL_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on CRAWL_ENV_LOCK.
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    let r = c
        .post(format!("{base}/v1/admin/connectors/{id}/run"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    drop(_guard);
    assert_eq!(status, 503, "expected 503 when crawl4ai unset: {body}");
}

#[tokio::test]
async fn run_web_connector_with_mock_crawls_and_inserts() {
    let (crawl_url, _crawl_handle) = start_mock_crawl4ai().await;
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    create_web_collection(&base, "web_col").await;

    let r = c
        .post(format!("{base}/v1/admin/connectors"))
        .json(&json!({
            "kind": "web",
            "name": "Docs crawler",
            "config": {
                "urls": ["https://example.com/doc"],
                "collection": "web_col",
                "chunker": "fixed_size",
                "chunk_size": 80,
                "chunk_overlap": 0
            }
        }))
        .send()
        .await
        .unwrap();
    let id = r.json::<Value>().await.unwrap()["id"].as_str().unwrap().to_owned();

    let _guard = CRAWL_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on CRAWL_ENV_LOCK.
    unsafe {
        std::env::set_var("VEDA_CRAWL4AI_ENDPOINT", &crawl_url);
    }
    let r = c
        .post(format!("{base}/v1/admin/connectors/{id}/run"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    drop(_guard);

    assert_eq!(status, 200, "run failed: {body}");
    assert_eq!(body["pages"], 1);
    assert!(body["chunks_inserted"].as_u64().unwrap() >= 1, "expected chunks: {body}");
    assert_eq!(body["per_url"][0]["title"], "Veda Docs");
}
