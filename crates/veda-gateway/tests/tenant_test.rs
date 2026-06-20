//! Tenant isolation property tests.
//!
//! Verifies the v1 invariant: requests carrying different
//! `X-Veda-Tenant` headers cannot see each other's collections or
//! records. The default tenant is "default" when the header is
//! absent.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base_url, tmp)
}

fn collection_payload(name: &str) -> Value {
    json!({
        "name": name,
        "fields": [
            { "name": "vec", "type": "vector", "dimension": 4, "metric": "cosine" },
            { "name": "label", "type": "string", "filterable": true },
        ],
        "shard_count": 1,
    })
}

#[tokio::test]
async fn two_tenants_with_same_collection_name_are_isolated() {
    let (base, _tmp) = start().await;
    let c = Client::new();

    // Tenant A creates "docs"
    let a = c
        .post(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "acme")
        .json(&collection_payload("docs"))
        .send()
        .await
        .unwrap();
    assert_eq!(a.status(), 201);

    // Tenant B creates "docs" — must succeed (different scoped key)
    let b = c
        .post(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "rival")
        .json(&collection_payload("docs"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        b.status(),
        201,
        "tenant B should be able to create its own 'docs'"
    );

    // A's list returns ["docs"]
    let body: Value = c
        .get(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "acme")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<String> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, vec!["docs"]);

    // B's list also returns ["docs"] — and only their own
    let body: Value = c
        .get(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "rival")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<String> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, vec!["docs"]);

    // Default tenant (no header) sees neither.
    let body: Value = c
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["collections"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn tenant_a_records_invisible_to_tenant_b() {
    let (base, _tmp) = start().await;
    let c = Client::new();

    // Both tenants create "docs"
    for tenant in ["acme", "rival"] {
        let resp = c
            .post(format!("{base}/v1/collections"))
            .header("x-veda-tenant", tenant)
            .json(&collection_payload("docs"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 201);
    }

    // Tenant A inserts a row
    let resp = c
        .post(format!("{base}/v1/collections/docs/insert"))
        .header("x-veda-tenant", "acme")
        .json(&json!({
            "records": [{ "vec": [1.0, 0.0, 0.0, 0.0], "label": "from-acme" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Tenant A's list shows 1 record
    let body: Value = c
        .get(format!("{base}/v1/collections/docs/records"))
        .header("x-veda-tenant", "acme")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["total"].as_u64().unwrap(), 1);

    // Tenant B's list shows 0 — same collection name, different scope
    let body: Value = c
        .get(format!("{base}/v1/collections/docs/records"))
        .header("x-veda-tenant", "rival")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["total"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn cross_tenant_describe_is_404() {
    let (base, _tmp) = start().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "acme")
        .json(&collection_payload("private_docs"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Rival tenant trying to describe acme's collection → 404
    let resp = c
        .get(format!("{base}/v1/collections/private_docs"))
        .header("x-veda-tenant", "rival")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn cross_tenant_drop_is_404() {
    let (base, _tmp) = start().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "acme")
        .json(&collection_payload("docs"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .delete(format!("{base}/v1/collections/docs"))
        .header("x-veda-tenant", "rival")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // acme's collection still exists
    let resp = c
        .get(format!("{base}/v1/collections/docs"))
        .header("x-veda-tenant", "acme")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn invalid_tenant_id_returns_400() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .get(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "has space")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn missing_tenant_header_uses_default() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&collection_payload("plain"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // The same request with X-Veda-Tenant: default sees it too
    let body: Value = c
        .get(format!("{base}/v1/collections"))
        .header("x-veda-tenant", "default")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<String> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, vec!["plain"]);
}
