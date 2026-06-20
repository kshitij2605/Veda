use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use veda_gateway::rest::{build_router, state::AppState};

/// Start a test server on a random port and return the base URL.
async fn start_test_server() -> (String, tempfile::TempDir) {
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

fn client() -> Client {
    Client::new()
}

// ── 1. Health check ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_health_check() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/admin/health"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["version"], "0.1.0");
}

// ── 2. Create collection ─────────────────────────────────────────────────

#[tokio::test]
async fn test_create_collection() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "test_col",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1,
            "replication_factor": 1
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "test_col");
    assert_eq!(body["status"], "created");
}

// ── 3. List collections ──────────────────────────────────────────────────

#[tokio::test]
async fn test_list_collections() {
    let (base, _tmp) = start_test_server().await;

    // Create a collection first
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "my_col",
            "fields": [
                { "name": "vec", "type": "vector", "dimension": 4, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();

    let resp = client()
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let collections = body["collections"].as_array().unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0]["name"], "my_col");
    assert_eq!(collections[0]["field_count"], 1);
}

// ── 4. Describe collection ───────────────────────────────────────────────

#[tokio::test]
async fn test_describe_collection() {
    let (base, _tmp) = start_test_server().await;

    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "desc_col",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "price", "type": "float64", "filterable": true },
                { "name": "emb", "type": "vector", "dimension": 8, "metric": "euclidean" }
            ],
            "shard_count": 2,
            "replication_factor": 1
        }))
        .send()
        .await
        .unwrap();

    let resp = client()
        .get(format!("{base}/v1/collections/desc_col"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "desc_col");
    assert_eq!(body["shard_count"], 2);
    assert_eq!(body["replication_factor"], 1);
    let fields = body["fields"].as_array().unwrap();
    assert_eq!(fields.len(), 3);
}

// ── 5. Insert records ────────────────────────────────────────────────────

#[tokio::test]
async fn test_insert_records() {
    let (base, _tmp) = start_test_server().await;

    // Create collection
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "insert_col",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert 10 records
    let mut records = Vec::new();
    for i in 0..10 {
        records.push(json!({
            "title": format!("Item {i}"),
            "embedding": [0.1 * i as f64, 0.2, 0.3, 0.4]
        }));
    }

    let resp = client()
        .post(format!("{base}/v1/collections/insert_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["inserted"], 10);
}

// ── 6. Search ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search() {
    let (base, _tmp) = start_test_server().await;

    // Create collection
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "search_col",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert records
    let mut records = Vec::new();
    for i in 0..5 {
        records.push(json!({
            "title": format!("Item {i}"),
            "embedding": [1.0, 0.0 + i as f64 * 0.1, 0.0, 0.0]
        }));
    }

    client()
        .post(format!("{base}/v1/collections/search_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    // Search
    let resp = client()
        .post(format!("{base}/v1/collections/search_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0] }
            },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);
    assert!(body["results"].as_array().unwrap().len() <= 3);
    assert!(body["took_ms"].as_u64().is_some());
}

// ── 7. Search with filter ────────────────────────────────────────────────

#[tokio::test]
async fn test_search_with_filter() {
    let (base, _tmp) = start_test_server().await;

    // Create collection
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "filter_col",
            "fields": [
                { "name": "category", "type": "string", "filterable": true },
                { "name": "price", "type": "float64", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert records
    let records = vec![
        json!({ "category": "electronics", "price": 99.99, "embedding": [1.0, 0.0, 0.0, 0.0] }),
        json!({ "category": "clothing", "price": 29.99, "embedding": [0.0, 1.0, 0.0, 0.0] }),
        json!({ "category": "electronics", "price": 499.99, "embedding": [0.5, 0.5, 0.0, 0.0] }),
    ];

    client()
        .post(format!("{base}/v1/collections/filter_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    // Search with filter
    let resp = client()
        .post(format!("{base}/v1/collections/filter_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0] }
            },
            "filter": "category = 'electronics'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    // Should return only electronics items (at most 2)
    let results = body["results"].as_array().unwrap();
    assert!(results.len() <= 2);
}

// ── 8. Delete records ────────────────────────────────────────────────────

#[tokio::test]
async fn test_delete_records() {
    let (base, _tmp) = start_test_server().await;

    // Create collection
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "delete_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert records
    client()
        .post(format!("{base}/v1/collections/delete_col/insert"))
        .json(&json!({
            "records": [
                { "embedding": [1.0, 0.0, 0.0, 0.0] },
                { "embedding": [0.0, 1.0, 0.0, 0.0] }
            ]
        }))
        .send()
        .await
        .unwrap();

    // Use a fake UUID for delete — we can't easily know the inserted IDs
    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client()
        .delete(format!("{base}/v1/collections/delete_col/records"))
        .json(&json!({ "ids": [fake_id] }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    // The fake ID likely won't match, so deleted should be 0
    assert_eq!(body["deleted"], 0);
}

// ── 9. Drop collection ───────────────────────────────────────────────────

#[tokio::test]
async fn test_drop_collection() {
    let (base, _tmp) = start_test_server().await;

    // Create collection
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "drop_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();

    // Drop it
    let resp = client()
        .delete(format!("{base}/v1/collections/drop_col"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "drop_col");
    assert_eq!(body["status"], "dropped");

    // Verify it's gone
    let resp = client()
        .get(format!("{base}/v1/collections/drop_col"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── 10. Not found ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_collection_not_found() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .get(format!("{base}/v1/collections/nonexistent"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("not found"));
}
