//! Phase G atomic create-and-load. POST /v1/collections with
//! `seed_records` bulk-inserts immediately after the collection
//! materialises. A seed failure rolls back the collection via the
//! standard drop path so create+seed is observable as
//! all-or-nothing.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
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

#[tokio::test]
async fn create_with_seed_records_returns_201_and_seeded_count() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let req = json!({
        "name": "seeded_docs",
        "shard_count": 1,
        "replication_factor": 1,
        "fields": [
            { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
            { "name": "label", "type": "string", "filterable": true }
        ],
        "seed_records": [
            { "v": [1.0, 0.0], "label": "alpha" },
            { "v": [0.0, 1.0], "label": "beta" },
            { "v": [1.0, 1.0], "label": "gamma" }
        ]
    });
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        201,
        "create+seed must succeed: {}",
        resp.text().await.unwrap()
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], json!("seeded_docs"));
    assert_eq!(body["status"], json!("created"));
    assert_eq!(body["seeded"], json!(3));

    // List records — all 3 must be there.
    let resp = c
        .get(format!("{base}/v1/collections/seeded_docs/records?limit=10"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total"], json!(3));
}

#[tokio::test]
async fn create_with_invalid_seed_rolls_back_the_collection() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // The third record's `v` field has wrong dimension — schema
    // expects dim=2 but we pass dim=3. The insert must fail and
    // roll back the collection.
    let req = json!({
        "name": "rollback_docs",
        "shard_count": 1,
        "replication_factor": 1,
        "fields": [
            { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
        ],
        "seed_records": [
            { "v": [1.0, 0.0] },
            { "v": [0.0, 1.0] },
            { "v": [1.0, 0.0, 0.0] }
        ]
    });
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&req)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body_text = resp.text().await.unwrap();
    assert!(
        !status.is_success(),
        "create+bad-seed must fail; got {status} body={body_text}"
    );
    assert!(
        body_text.contains("rolled back"),
        "error should mention rollback; got {body_text}"
    );

    // Re-creating the same name must work — the rollback dropped
    // the previous attempt cleanly.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "rollback_docs",
            "shard_count": 1,
            "replication_factor": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        201,
        "second create after rollback should succeed: {}",
        resp.text().await.unwrap()
    );
}

#[tokio::test]
async fn create_without_seed_records_omits_seeded_field() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "plain_docs",
            "shard_count": 1,
            "replication_factor": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], json!("plain_docs"));
    assert!(
        body.get("seeded").is_none() || body["seeded"].is_null(),
        "seeded should be absent when no seed_records were supplied; got {body}"
    );
}
