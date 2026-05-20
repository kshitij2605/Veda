//! Phase G atomic re-chunk smoke. POST /reindex creates name@v2
//! and points the alias at it. The full re-chunking pipeline
//! isn't part of this commit — the test only exercises the
//! atomic-swap primitive.

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
async fn reindex_in_non_raft_mode_returns_target_status_no_raft() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "docs",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    let resp = c
        .post(format!("{base}/v1/collections/docs/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "reindex: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("docs@v2"));
    // Non-Raft mode: status reflects the no-SM path.
    assert_eq!(body["status"], json!("no_raft"));
}

#[tokio::test]
async fn reindex_with_existing_v2_bumps_to_v3() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "logs@v2",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create v2: {}", resp.text().await.unwrap());

    let resp = c
        .post(format!("{base}/v1/collections/logs@v2/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("logs@v3"));
}

#[tokio::test]
async fn get_alias_returns_self_when_no_alias_set() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "plain",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .get(format!("{base}/v1/admin/collections/plain/alias"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["alias"], json!("plain"));
    assert_eq!(body["target"], json!("plain"));
}

#[tokio::test]
async fn reindex_unknown_collection_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections/ghost/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
