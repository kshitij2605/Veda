//! Transaction REST surface tests for the production
//! TxnCoordinator integration (Phase G).

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::Value;
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

#[tokio::test]
async fn begin_returns_txn_id() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c.post(format!("{base}/v1/txn")).send().await.unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert!(body["txn_id"].as_str().is_some_and(|s| !s.is_empty()));
}

#[tokio::test]
async fn commit_unknown_txn_is_404() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/txn/no-such-id/commit"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn abort_unknown_txn_is_idempotent_ok() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/txn/no-such-id/abort"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("aborted"));
}

#[tokio::test]
async fn empty_txn_begin_then_commit_round_trips() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let begin: Value = c
        .post(format!("{base}/v1/txn"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let txn_id = begin["txn_id"].as_str().unwrap();
    let commit_resp = c
        .post(format!("{base}/v1/txn/{txn_id}/commit"))
        .send()
        .await
        .unwrap();
    assert_eq!(commit_resp.status(), 200);
    let body: Value = commit_resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("committed"));
}

#[tokio::test]
async fn begin_then_abort_round_trips() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let begin: Value = c
        .post(format!("{base}/v1/txn"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let txn_id = begin["txn_id"].as_str().unwrap();
    let abort_resp = c
        .post(format!("{base}/v1/txn/{txn_id}/abort"))
        .send()
        .await
        .unwrap();
    assert_eq!(abort_resp.status(), 200);
    // Subsequent commit on the same id is 404 — handle is gone.
    let commit_resp = c
        .post(format!("{base}/v1/txn/{txn_id}/commit"))
        .send()
        .await
        .unwrap();
    assert_eq!(commit_resp.status(), 404);
}
