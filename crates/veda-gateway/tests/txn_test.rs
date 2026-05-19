//! Transaction REST surface smoke tests.
//!
//! v0 requires Raft mode — `start_test_server` runs without Raft,
//! so we verify the endpoints return 503 with a clear error
//! message. Full lifecycle correctness lives in
//! `veda-coordinator::store::tests` (txn_*).

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
async fn begin_without_raft_returns_503() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c.post(format!("{base}/v1/txn")).send().await.unwrap();
    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("raft"),
        "expected error to mention raft requirement, got {body:?}"
    );
}

#[tokio::test]
async fn commit_without_raft_returns_503() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/txn/fake-id/commit"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn abort_without_raft_returns_503() {
    let (base, _tmp) = start().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/txn/fake-id/abort"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}
