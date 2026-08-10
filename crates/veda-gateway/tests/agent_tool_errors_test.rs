//! Regression test for the documented error contract of
//! `POST /v1/agent-tools/{id}/execute`: invalid args → 400,
//! not found → 404, internal failure → 500.
//!
//! An unknown tool id was dispatched through `ToolOutcome::error`, which is
//! an alias for `internal`, so naming a tool that does not exist answered
//! 500 — telling the caller the server broke rather than that they asked
//! for something that isn't there.

use reqwest::Client;
use serde_json::json;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

fn client() -> Client {
    Client::new()
}

#[tokio::test]
async fn executing_an_unknown_tool_is_404() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .post(format!("{base}/v1/agent-tools/no-such-tool/execute"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(status, 404, "unknown tool should be 404: {body}");
    assert!(
        body.contains("unknown tool"),
        "message is matched by the MCP layer and must be preserved: {body}"
    );
}

#[tokio::test]
async fn executing_a_known_tool_with_bad_args_is_400() {
    let (base, _tmp) = start_test_server().await;

    // `search` requires a collection; omitting it is a caller error.
    let resp = client()
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&json!({ "query": "anything" }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(status, 400, "missing required args should be 400: {body}");
}

#[tokio::test]
async fn executing_a_known_tool_against_a_missing_collection_is_404() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .post(format!("{base}/v1/agent-tools/describe_collection/execute"))
        .json(&json!({ "name": "no_such_collection" }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(
        status, 404,
        "a missing collection should be 404, not 500: {body}"
    );
}
