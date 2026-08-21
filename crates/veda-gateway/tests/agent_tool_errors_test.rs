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

/// `mode: "sparse"` used to be refused outright by `build_query_body` with
/// "not yet supported". It now runs, so the failure modes that remain must be
/// caller errors an agent can act on, not opaque 500s.
///
/// Both cases below are deterministic and need no encoder: the first
/// collection has no sparse field at all, and the test server is started
/// without `VEDA_SPLADE_ENDPOINT`, so the second exercises the
/// missing-encoder path.
#[tokio::test]
async fn sparse_mode_fails_with_an_actionable_message_not_a_500() {
    let (base, _tmp) = start_test_server().await;

    // A dense-only collection.
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "dense_only",
            "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": 8,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "setup failed: {}",
        resp.text().await.unwrap()
    );

    let resp = client()
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&json!({ "collection": "dense_only", "query": "valves", "mode": "sparse" }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(
        status, 400,
        "no sparse field should be a caller error: {body}"
    );
    assert!(
        body.contains("sparse_vector"),
        "the message must name what is missing: {body}"
    );

    // A collection that does declare a sparse field, with no encoder wired up.
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "has_sparse",
            "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "sv", "type": "sparse_vector", "dimension": 30522 },
                { "name": "vec", "type": "vector", "dimension": 8,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "setup failed: {}",
        resp.text().await.unwrap()
    );

    let resp = client()
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&json!({ "collection": "has_sparse", "query": "valves", "mode": "sparse" }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(
        status, 400,
        "a missing encoder is a configuration error: {body}"
    );
    assert!(
        body.contains("VEDA_SPLADE_ENDPOINT"),
        "the message must say how to fix it: {body}"
    );
}
