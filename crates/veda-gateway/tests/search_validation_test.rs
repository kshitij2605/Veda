//! Regression tests for dense-query validation on `POST .../search`.
//!
//! Found by an end-to-end pass against a live server:
//!   - a typo'd vector field name returned `200` with zero results, because
//!     `resolve_metric` fell back to a default metric for an unknown field
//!     and the scan matched nothing — indistinguishable from "no matches";
//!   - a wrong-length query vector reached the query executor and surfaced
//!     as `500 search failed: Dimension mismatch`, telling the caller to
//!     retry a request that can never succeed.
//!
//! Both are caller mistakes and must be `400`.

use reqwest::Client;
use serde_json::{Value, json};
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

async fn setup(base: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "sv_test",
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": 4,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);

    let resp = client()
        .post(format!("{base}/v1/collections/sv_test/insert"))
        .json(&json!({ "records": [
            { "content": "alpha", "vec": [1.0, 0.0, 0.0, 0.0] },
            { "content": "beta",  "vec": [0.0, 1.0, 0.0, 0.0] }
        ]}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "seed: {:?}", resp.text().await);
}

async fn search(base: &str, body: Value) -> (u16, String) {
    let resp = client()
        .post(format!("{base}/v1/collections/sv_test/search"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

#[tokio::test]
async fn unknown_vector_field_is_a_400_not_an_empty_200() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    let (status, body) = search(
        &base,
        json!({ "query": { "vector": { "field": "nope", "values": [1.0, 0.0, 0.0, 0.0] } },
                "limit": 3 }),
    )
    .await;
    assert_eq!(
        status, 400,
        "expected 400 for an unknown field, got: {body}"
    );
    assert!(body.contains("nope"), "error should name the field: {body}");
}

#[tokio::test]
async fn non_vector_field_is_a_400() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    let (status, body) = search(
        &base,
        json!({ "query": { "vector": { "field": "content", "values": [1.0, 0.0, 0.0, 0.0] } },
                "limit": 3 }),
    )
    .await;
    assert_eq!(status, 400, "expected 400 for a non-vector field: {body}");
}

#[tokio::test]
async fn dimension_mismatch_is_a_400_not_a_500() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    let (status, body) = search(
        &base,
        json!({ "query": { "vector": { "field": "vec", "values": [1.0, 0.0] } },
                "limit": 3 }),
    )
    .await;
    assert_eq!(status, 400, "expected 400 for a dimension mismatch: {body}");
    assert!(
        body.contains('4') && body.contains('2'),
        "error should report expected and actual lengths: {body}"
    );
}

#[tokio::test]
async fn hybrid_dense_leg_is_validated_too() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    let (status, body) = search(
        &base,
        json!({ "query": { "hybrid": {
                    "dense": { "field": "vec", "values": [1.0, 0.0] },
                    "bm25":  { "field": "content", "query": "alpha" }
                }},
                "limit": 3 }),
    )
    .await;
    assert_eq!(
        status, 400,
        "hybrid's dense leg must be validated like a plain dense query: {body}"
    );
}

#[tokio::test]
async fn a_correct_dense_query_still_succeeds() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    let (status, body) = search(
        &base,
        json!({ "query": { "vector": { "field": "vec", "values": [1.0, 0.0, 0.0, 0.0] } },
                "limit": 3 }),
    )
    .await;
    assert_eq!(status, 200, "valid query rejected: {body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 2, "{body}");
}

#[tokio::test]
async fn truncate_to_still_accepts_a_shorter_query_vector() {
    let (base, _tmp) = start_test_server().await;
    setup(&base).await;

    // MRL truncation scores on the first N components, so a shorter query
    // vector is legitimate here and must not be rejected by the new check.
    let (status, body) = search(
        &base,
        json!({ "query": { "vector": { "field": "vec", "values": [1.0, 0.0] } },
                "truncate_to": 2, "limit": 3 }),
    )
    .await;
    assert_eq!(status, 200, "truncate_to query rejected: {body}");
}
