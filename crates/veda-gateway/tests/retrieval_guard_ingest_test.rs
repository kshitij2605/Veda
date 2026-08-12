//! Ingest-time screening stamps the reserved guardrail fields.
//!
//! No embedder is registered in these tests, so the semantic layer fails
//! open and the verdict is `skipped` — which still must stamp `_indexed_at`
//! and must NOT stamp a false `_guardrail_flagged = false` (that would claim
//! the chunk was screened when it was not).

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

#[tokio::test]
async fn plain_insert_stamps_indexed_at() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ig_test",
            "fields": [ { "name": "content", "type": "string", "bm25": true } ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);

    let resp = client()
        .post(format!("{base}/v1/collections/ig_test/insert"))
        .json(&json!({ "records": [ { "content": "an ordinary sentence" } ] }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "insert: {:?}",
        resp.text().await
    );

    let resp = client()
        .get(format!("{base}/v1/collections/ig_test/records?limit=10"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let fields = &body["records"][0]["fields"];
    assert!(
        fields["_indexed_at"].as_i64().is_some(),
        "_indexed_at must be stamped on every insert: {body}"
    );
    assert!(
        fields.get("_guardrail_flagged").is_none(),
        "with no embedder the chunk is unscreened; it must NOT claim to be clean: {body}"
    );
}
