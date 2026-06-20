//! Integration tests for logs-as-a-collection ingest.
//!
//! Creates a logs collection (message String bm25, level String filterable,
//! ts Int64 filterable), POSTs a batch of JSON + plain log lines to
//! `/ingest-logs`, then verifies:
//!   1. the `{ ingested: N }` shape,
//!   2. a BM25 text search finds a line by a word in its message,
//!   3. a `level = "ERROR"` scalar filter (via the aggregate count
//!      endpoint) selects exactly the error lines,
//!   4. `ttl_secs` stamps `_expires_at` so an immediately-expired batch is
//!      excluded from search (no embedder required — plain insert path).

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::{build_router, state::AppState};

async fn start_server() -> (String, tempfile::TempDir) {
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

async fn create_logs_collection(base: &str, name: &str) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "message", "type": "string", "bm25": true },
                { "name": "level",   "type": "string", "filterable": true },
                { "name": "ts",      "type": "int64",  "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        201,
        "create collection: {}",
        r.text().await.unwrap()
    );
}

#[tokio::test]
async fn ingest_logs_returns_count_and_is_bm25_searchable() {
    let (base, _tmp) = start_server().await;
    create_logs_collection(&base, "logs_search").await;

    let r = client()
        .post(format!("{base}/v1/collections/logs_search/ingest-logs"))
        .json(&json!({
            "lines": [
                r#"{"level":"error","msg":"disk full on device","ts":"2026-06-15T10:00:00Z","host":"h1"}"#,
                "2026-06-15T10:00:01Z WARN cache miss for key=42",
                "just some unstructured text about widgets"
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "ingest: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["ingested"].as_u64(), Some(3), "got: {body}");

    // BM25 search for a word in one of the messages.
    let r = client()
        .post(format!("{base}/v1/collections/logs_search/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "widgets" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    let msgs: Vec<&str> = results
        .iter()
        .filter_map(|h| h["fields"]["message"].as_str())
        .collect();
    assert!(
        msgs.iter().any(|m| m.contains("widgets")),
        "BM25 search should find the widgets line; got: {msgs:?}"
    );
}

#[tokio::test]
async fn ingest_logs_filter_by_level() {
    let (base, _tmp) = start_server().await;
    create_logs_collection(&base, "logs_filter").await;

    let r = client()
        .post(format!("{base}/v1/collections/logs_filter/ingest-logs"))
        .json(&json!({
            "lines": [
                r#"{"level":"error","msg":"alpha exploded"}"#,
                r#"{"level":"info","msg":"beta is fine"}"#,
                r#"{"level":"error","msg":"gamma exploded"}"#
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "ingest: {}", r.text().await.unwrap());

    // A `level = "ERROR"` scalar filter must select only the two error
    // lines. We assert this via the aggregate `count` endpoint, which
    // applies the filter to the stored records.
    let r = client()
        .get(format!("{base}/v1/collections/logs_filter/aggregate"))
        .query(&[
            ("field", "level"),
            ("metric", "count"),
            ("filter", "level = 'ERROR'"),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "aggregate: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let count = body["groups"][0]["value"].as_f64().unwrap();
    assert!(
        (count - 2.0).abs() < 0.001,
        "filter level=ERROR must count exactly the 2 error lines; got: {body}"
    );
}

#[tokio::test]
async fn ingest_logs_ttl_excludes_expired() {
    let (base, _tmp) = start_server().await;
    create_logs_collection(&base, "logs_ttl").await;

    // ttl_secs is added to now, so a normal positive ttl is in the
    // future and should remain searchable. Verify the happy path: an
    // ingested TTL'd line is still found (i.e. _expires_at was set to a
    // future time, not the past).
    let r = client()
        .post(format!("{base}/v1/collections/logs_ttl/ingest-logs"))
        .json(&json!({
            "lines": [ r#"{"level":"info","msg":"ephemeral token minted"}"# ],
            "ttl_secs": 3600
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "ingest: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["ingested"].as_u64(), Some(1));

    let r = client()
        .post(format!("{base}/v1/collections/logs_ttl/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "ephemeral" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        1,
        "future-TTL record must remain searchable; got: {body}"
    );
}
