//! Integration tests for TTL exclusion at query time (search / get / list).
//!
//! Expired records (with `_expires_at <= now_secs`) must NOT appear in:
//!   1. search results (vector search via populate_hit_fields)
//!   2. list records
//!   3. get record by id
//!
//! No TTL sweep is run — exclusion happens at query time.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
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

/// Unix epoch seconds far in the past so the record is definitely expired.
const EXPIRED: i64 = 1_000_000; // year 1970 + ~11 days — always in the past
/// Unix epoch seconds far in the future so the record is not expired.
const NOT_EXPIRED: i64 = 9_999_999_999; // year ~2286

// Helper: create the TTL test collection.
async fn create_ttl_collection(base: &str, name: &str) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "emb",         "type": "vector",  "dimension": 4, "metric": "cosine" },
                { "name": "label",       "type": "string" },
                { "name": "_expires_at", "type": "int64" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create collection: {}", r.text().await.unwrap());
}

// ── 1. Search excludes expired records ──────────────────────────────────────

#[tokio::test]
async fn ttl_search_excludes_expired() {
    let (base, _tmp) = start_server().await;
    create_ttl_collection(&base, "ttl_search1").await;

    // Insert expired record (expires_at in the past).
    let r = client()
        .post(format!("{base}/v1/collections/ttl_search1/insert"))
        .json(&json!({
            "records": [{
                "emb": [1.0, 0.0, 0.0, 0.0],
                "label": "expired",
                "_expires_at": EXPIRED
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert expired: {}", r.text().await.unwrap());

    // Insert non-expired record (expires_at in the future).
    let r = client()
        .post(format!("{base}/v1/collections/ttl_search1/insert"))
        .json(&json!({
            "records": [{
                "emb": [1.0, 0.1, 0.0, 0.0],
                "label": "valid",
                "_expires_at": NOT_EXPIRED
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert valid: {}", r.text().await.unwrap());

    // Insert record with no TTL (never expires).
    let r = client()
        .post(format!("{base}/v1/collections/ttl_search1/insert"))
        .json(&json!({
            "records": [{
                "emb": [0.9, 0.1, 0.0, 0.0],
                "label": "no-ttl"
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert no-ttl: {}", r.text().await.unwrap());

    // Search — query close to [1,0,0,0] should rank all 3 candidates.
    let r = client()
        .post(format!("{base}/v1/collections/ttl_search1/search"))
        .json(&json!({
            "query": { "vector": { "field": "emb", "values": [1.0, 0.0, 0.0, 0.0] } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "search: {}", r.text().await.unwrap());

    let body: Value = r.json().await.unwrap();
    let results = body["results"].as_array().unwrap();

    // Collect labels from returned hits.
    let labels: Vec<&str> = results
        .iter()
        .filter_map(|h| h["fields"]["label"].as_str())
        .collect();

    assert!(
        !labels.contains(&"expired"),
        "expired record must NOT appear in search results; got labels: {labels:?}"
    );
    assert!(
        labels.contains(&"valid"),
        "valid (future TTL) record MUST appear in search results; got labels: {labels:?}"
    );
    assert!(
        labels.contains(&"no-ttl"),
        "no-TTL record MUST appear in search results; got labels: {labels:?}"
    );
}

// ── 2. List excludes expired records ────────────────────────────────────────

#[tokio::test]
async fn ttl_list_excludes_expired() {
    let (base, _tmp) = start_server().await;
    create_ttl_collection(&base, "ttl_list1").await;

    // Insert expired.
    client()
        .post(format!("{base}/v1/collections/ttl_list1/insert"))
        .json(&json!({ "records": [{ "emb": [1.0,0.0,0.0,0.0], "label": "expired", "_expires_at": EXPIRED }] }))
        .send().await.unwrap();

    // Insert non-expired.
    client()
        .post(format!("{base}/v1/collections/ttl_list1/insert"))
        .json(&json!({ "records": [{ "emb": [0.0,1.0,0.0,0.0], "label": "valid", "_expires_at": NOT_EXPIRED }] }))
        .send().await.unwrap();

    let r = client()
        .get(format!("{base}/v1/collections/ttl_list1/records"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let records = body["records"].as_array().unwrap();
    let labels: Vec<&str> = records
        .iter()
        .filter_map(|h| h["fields"]["label"].as_str())
        .collect();

    assert!(
        !labels.contains(&"expired"),
        "expired record must NOT appear in list; got: {labels:?}"
    );
    assert!(
        labels.contains(&"valid"),
        "valid record MUST appear in list; got: {labels:?}"
    );
}

// ── 3. Get by id returns 404 for expired records ─────────────────────────────

#[tokio::test]
async fn ttl_get_excludes_expired() {
    let (base, _tmp) = start_server().await;
    create_ttl_collection(&base, "ttl_get1").await;

    // Insert expired.
    let r = client()
        .post(format!("{base}/v1/collections/ttl_get1/insert"))
        .json(&json!({ "records": [{ "emb": [1.0,0.0,0.0,0.0], "label": "expired", "_expires_at": EXPIRED }] }))
        .send().await.unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    // The non-log insert path always returns the IDs of inserted records.
    let expired_id: String = body["ids"][0].as_str()
        .expect("insert must return an id")
        .to_owned();

    // GET by ID for an expired record must return 404.
    let r = client()
        .get(format!("{base}/v1/collections/ttl_get1/records/{expired_id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        404,
        "expected 404 for expired record, got: {}",
        r.text().await.unwrap()
    );
}
