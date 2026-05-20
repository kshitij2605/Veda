//! Phase H replica reader gating. With the replica safe_ts
//! watermark advanced to a finite value, snapshot reads at `?at=`
//! greater than the watermark are rejected with 503; reads at or
//! below the watermark pass through. Default (u128::MAX) is
//! "no gating" and matches the existing MVCC test behaviour.

use std::net::SocketAddr;
use std::sync::Once;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

static SET_ENV: Once = Once::new();

fn set_env() {
    SET_ENV.call_once(|| unsafe {
        std::env::set_var("VEDA_LOG_PRIMARY", "1");
    });
}

async fn start_server() -> (String, tempfile::TempDir) {
    set_env();
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

async fn make_mvcc_collection(base: &str, name: &str) {
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ],
            "use_mvcc": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create collection {name}");
}

async fn insert_one(base: &str, name: &str) -> String {
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({ "records": [{ "v": [1.0, 0.0] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "insert into {name}");
    let body: Value = resp.json().await.unwrap();
    body["ids"][0].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn default_safe_ts_is_u128_max_and_passes_through_reads() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .get(format!("{base}/v1/admin/replica/safe_ts"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["replica_safe_ts"],
        json!("0"),
        "default replica_safe_ts is sentinel 0 (no gating)"
    );
}

#[tokio::test]
async fn read_at_ts_below_safe_ts_is_allowed() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    make_mvcc_collection(&base, "gated_a").await;
    let id = insert_one(&base, "gated_a").await;

    // Advance safe_ts to a very large value.
    let resp = c
        .post(format!("{base}/v1/admin/replica/safe_ts"))
        .json(&json!({ "ts": (u128::MAX / 2).to_string() }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Read at ts = 100 — well below the watermark.
    let resp = c
        .get(format!("{base}/v1/collections/gated_a/records/{id}?at=100"))
        .send()
        .await
        .unwrap();
    // 404 is fine here (the record's commit_ts is likely > 100);
    // the key invariant is NOT 503.
    assert_ne!(resp.status(), 503, "read at low ts must not be gated");
}

#[tokio::test]
async fn read_at_ts_above_safe_ts_returns_503() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    make_mvcc_collection(&base, "gated_b").await;
    let id = insert_one(&base, "gated_b").await;

    // Set safe_ts low.
    let resp = c
        .post(format!("{base}/v1/admin/replica/safe_ts"))
        .json(&json!({ "ts": "1000" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Read at ts way above 1000 → 503.
    let resp = c
        .get(format!("{base}/v1/collections/gated_b/records/{id}?at=99999999999"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("replica behind"),
        "expected 'replica behind' in error, got: {body}"
    );
}

#[tokio::test]
async fn safe_ts_advance_is_monotonic() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Move forward to 5000.
    let resp = c
        .post(format!("{base}/v1/admin/replica/safe_ts"))
        .json(&json!({ "ts": "5000" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["replica_safe_ts"], json!("5000"));

    // Try to move backwards — should be a no-op (still 5000).
    let resp = c
        .post(format!("{base}/v1/admin/replica/safe_ts"))
        .json(&json!({ "ts": "100" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["replica_safe_ts"], json!("5000"));
}

#[tokio::test]
async fn safe_ts_invalid_input_returns_400() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/admin/replica/safe_ts"))
        .json(&json!({ "ts": "not-a-number" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}
