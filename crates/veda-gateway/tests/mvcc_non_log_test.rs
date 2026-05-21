//! Regression for BUG-1 (2026-05-21 E2E campaign): the non-log
//! `insert_impl` fall-through used `sm.insert(record)` which stamps
//! `commit_ts = None`, so `?at=<past_ts>` reads on a `use_mvcc:true`
//! collection saw every record regardless of timestamp. Fix: mirror
//! the `insert_via_log` pattern — when `sm.has_mvcc()`, allocate a
//! commit_ts and call `sm.insert_mvcc`.
//!
//! This test deliberately runs in its OWN integration-test binary
//! so `VEDA_LOG_PRIMARY` is NOT set (`mvcc_test.rs` sets it via
//! `Once`; integration tests get separate processes). That keeps
//! the write on the non-log fall-through path that BUG-1 was on.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
    // Belt-and-suspenders: explicitly remove the env var so this
    // test exercises the non-log fall-through path even if some
    // ambient setting bleeds in.
    unsafe {
        std::env::remove_var("VEDA_LOG_PRIMARY");
    }
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
async fn mvcc_search_at_zero_via_non_log_path_returns_zero_hits() {
    // Search analogue of the GET test: search at ts=0 must see
    // no hits on a use_mvcc collection populated via the non-log
    // path.
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "snap_search_nolog",
            "fields": [
                { "name": "vec", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ],
            "shard_count": 1,
            "use_mvcc": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{base}/v1/collections/snap_search_nolog/insert"))
        .json(&json!({
            "records": [
                { "vec": [1.0, 0.0], "label": "v1" },
                { "vec": [0.0, 1.0], "label": "v2" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Search at ts=0.
    let resp = c
        .post(format!("{base}/v1/collections/snap_search_nolog/search"))
        .json(&json!({
            "query": { "vector": { "field": "vec", "values": [1.0, 0.0] } },
            "limit": 5,
            "at": "0"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["count"], json!(0),
        "search at ts=0 must see no records; body={body}"
    );

    // Search at TSO.now() (default) must see both.
    let resp = c
        .post(format!("{base}/v1/collections/snap_search_nolog/search"))
        .json(&json!({
            "query": { "vector": { "field": "vec", "values": [1.0, 0.0] } },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"], json!(2), "default search should see both; body={body}");
}
