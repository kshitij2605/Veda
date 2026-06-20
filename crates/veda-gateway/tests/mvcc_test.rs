//! Phase F live-read smoke: with `use_mvcc: true` (and
//! `use_log_primary: true` so writes flow through the log+MVCC
//! path), GET /records/:id?at=<ts> returns the version visible
//! at that snapshot timestamp. With no `?at=`, the read uses
//! TSO.now() and returns the latest version.

use std::net::SocketAddr;
use std::sync::Once;

use reqwest::Client;
use serde_json::{Value, json};
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

fn mvcc_schema() -> Value {
    json!({
        "name": "mvcc_docs",
        "fields": [
            { "name": "vec", "type": "vector", "dimension": 2, "metric": "cosine" },
            { "name": "label", "type": "string", "filterable": true }
        ],
        "shard_count": 1,
        "use_mvcc": true
    })
}

#[tokio::test]
async fn mvcc_get_with_at_far_future_returns_latest() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&mvcc_schema())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Insert one record under MVCC.
    let resp = c
        .post(format!("{base}/v1/collections/mvcc_docs/insert"))
        .json(&json!({
            "records": [
                { "vec": [1.0, 0.0], "label": "v1" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        201,
        "insert via mvcc should succeed: {:?}",
        resp.text().await
    );
    let body: Value = resp.json().await.unwrap();
    let id = body["ids"][0].as_str().unwrap().to_owned();

    // Read at far-future ts — must see the row.
    let resp = c
        .get(format!("{base}/v1/collections/mvcc_docs/records/{id}?at=340282366920938463463374607431768211455"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "far-future read should find: {:?}",
        resp.text().await
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["fields"]["label"], json!("v1"));
}

#[tokio::test]
async fn mvcc_get_at_zero_does_not_see_later_writes() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "snap_docs",
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
        .post(format!("{base}/v1/collections/snap_docs/insert"))
        .json(&json!({
            "records": [
                { "vec": [1.0, 0.0], "label": "v1" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let id = body["ids"][0].as_str().unwrap().to_owned();

    // Snapshot read at ts=0 must NOT see the row (its commit_ts > 0).
    let resp = c
        .get(format!("{base}/v1/collections/snap_docs/records/{id}?at=0"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        404,
        "read at ts=0 must not see later writes; body={:?}",
        resp.text().await
    );
}

#[tokio::test]
async fn mvcc_off_collection_uses_legacy_get_path() {
    // Same gateway, but the collection opts out (use_mvcc: false
    // wins over VEDA_MVCC env). The legacy segment-scan path
    // must serve the GET.
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "legacy_docs",
            "fields": [
                { "name": "vec", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ],
            "shard_count": 1,
            "use_mvcc": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{base}/v1/collections/legacy_docs/insert"))
        .json(&json!({
            "records": [
                { "vec": [1.0, 0.0], "label": "v1" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    let id = body["ids"][0].as_str().unwrap().to_owned();

    // Without ?at=, legacy collections use the segment scan and
    // `at` is ignored. With ?at=0 — also ignored on a legacy
    // collection (would otherwise wrongly 404).
    let resp = c
        .get(format!(
            "{base}/v1/collections/legacy_docs/records/{id}?at=0"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "legacy read must ignore ?at=: {:?}",
        resp.text().await
    );
}
