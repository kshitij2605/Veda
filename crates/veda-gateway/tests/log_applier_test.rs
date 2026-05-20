//! Phase C smoke: with `VEDA_LOG_APPLIER=1` (and the existing
//! `VEDA_LOG_PRIMARY=1`), the writer's `insert_via_log` skips
//! inline apply; visibility comes via the per-shard
//! `LogApplier` tick. The test sleeps briefly to let the
//! applier run, then confirms the record is queryable.

use std::net::SocketAddr;
use std::sync::Once;
use std::time::Duration;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

static SET_ENV: Once = Once::new();

fn set_env() {
    SET_ENV.call_once(|| unsafe {
        std::env::set_var("VEDA_LOG_PRIMARY", "1");
        std::env::set_var("VEDA_LOG_APPLIER", "1");
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

#[tokio::test]
async fn applier_enabled_picks_up_writes_after_a_short_delay() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "applier_docs",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    let resp = c
        .post(format!("{base}/v1/collections/applier_docs/insert"))
        .json(&json!({
            "records": [
                { "v": [1.0, 0.0], "label": "applied-by-applier" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "insert: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let id = body["ids"][0].as_str().unwrap().to_owned();

    // The applier ticks every ~20ms; give it a few ticks.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // The record is now visible — the applier picked it up.
    let resp = c
        .get(format!("{base}/v1/collections/applier_docs/records/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "applier should have made the record visible: {}",
        resp.text().await.unwrap()
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["fields"]["label"], json!("applied-by-applier"));
}
