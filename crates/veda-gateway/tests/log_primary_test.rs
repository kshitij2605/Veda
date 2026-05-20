//! Phase C smoke test: with `VEDA_LOG_PRIMARY=1`, record writes
//! go through veda-log instead of the cluster-wide Raft and are
//! immediately readable.

use std::net::SocketAddr;
use std::sync::Once;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

static SET_PRIMARY: Once = Once::new();

fn set_log_primary_env() {
    // Tests share process state; only need to set the env once.
    SET_PRIMARY.call_once(|| unsafe {
        std::env::set_var("VEDA_LOG_PRIMARY", "1");
    });
}

async fn start_log_primary_server() -> (String, tempfile::TempDir) {
    set_log_primary_env();
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

fn schema() -> Value {
    json!({
        "name": "primary_docs",
        "fields": [
            { "name": "vec", "type": "vector", "dimension": 4, "metric": "cosine" },
            { "name": "label", "type": "string", "filterable": true }
        ],
        "shard_count": 1
    })
}

#[tokio::test]
async fn log_primary_insert_then_read_finds_record() {
    let (base, tmp) = start_log_primary_server().await;
    let c = Client::new();

    // Create collection.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&schema())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Insert two records.
    let resp = c
        .post(format!("{base}/v1/collections/primary_docs/insert"))
        .json(&json!({
            "records": [
                { "vec": [1.0, 0.0, 0.0, 0.0], "label": "first" },
                { "vec": [0.0, 1.0, 0.0, 0.0], "label": "second" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "insert via log primary should succeed");
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["inserted"], 2);
    let ids = body["ids"].as_array().unwrap();
    assert_eq!(ids.len(), 2);

    // List records — should immediately see both.
    let body: Value = c
        .get(format!("{base}/v1/collections/primary_docs/records"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["total"].as_u64().unwrap(), 2);

    // Confirm the underlying shard log file exists on disk —
    // proof the durability hop fired.
    let log_dir = tmp
        .path()
        .join("_logs")
        .join("default__primary_docs")
        .join("0000");
    let entries = std::fs::read_dir(&log_dir).unwrap();
    let log_files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "log"))
        .collect();
    assert!(!log_files.is_empty(), "expected at least one .log file in {log_dir:?}");
}

#[tokio::test]
async fn log_primary_txn_write_for_unknown_txn_is_404() {
    // Prod-5 / Phase G: writes carrying `X-Veda-Txn-Id` route
    // through the TxnCoordinator instead of the v0 503 stub.
    // An unknown txn-id surfaces as 404 (TxnError::NotFound).
    let (base, _tmp) = start_log_primary_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&schema())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{base}/v1/collections/primary_docs/insert"))
        .header("x-veda-txn-id", "made-up-txn")
        .json(&json!({
            "records": [{ "vec": [1.0, 0.0, 0.0, 0.0], "label": "x" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
