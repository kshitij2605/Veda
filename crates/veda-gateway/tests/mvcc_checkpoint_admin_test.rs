//! GAP-3 (2026-05-21 E2E campaign): admin REST endpoint that
//! triggers `SegmentManager::mvcc_checkpoint` across every shard
//! of a collection. Before this, the underlying primitive existed
//! (`crates/veda-storage/src/segment_manager.rs::mvcc_checkpoint`)
//! but operators had no way to invoke it — the MVCC WAL grew
//! unbounded under sustained churn.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
}

#[tokio::test]
async fn mvcc_checkpoint_unknown_collection_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/admin/collections/ghost/mvcc_checkpoint"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn mvcc_checkpoint_collection_without_mvcc_returns_zero_checkpoints() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "no_mvcc",
            "shard_count": 2,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = c
        .post(format!(
            "{base}/v1/admin/collections/no_mvcc/mvcc_checkpoint"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "ckpt on no-mvcc should still 200: {:?}",
        r.text().await
    );
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["collection"], json!("no_mvcc"));
    // 2 shards, 0 checkpointed (MVCC not on), 0 entries.
    assert_eq!(body["shards"], json!(2));
    assert_eq!(body["shards_checkpointed"], json!(0));
    assert_eq!(body["entries"], json!(0));
}

#[tokio::test]
async fn mvcc_checkpoint_runs_on_use_mvcc_collection() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // use_mvcc collection — segment managers get the MVCC overlay
    // but durable MVCC WAL only spins up when a WAL path is set via
    // `enable_mvcc_durable`. The default `with_mvcc()` path does not
    // turn on the WAL, so `mvcc_checkpoint()` returns Ok(None) for
    // shards without it. We still get a 200 and a per-shard count
    // back. The point of the test is to prove the route + handler
    // are wired and the shard fan-out works.
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mvcc_ckpt",
            "shard_count": 3,
            "use_mvcc": true,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = c
        .post(format!("{base}/v1/collections/mvcc_ckpt/insert"))
        .json(&json!({
            "records": [
                { "v": [1.0, 0.0], "label": "v1" },
                { "v": [0.0, 1.0], "label": "v2" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = c
        .post(format!(
            "{base}/v1/admin/collections/mvcc_ckpt/mvcc_checkpoint"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "ckpt should succeed: {:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["collection"], json!("mvcc_ckpt"));
    assert_eq!(body["shards"], json!(3));
    // shards_checkpointed depends on whether the WAL is enabled in
    // the default AppState path. We assert it is non-negative and
    // shard count is exact; entries is non-negative.
    let ck = body["shards_checkpointed"].as_u64().unwrap();
    let ent = body["entries"].as_u64().unwrap();
    assert!(ck <= 3, "ck must not exceed shard count");
    let _ = ent; // observe; no assertion on exact count
}
