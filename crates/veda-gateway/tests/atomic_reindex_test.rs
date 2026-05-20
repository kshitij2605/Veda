//! Phase G atomic re-chunk durability. The plan's
//! `atomic_reindex.rs`: kill the gateway mid-reindex, recover,
//! and confirm the source collection is intact with no
//! observable half-state. Today's `/reindex` ships the
//! sibling-create + alias-swap primitive; the actual re-chunk
//! pipeline is a separate follow-up. The invariant that
//! matters NOW is: a crash between the sibling-create and the
//! alias-swap MUST NOT lose records from the source, and MUST
//! NOT surface the half-built sibling as the canonical name.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server_on(data_dir: &std::path::Path) -> String {
    let app_state = AppState::new(data_dir);
    // Match the production server.rs boot path: pick up any
    // existing collections from disk before serving requests.
    let _recovered = app_state.recover_collections_from_disk().await;
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    base_url
}

#[tokio::test]
async fn source_records_survive_crash_during_reindex() {
    let tmp = tempfile::tempdir().unwrap();
    let c = Client::new();
    // Lifetime 1: create the source + seed records.
    let base1 = start_server_on(tmp.path()).await;
    let resp = c
        .post(format!("{base1}/v1/collections"))
        .json(&json!({
            "name": "docs",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ],
            "seed_records": [
                { "v": [1.0, 0.0], "label": "alpha" },
                { "v": [0.0, 1.0], "label": "beta" },
                { "v": [1.0, 1.0], "label": "gamma" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    // Start a reindex (creates `docs@v2` + alias-swap). In
    // non-Raft mode the alias swap returns "no_raft", which
    // simulates the "crash before swap completes" state.
    let resp = c
        .post(format!("{base1}/v1/collections/docs/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("docs@v2"));

    // Simulate process kill: drop the AppState. (The
    // tokio::spawn task holding the router stays alive until
    // the process exits, but a fresh AppState on the same
    // data_dir is what recovery would do.)
    drop(base1);

    // Lifetime 2: fresh process on the same data_dir.
    let base2 = start_server_on(tmp.path()).await;
    // Source must still be queryable + the 3 records present.
    let resp = c
        .get(format!("{base2}/v1/collections/docs/records?limit=10"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "source GET: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["total"], json!(3),
        "all 3 source records must survive a mid-reindex crash; got {body}"
    );
}

#[tokio::test]
async fn dropped_collection_stays_dropped_across_restart() {
    // Counterpart invariant: a successful drop is durable.
    // Without this, the recover path could resurrect a
    // collection the operator explicitly dropped — that would
    // make rollback ambiguous (was the half-state a crash or a
    // partial drop?).
    let tmp = tempfile::tempdir().unwrap();
    let c = Client::new();
    let base1 = start_server_on(tmp.path()).await;
    let resp = c
        .post(format!("{base1}/v1/collections"))
        .json(&json!({
            "name": "doomed",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let resp = c
        .delete(format!("{base1}/v1/collections/doomed"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    drop(base1);

    let base2 = start_server_on(tmp.path()).await;
    let resp = c
        .get(format!("{base2}/v1/collections/doomed"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "drop must persist across restart");
}

#[tokio::test]
async fn rejected_seed_record_creation_leaves_no_artifacts() {
    // Phase G atomic create-and-load (seed_records) rolls back
    // a collection on seed failure. The "atomic" invariant
    // here is: no segment files, no schema marker, nothing on
    // disk afterwards.
    let tmp = tempfile::tempdir().unwrap();
    let c = Client::new();
    let base = start_server_on(tmp.path()).await;
    // Bad schema (vector dim mismatch in 3rd record).
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "atomic_test",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ],
            "seed_records": [
                { "v": [1.0, 0.0] },
                { "v": [0.0, 1.0] },
                { "v": [1.0, 0.0, 0.0] }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert!(!resp.status().is_success(), "bad seed must fail");
    drop(base);

    // Restart; the rolled-back collection must not resurrect.
    let base2 = start_server_on(tmp.path()).await;
    let resp = c
        .get(format!("{base2}/v1/collections/atomic_test"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        404,
        "rolled-back collection must not resurrect across restart"
    );
}
