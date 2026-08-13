//! `DELETE /v1/collections/{name}/records` must durably delete.
//!
//! `SegmentManager::delete_record` only marks a SEALED segment's delete
//! bitmap in memory. Without flushing that to `delete_bitmap.bin`, the
//! deletion is invisible to dense/ANN search (which reads a
//! `SegmentManager` rebuilt fresh from disk — see
//! `records::rebuild_executors`) the next time that rebuild runs, and to
//! every reader after a restart: a delete the caller was told succeeded,
//! and that a same-process `GET` even confirms, would silently
//! resurrect — including for a GDPR erasure request. This proves the
//! deletion survives exactly that: a rebuild from disk.

use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, records, state::AppState};
use veda_gateway::tenant::{self, TenantId};

fn client() -> Client {
    Client::new()
}

#[tokio::test]
async fn delete_of_a_sealed_record_survives_a_rebuild_from_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    // Keep a direct handle to the state so this test can trigger the same
    // `rebuild_executors` a restart (or any later mutation) would —
    // `build_router` below consumes `app_state` by value.
    let state_handle = app_state.clone();
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "del_persist_test",
            "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);

    let resp = client()
        .post(format!("{base}/v1/collections/del_persist_test/insert"))
        .json(&json!({ "records": [
            { "content": "keep me", "embedding": [1.0, 0.0, 0.0, 0.0] },
            { "content": "delete me", "embedding": [0.0, 1.0, 0.0, 0.0] }
        ] }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "insert: {:?}",
        resp.text().await
    );
    // Plain `/insert` seals its growing segment before returning (see
    // `insert_via_log`'s "seal growing segments so data is on disk"
    // step), so both records are now in a SEALED segment, not the growing
    // one — exactly the case that used to leak a stale copy back after a
    // rebuild.

    let resp = client()
        .get(format!(
            "{base}/v1/collections/del_persist_test/records?limit=10"
        ))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let records_json = body["records"].as_array().unwrap();
    assert_eq!(records_json.len(), 2, "{body}");
    let target_id = records_json
        .iter()
        .find(|r| r["fields"]["content"] == "delete me")
        .expect("seeded record must be present")["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let resp = client()
        .delete(format!("{base}/v1/collections/del_persist_test/records"))
        .json(&json!({ "ids": [target_id] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "delete: {:?}", resp.text().await);
    let del_body: Value = resp.json().await.unwrap();
    assert_eq!(del_body["deleted"], 1, "{del_body}");

    // Simulate what happens later — a restart, or any other mutation that
    // triggers `rebuild_executors` — by rebuilding the query executor
    // directly from disk. Before the fix, `delete_record` only marked the
    // sealed segment's bitmap in memory; this reload would resurrect the
    // "delete me" record because `delete_bitmap.bin` on disk never
    // learned about the delete.
    let scoped = tenant::scope(&TenantId::default_tenant(), "del_persist_test");
    records::rebuild_executors(&state_handle, &scoped)
        .await
        .unwrap();

    let resp = client()
        .post(format!("{base}/v1/collections/del_persist_test/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.0, 1.0, 0.0, 0.0] } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "search: {:?}", resp.text().await);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        1,
        "the deleted record must not survive a rebuild from disk: {body}"
    );
    assert_eq!(results[0]["fields"]["content"], "keep me", "{body}");
}
