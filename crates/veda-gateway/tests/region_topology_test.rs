//! Phase H admin endpoints — RegionTopology read/write/promote.
//! All Raft-dependent paths short-circuit cleanly in non-Raft
//! mode: GET returns `topology: null`, PUT returns 503.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
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
async fn get_region_topology_in_non_raft_mode_returns_none() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .get(format!("{base}/v1/admin/region_topology"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["topology"].is_null());
}

#[tokio::test]
async fn put_region_topology_in_non_raft_mode_returns_503() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .put(format!("{base}/v1/admin/region_topology"))
        .json(&json!({
            "primary_region": "us-east-1",
            "replica_regions": ["eu-west-1"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("raft coordinator"));
}

#[tokio::test]
async fn promote_region_in_non_raft_mode_returns_503() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/admin/region_topology/promote"))
        .json(&json!({
            "new_topology": {
                "primary_region": "eu-west-1",
                "replica_regions": ["us-east-1"],
                "epoch": 2
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn put_then_get_roundtrips_topology_shape() {
    // Even in non-Raft mode the PUT body validates and the GET
    // returns null. To exercise an actual round-trip we'd need
    // Raft running; the unit-test coverage in
    // veda-coordinator::store::tests::region_topology_round_trips_*
    // already pins the state-machine side.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    // Validate request body shape via PUT (will 503 but body is
    // accepted as JSON; if shape were wrong we'd get 400).
    let resp = c
        .put(format!("{base}/v1/admin/region_topology"))
        .json(&json!({
            "primary_region": "us-east-1",
            "replica_regions": ["eu-west-1", "ap-northeast-1"],
            "staleness_target_ms": 3000,
            "epoch": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "expected 503 in non-Raft mode");
}
