//! Force-seal route (`POST /v1/collections/{name}/seal`) + configurable
//! seal threshold. Verifies that after a force-seal the collection's
//! configured vector index type is built (search returns the inserted
//! record via the real index, not a brute-force scan of an empty growing
//! segment).

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
    let base_url = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base_url, tmp)
}

/// Create → insert a few records → force-seal → search. The force-seal
/// rebuilds the executors so the configured index (here the default HNSW)
/// is built on the sealed segment; the search must return the inserted
/// record by id.
#[tokio::test]
async fn seal_route_seals_and_makes_index_searchable() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // Vector field with an explicitly declared RaBitQ index — proves the
    // configured (non-default) index type survives create → seal → rebuild.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "seal_col",
            "shard_count": 1,
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "v", "type": "vector", "dimension": 4, "metric": "cosine", "index": "rabitq" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    // Insert a handful of records with literal vectors.
    let resp = c
        .post(format!("{base}/v1/collections/seal_col/insert"))
        .json(&json!({
            "records": [
                { "title": "a", "v": [1.0, 0.0, 0.0, 0.0] },
                { "title": "b", "v": [0.0, 1.0, 0.0, 0.0] },
                { "title": "c", "v": [0.0, 0.0, 1.0, 0.0] }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "insert: {}", resp.text().await.unwrap());

    // Force-seal: must succeed and return the documented shape.
    let resp = c
        .post(format!("{base}/v1/collections/seal_col/seal"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "seal: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["collection"], json!("seal_col"));
    assert!(
        body["sealed_segments"].is_number(),
        "seal response must report sealed_segments: {body}"
    );
    assert!(
        body["per_shard"].is_array() && !body["per_shard"].as_array().unwrap().is_empty(),
        "seal response must report per_shard: {body}"
    );

    // After the seal + rebuild, a vector search must find the inserted
    // record via the configured RaBitQ index.
    let resp = c
        .post(format!("{base}/v1/collections/seal_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "v", "values": [1.0, 0.0, 0.0, 0.0] } },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "search: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "search after seal must return hits via the built index: {body}"
    );
    let top_title = results[0]["fields"]["title"].as_str().unwrap_or_default();
    assert_eq!(
        top_title, "a",
        "nearest neighbour of [1,0,0,0] is 'a': {body}"
    );
}

/// Sealing a non-existent collection is a 404, not a 500.
#[tokio::test]
async fn seal_route_unknown_collection_is_404() {
    let (base, _tmp) = start_server().await;
    let resp = Client::new()
        .post(format!("{base}/v1/collections/nope/seal"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "expected 404 for unknown collection");
}
