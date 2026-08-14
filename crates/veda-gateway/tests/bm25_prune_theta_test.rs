//! `prune_theta` on the BM25 leg of `/v1/collections/{name}/search`:
//! rejected below 1.0, accepted at or above it, and rank-safe by default.
//!
//! The collection is force-sealed before querying, because block metadata —
//! and therefore pruning — exists only on sealed segments. Pruning
//! *effectiveness* is measured in `veda-storage`; this file covers request
//! plumbing and validation at BOTH entry points into `execute_bm25`
//! (plain BM25 and the hybrid BM25 leg), since a guard applied to only one
//! leaves the other unprotected.
//!
//! Harness mirrors `bm25_filter_test.rs` — gateway integration tests are
//! kept self-contained per file by convention.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

// ── Harness ──────────────────────────────────────────────────────────────

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

fn client() -> Client {
    Client::new()
}

/// Create a collection with a bm25 `message` field, insert ~400 records with
/// varying term frequencies, and force-seal so block metadata (and therefore
/// pruning) is built.
async fn create_seed_and_seal(base: &str, name: &str) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "message", "type": "string", "bm25": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        201,
        "create collection: {}",
        r.text().await.unwrap()
    );

    let records: Vec<Value> = (0..400)
        .map(|i| {
            // Every record shares "needle"; term frequency varies so
            // block-max scores differ across the corpus, giving pruning
            // something to actually prune.
            let reps = (i % 5) + 1;
            let needle = "needle ".repeat(reps);
            json!({ "message": format!("{needle}filler word number {i}") })
        })
        .collect();
    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/seal"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "seal: {}", r.text().await.unwrap());
}

fn ids(body: &Value) -> Vec<String> {
    body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_str().unwrap().to_owned())
        .collect()
}

// ── 1. Omitted prune_theta matches explicit 1.0 ─────────────────────────────

#[tokio::test]
async fn omitted_prune_theta_matches_explicit_one() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_default").await;

    let r = client()
        .post(format!("{base}/v1/collections/prune_default/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "needle" } },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "no-theta search: {}",
        r.text().await.unwrap()
    );
    let no_theta: Value = r.json().await.unwrap();
    let no_theta_ids = ids(&no_theta);
    assert!(
        !no_theta_ids.is_empty(),
        "no-theta search must return hits; got {no_theta}"
    );

    let r = client()
        .post(format!("{base}/v1/collections/prune_default/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "needle", "prune_theta": 1.0 } },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "explicit theta=1.0 search: {}",
        r.text().await.unwrap()
    );
    let explicit_theta: Value = r.json().await.unwrap();
    let explicit_ids = ids(&explicit_theta);

    assert_eq!(
        no_theta_ids, explicit_ids,
        "omitted prune_theta must match explicit 1.0; no_theta={no_theta} explicit={explicit_theta}"
    );
}

// ── 2. prune_theta < 1.0 is rejected ────────────────────────────────────────

#[tokio::test]
async fn prune_theta_below_one_is_rejected() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_low").await;

    let r = client()
        .post(format!("{base}/v1/collections/prune_low/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "needle", "prune_theta": 0.5 } },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        400,
        "prune_theta=0.5 must be rejected; body: {}",
        r.text().await.unwrap()
    );
}

// ── 3. prune_theta > 1.0 is accepted ────────────────────────────────────────

#[tokio::test]
async fn prune_theta_above_one_is_accepted() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_high").await;

    let r = client()
        .post(format!("{base}/v1/collections/prune_high/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "needle", "prune_theta": 4.0 } },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "prune_theta=4.0 must be accepted; body: {}",
        r.text().await.unwrap()
    );
    let body: Value = r.json().await.unwrap();
    let hits = body["results"].as_array().unwrap();
    assert!(
        hits.len() <= 20,
        "must return at most limit hits; got {} : {body}",
        hits.len()
    );
}

// ── 4. Hybrid validates its BM25 leg too ────────────────────────────────────
//
// This is the test that catches a guard applied to only one of the two
// dispatch sites into `execute_bm25`.

#[tokio::test]
async fn hybrid_validates_its_bm25_leg_too() {
    let (base, _tmp) = start_server().await;
    let name = "prune_hybrid";

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "emb",     "type": "vector", "dimension": 3, "metric": "cosine" },
                { "name": "message", "type": "string", "bm25": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({
            "records": [
                { "emb": [1.0, 0.0, 0.0], "message": "alpha needle" },
                { "emb": [0.0, 1.0, 0.0], "message": "beta needle" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/seal"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "seal: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/search"))
        .json(&json!({
            "query": {
                "hybrid": {
                    "dense": { "field": "emb", "values": [1.0, 0.0, 0.0] },
                    "bm25":  { "field": "message", "query": "needle", "prune_theta": 0.5 }
                }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        400,
        "hybrid's bm25 leg prune_theta=0.5 must be rejected; body: {}",
        r.text().await.unwrap()
    );
}
