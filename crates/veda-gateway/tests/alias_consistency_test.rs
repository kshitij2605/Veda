//! Regression tests for collection-alias resolution across the whole API.
//!
//! An alias points a collection name at a swapped-in sibling — that is what
//! `/reindex` and tuning-apply produce — so EVERY operation addressed to the
//! name must act on the alias target. `/search`, `/records` and `/insert`
//! always resolved; a live end-to-end run found that analytics, evaluate,
//! rerank, seal/optimize/describe, eval-set generation, and the
//! `embed-insert` / `ingest-logs` / `ingest-media` write paths did not. The
//! write paths were the damaging half: they returned `201 {"inserted":N}`
//! while the record landed in the abandoned base collection, so it was never
//! queryable through the name that was written to.
//!
//! Each test points an alias at a target collection and addresses the
//! endpoint by a name that has NO physical collection of its own. A handler
//! that resolves works; one that doesn't returns 404. That makes the
//! assertions deterministic and free of any external embedding/rerank service.

use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

async fn start_test_server() -> (String, tempfile::TempDir) {
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

/// Create `target` holding three records, then alias `ghost` -> `target`.
/// `ghost` itself is never created, so any handler that skips alias
/// resolution can only 404.
async fn setup_alias(base: &str, ghost: &str, target: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": target,
            "fields": [
                { "name": "content",  "type": "string",  "bm25": true },
                { "name": "category", "type": "string",  "filterable": true },
                { "name": "price",    "type": "float64", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create target: {:?}", resp.text().await);

    let resp = client()
        .post(format!("{base}/v1/collections/{target}/insert"))
        .json(&json!({ "records": [
            { "content": "first record",  "category": "v2", "price": 1.0 },
            { "content": "second record", "category": "v2", "price": 2.0 },
            { "content": "third record",  "category": "v2", "price": 3.0 }
        ]}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "seed: {:?}", resp.text().await);

    let resp = client()
        .put(format!("{base}/v1/admin/collections/{ghost}/alias"))
        .json(&json!({ "target": target }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "set alias: {:?}", resp.text().await);
}

async fn get_json(base: &str, path: &str) -> (u16, Value) {
    let resp = client().get(format!("{base}{path}")).send().await.unwrap();
    let status = resp.status().as_u16();
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    (status, body)
}

#[tokio::test]
async fn analytics_endpoints_follow_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "an_ghost", "an_target").await;

    let (status, body) = get_json(
        &base,
        "/v1/collections/an_ghost/aggregate?field=price&metric=count",
    )
    .await;
    assert_eq!(status, 200, "aggregate did not follow the alias: {body}");
    assert_eq!(
        body["groups"][0]["value"].as_f64().unwrap(),
        3.0,
        "aggregate counted the wrong collection: {body}"
    );

    let (status, body) = get_json(
        &base,
        "/v1/collections/an_ghost/histogram?field=price&bins=3",
    )
    .await;
    assert_eq!(status, 200, "histogram did not follow the alias: {body}");
    let binned: i64 = body["bins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["count"].as_i64().unwrap_or(0))
        .sum();
    assert_eq!(binned, 3, "histogram binned the wrong collection: {body}");

    let (status, body) =
        get_json(&base, "/v1/collections/an_ghost/field-stats?field=category").await;
    assert_eq!(status, 200, "field-stats did not follow the alias: {body}");
    assert_eq!(
        body["total_count"].as_i64().unwrap(),
        3,
        "field-stats read the wrong collection: {body}"
    );
}

#[tokio::test]
async fn evaluate_follows_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "ev_ghost", "ev_target").await;

    let (status, body) = get_json(&base, "/v1/collections/ev_ghost/evaluate?sample=200").await;
    assert_eq!(status, 200, "evaluate did not follow the alias: {body}");
    assert_eq!(
        body["records_evaluated"].as_i64().unwrap(),
        3,
        "evaluate reported on the wrong collection: {body}"
    );
}

#[tokio::test]
async fn describe_seal_and_optimize_follow_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "so_ghost", "so_target").await;

    let (status, body) = get_json(&base, "/v1/collections/so_ghost").await;
    assert_eq!(status, 200, "describe did not follow the alias: {body}");
    assert_eq!(
        body["fields"].as_array().unwrap().len(),
        3,
        "describe returned the wrong schema: {body}"
    );

    for op in ["seal", "optimize"] {
        let resp = client()
            .post(format!("{base}/v1/collections/so_ghost/{op}"))
            .json(&json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "{op} did not follow the alias: {:?}",
            resp.text().await
        );
    }
}

#[tokio::test]
async fn eval_set_size_recommendation_follows_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "sr_ghost", "sr_target").await;

    let (status, body) = get_json(
        &base,
        "/v1/collections/sr_ghost/eval-set-size-recommendation",
    )
    .await;
    assert_eq!(
        status, 200,
        "size recommendation did not follow the alias: {body}"
    );
    assert_eq!(
        body["chunk_count"].as_i64().unwrap(),
        3,
        "size recommendation counted the wrong collection: {body}"
    );
}

#[tokio::test]
async fn embed_insert_follows_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "ei_ghost", "ei_target").await;

    // The target has no vector field, so a resolved request gets as far as
    // the auto_embed check and fails with 400. An unresolved one cannot get
    // that far — it 404s on the collection lookup, which is the bug.
    let resp = client()
        .post(format!("{base}/v1/collections/ei_ghost/embed-insert"))
        .json(&json!({ "records": [ { "content": "written via the alias" } ] }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_ne!(status, 404, "embed-insert did not follow the alias: {body}");
    assert!(
        body.contains("auto_embed"),
        "expected embed-insert to reach the auto_embed check, got {status}: {body}"
    );
}

#[tokio::test]
async fn rerank_candidates_follow_an_alias() {
    let (base, _tmp) = start_test_server().await;
    setup_alias(&base, "rr_ghost", "rr_target").await;

    let (_, records) = get_json(&base, "/v1/collections/rr_target/records?limit=1").await;
    let id = records["records"][0]["id"].as_str().unwrap().to_owned();

    // Ids handed out by search/records live in the alias TARGET. Before the
    // fix the candidate lookup ran against the unresolved name and 404'd.
    let resp = client()
        .post(format!("{base}/v1/collections/rr_ghost/rerank"))
        .json(&json!({
            "query": "first",
            "candidates": [id],
            "field": "content",
            "top_k": 1
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_ne!(status, 404, "rerank did not follow the alias: {body}");
}

#[tokio::test]
async fn ingest_logs_follows_an_alias() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "lg_target",
            "fields": [
                { "name": "message",   "type": "string",    "bm25": true },
                { "name": "level",     "type": "string",    "filterable": true },
                { "name": "timestamp", "type": "timestamp", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);

    let resp = client()
        .put(format!("{base}/v1/admin/collections/lg_ghost/alias"))
        .json(&json!({ "target": "lg_target" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .post(format!("{base}/v1/collections/lg_ghost/ingest-logs"))
        .json(&json!({
            "lines": ["2026-08-10T00:00:00Z ERROR written via the alias"],
            "default_level": "error"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "ingest-logs did not follow the alias: {:?}",
        resp.text().await
    );

    // The write must be readable back through the same name it was written
    // to — that is exactly what was broken.
    let (status, body) = get_json(&base, "/v1/collections/lg_ghost/records?limit=10").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["total"].as_i64().unwrap(),
        1,
        "log line was written somewhere the alias name cannot read: {body}"
    );
}
