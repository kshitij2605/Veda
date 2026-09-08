//! `mode: "hybrid"` on the agent `search` tool: dense + BM25, fused.
//!
//! The dense leg needs an embedding backend the in-crate harness cannot
//! supply, so these tests cover the contract and the routing — the error
//! surface an agent can act on, and that the request reaches the shared REST
//! fusion path rather than a second implementation.
//!
//! Own file because it sets `EMBEDDING_ENDPOINT` to a stub listener, and
//! integration tests share one process per file.

use axum::{Json, Router, extract::State, routing::post};
use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

const DIM: usize = 8;

#[derive(Clone)]
struct Stub {
    calls: Arc<AtomicUsize>,
}

/// OpenAI-shaped `/v1/embeddings`. Returns a fixed unit vector: the dense
/// leg only has to run, not to rank meaningfully — BM25 supplies the signal
/// these tests assert on.
async fn embeddings(State(stub): State<Stub>, Json(_body): Json<Value>) -> Json<Value> {
    stub.calls.fetch_add(1, Ordering::SeqCst);
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    Json(
        json!({ "data": [ { "embedding": v, "index": 0, "object": "embedding" } ],
                 "model": "stub", "object": "list" }),
    )
}

async fn start_embed_stub() -> (String, Stub) {
    let stub = Stub {
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .with_state(stub.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), stub)
}

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    state.load_embedding_from_env().await;
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), tmp)
}

async fn create(c: &Client, base: &str, body: Value) {
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert!(
        r.status().is_success(),
        "setup failed: {}",
        r.text().await.unwrap()
    );
}

async fn hybrid(c: &Client, base: &str, mut args: Value) -> (u16, String) {
    args["mode"] = json!("hybrid");
    let r = c
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&args)
        .send()
        .await
        .unwrap();
    let s = r.status().as_u16();
    (s, r.text().await.unwrap())
}

#[tokio::test]
async fn hybrid_mode_contract() {
    let (embed_url, stub) = start_embed_stub().await;
    // SAFETY: single test in its own test binary.
    unsafe {
        std::env::set_var("EMBEDDING_ENDPOINT", &embed_url);
        std::env::set_var("EMBEDDING_MODEL", "stub");
        std::env::set_var("EMBEDDING_DIMENSION", DIM.to_string());
        std::env::set_var("EMBEDDING_PRESET", "");
    }
    let (base, _tmp) = start_test_server().await;
    let c = Client::new();

    // A collection with a dense field but NO bm25 field: hybrid has no
    // lexical leg, and that must be a caller error naming the fix.
    create(
        &c,
        &base,
        json!({
            "name": "no_bm25", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string" },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;
    let (status, body) = hybrid(&c, &base, json!({"collection":"no_bm25","query":"valves"})).await;
    assert_eq!(status, 400, "missing bm25 field should be 400: {body}");
    assert!(
        body.contains("bm25"),
        "the message must name what is missing: {body}"
    );

    // A well-formed hybrid collection.
    create(
        &c,
        &base,
        json!({
            "name": "kb", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;

    // An image query has no lexical half — refused, not silently degraded to
    // dense. Silent degradation is the failure mode worth guarding: the model
    // would believe it ran a hybrid search.
    let (status, body) = hybrid(
        &c,
        &base,
        json!({"collection":"kb","query_image_b64": "aGVsbG8="}),
    )
    .await;
    assert_eq!(status, 400, "image + hybrid should be 400: {body}");
    assert!(body.contains("lexical"), "{body}");

    // Knob validation: out-of-range weight and unknown fusion are caller
    // errors, not silently clamped or defaulted.
    let (status, body) = hybrid(
        &c,
        &base,
        json!({"collection":"kb","query":"v","dense_weight": 1.5}),
    )
    .await;
    assert_eq!(status, 400, "dense_weight 1.5 should be 400: {body}");
    assert!(body.contains("dense_weight"), "{body}");

    let (status, body) = hybrid(
        &c,
        &base,
        json!({"collection":"kb","query":"v","fusion":"magic"}),
    )
    .await;
    assert_eq!(status, 400, "unknown fusion should be 400: {body}");
    assert!(body.contains("magic"), "{body}");

    // A bm25_field that exists but is not bm25-enabled is also a caller error.
    create(
        &c,
        &base,
        json!({
            "name": "kb2", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "plain", "type": "string" },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;
    let (status, body) = hybrid(
        &c,
        &base,
        json!({"collection":"kb2","query":"v","bm25_field":"plain"}),
    )
    .await;
    assert_eq!(status, 400, "non-bm25 field should be 400: {body}");
    assert!(body.contains("plain"), "{body}");

    // ── Happy path ──────────────────────────────────────────────────────
    // Folded into this test rather than a second one: the file sets
    // process-global env vars, and two `#[tokio::test]`s race on them — the
    // loser reads the other's stub URL, whose listener dies with its runtime.
    //
    // The stub embedder returns the SAME vector for every query, so the dense
    // leg cannot distinguish the documents. Any ordering that tracks the query
    // text therefore came from the lexical leg — which is what proves BM25
    // actually ran rather than hybrid quietly behaving as dense.

    create(
        &c,
        &base,
        json!({
            "name": "docs", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;

    let v: Vec<f32> = {
        let mut v = vec![0.0; DIM];
        v[0] = 1.0;
        v
    };
    let r = c
        .post(format!("{base}/v1/collections/docs/insert"))
        .json(&json!({"records": [
            {"content": "nordwind pressure valves estonia", "vec": v},
            {"content": "kaito industrial bolts vietnam", "vec": v}
        ]}))
        .send()
        .await
        .unwrap();
    assert!(
        r.status().is_success(),
        "insert failed: {}",
        r.text().await.unwrap()
    );

    let before = stub.calls.load(Ordering::SeqCst);
    let (status, body) = hybrid(
        &c,
        &base,
        json!({"collection":"docs","query":"nordwind valves","limit":5}),
    )
    .await;
    assert_eq!(status, 200, "hybrid should succeed: {body}");
    let out: Value = serde_json::from_str(&body).unwrap();
    let hits = out["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "hybrid returned nothing: {body}");
    assert_eq!(
        hits[0]["fields"]["content"].as_str().unwrap_or_default(),
        "nordwind pressure valves estonia",
        "the lexical leg should decide the order: {body}"
    );
    assert!(
        stub.calls.load(Ordering::SeqCst) > before,
        "the dense leg must still embed the query"
    );
}
