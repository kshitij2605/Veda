//! Two defaults that used to be silent: binding a vector field to its
//! embedding model, and `auto` upgrading to hybrid.
//!
//! One test, not several: the file sets process-global env vars, and parallel
//! `#[tokio::test]`s race on them — the loser reads another test's stub URL,
//! whose listener dies with its runtime.

use axum::{Json, Router, extract::State, routing::post};
use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

const DIM: usize = 8;
const MODEL: &str = "stub-embedder-v1";

async fn embeddings(State(_): State<()>, Json(_b): Json<Value>) -> Json<Value> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    Json(
        json!({ "data": [ { "embedding": v, "index": 0, "object": "embedding" } ],
                 "model": MODEL, "object": "list" }),
    )
}

async fn start_stub() -> String {
    let app = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .with_state(());
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{addr}")
}

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    state.load_embedding_from_env().await;
    let router = build_router(state);
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
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
        "create failed: {}",
        r.text().await.unwrap()
    );
}

/// The `model` reported for a field by `GET /v1/collections/{name}`.
async fn field_model(c: &Client, base: &str, coll: &str, field: &str) -> Option<String> {
    let body = c
        .get(format!("{base}/v1/collections/{coll}"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let v: Value = serde_json::from_str(&body).unwrap();
    v["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == field)
        .and_then(|f| f["model"].as_str().map(str::to_owned))
}

#[tokio::test]
async fn auto_embed_fields_bind_and_auto_upgrades_to_hybrid() {
    let endpoint = start_stub().await;
    // SAFETY: the only test in this binary.
    unsafe {
        std::env::set_var("EMBEDDING_ENDPOINT", &endpoint);
        std::env::set_var("EMBEDDING_MODEL", MODEL);
        std::env::set_var("EMBEDDING_DIMENSION", DIM.to_string());
        std::env::set_var("EMBEDDING_PRESET", "");
    }
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    // ── model binding ───────────────────────────────────────────────────
    // A field that declares auto_embed gets bound: Veda produces those
    // vectors, so the model is unambiguous.
    create(
        &c,
        &base,
        json!({
            "name": "bound", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": DIM, "metric": "cosine",
                  "index": "hnsw",
                  "auto_embed": { "source": "content", "modality": "text" } }
            ]
        }),
    )
    .await;
    assert_eq!(
        field_model(&c, &base, "bound", "vec").await.as_deref(),
        Some(MODEL),
        "an auto_embed field must be bound to the model that will fill it"
    );

    // A field WITHOUT auto_embed is left unbound: the caller supplies those
    // vectors, so binding them would reject inserts after any embedder change
    // for a field Veda never embedded.
    create(
        &c,
        &base,
        json!({
            "name": "unbound", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;
    assert_eq!(
        field_model(&c, &base, "unbound", "vec").await,
        None,
        "a caller-supplied vector field must stay unbound"
    );

    // An explicit `model` still wins over the inferred one.
    create(
        &c,
        &base,
        json!({
            "name": "explicit", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string" },
                { "name": "vec", "type": "vector", "dimension": DIM, "metric": "cosine",
                  "index": "hnsw", "model": "caller-choice",
                  "auto_embed": { "source": "content", "modality": "text" } }
            ]
        }),
    )
    .await;
    assert_eq!(
        field_model(&c, &base, "explicit", "vec").await.as_deref(),
        Some("caller-choice"),
        "an explicit model must not be overwritten"
    );

    // ── auto routing ────────────────────────────────────────────────────
    // A question-form query on a bm25 collection: `auto` upgrades to hybrid,
    // and says so.
    let out = search_auto(&c, &base, "bound", "Which supplier is in Estonia?").await;
    assert_eq!(
        out["mode"], "hybrid",
        "auto should upgrade a lexical query on a bm25 collection: {out}"
    );
    assert_eq!(out["mode_selected_by"], "auto", "{out}");

    // The SAME query on a collection with NO bm25 field stays dense — the
    // upgrade is gated on the collection actually having a lexical leg.
    // (`unbound` is not that collection: it has bm25 on `content`, which is
    // why it was upgrading. A gate test needs a fixture that actually lacks
    // the thing being gated on.)
    create(
        &c,
        &base,
        json!({
            "name": "no_lexical", "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string" },
                { "name": "vec", "type": "vector", "dimension": DIM,
                  "metric": "cosine", "index": "hnsw" }
            ]
        }),
    )
    .await;
    let out = search_auto(&c, &base, "no_lexical", "Which supplier is in Estonia?").await;
    assert!(
        out.get("mode").is_none(),
        "no bm25 field means no upgrade: {out}"
    );

    // A short keyword query routes dense, not hybrid — the router only
    // upgrades what it classifies as lexical/multi-faceted.
    let out = search_auto(&c, &base, "bound", "valves").await;
    assert!(
        out.get("mode").is_none(),
        "a short query should not upgrade: {out}"
    );
}

async fn search_auto(c: &Client, base: &str, coll: &str, q: &str) -> Value {
    let body = c
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&json!({ "collection": coll, "query": q, "limit": 5 }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    serde_json::from_str(&body).unwrap_or_else(|_| panic!("non-JSON response: {body}"))
}
