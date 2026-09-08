//! `embed-insert` populating a `sparse_vector` field that declares
//! `auto_embed`, driven against a real HTTP encoder.
//!
//! Two defects, both live before this test existed:
//!
//! 1. The handler never encoded sparse fields at all. `SparseVector`'s
//!    `auto_embed` doc comment promised "embed-insert and process-document
//!    route the named source field through a sparse encoder"; only
//!    `process-document` ever did, so a record inserted here kept that column
//!    empty and said `{"inserted":1}` while doing it.
//! 2. A collection whose ONLY auto-embed field is sparse was rejected with
//!    400 before reaching any of that, because mapping inference scanned
//!    `FieldType::Vector` alone and treated "nothing dense to infer" as
//!    "nothing to do".
//!
//! Real HTTP for the same reason as `agent_sparse_search_live_test`: the
//! SPLADE panic fixed in #22 survived a full test suite that never made a
//! request. Own file, so setting `VEDA_SPLADE_ENDPOINT` cannot race the tests
//! that require it unset.

use axum::{Json, Router, extract::State, routing::post};
use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

/// What the stub encodes "widget" to. The search at the end queries these
/// exact indices, so the assertion is that the *stored column* holds them —
/// not merely that the encoder was called.
const WIDGET_IDX: [u32; 3] = [7, 11, 13];

#[derive(Clone)]
struct Stub {
    calls: Arc<AtomicUsize>,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
}

async fn embed_sparse(State(stub): State<Stub>, Json(body): Json<Value>) -> Json<Value> {
    stub.calls.fetch_add(1, Ordering::SeqCst);
    let text = body["inputs"][0]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    stub.seen.lock().unwrap().push(text.clone());
    let idx: &[u32] = if text.contains("widget") {
        &WIDGET_IDX
    } else {
        &[]
    };
    let entries: Vec<Value> = idx
        .iter()
        .map(|i| json!({ "index": i, "value": 0.5 }))
        .collect();
    Json(json!([entries]))
}

async fn start_splade_stub() -> (String, Stub) {
    let stub = Stub {
        calls: Arc::new(AtomicUsize::new(0)),
        seen: Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route("/embed_sparse", post(embed_sparse))
        .with_state(stub.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), stub)
}

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    state.load_splade_from_env().await;
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), tmp)
}

#[tokio::test]
async fn embed_insert_populates_a_sparse_auto_embed_field() {
    let (splade_url, stub) = start_splade_stub().await;
    // SAFETY: this file is its own test binary and holds a single test.
    unsafe {
        std::env::set_var("VEDA_SPLADE_ENDPOINT", &splade_url);
    }

    let (base, _tmp) = start_test_server().await;
    let c = Client::new();

    // Sparse-ONLY auto-embed: no dense vector field anywhere in the schema.
    // Reaching a 2xx at all is defect (2) fixed; before, this 400'd.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "sparse_auto",
            "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string" },
                { "name": "sv", "type": "sparse_vector", "dimension": 30522,
                  "metric": "cosine",
                  "auto_embed": { "source": "content", "modality": "text" } }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "collection setup failed: {}",
        resp.text().await.unwrap()
    );

    let resp = c
        .post(format!("{base}/v1/collections/sparse_auto/embed-insert"))
        .json(&json!({ "records": [ { "content": "a widget for testing" } ] }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert!(
        (200..300).contains(&status),
        "embed-insert on a sparse-only auto_embed collection should succeed, \
         got {status}: {body}"
    );
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["inserted"], 1, "expected one row inserted: {body}");

    // The encoder was actually reached, with the source field's text.
    assert_eq!(
        stub.calls.load(Ordering::SeqCst),
        1,
        "embed-insert must call the sparse encoder once per record"
    );
    assert_eq!(
        stub.seen.lock().unwrap().clone(),
        vec!["a widget for testing".to_owned()],
        "the encoder must receive the auto_embed source field's text"
    );

    // And — the assertion that actually distinguishes "encoded" from
    // "stored" — the column holds that vector, so a sparse query for the
    // same indices retrieves the row.
    let resp = c
        .post(format!("{base}/v1/collections/sparse_auto/search"))
        .json(&json!({
            "query": { "sparse_vector": {
                "field": "sv",
                "indices": WIDGET_IDX,
                "values": [0.5, 0.5, 0.5]
            } },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    let body = resp.text().await.unwrap();
    let out: Value = serde_json::from_str(&body).unwrap();
    let results = out["results"].as_array().unwrap_or_else(|| {
        panic!("no results array — the sparse column was never populated: {body}")
    });
    assert_eq!(
        results.len(),
        1,
        "the auto-embedded sparse column should be searchable: {body}"
    );
    let distance = results[0]["distance"].as_f64().unwrap();
    assert!(
        distance < 0.01,
        "an identical query should match near-exactly, got {distance}: {body}"
    );
}
