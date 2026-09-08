//! The sparse leg of the agent `search` tool, driven end to end against a
//! real HTTP encoder.
//!
//! **Why this is not mocked.** The SPLADE panic fixed in #22 shipped because
//! every test in that module drove `parse_response` directly and none made an
//! HTTP call, so a `reqwest::blocking`-inside-async landmine sat invisible
//! behind green tests. A stub that handed back a canned `SpladeEncoding`
//! would reproduce that blind spot exactly. This drives a real listener over
//! real TCP from inside `#[tokio::test]` — the context the product actually
//! encodes from — which is the shape the process note in the task list asks
//! for.
//!
//! **Why its own file.** Integration tests share one process per file. This
//! one sets `VEDA_SPLADE_ENDPOINT`; `agent_tool_errors_test.rs` asserts the
//! behaviour when that variable is *unset*. Sharing a process would race.

use axum::{Json, Router, extract::State, routing::post};
use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

/// Vocabulary indices the stub assigns to each recognised term, matching the
/// vectors inserted below. Two *disjoint* sets, so the query text decides the
/// ranking: a stub that ignored its input could not tell them apart, and
/// neither could a test that only asserted "some hits came back".
const ALPHA_IDX: [u32; 3] = [10, 20, 30];
const GAMMA_IDX: [u32; 3] = [500, 600, 700];

#[derive(Clone)]
struct Stub {
    calls: Arc<AtomicUsize>,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
}

/// TEI's `/embed_sparse` contract: `{"inputs": [text]}` in, a list-of-lists
/// of `{index, value}` out (outer dimension is one entry per input string).
async fn embed_sparse(State(stub): State<Stub>, Json(body): Json<Value>) -> Json<Value> {
    stub.calls.fetch_add(1, Ordering::SeqCst);
    let text = body["inputs"][0].as_str().unwrap_or_default().to_lowercase();
    stub.seen.lock().unwrap().push(text.clone());

    let idx: &[u32] = if text.contains("alpha") {
        &ALPHA_IDX
    } else if text.contains("gamma") {
        &GAMMA_IDX
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
    // `AppState::new` leaves the encoder unset; the server binary installs it
    // at startup. Without this the sparse leg reports "no encoder configured"
    // and the test would pass for the wrong reason.
    state.load_splade_from_env().await;
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), tmp)
}

async fn sparse_search(c: &Client, base: &str, query: &str) -> Value {
    let resp = c
        .post(format!("{base}/v1/agent-tools/search/execute"))
        .json(&json!({
            "collection": "sparse_kb",
            "query": query,
            "mode": "sparse",
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    assert_eq!(status, 200, "sparse search should succeed: {body}");
    serde_json::from_str(&body).unwrap()
}

fn top_content(out: &Value) -> String {
    out["hits"][0]["fields"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no content on the top hit: {out}"))
        .to_owned()
}

/// The whole point: `mode: "sparse"` returns real hits, and *which* hits
/// depends on the query text having actually gone through the encoder.
#[tokio::test]
async fn sparse_mode_returns_hits_ranked_by_the_encoded_query() {
    let (splade_url, stub) = start_splade_stub().await;
    // SAFETY: this file is its own test binary and holds a single test, so no
    // other thread in this process reads or writes the environment.
    unsafe {
        std::env::set_var("VEDA_SPLADE_ENDPOINT", &splade_url);
    }

    let (base, _tmp) = start_test_server().await;
    let c = Client::new();

    // A sparse-ONLY collection: no dense vector field at all. This is exactly
    // the shape the old code rejected as "no vector fields" before a sparse
    // query could run, so it is the shape worth testing.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "sparse_kb",
            "shard_count": 1,
            "fields": [
                { "name": "content", "type": "string" },
                { "name": "sv", "type": "sparse_vector",
                  "dimension": 30522, "metric": "cosine" }
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
        .post(format!("{base}/v1/collections/sparse_kb/insert"))
        .json(&json!({
            "records": [
                { "content": "doc-alpha",
                  "sv": { "indices": ALPHA_IDX, "values": [0.5, 0.5, 0.5] } },
                { "content": "doc-gamma",
                  "sv": { "indices": GAMMA_IDX, "values": [0.5, 0.5, 0.5] } }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "insert failed: {}",
        resp.text().await.unwrap()
    );

    // Query one: the encoder maps "alpha" onto doc-alpha's indices.
    let out = sparse_search(&c, &base, "alpha valves").await;
    assert!(
        out["count"].as_u64().unwrap_or(0) > 0,
        "sparse search returned no hits: {out}"
    );
    assert_eq!(
        top_content(&out),
        "doc-alpha",
        "the encoded query should rank doc-alpha first: {out}"
    );

    // Query two: a different query must reorder the results. This is what
    // separates "the encoder ran" from "something returned rows".
    let out = sparse_search(&c, &base, "gamma valves").await;
    assert_eq!(
        top_content(&out),
        "doc-gamma",
        "a different query must produce a different top hit: {out}"
    );

    // And the encoder was genuinely reached over HTTP both times — the
    // assertion the SPLADE panic would have failed.
    assert_eq!(
        stub.calls.load(Ordering::SeqCst),
        2,
        "the sparse leg must call the encoder once per query"
    );
    let seen = stub.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec!["alpha valves".to_owned(), "gamma valves".to_owned()],
        "the encoder must receive the caller's query text verbatim"
    );
}
