//! Integration tests for the web-ingestion connector
//! (`POST /v1/collections/{name}/ingest-url`) and the shared
//! `ingest_markdown` pipeline it rides on.
//!
//! Covers:
//!   - A mock crawl4ai server returns canned markdown; `/ingest-url`
//!     chunks/embeds/inserts it and the records carry `source_url`.
//!   - `VEDA_CRAWL4AI_ENDPOINT` unset → clean 503 (no panic).

use reqwest::Client;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

/// Serializes the env-var-driven crawl4ai tests so they don't race on the
/// process-global `VEDA_CRAWL4AI_ENDPOINT`.
static CRAWL_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Start a gateway with a pre-registered text embedding model (a "mock"
/// endpoint → deterministic vectors, no real HTTP).
async fn start_gateway() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    {
        let mut registry = state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "mock-text".into(),
            endpoint: "http://mock:9999/embed".into(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 1024 * 1024,
            timeout_seconds: 10,
            batch_size: 32,
        });
    }
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

/// Captures what the client sent to the mock crawl4ai `/md` endpoint so a
/// test can assert the request contract (single `url` field + bearer).
#[derive(Clone, Default)]
struct CrawlSpy {
    body: Arc<Mutex<Option<Value>>>,
    had_bearer: Arc<std::sync::atomic::AtomicBool>,
}

/// A mock crawl4ai server: `POST /md` returns the real per-URL shape
/// `{ "url", "markdown": "<string>", "success": true }`, and records the
/// request body + whether an Authorization bearer was present.
async fn start_mock_crawl4ai() -> (String, CrawlSpy, tokio::task::JoinHandle<()>) {
    use axum::extract::State as AxState;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::Router;

    let spy = CrawlSpy::default();

    async fn md(
        AxState(spy): AxState<CrawlSpy>,
        headers: HeaderMap,
        axum::Json(body): axum::Json<Value>,
    ) -> axum::Json<Value> {
        *spy.body.lock().unwrap() = Some(body);
        spy.had_bearer.store(
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.starts_with("Bearer "))
                .unwrap_or(false),
            Ordering::SeqCst,
        );
        axum::Json(json!({
            "url": "https://example.com/doc",
            "markdown": "# Veda\n\nVeda is a multimodal vector database. It supports many modalities.\n\n## Indexing\n\nVeda offers HNSW, IVF-PQ, and DiskANN indexes for approximate nearest neighbour search.",
            "title": "Veda Docs",
            "success": true
        }))
    }

    let router = Router::new()
        .route("/md", post(md))
        .with_state(spy.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, spy, handle)
}

/// A mock crawl4ai server that counts how many `POST /md` it receives,
/// so a wiring test can assert the SSRF guard short-circuits BEFORE any
/// crawl is attempted (hit count stays 0).
async fn start_counting_crawl4ai() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    use axum::extract::State as AxState;
    use axum::routing::post;
    use axum::Router;

    let hits = Arc::new(AtomicUsize::new(0));

    async fn md(AxState(hits): AxState<Arc<AtomicUsize>>) -> axum::Json<Value> {
        hits.fetch_add(1, Ordering::SeqCst);
        axum::Json(json!({
            "url": "https://example.com/doc", "markdown": "# x\n\nbody", "title": "X", "success": true
        }))
    }

    let router = Router::new()
        .route("/md", post(md))
        .with_state(hits.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, hits, handle)
}

fn client() -> Client {
    Client::new()
}

async fn create_web_collection(base: &str, name: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "content", "type": "string", "filterable": true },
                { "name": "source_url", "type": "string", "filterable": true },
                { "name": "title", "type": "string" },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create collection failed: {:?}", resp.text().await);
}

#[tokio::test]
async fn test_ingest_url_chunks_embeds_and_inserts_with_source_url() {
    let (crawl_url, spy, _crawl_handle) = start_mock_crawl4ai().await;
    let (base, _tmp) = start_gateway().await;
    create_web_collection(&base, "web_col").await;

    // The handler reads VEDA_CRAWL4AI_ENDPOINT at request time, so the env
    // var must be set for the duration of the request. Serialize on the
    // env lock and hold it across the request.
    let _guard = CRAWL_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on CRAWL_ENV_LOCK; no concurrent crawl-env access.
    unsafe {
        std::env::set_var("VEDA_CRAWL4AI_ENDPOINT", &crawl_url);
        std::env::set_var("VEDA_CRAWL4AI_TOKEN", "test-token");
    }

    let resp = client()
        .post(format!("{base}/v1/collections/web_col/ingest-url"))
        .json(&json!({
            "url": "https://example.com/doc",
            "chunker": "fixed_size",
            "chunk_size": 80,
            "chunk_overlap": 0
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let body: Value = resp.json().await.unwrap();

    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
        std::env::remove_var("VEDA_CRAWL4AI_TOKEN");
    }
    drop(_guard);

    assert_eq!(status, 201, "ingest-url failed: {body}");
    assert_eq!(body["pages"], 1);
    let inserted = body["chunks_inserted"].as_u64().unwrap();
    assert!(inserted >= 1, "expected at least one chunk inserted: {body}");
    assert_eq!(body["per_url"][0]["title"], "Veda Docs");

    // The client must hit `POST /md` with a single `url` string (NOT an
    // `urls` array) and forward the bearer token.
    let sent = spy.body.lock().unwrap().clone().expect("crawl4ai /md was not called");
    assert_eq!(
        sent["url"], "https://example.com/doc",
        "client must POST a single `url` string to /md: {sent}"
    );
    assert!(sent.get("urls").is_none(), "client must not send an `urls` array: {sent}");
    assert!(
        spy.had_bearer.load(Ordering::SeqCst),
        "client must forward the bearer token to crawl4ai"
    );

    // Verify the records were inserted and carry source_url.
    let list = client()
        .get(format!("{base}/v1/collections/web_col/records?limit=100"))
        .send()
        .await
        .unwrap();
    assert_eq!(list.status(), 200);
    let list_body: Value = list.json().await.unwrap();
    let records = list_body["records"].as_array().unwrap();
    assert_eq!(records.len() as u64, inserted, "record count mismatch: {list_body}");
    for rec in records {
        assert_eq!(
            rec["fields"]["source_url"], "https://example.com/doc",
            "record missing source_url: {rec}"
        );
        assert!(
            rec["fields"]["content"].as_str().is_some(),
            "record missing content: {rec}"
        );
    }
}

#[tokio::test]
async fn test_ingest_url_not_configured_returns_503() {
    let (base, _tmp) = start_gateway().await;
    create_web_collection(&base, "web_col_unset").await;

    // Ensure the endpoint is unset for this request.
    let _guard = CRAWL_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on CRAWL_ENV_LOCK.
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }

    let resp = client()
        .post(format!("{base}/v1/collections/web_col_unset/ingest-url"))
        .json(&json!({ "url": "https://example.com/doc" }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let body: Value = resp.json().await.unwrap();
    drop(_guard);

    assert_eq!(status, 503, "expected 503 when crawl4ai unset: {body}");
    assert_eq!(body["error"]["code"], "CRAWL_SERVICE_UNAVAILABLE");
}

#[tokio::test]
async fn test_ingest_url_ssrf_blocked_returns_403_without_crawling() {
    // The mock crawl4ai is fully configured and counts hits; the SSRF
    // guard must reject the private target URL and short-circuit so the
    // crawl server is NEVER contacted.
    let (crawl_url, hits, _crawl_handle) = start_counting_crawl4ai().await;
    let (base, _tmp) = start_gateway().await;
    create_web_collection(&base, "web_col_ssrf").await;

    let _guard = CRAWL_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on CRAWL_ENV_LOCK.
    unsafe {
        std::env::set_var("VEDA_CRAWL4AI_ENDPOINT", &crawl_url);
    }

    let resp = client()
        .post(format!("{base}/v1/collections/web_col_ssrf/ingest-url"))
        .json(&json!({ "url": "http://169.254.169.254/latest/meta-data/" }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let body: Value = resp.json().await.unwrap();

    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    drop(_guard);

    assert_eq!(status, 403, "expected 403 for blocked SSRF URL: {body}");
    assert_eq!(body["error"]["code"], "URL_BLOCKED_PRIVATE");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "crawl4ai must NOT be contacted for a blocked URL"
    );
}
