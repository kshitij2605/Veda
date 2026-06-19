//! Integration tests for the agentic web-search tools: the
//! [`SearxngClient`] against a mock searxng, the `web_search` / `fetch_url`
//! agent-catalog tools against mock backends, the opt-in behaviour
//! (endpoints unset → clean not-configured, web tools excluded from the
//! default agent), and the built-in `veda-web` agent's whitelist.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

use veda_gateway::agent::catalog::{ToolCatalog, ToolContext};
use veda_gateway::agent::model::{default_agent, default_web_agent};
use veda_gateway::rest::searxng::SearxngClient;
use veda_gateway::rest::state::AppState;
use veda_gateway::tenant::TenantId;

/// Serializes the env-var-driven tests so they don't race on the
/// process-global `VEDA_SEARXNG_ENDPOINT` / `VEDA_CRAWL4AI_ENDPOINT`.
static WEB_ENV_LOCK: Mutex<()> = Mutex::new(());

/// A mock searxng server: `GET /search` returns three canned results,
/// mirroring `{ "results": [ {title, url, content, engine} ] }`.
async fn start_mock_searxng() -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::get;
    use axum::Router;

    async fn search() -> axum::Json<Value> {
        axum::Json(json!({
            "results": [
                { "title": "Veda", "url": "https://veda.dev", "content": "multimodal vector DB", "engine": "google" },
                { "title": "HNSW", "url": "https://example.com/hnsw", "content": "graph ANN index", "engine": "bing" },
                { "title": "RAG", "url": "https://example.com/rag", "content": "retrieval augmented generation" }
            ]
        }))
    }

    let router = Router::new().route("/search", get(search));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, handle)
}

/// A mock crawl4ai server: `POST /crawl` returns one result with canned
/// markdown + a title.
async fn start_mock_crawl4ai() -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    use axum::Router;

    async fn crawl() -> axum::Json<Value> {
        axum::Json(json!({
            "results": [{
                "url": "https://veda.dev",
                "markdown": "# Veda\n\nVeda is a multimodal vector database.",
                "metadata": { "title": "Veda" }
            }]
        }))
    }

    let router = Router::new().route("/crawl", post(crawl));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, handle)
}

// ── SearxngClient ────────────────────────────────────────────────────────

#[tokio::test]
async fn searxng_client_parses_and_caps_to_limit() {
    let (searx_url, _h) = start_mock_searxng().await;
    let client = SearxngClient::new(searx_url, None);

    // Three results available; cap to two.
    let results = client.search("vector db", 2).await.unwrap();
    assert_eq!(results.len(), 2, "limit not applied");
    assert_eq!(results[0].title, "Veda");
    assert_eq!(results[0].url, "https://veda.dev");
    assert_eq!(results[0].snippet, "multimodal vector DB");
    assert_eq!(results[0].engine.as_deref(), Some("google"));

    // A larger limit returns all three.
    let results = client.search("vector db", 10).await.unwrap();
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn searxng_client_unreachable_is_clean_error() {
    // Nothing listening on this port → SEARCH_UNAVAILABLE, not a panic.
    let client = SearxngClient::new("http://127.0.0.1:1", None);
    let err = client.search("x", 5).await.unwrap_err();
    assert_eq!(err.code, "SEARCH_UNAVAILABLE");
}

// ── web_search tool ──────────────────────────────────────────────────────

#[tokio::test]
async fn web_search_tool_dispatches_and_returns_results() {
    let (searx_url, _h) = start_mock_searxng().await;
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let tenant = TenantId::default_tenant();
    let ctx = ToolContext::new(&state, &tenant);
    let cat = ToolCatalog::new();

    let _guard = WEB_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on WEB_ENV_LOCK.
    unsafe {
        std::env::set_var("VEDA_SEARXNG_ENDPOINT", &searx_url);
    }
    let out = cat
        .dispatch(&ctx, "web_search", json!({ "query": "vector db", "limit": 2 }))
        .await;
    unsafe {
        std::env::remove_var("VEDA_SEARXNG_ENDPOINT");
    }
    drop(_guard);

    assert!(!out.is_error, "web_search errored: {:?}", out.content);
    let results = out.content["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(out.content["count"], 2);
    assert_eq!(results[0]["title"], "Veda");
    assert_eq!(results[0]["url"], "https://veda.dev");
    assert!(results[0]["snippet"].as_str().is_some());
}

#[tokio::test]
async fn web_search_tool_unset_is_clean_not_configured() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let tenant = TenantId::default_tenant();
    let ctx = ToolContext::new(&state, &tenant);
    let cat = ToolCatalog::new();

    let _guard = WEB_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on WEB_ENV_LOCK.
    unsafe {
        std::env::remove_var("VEDA_SEARXNG_ENDPOINT");
    }
    let out = cat
        .dispatch(&ctx, "web_search", json!({ "query": "vector db" }))
        .await;
    drop(_guard);

    // Not configured is a clean success result, not an error.
    assert!(!out.is_error, "should not be a hard error: {:?}", out.content);
    assert_eq!(out.content["not_configured"], true);
    assert!(out.content["message"].as_str().is_some());
}

// ── fetch_url tool ───────────────────────────────────────────────────────

#[tokio::test]
async fn fetch_url_tool_dispatches_and_returns_markdown() {
    let (crawl_url, _h) = start_mock_crawl4ai().await;
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let tenant = TenantId::default_tenant();
    let ctx = ToolContext::new(&state, &tenant);
    let cat = ToolCatalog::new();

    let _guard = WEB_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on WEB_ENV_LOCK.
    unsafe {
        std::env::set_var("VEDA_CRAWL4AI_ENDPOINT", &crawl_url);
    }
    let out = cat
        .dispatch(&ctx, "fetch_url", json!({ "url": "https://veda.dev" }))
        .await;
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    drop(_guard);

    assert!(!out.is_error, "fetch_url errored: {:?}", out.content);
    assert_eq!(out.content["url"], "https://veda.dev");
    assert_eq!(out.content["title"], "Veda");
    assert!(out.content["markdown"].as_str().unwrap().contains("multimodal vector database"));
    assert_eq!(out.content["truncated"], false);
}

#[tokio::test]
async fn fetch_url_tool_ssrf_blocked_without_crawling() {
    // crawl4ai is configured and counts hits; a private target URL must be
    // blocked by the SSRF guard with a clean `{ blocked: true }` outcome
    // (not a hard error) and the crawl server must never be contacted.
    use axum::extract::State as AxState;
    use axum::routing::post;
    use axum::Router;

    let hits = Arc::new(AtomicUsize::new(0));
    async fn crawl(AxState(hits): AxState<Arc<AtomicUsize>>) -> axum::Json<Value> {
        hits.fetch_add(1, Ordering::SeqCst);
        axum::Json(json!({ "results": [{ "url": "x", "markdown": "# x" }] }))
    }
    let router = Router::new()
        .route("/crawl", post(crawl))
        .with_state(hits.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let crawl_url = format!("http://{}", listener.local_addr().unwrap());
    let _h = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let tenant = TenantId::default_tenant();
    let ctx = ToolContext::new(&state, &tenant);
    let cat = ToolCatalog::new();

    let _guard = WEB_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on WEB_ENV_LOCK.
    unsafe {
        std::env::set_var("VEDA_CRAWL4AI_ENDPOINT", &crawl_url);
    }
    let out = cat
        .dispatch(
            &ctx,
            "fetch_url",
            json!({ "url": "http://169.254.169.254/latest/meta-data/" }),
        )
        .await;
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    drop(_guard);

    assert!(!out.is_error, "blocked URL should be a clean outcome: {:?}", out.content);
    assert_eq!(out.content["blocked"], true);
    assert_eq!(out.content["code"], "URL_BLOCKED_PRIVATE");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "crawl4ai must NOT be contacted for a blocked URL"
    );
}

#[tokio::test]
async fn fetch_url_tool_unset_is_clean_not_configured() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let tenant = TenantId::default_tenant();
    let ctx = ToolContext::new(&state, &tenant);
    let cat = ToolCatalog::new();

    let _guard = WEB_ENV_LOCK.lock().unwrap();
    // SAFETY: serialized on WEB_ENV_LOCK.
    unsafe {
        std::env::remove_var("VEDA_CRAWL4AI_ENDPOINT");
    }
    let out = cat
        .dispatch(&ctx, "fetch_url", json!({ "url": "https://veda.dev" }))
        .await;
    drop(_guard);

    assert!(!out.is_error, "should not be a hard error: {:?}", out.content);
    assert_eq!(out.content["not_configured"], true);
}

// ── opt-in / agents ──────────────────────────────────────────────────────

#[test]
fn web_tools_are_registered_builtins_but_opt_in() {
    let cat = ToolCatalog::new();
    // Present in the full builtin set (count bumped 13 → 15).
    assert_eq!(cat.builtin_specs().len(), 15);
    let all: Vec<&str> = cat.builtin_specs().iter().map(|s| s.id.as_str()).collect();
    assert!(all.contains(&"web_search"));
    assert!(all.contains(&"fetch_url"));

    // ...but excluded from the "empty whitelist == all" expansion that
    // the default agent uses.
    let default_specs = cat.specs_for(&default_agent().tool_ids);
    let default_ids: Vec<&str> = default_specs.iter().map(|s| s.id.as_str()).collect();
    assert!(!default_ids.contains(&"web_search"), "web_search must be opt-in");
    assert!(!default_ids.contains(&"fetch_url"), "fetch_url must be opt-in");
}

#[test]
fn veda_web_agent_resolves_with_web_whitelist() {
    let agent = default_web_agent();
    assert_eq!(agent.id, "veda-web");
    let cat = ToolCatalog::new();
    let specs = cat.specs_for(&agent.tool_ids);
    let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
    assert!(ids.contains(&"web_search"));
    assert!(ids.contains(&"fetch_url"));
    assert!(ids.contains(&"search"));
    // Only the three whitelisted tools — nothing else leaks in.
    assert_eq!(specs.len(), 3);
}
