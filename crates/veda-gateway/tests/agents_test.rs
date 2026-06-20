//! Integration tests for the Veda Agents REST surface.
//!
//! Covers:
//!   - Agent CRUD over HTTP (create → get → list → update → delete)
//!   - Tenant isolation of agents (X-Veda-Tenant scoping)
//!   - AgentTool CRUD round-trip
//!   - GET /v1/agent-tools/catalog returns the built-in specs
//!   - POST /v1/agent-tools/{id}/execute against a seeded collection
//!     (catalog dispatch over HTTP, no LLM required)
//!   - POST /v1/agents/veda-agent/converse with no LLM configured → 503
//!   - Conversation GET / list / delete round-trip

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

// ── Agent CRUD ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_crud_round_trip() {
    let (base, _tmp) = start_test_server().await;

    // Create (no id → server assigns one).
    let resp = client()
        .post(format!("{base}/v1/agents"))
        .json(&json!({
            "id": "",
            "name": "Researcher",
            "description": "Searches collections",
            "instructions": "Be concise.",
            "tool_ids": ["search", "aggregate"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: Value = resp.json().await.unwrap();
    let id = created["id"].as_str().unwrap().to_owned();
    assert!(!id.is_empty());
    // The returned id must NOT carry the tenant prefix.
    assert!(!id.contains("__"), "returned id leaked tenant prefix: {id}");
    assert_eq!(created["name"], "Researcher");

    // Get.
    let resp = client()
        .get(format!("{base}/v1/agents/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let got: Value = resp.json().await.unwrap();
    assert_eq!(got["id"], id);
    assert_eq!(got["name"], "Researcher");

    // List.
    let resp = client()
        .get(format!("{base}/v1/agents"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let list: Value = resp.json().await.unwrap();
    let agents = list["agents"].as_array().unwrap();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0]["id"], id);

    // Update (PUT with explicit id).
    let resp = client()
        .put(format!("{base}/v1/agents/{id}"))
        .json(&json!({
            "id": id,
            "name": "Researcher v2",
            "instructions": "Be very concise.",
            "tool_ids": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let updated: Value = resp.json().await.unwrap();
    assert_eq!(updated["name"], "Researcher v2");

    let resp = client()
        .get(format!("{base}/v1/agents/{id}"))
        .send()
        .await
        .unwrap();
    let got: Value = resp.json().await.unwrap();
    assert_eq!(got["name"], "Researcher v2");

    // Delete.
    let resp = client()
        .delete(format!("{base}/v1/agents/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);

    let resp = client()
        .get(format!("{base}/v1/agents/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_agent_tenant_isolation() {
    let (base, _tmp) = start_test_server().await;

    // Create an agent under tenant "acme".
    let resp = client()
        .post(format!("{base}/v1/agents"))
        .header("X-Veda-Tenant", "acme")
        .json(&json!({ "id": "shared-id", "name": "Acme Agent", "tool_ids": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Tenant "rival" must not see it in its list.
    let resp = client()
        .get(format!("{base}/v1/agents"))
        .header("X-Veda-Tenant", "rival")
        .send()
        .await
        .unwrap();
    let list: Value = resp.json().await.unwrap();
    assert_eq!(list["agents"].as_array().unwrap().len(), 0);

    // ...nor fetch it by id.
    let resp = client()
        .get(format!("{base}/v1/agents/shared-id"))
        .header("X-Veda-Tenant", "rival")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // The owning tenant still sees it.
    let resp = client()
        .get(format!("{base}/v1/agents/shared-id"))
        .header("X-Veda-Tenant", "acme")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let got: Value = resp.json().await.unwrap();
    assert_eq!(got["id"], "shared-id");
}

// ── AgentTool CRUD ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_tool_crud_round_trip() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .post(format!("{base}/v1/agent-tools"))
        .json(&json!({
            "id": "weather",
            "name": "Weather",
            "description": "Fetch weather",
            "kind": "http",
            "config": { "url": "https://example.com", "method": "GET" }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: Value = resp.json().await.unwrap();
    assert_eq!(created["id"], "weather");

    let resp = client()
        .get(format!("{base}/v1/agent-tools/weather"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let got: Value = resp.json().await.unwrap();
    assert_eq!(got["kind"], "http");

    let resp = client()
        .get(format!("{base}/v1/agent-tools"))
        .send()
        .await
        .unwrap();
    let list: Value = resp.json().await.unwrap();
    assert_eq!(list["tools"].as_array().unwrap().len(), 1);

    let resp = client()
        .delete(format!("{base}/v1/agent-tools/weather"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
    let resp = client()
        .get(format!("{base}/v1/agent-tools/weather"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── Built-in tool catalog ────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_catalog_returns_builtins() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/agent-tools/catalog"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let tools = body["tools"].as_array().unwrap();
    // Mirrors ToolCatalog's builtin count (15 incl. the opt-in web tools).
    assert_eq!(tools.len(), 15);
    let ids: Vec<&str> = tools.iter().map(|t| t["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"search"));
    assert!(ids.contains(&"list_collections"));
    assert!(ids.contains(&"web_search"));
    assert!(ids.contains(&"fetch_url"));
    // Each spec carries an input_schema.
    assert!(tools[0]["input_schema"].is_object());
}

// ── Tool execute over HTTP (no LLM) ──────────────────────────────────────

async fn seed_collection(base: &str) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "agg_test",
            "fields": [
                { "name": "category", "type": "string",  "filterable": true },
                { "name": "price",    "type": "float64", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        201,
        "create failed: {}",
        r.text().await.unwrap()
    );

    let r = client()
        .post(format!("{base}/v1/collections/agg_test/insert"))
        .json(&json!({ "records": [
            { "category": "electronics", "price": 100.0 },
            { "category": "electronics", "price": 200.0 },
            { "category": "books",       "price": 10.0  }
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        201,
        "insert failed: {}",
        r.text().await.unwrap()
    );
}

#[tokio::test]
async fn test_execute_aggregate_tool() {
    let (base, _tmp) = start_test_server().await;
    seed_collection(&base).await;

    // `{ "args": {...} }` form.
    let resp = client()
        .post(format!("{base}/v1/agent-tools/aggregate/execute"))
        .json(&json!({ "args": {
            "collection": "agg_test", "field": "price", "metric": "sum"
        }}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 310.0).abs() < 0.001, "sum expected 310, got {val}");
}

#[tokio::test]
async fn test_execute_list_collections_tool_direct_args() {
    let (base, _tmp) = start_test_server().await;
    seed_collection(&base).await;

    // Args object directly (no "args" wrapper).
    let resp = client()
        .post(format!("{base}/v1/agent-tools/list_collections/execute"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let cols = body["collections"].as_array().unwrap();
    assert!(cols.iter().any(|c| c.as_str() == Some("agg_test")));
}

#[tokio::test]
async fn test_execute_tool_not_found_maps_404() {
    let (base, _tmp) = start_test_server().await;
    // describe_collection on a missing collection → NotFound → 404.
    let resp = client()
        .post(format!("{base}/v1/agent-tools/describe_collection/execute"))
        .json(&json!({ "args": { "name": "no_such" } }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── Converse without an LLM → 503 ────────────────────────────────────────

#[tokio::test]
async fn test_converse_without_llm_returns_503() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/agents/veda-agent/converse"))
        .json(&json!({ "input": "hello" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "{}", resp.text().await.unwrap());
}

// ── Conversation GET / list / delete ─────────────────────────────────────

#[tokio::test]
async fn test_conversation_get_list_delete() {
    let (base, _tmp) = start_test_server().await;

    // No converse endpoint without an LLM, so seed a conversation by
    // persisting one through the saved-object store directly would require
    // internal access; instead drive the public CRUD: list is empty,
    // GET/DELETE of a missing conversation behave correctly.
    let resp = client()
        .get(format!("{base}/v1/conversations"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let list: Value = resp.json().await.unwrap();
    assert_eq!(list["conversations"].as_array().unwrap().len(), 0);

    // GET missing → 404.
    let resp = client()
        .get(format!("{base}/v1/conversations/nope"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // DELETE missing → 204 (idempotent).
    let resp = client()
        .delete(format!("{base}/v1/conversations/nope"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
}

#[tokio::test]
async fn test_feedback_unknown_conversation_404_and_bad_rating_400() {
    let (base, _tmp) = start_test_server().await;

    // Feedback on a missing conversation → 404 (route is wired).
    let resp = client()
        .post(format!("{base}/v1/conversations/nope/feedback"))
        .json(&json!({ "rating": "up" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Invalid rating → 400 (validation runs before the lookup).
    let resp = client()
        .post(format!("{base}/v1/conversations/whatever/feedback"))
        .json(&json!({ "rating": "sideways" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}
