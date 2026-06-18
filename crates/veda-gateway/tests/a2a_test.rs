//! Integration tests for the A2A (Agent-to-Agent) endpoint.
//!
//! Covers:
//!   - GET  /v1/a2a/veda-agent/card        → AgentCard shape + skills
//!   - GET  /v1/a2a/{unknown}/card         → 404
//!   - POST /v1/a2a/veda-agent (no LLM)    → 503 for a valid message/send
//!   - POST /v1/a2a/veda-agent (bad method)→ JSON-RPC -32601
//!
//! The full message/send loop is covered by the executor unit tests; here
//! we only need the deterministic, LLM-free paths.

use reqwest::Client;
use serde_json::{json, Value};
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

#[tokio::test]
async fn test_a2a_agent_card_default_agent() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .get(format!("{base}/v1/a2a/veda-agent/card"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let card: Value = resp.json().await.unwrap();

    assert_eq!(card["name"], "Veda Agent");
    assert!(card["version"].as_str().is_some());
    assert_eq!(card["capabilities"]["streaming"], false);
    assert_eq!(card["capabilities"]["pushNotifications"], false);
    assert_eq!(card["url"], "/v1/a2a/veda-agent");

    // Default agent has an empty tool whitelist → all 13 builtins as skills.
    let skills = card["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 13);
    let ids: Vec<&str> = skills.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"search"));
    assert!(ids.contains(&"aggregate"));
    // Each skill carries a name + description + tags.
    assert!(skills[0]["name"].as_str().is_some());
    assert!(skills[0]["description"].as_str().is_some());
    assert_eq!(skills[0]["tags"][0], "tool");
}

#[tokio::test]
async fn test_a2a_agent_card_unknown_agent_404() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/a2a/no-such-agent/card"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_a2a_message_send_without_llm_returns_503() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/a2a/veda-agent"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [ { "kind": "text", "text": "hello" } ]
                }
            }
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 503, "{}", resp.text().await.unwrap());
}

#[tokio::test]
async fn test_a2a_unknown_method_returns_jsonrpc_minus_32601() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/a2a/veda-agent"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tasks/get",
            "params": {}
        }))
        .send().await.unwrap();
    // Unknown method is a JSON-RPC error returned with HTTP 200.
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["id"], 7);
    assert_eq!(body["error"]["code"], -32601);
}
