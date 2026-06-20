//! Integration tests for streaming `/converse` over Server-Sent Events
//! (`POST /v1/agents/{id}/converse/stream`).
//!
//! Covers:
//!   - No LLM configured → 503 *before* any streaming starts.
//!   - The blocking `/converse` is unchanged (still 503 with no LLM).
//!   - A full streaming run against a mock OpenAI-compatible SSE endpoint:
//!     ordered named SSE frames (`reasoning` / `tool_call` / `tool_result`
//!     / `answer`) terminating in a `done` frame carrying metrics + the
//!     conversation id, and the conversation is persisted.

use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Mutex;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

/// Serializes the env-var-driven mock-LLM tests so they don't race on the
/// process-global `VEDA_LLM_*` env.
static LLM_ENV_LOCK: Mutex<()> = Mutex::new(());

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

/// Start a server whose `AppState` has loaded its LLM config from the
/// (caller-set) `VEDA_LLM_*` env vars.
async fn start_test_server_with_llm() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    state.load_llm_from_env().await; // reads env set by the caller
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

/// A mock OpenAI-compatible `chat/completions` server. The streaming
/// executor drives its tool-calling turns with the blocking (JSON)
/// `complete_with_tools`, so this mock returns a single non-streaming JSON
/// completion with a final answer and no tool calls. The gateway then
/// streams that answer to the SSE client as multiple `answer` deltas.
async fn start_mock_llm() -> (String, tokio::task::JoinHandle<()>) {
    use axum::Router;
    use axum::routing::post;

    async fn completions() -> axum::Json<Value> {
        axum::Json(json!({
            "choices": [{
                "message": { "role": "assistant", "content": "Veda uses HNSW." },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }
        }))
    }

    let router = Router::new().route("/v1/chat/completions", post(completions));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/v1/chat/completions");
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, handle)
}

fn client() -> Client {
    Client::new()
}

#[tokio::test]
async fn test_converse_stream_without_llm_returns_503() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/agents/veda-agent/converse/stream"))
        .json(&json!({ "input": "hello" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "{}", resp.text().await.unwrap());
}

#[tokio::test]
async fn test_blocking_converse_still_503_without_llm() {
    // The blocking path is untouched and still returns 503 with no LLM.
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/agents/veda-agent/converse"))
        .json(&json!({ "input": "hello" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn test_converse_stream_emits_ordered_frames_ending_in_done() {
    // Spin a mock OpenAI SSE endpoint, point the gateway's LLM at it via
    // env (serialized), and assert the streamed frames.
    let (llm_url, _llm_handle) = start_mock_llm().await;

    let base = {
        let _guard = LLM_ENV_LOCK.lock().unwrap();
        // SAFETY: tests are serialized on LLM_ENV_LOCK; no concurrent env
        // access from other LLM-env tests.
        unsafe {
            std::env::set_var("VEDA_LLM_ENDPOINT", &llm_url);
            std::env::set_var("VEDA_LLM_MODEL", "mock-model");
        }
        let (base, tmp) = start_test_server_with_llm().await;
        unsafe {
            std::env::remove_var("VEDA_LLM_ENDPOINT");
            std::env::remove_var("VEDA_LLM_MODEL");
        }
        // Keep tmp alive for the duration of the test.
        std::mem::forget(tmp);
        base
    };

    let resp = client()
        .post(format!("{base}/v1/agents/veda-agent/converse/stream"))
        .json(&json!({ "input": "what index does veda use?" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(ct.starts_with("text/event-stream"), "content-type: {ct}");

    let body = resp.text().await.unwrap();

    // Parse the SSE frames: collect (event, data) pairs.
    let mut frames: Vec<(String, String)> = Vec::new();
    let mut cur_event: Option<String> = None;
    for line in body.lines() {
        if let Some(ev) = line.strip_prefix("event:") {
            cur_event = Some(ev.trim().to_owned());
        } else if let Some(data) = line.strip_prefix("data:")
            && let Some(ev) = cur_event.take()
        {
            frames.push((ev, data.trim().to_owned()));
        }
    }

    assert!(!frames.is_empty(), "no SSE frames parsed from:\n{body}");

    // At least one `answer` frame, and the last frame is `done`.
    let answer_count = frames.iter().filter(|(e, _)| e == "answer").count();
    assert!(
        answer_count >= 1,
        "expected >=1 answer frame, frames: {frames:?}"
    );

    let (last_event, last_data) = frames.last().unwrap();
    assert_eq!(last_event, "done", "last frame must be `done`: {frames:?}");

    // Concatenated answer deltas reconstruct the model's text.
    let answer: String = frames
        .iter()
        .filter(|(e, _)| e == "answer")
        .map(|(_, d)| {
            let v: Value = serde_json::from_str(d).unwrap();
            v["text"].as_str().unwrap_or("").to_owned()
        })
        .collect();
    assert_eq!(answer, "Veda uses HNSW.");

    // The `done` payload carries metrics + a conversation id.
    let done: Value = serde_json::from_str(last_data).unwrap();
    assert_eq!(done["response"], "Veda uses HNSW.");
    assert!(
        done["conversation_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
    assert_eq!(done["metrics"]["total_tokens"], 10);

    // The conversation was persisted and is fetchable.
    let cid = done["conversation_id"].as_str().unwrap();
    let resp = client()
        .get(format!("{base}/v1/conversations/{cid}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "conversation should be persisted");
    let conv: Value = resp.json().await.unwrap();
    let msgs = conv["messages"].as_array().unwrap();
    assert!(
        msgs.iter().any(|m| m["role"] == "assistant"),
        "persisted conversation should carry the assistant turn: {conv}"
    );
}
