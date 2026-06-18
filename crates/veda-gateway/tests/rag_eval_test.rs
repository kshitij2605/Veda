//! Integration tests for the RAG/agent eval harness REST surface.
//!
//! Covers (no LLM required for any of these):
//!   - Eval-set CRUD over HTTP (create → get → list → update → delete)
//!   - Tenant isolation of eval sets (X-Veda-Tenant scoping)
//!   - POST /v1/eval-sets/{id}/run with no LLM configured → 503
//!   - Flywheel: from-conversation builds a case with query +
//!     expected_chunk_ids + gold_answer from an explicit turn
//!   - GET /v1/eval-sets/{id}/runs is empty before any run

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
async fn test_eval_set_crud_round_trip() {
    let (base, _tmp) = start_test_server().await;

    // Create (no id → server assigns one).
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({
            "id": "",
            "name": "Golden set",
            "agent_id": "veda-agent",
            "cases": [
                { "id": "c1", "query": "what index does veda use?",
                  "expected_chunk_ids": ["id-1"], "gold_answer": "HNSW" }
            ]
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: Value = resp.json().await.unwrap();
    let id = created["id"].as_str().unwrap().to_owned();
    assert!(!id.is_empty());
    assert!(!id.contains("__"), "returned id leaked tenant prefix: {id}");
    assert_eq!(created["name"], "Golden set");
    assert_eq!(created["cases"][0]["expected_chunk_ids"][0], "id-1");

    // Get.
    let resp = client().get(format!("{base}/v1/eval-sets/{id}")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let got: Value = resp.json().await.unwrap();
    assert_eq!(got["id"], id);
    assert_eq!(got["cases"][0]["query"], "what index does veda use?");

    // List.
    let resp = client().get(format!("{base}/v1/eval-sets")).send().await.unwrap();
    let list: Value = resp.json().await.unwrap();
    let sets = list["eval_sets"].as_array().unwrap();
    assert_eq!(sets.len(), 1);
    assert_eq!(sets[0]["id"], id);

    // Update (PUT).
    let resp = client()
        .put(format!("{base}/v1/eval-sets/{id}"))
        .json(&json!({ "id": id, "name": "Golden set v2", "cases": [] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let updated: Value = resp.json().await.unwrap();
    assert_eq!(updated["name"], "Golden set v2");
    assert_eq!(updated["cases"].as_array().unwrap().len(), 0);

    // Delete.
    let resp = client().delete(format!("{base}/v1/eval-sets/{id}")).send().await.unwrap();
    assert_eq!(resp.status(), 204);
    let resp = client().get(format!("{base}/v1/eval-sets/{id}")).send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_create_assigns_ids_when_absent() {
    let (base, _tmp) = start_test_server().await;

    // Truly OMIT the set id and the case ids (the real client case).
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({
            "name": "No ids",
            "cases": [
                { "query": "what index does veda use?", "expected_chunk_ids": ["id-1"] },
                { "query": "another", "expected_chunk_ids": [] }
            ]
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: Value = resp.json().await.unwrap();

    let set_id = created["id"].as_str().unwrap();
    assert!(set_id.starts_with("eval-"), "server should assign set id, got {set_id}");

    let cases = created["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for c in cases {
        let cid = c["id"].as_str().unwrap();
        assert!(cid.starts_with("case-"), "server should assign case id, got {cid}");
    }
}

#[tokio::test]
async fn test_create_preserves_explicit_ids_and_update_preserves() {
    let (base, _tmp) = start_test_server().await;

    // Explicit case id is preserved.
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({
            "id": "explicit-set",
            "name": "Explicit",
            "cases": [{ "id": "c1", "query": "q", "expected_chunk_ids": [] }]
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: Value = resp.json().await.unwrap();
    assert_eq!(created["id"], "explicit-set");
    assert_eq!(created["cases"][0]["id"], "c1");

    // Update preserves the explicit case id and assigns one for the new case.
    let resp = client()
        .put(format!("{base}/v1/eval-sets/explicit-set"))
        .json(&json!({
            "id": "explicit-set",
            "name": "Explicit v2",
            "cases": [
                { "id": "c1", "query": "q", "expected_chunk_ids": [] },
                { "query": "new", "expected_chunk_ids": [] }
            ]
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let updated: Value = resp.json().await.unwrap();
    assert_eq!(updated["cases"][0]["id"], "c1");
    let new_id = updated["cases"][1]["id"].as_str().unwrap();
    assert!(new_id.starts_with("case-"), "new case should get an id, got {new_id}");
}

#[tokio::test]
async fn test_eval_set_tenant_isolation() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .header("X-Veda-Tenant", "acme")
        .json(&json!({ "id": "shared", "name": "Acme set", "cases": [] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    // rival sees nothing.
    let resp = client()
        .get(format!("{base}/v1/eval-sets"))
        .header("X-Veda-Tenant", "rival")
        .send().await.unwrap();
    let list: Value = resp.json().await.unwrap();
    assert_eq!(list["eval_sets"].as_array().unwrap().len(), 0);

    let resp = client()
        .get(format!("{base}/v1/eval-sets/shared"))
        .header("X-Veda-Tenant", "rival")
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);

    // acme still sees it.
    let resp = client()
        .get(format!("{base}/v1/eval-sets/shared"))
        .header("X-Veda-Tenant", "acme")
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_run_eval_without_llm_returns_503() {
    let (base, _tmp) = start_test_server().await;

    // A set with one case so the run actually attempts a turn.
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({
            "id": "s1", "name": "set",
            "cases": [{ "id": "c1", "query": "hello", "expected_chunk_ids": [] }]
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .post(format!("{base}/v1/eval-sets/s1/run"))
        .json(&json!({}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 503, "{}", resp.text().await.unwrap());
}

#[tokio::test]
async fn test_runs_list_empty_before_any_run() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({ "id": "s1", "name": "set", "cases": [] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client().get(format!("{base}/v1/eval-sets/s1/runs")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["runs"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_flywheel_from_explicit_turn_adds_case() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({ "id": "s1", "name": "set", "cases": [] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    // Graduate an explicit turn (no conversation needed).
    let resp = client()
        .post(format!("{base}/v1/eval-sets/s1/cases/from-conversation"))
        .json(&json!({
            "query": "what index does veda use?",
            "retrieved_chunk_ids": ["id-1", "id-2"],
            "answer": "Veda uses HNSW."
        }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let set: Value = resp.json().await.unwrap();
    let cases = set["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0]["query"], "what index does veda use?");
    assert_eq!(cases[0]["expected_chunk_ids"][0], "id-1");
    assert_eq!(cases[0]["expected_chunk_ids"][1], "id-2");
    assert_eq!(cases[0]["gold_answer"], "Veda uses HNSW.");
}

#[tokio::test]
async fn test_flywheel_requires_query_or_conversation() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .post(format!("{base}/v1/eval-sets"))
        .json(&json!({ "id": "s1", "name": "set", "cases": [] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    // Empty body → 400 (neither conversation_id nor query).
    let resp = client()
        .post(format!("{base}/v1/eval-sets/s1/cases/from-conversation"))
        .json(&json!({}))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400);
}
