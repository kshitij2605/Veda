//! Integration tests: lexical (BM25 / match_phrase / hybrid) search paths
//! must honor the request scalar `filter` AND the RLS-injected `filter_expr`.
//!
//! Regression coverage for the bug where the built-in BM25 leg ignored the
//! request filter, so scalar filtering and row-level security were not
//! applied to full-text queries (only the dense path enforced them).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::{generate_token, Role};
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

// ── Harness ──────────────────────────────────────────────────────────────

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert("admin".into(), UserRecord { password: "admin".into(), role: Role::Admin });
    let state = AppState::with_auth_and_users(tmp.path(), "bm25-rls-secret", users);
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

/// Create a collection with a bm25 `message` field and a filterable `level`
/// field, then insert a fixed mix of ERROR / INFO records that all share the
/// term "exploded" in their message.
async fn create_and_seed(base: &str, name: &str) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "message", "type": "string", "bm25": true },
                { "name": "level",   "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create collection: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({
            "records": [
                { "message": "alpha exploded badly",   "level": "ERROR" },
                { "message": "beta exploded too",       "level": "ERROR" },
                { "message": "gamma exploded slightly", "level": "INFO"  },
                { "message": "delta exploded later",    "level": "INFO"  },
                { "message": "epsilon exploded once",   "level": "INFO"  }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());
}

fn levels(body: &Value) -> Vec<String> {
    body["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|h| h["fields"]["level"].as_str().map(str::to_owned))
        .collect()
}

// ── 1. BM25 + scalar filter (core regression) ─────────────────────────────

#[tokio::test]
async fn bm25_honors_scalar_filter() {
    let (base, _tmp) = start_server().await;
    create_and_seed(&base, "bm25_filter").await;

    // Without the filter: all 5 records share "exploded".
    let r = client()
        .post(format!("{base}/v1/collections/bm25_filter/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "exploded" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["results"].as_array().unwrap().len(), 5, "no-filter: {body}");

    // With level = 'ERROR': only the 2 ERROR records survive, and they must
    // actually contain the term.
    let r = client()
        .post(format!("{base}/v1/collections/bm25_filter/search"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "exploded" } },
            "filter": "level = 'ERROR'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "filtered search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let lv = levels(&body);
    assert_eq!(lv.len(), 2, "expected exactly 2 ERROR hits; got {body}");
    assert!(lv.iter().all(|l| l == "ERROR"), "all hits must be ERROR; got {lv:?}");
    for h in body["results"].as_array().unwrap() {
        assert!(
            h["fields"]["message"].as_str().unwrap().contains("exploded"),
            "hit must contain the term; got {h}"
        );
    }
}

// ── 2. match_phrase + scalar filter ────────────────────────────────────────

#[tokio::test]
async fn match_phrase_honors_scalar_filter() {
    let (base, _tmp) = start_server().await;
    create_and_seed(&base, "phrase_filter").await;

    // Phrase "exploded" present in all; filter to INFO -> 3 survivors.
    let r = client()
        .post(format!("{base}/v1/collections/phrase_filter/search"))
        .json(&json!({
            "query": { "match_phrase": { "field": "message", "query": "exploded", "slop": 0 } },
            "filter": "level = 'INFO'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "phrase search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let lv = levels(&body);
    assert_eq!(lv.len(), 3, "expected 3 INFO hits; got {body}");
    assert!(lv.iter().all(|l| l == "INFO"), "all hits must be INFO; got {lv:?}");
}

// ── 3. Hybrid: BM25 leg is filtered ────────────────────────────────────────
//
// No live embedder is required: a hybrid query supplies the dense vector
// directly. The collection has a dense field so the dense leg runs against a
// flat index; the BM25 leg must drop non-matching records. We assert the
// fused result set contains only ERROR records.

#[tokio::test]
async fn hybrid_bm25_leg_honors_scalar_filter() {
    let (base, _tmp) = start_server().await;
    let name = "hybrid_filter";

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "emb",     "type": "vector", "dimension": 3, "metric": "cosine" },
                { "name": "message", "type": "string", "bm25": true },
                { "name": "level",   "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({
            "records": [
                { "emb": [1.0, 0.0, 0.0], "message": "alpha exploded", "level": "ERROR" },
                { "emb": [0.9, 0.1, 0.0], "message": "beta exploded",  "level": "ERROR" },
                { "emb": [0.0, 1.0, 0.0], "message": "gamma exploded", "level": "INFO"  },
                { "emb": [0.0, 0.0, 1.0], "message": "delta exploded", "level": "INFO"  }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/search"))
        .json(&json!({
            "query": {
                "hybrid": {
                    "dense": { "field": "emb", "values": [1.0, 0.0, 0.0] },
                    "bm25":  { "field": "message", "query": "exploded" }
                }
            },
            "filter": "level = 'ERROR'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "hybrid search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let lv = levels(&body);
    assert!(!lv.is_empty(), "hybrid must return survivors; got {body}");
    assert!(
        lv.iter().all(|l| l == "ERROR"),
        "every hybrid hit must be ERROR (both legs filtered); got {lv:?} / {body}"
    );
}

// ── 4. RLS: BM25 search only returns the caller's own records ──────────────

#[tokio::test]
async fn bm25_enforces_rls() {
    let (base, _tmp) = start_auth_server().await;

    // admin login
    let admin = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "admin" }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned();

    // RLS-enabled collection.
    let r = client()
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {admin}"))
        .json(&json!({
            "name": "bm25_rls",
            "fields": [
                { "name": "message", "type": "string", "bm25": true },
                { "name": "owner",   "type": "string", "filterable": true }
            ],
            "rls_owner_field": "owner"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // Admin seeds records owned by alice and by bob (both mention "exploded").
    let r = client()
        .post(format!("{base}/v1/collections/bm25_rls/insert"))
        .header("Authorization", format!("Bearer {admin}"))
        .json(&json!({
            "records": [
                { "message": "alice exploded thing", "owner": "alice" },
                { "message": "bob exploded thing",   "owner": "bob"   }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());

    // Non-admin alice token (no user-store entry needed).
    let alice = generate_token("alice", Role::Editor, "bm25-rls-secret").unwrap();

    let r = client()
        .post(format!("{base}/v1/collections/bm25_rls/search"))
        .header("Authorization", format!("Bearer {alice}"))
        .json(&json!({
            "query": { "bm25": { "field": "message", "query": "exploded" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "alice search: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    let owners: Vec<String> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|h| h["fields"]["owner"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(owners.len(), 1, "alice must see exactly her record; got {body}");
    assert_eq!(owners[0], "alice", "RLS leak: alice saw {owners:?}");
}
