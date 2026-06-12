//! Integration tests for Feature D: Row-Level Security.
//!
//! Spin up a real HTTP server with JWT auth and verify:
//!   1. alice sees only her records; bob sees only his (read isolation)
//!   2. non-admin cannot insert a record with a different owner (write enforcement)
//!   3. Admin user sees ALL records (admin bypass)
//!   4. collection without rls_owner_field behaves as before (backward compat)

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::state::{AppState, UserRecord};
use veda_gateway::rest::build_router;

async fn start_rls_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert("admin".into(), UserRecord { password: "admin".into(), role: Role::Admin });
    users.insert("alice".into(), UserRecord { password: "alice".into(), role: Role::Editor });
    users.insert("bob".into(),   UserRecord { password: "bob".into(),   role: Role::Editor });
    let state = AppState::with_auth_and_users(tmp.path(), "rls-test-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

async fn login_user(base: &str, user: &str, pass: &str) -> String {
    let resp = Client::new()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": user, "password": pass }))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "login {user} failed");
    resp.json::<Value>().await.unwrap()["token"].as_str().unwrap().to_owned()
}

fn post_req(base: &str, path: &str, tok: &str) -> reqwest::RequestBuilder {
    Client::new()
        .post(format!("{base}{path}"))
        .header("Authorization", format!("Bearer {tok}"))
}

fn get_req(base: &str, path: &str, tok: &str) -> reqwest::RequestBuilder {
    Client::new()
        .get(format!("{base}{path}"))
        .header("Authorization", format!("Bearer {tok}"))
}

// ── 1. alice and bob see only their own records ──────────────────────────────

#[tokio::test]
async fn rls_read_isolation() {
    let (base, _tmp) = start_rls_server().await;
    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;
    let bob   = login_user(&base, "bob",   "bob").await;

    let r = post_req(&base, "/v1/collections", &admin)
        .json(&json!({
            "name": "rls_col1",
            "fields": [
                { "name": "emb",     "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "owner",   "type": "string", "filterable": true },
                { "name": "content", "type": "string" }
            ],
            "rls_owner_field": "owner"
        })).send().await.unwrap();
    assert_eq!(r.status(), 201, "{}", r.text().await.unwrap());

    // Alice inserts — owner auto-set to "alice" (field omitted)
    let r = post_req(&base, "/v1/collections/rls_col1/insert", &alice)
        .json(&json!({ "records": [{ "emb": [0.1,0.2,0.3,0.4], "content": "alice-doc" }] }))
        .send().await.unwrap();
    assert_eq!(r.status(), 201, "alice insert: {}", r.text().await.unwrap());

    // Bob inserts — owner auto-set to "bob"
    let r = post_req(&base, "/v1/collections/rls_col1/insert", &bob)
        .json(&json!({ "records": [{ "emb": [0.5,0.6,0.7,0.8], "content": "bob-doc" }] }))
        .send().await.unwrap();
    assert_eq!(r.status(), 201, "bob insert: {}", r.text().await.unwrap());

    // Alice list -> must see exactly 1 record with owner == "alice"
    let r = get_req(&base, "/v1/collections/rls_col1/records", &alice).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "alice: expected 1, got {:?}", body);
    assert_eq!(recs[0]["fields"]["owner"].as_str().unwrap(), "alice");

    // Bob list -> must see exactly 1 record with owner == "bob"
    let r = get_req(&base, "/v1/collections/rls_col1/records", &bob).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    let recs = body["records"].as_array().unwrap();
    assert_eq!(recs.len(), 1, "bob: expected 1, got {:?}", body);
    assert_eq!(recs[0]["fields"]["owner"].as_str().unwrap(), "bob");
}

// ── 2. non-admin cannot spoof owner ──────────────────────────────────────────

#[tokio::test]
async fn rls_write_enforcement() {
    let (base, _tmp) = start_rls_server().await;
    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;

    let r = post_req(&base, "/v1/collections", &admin)
        .json(&json!({
            "name": "rls_col2",
            "fields": [
                { "name": "emb",   "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "owner", "type": "string", "filterable": true }
            ],
            "rls_owner_field": "owner"
        })).send().await.unwrap();
    assert_eq!(r.status(), 201);

    // Alice claims owner = "bob" -> 403
    let r = post_req(&base, "/v1/collections/rls_col2/insert", &alice)
        .json(&json!({ "records": [{ "emb": [0.1,0.2,0.3,0.4], "owner": "bob" }] }))
        .send().await.unwrap();
    assert_eq!(r.status(), 403, "expected 403, got: {}", r.text().await.unwrap());
}

// ── 3. Admin bypasses RLS and sees all records ───────────────────────────────

#[tokio::test]
async fn rls_admin_bypass() {
    let (base, _tmp) = start_rls_server().await;
    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;
    let bob   = login_user(&base, "bob",   "bob").await;

    let r = post_req(&base, "/v1/collections", &admin)
        .json(&json!({
            "name": "rls_col3",
            "fields": [
                { "name": "emb",   "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "owner", "type": "string", "filterable": true }
            ],
            "rls_owner_field": "owner"
        })).send().await.unwrap();
    assert_eq!(r.status(), 201);

    post_req(&base, "/v1/collections/rls_col3/insert", &alice)
        .json(&json!({ "records": [{ "emb": [0.1,0.2,0.3,0.4] }] }))
        .send().await.unwrap();
    post_req(&base, "/v1/collections/rls_col3/insert", &bob)
        .json(&json!({ "records": [{ "emb": [0.5,0.6,0.7,0.8] }] }))
        .send().await.unwrap();

    // Admin sees 2 records (total)
    let r = get_req(&base, "/v1/collections/rls_col3/records", &admin).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["total"].as_u64().unwrap_or(0), 2, "admin: expected 2, got {:?}", body);
}

// ── 4. collection WITHOUT rls_owner_field: backward compat ───────────────────

#[tokio::test]
async fn rls_no_field_backward_compat() {
    let (base, _tmp) = start_rls_server().await;
    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;
    let bob   = login_user(&base, "bob",   "bob").await;

    let r = post_req(&base, "/v1/collections", &admin)
        .json(&json!({
            "name": "norls_col",
            "fields": [
                { "name": "emb", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "tag", "type": "string" }
            ]
        })).send().await.unwrap();
    assert_eq!(r.status(), 201);

    post_req(&base, "/v1/collections/norls_col/insert", &alice)
        .json(&json!({ "records": [{ "emb": [0.1,0.2,0.3,0.4], "tag": "alice" }] }))
        .send().await.unwrap();
    post_req(&base, "/v1/collections/norls_col/insert", &bob)
        .json(&json!({ "records": [{ "emb": [0.5,0.6,0.7,0.8], "tag": "bob" }] }))
        .send().await.unwrap();

    // Alice sees both records (no RLS filter applied)
    let r = get_req(&base, "/v1/collections/norls_col/records", &alice).send().await.unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["total"].as_u64().unwrap_or(0), 2, "no-rls: expected 2, got {:?}", body);
}
