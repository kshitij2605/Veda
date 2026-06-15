//! Integration tests for the Log Insights REST API
//! (`/v1/admin/logs` + `/v1/admin/slow-queries`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::logs::LogRecord;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server over a fresh temp dir. Returns the base URL,
/// a clone of the `AppState` (so tests can seed the log buffer
/// directly), and the TempDir guard.
async fn start_server() -> (String, AppState, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, app_state, tmp)
}

async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert("admin".to_owned(), UserRecord::with_password("admin", Role::Admin));
    users.insert("viewer".to_owned(), UserRecord::with_password("viewer", Role::Viewer));
    let app_state = AppState::with_auth_and_users(tmp.path(), "test-jwt-secret", users);
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
}

async fn login(base: &str, user: &str, pass: &str) -> String {
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": user, "password": pass }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "login failed for {user}");
    let body: Value = r.json().await.unwrap();
    body["token"].as_str().unwrap().to_owned()
}

fn rec(level: &str, target: &str, message: &str) -> LogRecord {
    LogRecord {
        ts_ms: 0,
        level: level.to_string(),
        target: target.to_string(),
        message: message.to_string(),
        fields: serde_json::Map::new(),
        trace_id: None,
        span_id: None,
    }
}

#[tokio::test]
async fn logs_returns_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/logs")).send().await.unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["logs"].is_array());
    assert_eq!(body["logs"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn logs_returns_buffered_records_newest_first() {
    let (base, state, _tmp) = start_server().await;
    state.log_buffer().push(rec("INFO", "mod_a", "first"));
    state.log_buffer().push(rec("WARN", "mod_b", "second"));

    let c = Client::new();
    let r = c.get(format!("{base}/v1/admin/logs")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let logs = body["logs"].as_array().unwrap();
    assert_eq!(logs.len(), 2);
    // newest-first
    assert_eq!(logs[0]["message"], json!("second"));
    assert_eq!(logs[1]["message"], json!("first"));
}

#[tokio::test]
async fn logs_level_filter_keeps_at_or_above_severity() {
    let (base, state, _tmp) = start_server().await;
    state.log_buffer().push(rec("INFO", "m", "info-line"));
    state.log_buffer().push(rec("WARN", "m", "warn-line"));
    state.log_buffer().push(rec("ERROR", "m", "error-line"));

    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/logs?level=ERROR"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let logs = body["logs"].as_array().unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["message"], json!("error-line"));
}

#[tokio::test]
async fn logs_q_substring_filter_works() {
    let (base, state, _tmp) = start_server().await;
    state.log_buffer().push(rec("INFO", "m", "connection refused"));
    state.log_buffer().push(rec("INFO", "m", "all good"));

    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/logs?q=connection"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let logs = body["logs"].as_array().unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["message"], json!("connection refused"));
}

#[tokio::test]
async fn slow_queries_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/slow-queries"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["slow_queries"].is_array());
    assert_eq!(body["slow_queries"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn slow_queries_sorted_by_took_ms_desc() {
    let (base, _state, tmp) = start_server().await;
    let c = Client::new();
    // Create a collection via the API so it is registered in `schemas`
    // (scoped as `default__widgets` for the default tenant).
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "widgets",
            "fields": [{ "name": "category", "type": "string", "filterable": true }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create failed: {:?}", r.text().await);

    // Seed the per-collection query log at the tenant-scoped path.
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let scoped = "default__widgets";
    let dir = tmp.path().join(scoped);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("_query_log.jsonl");
    let lines = [
        json!({"ts_ms": now_ms, "top_k": 5, "doc_ids": ["a"], "took_ms": 12, "trace_id": null}),
        json!({"ts_ms": now_ms, "top_k": 5, "doc_ids": ["b"], "took_ms": 99, "trace_id": "t99"}),
        json!({"ts_ms": now_ms, "top_k": 5, "doc_ids": ["c"], "took_ms": 50, "trace_id": null}),
    ];
    let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(&log, body).unwrap();

    let r = c
        .get(format!("{base}/v1/admin/slow-queries"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    let sq = body["slow_queries"].as_array().unwrap();
    assert_eq!(sq.len(), 3);
    assert_eq!(sq[0]["took_ms"], json!(99));
    assert_eq!(sq[1]["took_ms"], json!(50));
    assert_eq!(sq[2]["took_ms"], json!(12));
    assert_eq!(sq[0]["collection"], json!("widgets"));
    assert_eq!(sq[0]["trace_id"], json!("t99"));
}

#[tokio::test]
async fn logs_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/logs"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/logs"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
