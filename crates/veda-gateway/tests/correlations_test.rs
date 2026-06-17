//! Integration tests for the correlation analyzer REST API
//! (`GET /v1/admin/correlations`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server over a fresh temp dir. Returns the base URL and
/// the TempDir guard (tests seed `_query_log.jsonl` directly on it).
async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
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

/// Register `widgets` and seed its query log with a clear pattern:
/// `hybrid` queries are much slower and some of them error; `dense`
/// queries are fast and always succeed.
async fn seed(base: &str, tmp: &tempfile::TempDir) {
    let c = Client::new();
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

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let scoped = "default__widgets";
    let dir = tmp.path().join(scoped);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("_query_log.jsonl");

    let mk = |qt: &str, took: u64, status: &str| {
        json!({
            "ts_ms": now_ms,
            "top_k": 10,
            "doc_ids": [],
            "took_ms": took,
            "collection": "widgets",
            "query_type": qt,
            "has_filter": false,
            "reranked": false,
            "result_count": 3,
            "status": status,
            "error_kind": if status == "error" { json!("internal") } else { json!(null) },
            "index_type": "hnsw"
        })
    };

    let lines = [
        mk("dense", 8, "ok"),
        mk("dense", 12, "ok"),
        mk("dense", 10, "ok"),
        mk("hybrid", 2200, "ok"),
        mk("hybrid", 2600, "ok"),
        mk("hybrid", 3000, "error"),
    ];
    let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(&log, body).unwrap();
}

#[tokio::test]
async fn correlations_empty_on_fresh_server() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/correlations"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["summary"]["total_count"], json!(0));
    assert!(body["dimensions"].is_array());
}

#[tokio::test]
async fn correlations_ranks_slow_and_errored_bucket_first() {
    let (base, tmp) = start_server().await;
    seed(&base, &tmp).await;

    let c = Client::new();
    let r = c
        // wide window so the just-written entries are in range.
        .get(format!("{base}/v1/admin/correlations?window_secs=86400"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();

    // summary: 6 entries, 1 error.
    assert_eq!(body["summary"]["total_count"], json!(6));
    assert_eq!(body["summary"]["error_count"], json!(1));
    assert!(body["summary"]["p95_ms"].as_f64().unwrap() >= 2600.0);

    // query_type dimension: hybrid (slow + errored) must rank first.
    let dims = body["dimensions"].as_array().unwrap();
    let qt = dims
        .iter()
        .find(|d| d["dimension"] == json!("query_type"))
        .expect("query_type dimension present");
    let buckets = qt["buckets"].as_array().unwrap();
    assert_eq!(buckets[0]["key"], json!("hybrid"), "hybrid ranked first");
    assert!(buckets[0]["error_rate"].as_f64().unwrap() > 0.0);
    assert!(buckets[0]["p95_ms"].as_f64().unwrap() >= 2600.0);
    assert!(buckets[0]["slow_count"].as_u64().unwrap() >= 3);
    // dense bucket present and fast.
    let dense = buckets
        .iter()
        .find(|b| b["key"] == json!("dense"))
        .expect("dense bucket present");
    assert_eq!(dense["error_count"], json!(0));
    assert!(dense["p95_ms"].as_f64().unwrap() < 100.0);
}

#[tokio::test]
async fn correlations_single_dimension_filter() {
    let (base, tmp) = start_server().await;
    seed(&base, &tmp).await;

    let c = Client::new();
    let r = c
        .get(format!(
            "{base}/v1/admin/correlations?window_secs=86400&dimension=query_type"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let dims = body["dimensions"].as_array().unwrap();
    assert_eq!(dims.len(), 1);
    assert_eq!(dims[0]["dimension"], json!("query_type"));
}

/// Read + parse the per-collection query log, polling briefly because
/// the write is spawned (`tokio::spawn_blocking`) off the request path.
async fn read_log(tmp: &tempfile::TempDir, scoped: &str) -> Vec<Value> {
    let path = tmp.path().join(scoped).join("_query_log.jsonl");
    for _ in 0..50 {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let lines: Vec<Value> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            if !lines.is_empty() {
                return lines;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    Vec::new()
}

#[tokio::test]
async fn successful_search_logs_enriched_ok_entry() {
    let (base, tmp) = start_server().await;
    let c = Client::new();
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "qcol",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    let records: Vec<Value> = (0..5)
        .map(|i| json!({ "title": format!("Item {i}"), "embedding": [1.0, i as f64 * 0.1, 0.0, 0.0] }))
        .collect();
    c.post(format!("{base}/v1/collections/qcol/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    let r = c
        .post(format!("{base}/v1/collections/qcol/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0] } },
            "filter": "title = \"Item 1\"",
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);

    let entries = read_log(&tmp, "default__qcol").await;
    assert_eq!(entries.len(), 1, "exactly one query-log write per request");
    let e = &entries[0];
    assert_eq!(e["status"], json!("ok"));
    assert_eq!(e["query_type"], json!("dense"));
    assert_eq!(e["has_filter"], json!(true));
    assert_eq!(e["reranked"], json!(false));
    assert_eq!(e["collection"], json!("qcol"));
    assert_eq!(e["index_type"], json!("hnsw"));
    assert!(e["result_count"].as_u64().is_some());
}

#[tokio::test]
async fn failed_search_logs_error_entry_with_kind() {
    let (base, tmp) = start_server().await;
    let c = Client::new();
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ecol",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Empty query body → no query variant set → BAD_REQUEST. The error
    // path must still write exactly one query-log entry, status=error.
    let r = c
        .post(format!("{base}/v1/collections/ecol/search"))
        .json(&json!({ "query": {}, "limit": 5 }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "{:?}", r.text().await);

    let entries = read_log(&tmp, "default__ecol").await;
    assert_eq!(entries.len(), 1, "exactly one query-log write per request");
    let e = &entries[0];
    assert_eq!(e["status"], json!("error"));
    assert_eq!(e["error_kind"], json!("bad_request"));
    assert_eq!(e["result_count"], json!(0));
}

#[tokio::test]
async fn correlations_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/correlations"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/correlations"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
