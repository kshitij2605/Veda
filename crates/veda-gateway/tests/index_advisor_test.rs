//! Integration tests for the ANN index-advisor REST API
//! (`GET /v1/admin/index-advisor`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::advisor::recall_probe::{IndexParams, RecallSample};
use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server. Returns the base URL, the live `AppState` clone
/// (so the test can seed recall results), and the TempDir guard.
async fn start_server() -> (String, AppState, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let state_handle = app_state.clone();
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, state_handle, tmp)
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

/// Create a `docs` collection with a vector field, seed its query log, and
/// record a degraded (0.70) proxy recall sample into the advisor. Then
/// trigger an advisor tick so a snapshot with `index_health` exists.
async fn seed_degraded(base: &str, state: &AppState, tmp: &tempfile::TempDir) {
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "docs",
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine", "index": "hnsw" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create failed: {:?}", r.text().await);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let scoped = "default__docs";
    let dir = tmp.path().join(scoped);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("_query_log.jsonl");
    let mk = |took: u64| {
        json!({
            "ts_ms": now_ms,
            "top_k": 10,
            "doc_ids": [],
            "took_ms": took,
            "collection": "docs",
            "query_type": "dense",
            "has_filter": false,
            "reranked": false,
            "result_count": 3,
            "status": "ok",
            "error_kind": null,
            "index_type": "hnsw"
        })
    };
    let lines = [mk(10), mk(20), mk(30), mk(40)];
    let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(&log, body).unwrap();

    // Seed a degraded recall sample for "docs".
    {
        let mut results = state.advisor().recall_results.lock().await;
        results.record(RecallSample {
            collection: "docs".into(),
            index_type: "hnsw".into(),
            recall_at_k: 0.70,
            k: 10,
            sampled_queries: 20,
            p95_ms: Some(35.0),
            params: IndexParams::default(),
            ts_ms: now_ms,
            is_proxy: true,
        });
    }

    // Trigger an advisor tick: collects a snapshot (assembling
    // index_health from schema + query log + the recall sample) and runs
    // the rules (opening an index_recall_degraded event).
    let r = c
        .post(format!("{base}/v1/admin/insights/run"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "run tick failed: {:?}", r.text().await);
}

#[tokio::test]
async fn index_advisor_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/index-advisor"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["rows"].is_array());
    assert_eq!(body["rows"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn index_advisor_returns_row_with_recommendation() {
    let (base, state, tmp) = start_server().await;
    seed_degraded(&base, &state, &tmp).await;

    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/index-advisor"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();

    let rows = body["rows"].as_array().expect("rows array");
    let row = rows
        .iter()
        .find(|row| row["collection"] == json!("docs"))
        .expect("docs row present");

    assert_eq!(row["index_type"], json!("hnsw"));
    // recall threaded from the seeded sample; labelled proxy.
    assert_eq!(row["recall"], json!(0.70));
    assert_eq!(row["is_proxy"], json!(true));
    // p95 from the query log.
    assert!(row["p95_ms"].as_f64().unwrap() > 0.0);

    // The degraded-recall rule must have produced a recommendation.
    let recs = row["recommendations"].as_array().expect("recommendations array");
    assert!(
        recs.iter().any(|x| x["rule_id"] == json!("index_recall_degraded")),
        "index_recall_degraded recommendation present; got {recs:?}"
    );
    // A suggested reindex body is offered for the opt-in button.
    assert!(row["suggested_reindex"].is_object());

    // The open index event for the collection is surfaced.
    let open = row["open_events"].as_array().expect("open_events array");
    assert!(
        open.iter().any(|e| e["rule_id"] == json!("index_recall_degraded")),
        "open index_recall_degraded event present; got {open:?}"
    );
}

#[tokio::test]
async fn index_advisor_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/index-advisor"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/index-advisor"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
