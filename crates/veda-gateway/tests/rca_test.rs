//! Integration tests for the cross-signal root-cause bundle endpoint
//! (`GET /v1/admin/insights/{event_id}/rca`) + its admin gate.

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Seed one open advisor event + an in-window ERROR log carrying a trace
/// id, then return (base_url, tmpdir, event_id). Reuses the public
/// `force a tick` is not needed: we seed the event store directly via the
/// advisor handle on `AppState` before building the router.
async fn server_with_seeded_event() -> (String, tempfile::TempDir, String) {
    use veda_gateway::advisor::engine::now_ms;
    use veda_gateway::advisor::event::{OpsEvent, RemediationAction, Scope, Severity};
    use veda_gateway::advisor::rules::RuleHit;
    use veda_gateway::logs::buffer::LogRecord;

    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());

    let now = now_ms();
    {
        let mut store = app_state.advisor().events.lock().await;
        store.reconcile(
            vec![RuleHit {
                rule_id: "high_error_rate".to_owned(),
                scope: Scope::Global,
                severity: Severity::High,
                title: "errors spiking".to_owned(),
                detail: "5xx ratio over threshold".to_owned(),
                recommendation: "inspect logs".to_owned(),
                runbook: None,
                remediation: RemediationAction::None,
            }],
            now,
        );
    }
    // An in-window ERROR log with a trace id → joined into the bundle.
    app_state.log_buffer().push(LogRecord {
        ts_ms: now,
        level: "ERROR".to_owned(),
        target: "veda_gateway::search".to_owned(),
        message: "downstream connection refused".to_owned(),
        fields: serde_json::Map::new(),
        trace_id: Some("trace-abc".to_owned()),
        span_id: None,
    });

    let event_id = OpsEvent::make_id("high_error_rate", &Scope::Global);

    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp, event_id)
}

async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert(
        "admin".to_owned(),
        UserRecord::with_password("admin", Role::Admin),
    );
    users.insert(
        "viewer".to_owned(),
        UserRecord::with_password("viewer", Role::Viewer),
    );
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

#[tokio::test]
async fn rca_returns_bundle_for_seeded_event() {
    let (base, _tmp, event_id) = server_with_seeded_event().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/insights/{event_id}/rca"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["event_id"], json!("high_error_rate:global"));
    assert!(body["logs"].is_array());
    assert!(body["slow_queries"].is_array());
    assert!(body["trace_ids"].is_array());
    // The seeded ERROR log was joined; its trace id is in the correlation set.
    let logs = body["logs"].as_array().unwrap();
    assert_eq!(
        logs.len(),
        1,
        "expected the one in-window ERROR log: {logs:?}"
    );
    assert_eq!(logs[0]["message"], "downstream connection refused");
    assert_eq!(body["trace_ids"][0], "trace-abc");
}

#[tokio::test]
async fn rca_unknown_event_returns_404() {
    let (base, _tmp, _id) = server_with_seeded_event().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/insights/bogus:global/rca"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404, "{:?}", r.text().await);
}

#[tokio::test]
async fn rca_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer is rejected by the require_admin route layer (403), before any
    // event lookup.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/insights/any:global/rca"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin passes the gate; with no such event it is 404 (not 403).
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/insights/any:global/rca"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404, "admin passes gate; missing event → 404");
}
