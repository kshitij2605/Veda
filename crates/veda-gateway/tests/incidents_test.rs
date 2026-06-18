//! Integration tests for the semantic incident-correlation REST API
//! (`GET /v1/admin/incidents`).

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::advisor::event::{RemediationAction, Scope, Severity};
use veda_gateway::advisor::rules::RuleHit;
use veda_gateway::advisor::run_incident_correlation;
use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server. Returns the base URL, the live `AppState` clone
/// (so the test can seed events + run correlation), and the TempDir guard.
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

fn hit(rule_id: &str, scope: Scope, sev: Severity, title: &str, detail: &str) -> RuleHit {
    RuleHit {
        rule_id: rule_id.to_owned(),
        scope,
        severity: sev,
        title: title.to_owned(),
        detail: detail.to_owned(),
        recommendation: "fix it".to_owned(),
        runbook: None,
        remediation: RemediationAction::None,
    }
}

/// Seed three events (two same rule+scope = recurring, one distinct) into
/// the event store, then run the off-tick incident correlator so the
/// store has incidents. With no embedder configured this exercises the
/// rule/scope fallback grouping.
async fn seed_and_correlate(state: &AppState) {
    {
        let mut store = state.advisor().events.lock().await;
        // First round opens two events.
        store.reconcile(
            vec![
                hit(
                    "compaction_backlog",
                    Scope::Collection("docs".into()),
                    Severity::Medium,
                    "Compaction backlog on docs",
                    "25 sealed segments",
                ),
                hit(
                    "disk_pressure",
                    Scope::Node("n1".into()),
                    Severity::High,
                    "Disk pressure on n1",
                    "free space below 10%",
                ),
            ],
            1000,
        );
        // The same compaction_backlog event refreshing is dedup'd by the
        // store, but its presence in open+history still feeds one incident.
        store.reconcile(
            vec![hit(
                "compaction_backlog",
                Scope::Collection("docs".into()),
                Severity::High,
                "Compaction backlog on docs (worse)",
                "30 sealed segments",
            )],
            2000,
        );
    }
    run_incident_correlation(state).await;
}

#[tokio::test]
async fn incidents_empty_on_fresh_server() {
    let (base, _state, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/incidents"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["incidents"].is_array());
    assert_eq!(body["incidents"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn incidents_returns_clusters() {
    let (base, state, _tmp) = start_server().await;
    seed_and_correlate(&state).await;

    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/incidents"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();

    let incidents = body["incidents"].as_array().expect("incidents array");
    // Two distinct rule/scope incidents (docs compaction + n1 disk). The
    // recurring compaction_backlog alert collapses into a single incident.
    assert_eq!(incidents.len(), 2, "got {incidents:?}");

    // The disk-pressure incident is present and High severity.
    let disk = incidents
        .iter()
        .find(|i| i["representative"].as_str().unwrap().contains("Disk pressure"))
        .expect("disk pressure incident present");
    assert_eq!(disk["max_severity"], json!("high"));

    // Every incident exposes the dedup shape.
    for inc in incidents {
        assert!(inc["id"].is_string());
        assert!(inc["member_event_ids"].is_array());
        assert!(inc["count"].as_u64().unwrap() >= 1);
        assert!(inc["rule_ids"].is_array());
        assert!(inc["first_seen_ms"].as_i64().unwrap() > 0);
        assert!(inc["last_seen_ms"].as_i64().unwrap() > 0);
    }
}

#[tokio::test]
async fn incidents_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/incidents"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/incidents"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "admin must be allowed: {:?}", r.text().await);
}
