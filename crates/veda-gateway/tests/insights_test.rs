//! Integration tests for the ops-advisor "Insights" REST API
//! (`/v1/admin/insights/*`) + the engine wiring on `AppState`.

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

/// Auth-disabled server over a fresh temp dir (mirrors the other admin
/// endpoint tests). Returns the base URL and the TempDir guard.
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

/// Auth-enabled server with an admin + a viewer user.
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
async fn run_returns_opened_closed_counts() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/admin/insights/run"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    // Clean server: nothing trips, so 0/0 — but the tick ran without error.
    assert_eq!(body["opened"], json!(0));
    assert_eq!(body["closed"], json!(0));
}

#[tokio::test]
async fn rules_lists_known_threshold_and_anomaly_ids() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .get(format!("{base}/v1/admin/insights/rules"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let ids: Vec<String> = body["rules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    assert!(ids.contains(&"compaction_backlog".to_owned()), "{ids:?}");
    assert!(ids.contains(&"high_error_rate".to_owned()), "{ids:?}");
    assert!(
        ids.iter().any(|i| i.starts_with("anomaly_")),
        "expected an anomaly_* rule in {ids:?}"
    );
}

#[tokio::test]
async fn put_rule_mutes_and_persists() {
    let (base, tmp) = start_server().await;
    let c = Client::new();

    // Mute compaction_backlog.
    let r = c
        .put(format!("{base}/v1/admin/insights/rules/compaction_backlog"))
        .json(&json!({ "muted": true }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["muted"], json!(true));

    // GET /rules now shows it muted.
    let r = c
        .get(format!("{base}/v1/admin/insights/rules"))
        .send()
        .await
        .unwrap();
    let body: Value = r.json().await.unwrap();
    let entry = body["rules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == json!("compaction_backlog"))
        .unwrap();
    assert_eq!(entry["effective_config"]["muted"], json!(true));

    // Persisted to disk: the config file reflects the mute.
    let cfg_path = tmp.path().join("_advisor_config.json");
    let raw = std::fs::read_to_string(&cfg_path).expect("config persisted");
    let cfg: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(cfg["rules"]["compaction_backlog"]["muted"], json!(true));
}

#[tokio::test]
async fn channels_get_redacts_webhook_url() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let secret_url = "https://hooks.example.com/super-secret-token-12345";
    let r = c
        .put(format!("{base}/v1/admin/insights/channels"))
        .json(&json!({
            "webhook": { "enabled": true, "url": secret_url, "min_severity": "high" }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);

    let r = c
        .get(format!("{base}/v1/admin/insights/channels"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let text = r.text().await.unwrap();
    // The secret URL must NOT appear anywhere in the GET response.
    assert!(
        !text.contains(secret_url),
        "secret webhook URL leaked in channels GET: {text}"
    );
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["webhook"]["configured"], json!(true));
    assert_eq!(body["webhook"]["enabled"], json!(true));
    assert_eq!(body["webhook"]["min_severity"], json!("high"));
}

#[tokio::test]
async fn ack_unknown_event_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/admin/insights/bogus:global/ack"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn insights_requires_admin_role() {
    let (base, _tmp) = start_auth_server().await;
    let c = Client::new();

    // Viewer token is rejected by the require_admin route layer.
    let viewer = login(&base, "viewer", "viewer").await;
    let r = c
        .get(format!("{base}/v1/admin/insights/rules"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "viewer must be forbidden");

    // Admin token passes.
    let admin = login(&base, "admin", "admin").await;
    let r = c
        .get(format!("{base}/v1/admin/insights/rules"))
        .bearer_auth(&admin)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "admin must be allowed: {:?}",
        r.text().await
    );
}
