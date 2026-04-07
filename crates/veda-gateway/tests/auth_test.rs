use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::{self, Role};
use veda_gateway::rest::state::{AppState, UserRecord};
use veda_gateway::rest::build_router;

/// Start a test server with auth enabled and the given users.
async fn start_auth_server(
    users: HashMap<String, UserRecord>,
) -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state =
        AppState::with_auth_and_users(tmp.path(), "test-jwt-secret", users);
    let router = build_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp)
}

fn default_users() -> HashMap<String, UserRecord> {
    let mut users = HashMap::new();
    users.insert(
        "admin".into(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );
    users.insert(
        "viewer".into(),
        UserRecord {
            password: "viewer".into(),
            role: Role::Viewer,
        },
    );
    users
}

fn client() -> Client {
    Client::new()
}

// ── 1. Login with valid credentials returns token ───────────────────────

#[tokio::test]
async fn test_login_valid_credentials() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    let resp = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "admin" }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["token"].as_str().is_some());
    assert!(!body["token"].as_str().unwrap().is_empty());
}

// ── 2. Login with invalid credentials returns 401 ───────────────────────

#[tokio::test]
async fn test_login_invalid_credentials() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    let resp = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "wrong" }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

// ── 3. Access protected endpoint with valid token returns 200 ───────────

#[tokio::test]
async fn test_access_with_valid_token() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    // Login first
    let login_resp = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "admin" }))
        .send()
        .await
        .unwrap();
    let login_body: Value = login_resp.json().await.unwrap();
    let token = login_body["token"].as_str().unwrap();

    // Access protected endpoint
    let resp = client()
        .get(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
}

// ── 4. Access protected endpoint without token returns 401 ──────────────

#[tokio::test]
async fn test_access_without_token() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    let resp = client()
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

// ── 5. Access admin endpoint with viewer role returns 200 ───────────────
//    (RBAC is enforced at the middleware layer per-route; the auth middleware
//     itself only validates the token. This test verifies the viewer can
//     access generic endpoints. See the RBAC unit test for permission checks.)

#[tokio::test]
async fn test_viewer_can_access_collections_list() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    let login_resp = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "viewer", "password": "viewer" }))
        .send()
        .await
        .unwrap();
    let login_body: Value = login_resp.json().await.unwrap();
    let token = login_body["token"].as_str().unwrap();

    let resp = client()
        .get(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
}

// ── 6. Token expiration returns 401 ─────────────────────────────────────

#[tokio::test]
async fn test_expired_token() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    // Manually create an expired token.
    let expired_claims = auth::Claims {
        sub: "admin".into(),
        role: Role::Admin,
        exp: 1, // epoch + 1 second — long expired
    };
    let token = jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &expired_claims,
        &jsonwebtoken::EncodingKey::from_secret(b"test-jwt-secret"),
    )
    .unwrap();

    let resp = client()
        .get(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

// ── 7. Health endpoint is accessible without token (public) ─────────────

#[tokio::test]
async fn test_health_no_auth_required() {
    let (base, _tmp) = start_auth_server(default_users()).await;

    let resp = client()
        .get(format!("{base}/v1/admin/health"))
        .send()
        .await
        .unwrap();

    // Health is a public endpoint, not behind auth middleware.
    assert_eq!(resp.status(), 200);
}

// ── 8. RBAC permission checks (unit tests) ──────────────────────────────

#[test]
fn test_rbac_admin_can_do_everything() {
    assert!(veda_gateway::rbac::check_permission(Role::Admin, "search"));
    assert!(veda_gateway::rbac::check_permission(Role::Admin, "insert"));
    assert!(veda_gateway::rbac::check_permission(
        Role::Admin,
        "delete_records"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Admin,
        "list_collections"
    ));
    assert!(veda_gateway::rbac::check_permission(Role::Admin, "admin"));
}

#[test]
fn test_rbac_editor_permissions() {
    assert!(veda_gateway::rbac::check_permission(
        Role::Editor,
        "search"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Editor,
        "insert"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Editor,
        "delete_records"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Editor,
        "list_collections"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Editor,
        "describe_collection"
    ));
    assert!(!veda_gateway::rbac::check_permission(
        Role::Editor,
        "admin"
    ));
}

#[test]
fn test_rbac_viewer_permissions() {
    assert!(veda_gateway::rbac::check_permission(
        Role::Viewer,
        "search"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Viewer,
        "get_record"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Viewer,
        "list_collections"
    ));
    assert!(veda_gateway::rbac::check_permission(
        Role::Viewer,
        "describe_collection"
    ));
    assert!(!veda_gateway::rbac::check_permission(
        Role::Viewer,
        "insert"
    ));
    assert!(!veda_gateway::rbac::check_permission(
        Role::Viewer,
        "delete_records"
    ));
    assert!(!veda_gateway::rbac::check_permission(
        Role::Viewer,
        "admin"
    ));
}
