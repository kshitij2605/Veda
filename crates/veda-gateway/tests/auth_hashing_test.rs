//! SEC-1 / SEC-2 (2026-05-21 E2E campaign): verifies the auth path
//! after switching from plaintext compare to Argon2id PHC hashes,
//! including back-compat with legacy plaintext records and the
//! `VEDA_BOOTSTRAP_ADMIN_PASSWORD` env-driven default.

use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{hash_password, AppState, UserRecord};

async fn start_with(users: HashMap<String, UserRecord>) -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
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

#[tokio::test]
async fn login_with_argon2_hashed_password_returns_token() {
    let mut users = HashMap::new();
    users.insert(
        "alice".to_owned(),
        UserRecord::with_password("correct horse battery staple", Role::Admin),
    );

    let (base, _tmp) = start_with(users).await;
    let c = Client::new();

    let r = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "alice", "password": "correct horse battery staple" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "login should succeed: {:?}", r.text().await);
    let body: Value = r.json().await.unwrap();
    assert!(body["token"].as_str().is_some());

    // Wrong password
    let r = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "alice", "password": "wrong" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test]
async fn login_with_unknown_username_returns_401_in_constant_time() {
    // Constant-work check on the lookup miss path. We don't assert
    // exact timing (CI variance), only that the response is 401 and
    // that a username that doesn't exist behaves identically to a
    // bad password (no info leak via the error body).
    let mut users = HashMap::new();
    users.insert(
        "alice".to_owned(),
        UserRecord::with_password("hunter2", Role::Admin),
    );
    let (base, _tmp) = start_with(users).await;
    let c = Client::new();

    let r1 = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "ghost", "password": "anything" }))
        .send()
        .await
        .unwrap();
    let r2 = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "alice", "password": "wrong" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 401);
    assert_eq!(r2.status(), 401);
    assert_eq!(r1.text().await.unwrap(), r2.text().await.unwrap());
}

#[tokio::test]
async fn login_with_legacy_plaintext_record_still_works() {
    // Back-compat path: a UserRecord constructed by hand with a
    // cleartext password (as the pre-2026-05-21 tests did) is still
    // accepted via constant_time_eq, with a one-shot warn.
    let mut users = HashMap::new();
    users.insert(
        "legacy".to_owned(),
        UserRecord {
            password: "raw-plain".to_owned(),
            role: Role::Viewer,
        },
    );
    let (base, _tmp) = start_with(users).await;
    let c = Client::new();

    let r = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "legacy", "password": "raw-plain" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "legacy plaintext should still verify: {:?}", r.text().await);
}

#[test]
fn hash_password_round_trip_works_and_is_salted() {
    use veda_gateway::rest::state::verify_credentials;
    let h1 = hash_password("hunter2");
    let h2 = hash_password("hunter2");
    // Same password, different salts → different stored values.
    assert_ne!(h1, h2);
    // Both verify.
    assert!(verify_credentials(&h1, "hunter2", "u"));
    assert!(verify_credentials(&h2, "hunter2", "u"));
    // Wrong password rejected.
    assert!(!verify_credentials(&h1, "wrong", "u"));
    // Both are PHC strings — start with $argon2id$.
    assert!(h1.starts_with("$argon2id$"));
    assert!(h2.starts_with("$argon2id$"));
}
