//! Regression tests for how auto-optimization config is keyed and validated.
//!
//! The config has to be stored under the *resolved* collection, because that
//! is the name `ingest_markdown` passes to `maybe_trigger_auto_pipeline` when
//! it looks the config up after an ingestion. Keying it by the caller's name
//! meant an aliased collection — exactly what applying a tier-2/3 tuning
//! trial produces — never fired its own auto-optimization: the config sat
//! under the pre-swap name and the post-ingest lookup never found it.
//!
//! `GET` also used to answer 200 with a synthesized default for any name at
//! all, so a typo'd collection looked real and un-opted-in while `PUT` on
//! that same name correctly refused with 404.

use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
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

async fn create(base: &str, name: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [ { "name": "content", "type": "string", "bm25": true } ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create {name}: {:?}", resp.text().await);
}

#[tokio::test]
async fn get_auto_optimization_404s_for_an_unknown_collection() {
    let (base, _tmp) = start_test_server().await;

    let resp = client()
        .get(format!(
            "{base}/v1/collections/no_such_coll/auto-optimization"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        404,
        "unknown collection must 404 like PUT does: {:?}",
        resp.text().await
    );
}

#[tokio::test]
async fn config_is_keyed_by_the_alias_target() {
    let (base, _tmp) = start_test_server().await;
    create(&base, "ao_target").await;

    let resp = client()
        .put(format!("{base}/v1/admin/collections/ao_ghost/alias"))
        .json(&json!({ "target": "ao_target" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Write the config through the alias name.
    let resp = client()
        .put(format!("{base}/v1/collections/ao_ghost/auto-optimization"))
        .json(&json!({ "enabled": true, "target_questions": 7, "mode": "Performance" }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "PUT through the alias should resolve: {:?}",
        resp.text().await
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["collection"], "ao_ghost",
        "the response should echo the caller's name: {body}"
    );

    // It must be readable under the TARGET name, because that is the name
    // the post-ingest trigger uses.
    let resp = client()
        .get(format!("{base}/v1/collections/ao_target/auto-optimization"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["enabled"], true,
        "config written through the alias must be found under the target: {body}"
    );
    assert_eq!(body["mode"], "Performance", "{body}");
    assert_eq!(body["target_questions"], 7, "{body}");

    // And still readable through the alias name.
    let resp = client()
        .get(format!("{base}/v1/collections/ao_ghost/auto-optimization"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"], true, "{body}");
    assert_eq!(body["collection"], "ao_ghost", "{body}");
}

#[tokio::test]
async fn round_trip_without_an_alias_still_works() {
    let (base, _tmp) = start_test_server().await;
    create(&base, "ao_plain").await;

    let resp = client()
        .get(format!("{base}/v1/collections/ao_plain/auto-optimization"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"], false, "default is opted out: {body}");

    let resp = client()
        .put(format!("{base}/v1/collections/ao_plain/auto-optimization"))
        .json(&json!({ "enabled": true, "mode": "Quality" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client()
        .get(format!("{base}/v1/collections/ao_plain/auto-optimization"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"], true, "{body}");
    assert_eq!(body["collection"], "ao_plain", "{body}");
}
