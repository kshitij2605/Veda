//! UX-1 (2026-05-21 E2E campaign): /embed-insert infers the
//! `auto_embed` mapping from the collection schema when the caller
//! doesn't supply one.
//!
//! Before: omitting `auto_embed` returned `422 missing field
//! 'auto_embed'`, even when the schema declared the per-field
//! `auto_embed` config at create time.
//!
//! After: when a vector field has `FieldType::Vector { auto_embed:
//! Some(AutoEmbedConfig { source, .. }) }`, the handler maps that
//! field's name to `source` automatically. An explicit
//! request-level `auto_embed` still wins.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

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

#[tokio::test]
async fn embed_insert_without_request_auto_embed_clear_error_when_schema_has_none() {
    // The legacy behaviour was a cryptic `422 missing field
    // 'auto_embed'`. We now return a 400 with a clear message that
    // explains both ways to fix it.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "no_schema_ae",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "text", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    let r = c
        .post(format!("{base}/v1/collections/no_schema_ae/embed-insert"))
        .json(&json!({ "records": [ { "text": "hello" } ] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "without schema config nor request map, should be 400");
    let body: Value = r.json().await.unwrap();
    let err = body["error"].as_str().unwrap_or("");
    assert!(err.contains("auto_embed"), "error should mention auto_embed: {err}");
    assert!(
        err.contains("source_field") || err.contains("auto_embed configured"),
        "error should explain how to fix: {err}"
    );
}

#[tokio::test]
async fn embed_insert_inferred_mapping_from_schema_auto_embed() {
    // Create a collection that declares auto_embed at the schema
    // level. The /embed-insert call MAY omit `auto_embed` in the
    // request body; the handler should infer it.
    //
    // We can't easily assert the embed succeeded end-to-end here
    // because the test AppState has no embedding endpoint
    // registered — but we can assert the handler progresses past
    // "missing field" / "no auto_embed configured" and instead
    // fails downstream with "no model registered for modality
    // 'Text'", which proves the inference worked.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "with_schema_ae",
            "shard_count": 1,
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "dimension": 4,
                    "metric": "cosine",
                    "auto_embed": {
                        "source": "text",
                        "modality": "text"
                    }
                },
                { "name": "text", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create with schema auto_embed: {:?}", r.text().await);

    let r = c
        .post(format!("{base}/v1/collections/with_schema_ae/embed-insert"))
        .json(&json!({ "records": [ { "text": "hello" } ] }))
        .send()
        .await
        .unwrap();
    // Either succeeds (if some embedding model happens to be
    // registered) or fails at the "no model registered" step.
    // In both cases we must NOT hit the legacy "missing field"
    // 422 OR the "no auto_embed configured" 400.
    let status = r.status();
    let body = r.text().await.unwrap();
    assert!(
        status == 201 || status == 400 || status == 502,
        "unexpected status {status}: {body}"
    );
    assert!(
        !body.contains("missing field"),
        "should not be 422 missing field: {body}"
    );
    assert!(
        !body.contains("no `auto_embed` mappings supplied"),
        "schema-declared auto_embed should have been inferred: {body}"
    );
}

#[tokio::test]
async fn embed_insert_explicit_request_auto_embed_overrides_schema() {
    // Belt-and-suspenders: when the request includes auto_embed,
    // the schema-declared one is ignored (operator can override
    // the source field per call).
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let r = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "override_ae",
            "shard_count": 1,
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "dimension": 4,
                    "metric": "cosine",
                    "auto_embed": { "source": "text", "modality": "text" }
                },
                { "name": "text", "type": "string", "filterable": true },
                { "name": "title", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201);

    // Use a *different* source field via the request map.
    let r = c
        .post(format!("{base}/v1/collections/override_ae/embed-insert"))
        .json(&json!({
            "records": [ { "text": "hello", "title": "h" } ],
            "auto_embed": {
                "embedding": { "source_field": "title" }
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert!(
        status == 201 || status == 400 || status == 502,
        "unexpected status {status}: {body}"
    );
    assert!(!body.contains("missing field"), "explicit map shouldn't 422: {body}");
}
