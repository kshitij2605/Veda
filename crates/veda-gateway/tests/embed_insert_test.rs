use reqwest::Client;
use serde_json::{json, Value};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

/// Start a test server with a pre-registered embedding model.
async fn start_test_server_with_models() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());

    // Register embedding models.
    {
        let mut registry = app_state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "clip-vit-l-14".into(),
            endpoint: "http://localhost:8080/embed".into(),
            supported_modalities: vec![Modality::Image, Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 10 * 1024 * 1024,
            timeout_seconds: 30,
            batch_size: 32,
        });
        registry.register(EmbeddingModelConfig {
            name: "text-only-model".into(),
            endpoint: "http://localhost:8081/embed".into(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 512 * 1024,
            timeout_seconds: 10,
            batch_size: 64,
        });
    }

    let router = build_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp)
}

fn client() -> Client {
    Client::new()
}

/// Helper: create a collection with a string field and a 4-dim vector field.
async fn create_test_collection(base: &str, name: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "image", "type": "string" },
                { "name": "image_embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 201, "failed to create collection: {:?}", resp.text().await);
}

// ── 1. Embed-insert with mock model → records inserted with vectors ─────

#[tokio::test]
async fn test_embed_insert_success() {
    let (base, _tmp) = start_test_server_with_models().await;
    create_test_collection(&base, "embed_col").await;

    let resp = client()
        .post(format!("{base}/v1/collections/embed_col/embed-insert"))
        .json(&json!({
            "records": [
                { "title": "Red Shoes", "image": "aW1hZ2VkYXRhMQ==" },
                { "title": "Blue Hat", "image": "aW1hZ2VkYXRhMg==" }
            ],
            "auto_embed": {
                "image_embedding": { "source_field": "image", "model": "clip-vit-l-14" }
            }
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(status, 201, "embed-insert failed: {body}");
    assert_eq!(body["inserted"], 2);

    // Verify we can search the collection (data was actually inserted).
    let search_resp = client()
        .post(format!("{base}/v1/collections/embed_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "image_embedding", "values": [0.5, 0.5, 0.5, 0.5] }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(search_resp.status(), 200);
    let search_body: Value = search_resp.json().await.unwrap();
    assert_eq!(search_body["count"], 2, "should find 2 inserted records");
}

// ── 2. Embed-insert with unsupported modality → 422 error ──────────────

#[tokio::test]
async fn test_embed_insert_unsupported_modality() {
    let (base, _tmp) = start_test_server_with_models().await;
    create_test_collection(&base, "modality_col").await;

    // text-only-model doesn't support Image. The source field "image" maps
    // to Image modality via field name detection.
    let resp = client()
        .post(format!("{base}/v1/collections/modality_col/embed-insert"))
        .json(&json!({
            "records": [
                { "title": "Test", "image": "aW1hZ2VkYXRh" }
            ],
            "auto_embed": {
                "image_embedding": { "source_field": "image", "model": "text-only-model" }
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 422);
    let body: Value = resp.json().await.unwrap();
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("does not support"), "error should mention modality mismatch: {error}");
}

// ── 3. Embed-insert with missing model → 400 error ─────────────────────

#[tokio::test]
async fn test_embed_insert_missing_model() {
    let (base, _tmp) = start_test_server_with_models().await;
    create_test_collection(&base, "missing_model_col").await;

    let resp = client()
        .post(format!("{base}/v1/collections/missing_model_col/embed-insert"))
        .json(&json!({
            "records": [
                { "title": "Test", "image": "aW1hZ2U=" }
            ],
            "auto_embed": {
                "image_embedding": { "source_field": "image", "model": "nonexistent-model" }
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("not found"), "error should mention model not found: {error}");
}
