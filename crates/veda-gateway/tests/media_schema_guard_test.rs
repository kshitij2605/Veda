//! Regression test: `ingest-media` must reject a collection that declares
//! more than one dense vector field, up front and with an actionable error.
//!
//! Every dense vector field is a dense column that each record has to fill,
//! but a media upload only fills the one vector field whose `auto_embed`
//! modality matches the file. A collection with (say) an image field and a
//! video field therefore accepts neither: the first upload got as far as the
//! storage layer and came back as
//! `500 INGEST_FAILED: record missing required vector field 'vid_vec'`,
//! which says nothing about what to change — and the docs used to recommend
//! exactly that layout for a multi-modality collection.

use reqwest::Client;
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    {
        let mut registry = app_state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "omni-test".into(),
            endpoint: "http://mock:8080/embed".into(),
            api_key: String::new(),
            supported_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
            ],
            output_dimension: 4,
            max_input_size_bytes: 10 * 1024 * 1024,
            timeout_seconds: 30,
            batch_size: 32,
        });
    }
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

fn client() -> Client {
    Client::new()
}

async fn create(base: &str, body: Value) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);
}

/// Upload a tiny PNG as `file`. The embedding call would fail against the
/// mock endpoint, so these tests only assert what happens *before* it.
async fn upload_png(base: &str, collection: &str) -> (u16, String) {
    let png: &[u8] = b"\x89PNG\r\n\x1a\n-not-a-real-png-but-typed-as-one";
    let part = reqwest::multipart::Part::bytes(png.to_vec())
        .file_name("probe.png")
        .mime_str("image/png")
        .unwrap();
    let form = reqwest::multipart::Form::new().part("file", part);
    let resp = client()
        .post(format!("{base}/v1/collections/{collection}/ingest-media"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

#[tokio::test]
async fn ingest_media_rejects_a_collection_with_two_vector_fields() {
    let (base, _tmp) = start_test_server().await;
    create(
        &base,
        json!({
            "name": "mm_two",
            "fields": [
                { "name": "class", "type": "string", "filterable": true },
                { "name": "media_blob", "type": "blob" },
                { "name": "img_vec", "type": "vector", "dimension": 4, "metric": "cosine",
                  "index": "hnsw", "auto_embed": { "source": "media_blob", "modality": "image" } },
                { "name": "vid_vec", "type": "vector", "dimension": 4, "metric": "cosine",
                  "index": "hnsw", "auto_embed": { "source": "media_blob", "modality": "video" } }
            ]
        }),
    )
    .await;

    let (status, body) = upload_png(&base, "mm_two").await;
    assert_eq!(
        status, 409,
        "expected an up-front schema rejection, got {status}: {body}"
    );
    assert!(
        body.contains("INCOMPATIBLE_SCHEMA"),
        "expected the INCOMPATIBLE_SCHEMA code: {body}"
    );
    assert!(
        body.contains("vid_vec"),
        "the error must name the field that cannot be filled: {body}"
    );
}

#[tokio::test]
async fn ingest_media_accepts_a_single_vector_field_collection() {
    let (base, _tmp) = start_test_server().await;
    create(
        &base,
        json!({
            "name": "mm_one",
            "fields": [
                { "name": "class", "type": "string", "filterable": true },
                { "name": "media_blob", "type": "blob" },
                { "name": "img_vec", "type": "vector", "dimension": 4, "metric": "cosine",
                  "index": "hnsw", "auto_embed": { "source": "media_blob", "modality": "image" } }
            ]
        }),
    )
    .await;

    // The schema is fine, so this must get past the guard and fail later on
    // the unreachable mock embedding endpoint instead.
    let (status, body) = upload_png(&base, "mm_one").await;
    assert_ne!(
        status, 409,
        "a single-vector-field media collection must not be rejected: {body}"
    );
    assert!(
        !body.contains("INCOMPATIBLE_SCHEMA"),
        "unexpected schema rejection: {body}"
    );
}
