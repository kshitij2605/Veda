//! `process-document` must be able to complete on a multimodal collection.
//!
//! A dense vector column is MANDATORY on every record (`growing_segment.rs` /
//! `segment.rs`: "record missing required vector field"), but the pipeline's
//! two ingest loops used to build disjoint target sets — `ingest_common` fed
//! `auto_embed: Text` and non-`*image*`-named columns, `document.rs` fed
//! `auto_embed` image/video/audio and `*image*`-named ones — so no column was
//! ever filled by both record types and the ingest failed in both directions:
//! an `image_blob` collection with a plain `embedding` field died as soon as
//! OCR returned an image, and a collection with an image column died on its
//! text chunks even for a document containing none.
//!
//! Both loops now resolve their columns through the same
//! `ingest_common::resolve_dense_targets`, so a collection with one vector
//! field backed by a multimodal model holds its text chunks and its images in
//! one unified vector space — Veda's design point. A schema that genuinely
//! cannot be satisfied is rejected up front with `409 INCOMPATIBLE_SCHEMA`,
//! matching `/ingest-media`'s guard for the identical invariant, instead of
//! reaching the storage layer.
//!
//! FlashParse is mocked: the handler POSTs the file and expects a ZIP back
//! containing markdown and images, so a local axum server returning a
//! hand-built ZIP exercises the real code path with no OCR service.

use reqwest::Client;
use serde_json::{Value, json};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Mutex;
use tokio::net::TcpListener;
use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

const MOCK_DIM: u32 = 64;

/// `AppState::new` reads `VEDA_FLASHPARSE_ENDPOINT` out of the process
/// environment, so the two tests in this binary must not construct their
/// servers concurrently.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A ZIP shaped like FlashParse's response: one markdown page plus one
/// extracted image.
fn flashparse_zip() -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file("page_001.md", opts).unwrap();
        zip.write_all(b"# Printer manual\n\nThe printer supports duplex printing.\n")
            .unwrap();
        zip.start_file("img_001.png", opts).unwrap();
        // Content is irrelevant — the pipeline only stores the bytes as a
        // blob and embeds them; it must merely be non-empty.
        zip.write_all(b"\x89PNG\r\n\x1a\nfake-image-bytes").unwrap();
        zip.finish().unwrap();
    }
    buf.into_inner()
}

async fn start_mock_flashparse() -> String {
    use axum::Router;
    use axum::routing::post;

    async fn process() -> impl axum::response::IntoResponse {
        ([("content-type", "application/zip")], flashparse_zip())
    }

    let router = Router::new().route("/api/v1/pdf/process", post(process));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

/// `modalities` is what makes or breaks the schema: an omni model fills one
/// column from both text and images; a text-only model cannot fill it from an
/// image at all.
async fn start_test_server(
    flashparse_url: &str,
    model: &str,
    modalities: Vec<Modality>,
) -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized on ENV_LOCK; the variable is read back
        // synchronously by `AppState::new` and then cleared.
        unsafe {
            std::env::set_var("VEDA_FLASHPARSE_ENDPOINT", flashparse_url);
        }
        let state = AppState::new(tmp.path());
        unsafe {
            std::env::remove_var("VEDA_FLASHPARSE_ENDPOINT");
        }
        state
    };
    {
        let mut registry = state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: model.into(),
            endpoint: "http://mock:8000/embed".into(),
            api_key: String::new(),
            supported_modalities: modalities,
            output_dimension: MOCK_DIM,
            max_input_size_bytes: 1024 * 1024,
            timeout_seconds: 10,
            batch_size: 32,
        });
    }
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

/// One dense vector column + an image blob column: the multimodal document
/// collection the docs recommend.
async fn create_collection(base: &str, name: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "source_file", "type": "string", "filterable": true },
                { "name": "image_blob", "type": "blob" },
                { "name": "embedding", "type": "vector", "dimension": MOCK_DIM,
                  "metric": "cosine",
                  "auto_embed": { "source": "content", "modality": "text" } }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);
}

async fn process_document(base: &str, name: &str) -> (reqwest::StatusCode, String) {
    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(b"%PDF-1.4 fake".to_vec())
            .file_name("manual.pdf")
            .mime_str("application/pdf")
            .unwrap(),
    );
    let resp = client()
        .post(format!("{base}/v1/collections/{name}/process-document"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    (status, resp.text().await.unwrap())
}

/// Every inserted record, with its fields.
///
/// `filter_query` is an agent-facing tool, so the dispatch choke point
/// summarises embeddings rather than serialising their components (see
/// `agent::field_guard::strip_embeddings`). The summary still carries the
/// dimension, so this is still how the vector column is verified as
/// actually filled — see [`vector_len`].
async fn all_records(base: &str, name: &str) -> Vec<Value> {
    let body: Value = client()
        .post(format!("{base}/v1/agent-tools/filter_query/execute"))
        .json(&json!({ "collection": name, "filter": "_indexed_at > 0", "limit": 50 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["records"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| panic!("records array: {body}"))
}

/// Width of the record's embedding column.
///
/// Agent-facing payloads render a dense vector as
/// `{"vector": {"dim": N}}` instead of N floats, so this reads the
/// reported dimension. The invariant under test is unchanged: the key is
/// absent entirely when the column was never filled, so a present
/// summary with a non-zero dim still proves the round trip populated it.
fn vector_len(record: &Value) -> usize {
    let embedding = &record["fields"]["embedding"];
    embedding["vector"]["dim"]
        .as_u64()
        .map(|d| d as usize)
        // Tolerate a raw component list too, so this helper keeps working
        // on any path that does not go through the agent dispatch.
        .or_else(|| embedding.as_array().map(|a| a.len()))
        .unwrap_or_else(|| panic!("record has no embedding vector: {record}"))
}

/// The end-to-end round trip: a document WITH an image, into a collection
/// with one unified vector column. Before the fix this returned
/// `500 INSERT_FAILED: record missing required vector field 'embedding'`.
#[tokio::test]
async fn process_document_round_trips_images_into_the_unified_vector_field() {
    let flashparse = start_mock_flashparse().await;
    let (base, _tmp) = start_test_server(
        &flashparse,
        "mock-omni-embedder",
        vec![Modality::Text, Modality::Image],
    )
    .await;
    create_collection(&base, "doc_mm").await;

    let (status, body_text) = process_document(&base, "doc_mm").await;
    assert_eq!(status, 201, "process-document: {status} {body_text}");
    let body: Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(body["images_extracted"], 1, "{body}");
    let kinds: Vec<&str> = body["chunks"]
        .as_array()
        .expect("chunks")
        .iter()
        .filter_map(|c| c["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"image") && kinds.contains(&"text"),
        "both record types must be inserted: {body}"
    );

    let records = all_records(&base, "doc_mm").await;
    let image_rec = records
        .iter()
        .find(|r| {
            r["fields"]["content"]
                .as_str()
                .is_some_and(|c| c.starts_with("[image:"))
        })
        .unwrap_or_else(|| panic!("no image record was inserted: {records:?}"));
    let text_rec = records
        .iter()
        .find(|r| {
            r["fields"]["content"]
                .as_str()
                .is_some_and(|c| !c.starts_with("[image:"))
        })
        .unwrap_or_else(|| panic!("no text record was inserted: {records:?}"));

    // Both record types filled the SAME column — that is the unified vector
    // space, and the only shape the mandatory-dense-column invariant allows.
    assert_eq!(vector_len(image_rec), MOCK_DIM as usize, "{image_rec}");
    assert_eq!(vector_len(text_rec), MOCK_DIM as usize, "{text_rec}");
}

/// The same schema with a text-only embedder is genuinely unsatisfiable once
/// the document contains an image. It must be refused up front, naming the
/// column and the fix — never with the storage layer's message, and never
/// after half the document is already inserted.
#[tokio::test]
async fn process_document_rejects_a_schema_whose_column_images_cannot_fill() {
    let flashparse = start_mock_flashparse().await;
    let (base, _tmp) =
        start_test_server(&flashparse, "mock-text-embedder", vec![Modality::Text]).await;
    create_collection(&base, "doc_bad").await;

    let (status, body_text) = process_document(&base, "doc_bad").await;
    assert_eq!(status, 409, "expected a schema rejection: {body_text}");
    let body: Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(body["error"]["code"], "INCOMPATIBLE_SCHEMA", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("embedding"),
        "the offending column must be named: {body}"
    );
    assert!(
        message.contains("multimodal") || message.contains("one collection per modality"),
        "the fix must be stated: {body}"
    );
    assert!(
        !message.contains("record missing required vector field"),
        "the storage-level message must never be what the caller sees: {body}"
    );

    // Rejected up front: not one text chunk was written before the failure.
    assert!(
        all_records(&base, "doc_bad").await.is_empty(),
        "the guard must run before any insert"
    );
}
