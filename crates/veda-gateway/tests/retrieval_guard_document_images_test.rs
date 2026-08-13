//! `process-document`'s image records are stamped, per design §4.10.
//!
//! The OCR pipeline inserts one record per extracted image, carrying an
//! `[image: x.png]` placeholder and an image (not text) embedding — exactly
//! the "no meaningful text to screen" case §4.10 describes, and exactly what
//! `/ingest-media` already stamps as `skipped`. This path inserted them bare,
//! with no `_indexed_at` and no verdict, so they classified `Unscreened`.
//!
//! That is not a cosmetic mismatch. `unscreened` is the single signal the
//! deliberate fail-open decision (§6) rests on: an operator runs a full
//! rescan, sees `unscreened > 0` still, and cannot tell "pre-feature data I
//! have not converted" from "the document pipeline emits these on every
//! ingest". Two media paths disagreeing about the same situation makes the
//! count unreadable.
//!
//! FlashParse is mocked: the handler POSTs the file and expects a ZIP back
//! containing markdown and images, so a local axum server returning a
//! hand-built ZIP exercises the real code path with no OCR service.

use reqwest::Client;
use serde_json::{Value, json};
use std::io::Write;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

const MOCK_DIM: u32 = 64;

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
        // blob; it must merely be non-empty.
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

/// `flashparse_endpoint` is read out of the environment by `AppState::new`,
/// so the variable must be set across construction only. This binary holds a
/// single test, so nothing races it.
async fn start_test_server(flashparse_url: &str) -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    // SAFETY: this test binary contains exactly one test; no other thread
    // touches the process environment.
    unsafe {
        std::env::set_var("VEDA_FLASHPARSE_ENDPOINT", flashparse_url);
    }
    let state = AppState::new(tmp.path());
    unsafe {
        std::env::remove_var("VEDA_FLASHPARSE_ENDPOINT");
    }
    {
        let mut registry = state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "mock-text-embedder".into(),
            endpoint: "http://mock:8000/embed".into(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
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

#[tokio::test]
async fn process_document_image_records_are_stamped_skipped_not_unscreened() {
    let flashparse = start_mock_flashparse().await;
    let (base, _tmp) = start_test_server(&flashparse).await;

    // `image_blob` is what makes the pipeline insert image records at all
    // (`should_insert_images`).
    //
    // No dense vector field on purpose: this test is about the stamping, and
    // the registered embedder is text-only, so a vector column would need the
    // multimodal setup `document_multimodal_test.rs` covers (both ingest
    // loops fill the same mandatory dense column, which needs a model that
    // accepts text AND images).
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "rg_doc",
            "fields": [
                { "name": "content", "type": "string", "bm25": true },
                { "name": "source_file", "type": "string", "filterable": true },
                { "name": "image_blob", "type": "blob" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {:?}", resp.text().await);

    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(b"%PDF-1.4 fake".to_vec())
            .file_name("manual.pdf")
            .mime_str("application/pdf")
            .unwrap(),
    );
    let resp = client()
        .post(format!("{base}/v1/collections/rg_doc/process-document"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body_text = resp.text().await.unwrap();
    assert_eq!(status, 201, "process-document: {status} {body_text}");

    // Every record — text chunks AND the image placeholder — must carry
    // `_indexed_at`, so this filter selects all of them.
    let body: Value = client()
        .post(format!("{base}/v1/agent-tools/filter_query/execute"))
        .json(&json!({ "collection": "rg_doc", "filter": "_indexed_at > 0", "limit": 50 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let records = body["records"].as_array().expect("records array");

    let image_rec = records
        .iter()
        .find(|r| {
            r["fields"]["content"]
                .as_str()
                .is_some_and(|c| c.starts_with("[image:"))
        })
        .unwrap_or_else(|| panic!("no image record was inserted: {body}"));

    assert!(
        image_rec["fields"]["_indexed_at"].as_i64().unwrap_or(0) > 0,
        "the image record must carry _indexed_at: {image_rec}"
    );
    // `_indexed_at` present + no `_guardrail_flagged` is precisely what
    // `classify_hit` reads as `screened_skipped` rather than `unscreened`.
    assert!(
        image_rec["fields"].get("_guardrail_flagged").is_none(),
        "screening did not apply, so no verdict may be claimed: {image_rec}"
    );
}
