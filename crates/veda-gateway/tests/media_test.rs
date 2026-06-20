//! Integration tests for media ingestion
//! (`POST /v1/collections/{name}/ingest-media`).
//!
//! Uses a mock omni multimodal embedder (a `mock` endpoint → deterministic
//! vectors, no real HTTP) so image/audio/video all embed into one shared
//! vector space. Covers:
//!   - audio file (audio/wav) → 1 record inserted, vector + metadata stamped.
//!   - video file (video/mp4) → 1 record inserted.
//!   - non-media MIME (text/plain) → clean rejection, no record.

use reqwest::Client;
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_gateway::rest::{build_router, state::AppState};

const DIM: u32 = 1024;

/// Start a gateway with a pre-registered omni multimodal embedding model (a
/// `mock` endpoint → deterministic vectors). Image/Audio/Video all resolve
/// to this single model — one shared vector space.
async fn start_gateway() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    {
        let mut registry = state.embedding_registry().write().await;
        registry.register(EmbeddingModelConfig {
            name: "jina-v5-omni-small".into(),
            endpoint: "http://mock-omni:9999/embed".into(),
            api_key: String::new(),
            supported_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
            ],
            output_dimension: DIM,
            max_input_size_bytes: 20 * 1024 * 1024,
            timeout_seconds: 60,
            batch_size: 8,
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

/// Create a collection with a `media` auto_embed vector field of the given
/// modality plus the metadata fields `ingest_media` stamps.
async fn create_media_collection(base: &str, name: &str, modality: &str) {
    let resp = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "content", "type": "string", "filterable": true },
                { "name": "filename", "type": "string", "filterable": true },
                { "name": "source_path", "type": "string" },
                { "name": "class", "type": "string", "filterable": true },
                { "name": "duration_secs", "type": "float64" },
                { "name": "media_blob", "type": "blob" },
                {
                    "name": "media",
                    "type": "vector",
                    "dimension": DIM,
                    "metric": "cosine",
                    "auto_embed": { "source": "media_blob", "modality": modality }
                }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        201,
        "create collection failed: {:?}",
        resp.text().await
    );
}

async fn ingest(base: &str, col: &str, filename: &str, mime: &str, bytes: Vec<u8>) -> (u16, Value) {
    let part = Part::bytes(bytes)
        .file_name(filename.to_owned())
        .mime_str(mime)
        .unwrap();
    let form = Form::new()
        .part("file", part)
        .text("class", "demo")
        .text("source_path", format!("/uploads/{filename}"));
    let resp = client()
        .post(format!("{base}/v1/collections/{col}/ingest-media"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body: Value = resp.json().await.unwrap();
    (status, body)
}

#[tokio::test]
async fn ingest_media_audio_inserts_record() {
    let (base, _tmp) = start_gateway().await;
    create_media_collection(&base, "audio_col", "audio").await;

    // Fake WAV: a minimal RIFF/WAVE header is enough — the mock embedder
    // hashes bytes deterministically and never parses the audio.
    let mut wav = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
    wav.extend_from_slice(&[0u8; 32]);
    let (status, body) = ingest(&base, "audio_col", "clip.wav", "audio/wav", wav).await;

    assert_eq!(status, 201, "ingest-media audio failed: {body}");
    assert_eq!(body["records_inserted"], 1);
    assert_eq!(body["vector_dim"], DIM);
    assert_eq!(body["modality"], "Audio");
    assert!(
        body["blob_key"].as_str().is_some(),
        "blob_key missing: {body}"
    );

    // The record is searchable (data inserted with a vector) and carries the
    // stamped metadata.
    let search = client()
        .post(format!("{base}/v1/collections/audio_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "media", "values": vec![0.1f32; DIM as usize] } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(search.status(), 200);
    let sb: Value = search.json().await.unwrap();
    assert_eq!(
        sb["count"], 1,
        "should find the inserted audio record: {sb}"
    );
    let rec = &sb["results"][0]["fields"];
    assert_eq!(rec["filename"], "clip.wav", "filename not stamped: {sb}");
    assert_eq!(rec["class"], "demo", "class not stamped: {sb}");
}

#[tokio::test]
async fn ingest_media_video_inserts_record() {
    let (base, _tmp) = start_gateway().await;
    create_media_collection(&base, "video_col", "video").await;

    // Fake MP4: an ftyp box prefix is enough for a deterministic mock embed.
    let mut mp4 = b"\x00\x00\x00\x18ftypmp42".to_vec();
    mp4.extend_from_slice(&[0u8; 64]);
    let (status, body) = ingest(&base, "video_col", "clip.mp4", "video/mp4", mp4).await;

    assert_eq!(status, 201, "ingest-media video failed: {body}");
    assert_eq!(body["records_inserted"], 1);
    assert_eq!(body["vector_dim"], DIM);
    assert_eq!(body["modality"], "Video");
    assert!(
        body["blob_key"].as_str().is_some(),
        "blob_key missing: {body}"
    );
}

#[tokio::test]
async fn ingest_media_unknown_mime_errors() {
    let (base, _tmp) = start_gateway().await;
    create_media_collection(&base, "reject_col", "audio").await;

    let (status, body) = ingest(
        &base,
        "reject_col",
        "notes.txt",
        "text/plain",
        b"just some text".to_vec(),
    )
    .await;

    assert_eq!(status, 400, "non-media MIME must be rejected: {body}");
    assert_eq!(body["error"]["code"], "UNSUPPORTED_MEDIA");
}
