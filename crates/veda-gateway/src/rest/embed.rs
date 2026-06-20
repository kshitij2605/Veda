//! POST /v1/embed — one-shot text/image -> vector for the search
//! playground. Without this, the only way to drive `/search` is to
//! hand-craft a 1024-dim JSON array, which gates the entire UX.
//!
//! Reuses the same `EmbeddingClient` + `EmbeddingModelRegistry` already
//! wired by `embed_insert` and `process-document`. Model selection:
//!   - explicit `model` name if provided
//!   - else: registry's `find_by_modality(modality)` first match
//!
//! Modality: exactly one of `text` / `image_b64` / `video_b64` /
//! `audio_b64` selects Text / Image / Video / Audio respectively;
//! zero or more-than-one set -> 400.
//!
//! Response carries the vector + the model that produced it + a wall
//! clock so the playground can show "embedded in 47 ms".

use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;

use super::collections::ErrorResponse;
use super::state::AppState;

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub text: Option<String>,
    pub image_b64: Option<String>,
    pub video_b64: Option<String>,
    pub audio_b64: Option<String>,
    /// Optional explicit model id. If omitted, picks the first model in
    /// the registry that supports the inferred modality.
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub modality: String,
    pub dimension: u32,
    pub values: Vec<f32>,
    pub elapsed_ms: f64,
}

#[instrument(name = "embed", skip(state, req))]
pub async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();

    // 1) Infer modality from which field is populated.
    let (modality, input) = resolve_modality(&req).map_err(bad_request)?;

    // 2) Resolve which model to use.
    let registry = state.inner.embedding_registry.read().await;
    let model_config = if let Some(name) = req.model.as_deref() {
        registry
            .get(name)
            .ok_or_else(|| bad_request(format!("model '{name}' not in registry")))?
            .clone()
    } else {
        let candidates = registry.find_by_modality(modality);
        let first = *candidates.first().ok_or_else(|| {
            bad_request(format!(
                "no model registered for modality '{modality}'. \
                 Available: {:?}",
                registry
                    .list()
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
            ))
        })?;
        first.clone()
    };
    drop(registry);

    let model_name = model_config.name.clone();
    let dim = model_config.output_dimension;
    let client = EmbeddingClient::new(model_config);

    // 3) Embed.
    let values = match input {
        EmbedInput::Text(t) => client.embed_text(&t).await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("embed_text failed: {e}"),
                }),
            )
        })?,
        EmbedInput::Bytes(b) => client.embed_bytes(&b, modality).await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("embed_bytes failed: {e}"),
                }),
            )
        })?,
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(EmbedResponse {
        model: model_name,
        modality: modality.to_string(),
        dimension: dim,
        values,
        elapsed_ms,
    }))
}

#[derive(Debug)]
enum EmbedInput {
    Text(String),
    Bytes(Vec<u8>),
}

/// Resolve the request to exactly one modality + decoded input.
///
/// Enforces exactly-one-of over `{text, image_b64, video_b64, audio_b64}`:
/// zero set or two-or-more set is an error. Base64 fields are decoded here
/// so a malformed payload fails fast before model resolution.
fn resolve_modality(req: &EmbedRequest) -> Result<(Modality, EmbedInput), String> {
    let candidates = [
        ("text", req.text.as_deref()),
        ("image_b64", req.image_b64.as_deref()),
        ("video_b64", req.video_b64.as_deref()),
        ("audio_b64", req.audio_b64.as_deref()),
    ];

    let set: Vec<(&str, &str)> = candidates
        .into_iter()
        .filter_map(|(name, val)| match val {
            Some(v) if !v.is_empty() => Some((name, v)),
            _ => None,
        })
        .collect();

    match set.as_slice() {
        [] => Err(
            "exactly one of `text`, `image_b64`, `video_b64`, or `audio_b64` is required".into(),
        ),
        [(field, value)] => match *field {
            "text" => Ok((Modality::Text, EmbedInput::Text((*value).to_owned()))),
            "image_b64" => Ok((
                Modality::Image,
                EmbedInput::Bytes(decode_b64(field, value)?),
            )),
            "video_b64" => Ok((
                Modality::Video,
                EmbedInput::Bytes(decode_b64(field, value)?),
            )),
            "audio_b64" => Ok((
                Modality::Audio,
                EmbedInput::Bytes(decode_b64(field, value)?),
            )),
            _ => unreachable!("candidate set is fixed"),
        },
        _ => Err(format!(
            "only one of `text`, `image_b64`, `video_b64`, or `audio_b64` may be set (got {})",
            set.iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn decode_b64(field: &str, value: &str) -> Result<Vec<u8>, String> {
    BASE64_STANDARD
        .decode(value)
        .map_err(|e| format!("invalid base64 in {field}: {e}"))
}

fn bad_request(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(
        text: Option<&str>,
        image: Option<&str>,
        video: Option<&str>,
        audio: Option<&str>,
    ) -> EmbedRequest {
        EmbedRequest {
            text: text.map(str::to_owned),
            image_b64: image.map(str::to_owned),
            video_b64: video.map(str::to_owned),
            audio_b64: audio.map(str::to_owned),
            model: None,
        }
    }

    // BAStandard("hi") == "aGk=" ; reuse a known-valid base64 blob.
    const VALID_B64: &str = "aGVsbG8="; // "hello"

    #[test]
    fn text_resolves_to_text() {
        let (m, input) = resolve_modality(&req(Some("hi"), None, None, None)).unwrap();
        assert_eq!(m, Modality::Text);
        assert!(matches!(input, EmbedInput::Text(t) if t == "hi"));
    }

    #[test]
    fn image_resolves_to_image() {
        let (m, input) = resolve_modality(&req(None, Some(VALID_B64), None, None)).unwrap();
        assert_eq!(m, Modality::Image);
        assert!(matches!(input, EmbedInput::Bytes(b) if b == b"hello"));
    }

    #[test]
    fn video_resolves_to_video() {
        let (m, _) = resolve_modality(&req(None, None, Some(VALID_B64), None)).unwrap();
        assert_eq!(m, Modality::Video);
    }

    #[test]
    fn audio_resolves_to_audio() {
        let (m, _) = resolve_modality(&req(None, None, None, Some(VALID_B64))).unwrap();
        assert_eq!(m, Modality::Audio);
    }

    #[test]
    fn two_fields_set_is_error() {
        let err = resolve_modality(&req(Some("hi"), Some(VALID_B64), None, None)).unwrap_err();
        assert!(err.contains("only one of"));
    }

    #[test]
    fn no_fields_set_is_error() {
        let err = resolve_modality(&req(None, None, None, None)).unwrap_err();
        assert!(err.contains("exactly one of"));
    }

    #[test]
    fn bad_base64_is_error() {
        let err = resolve_modality(&req(None, Some("!!!not-base64!!!"), None, None)).unwrap_err();
        assert!(err.contains("invalid base64 in image_b64"));
    }
}
