//! POST /v1/embed — one-shot text/image -> vector for the search
//! playground. Without this, the only way to drive `/search` is to
//! hand-craft a 1024-dim JSON array, which gates the entire UX.
//!
//! Reuses the same `EmbeddingClient` + `EmbeddingModelRegistry` already
//! wired by `embed_insert` and `process-document`. Model selection:
//!   - explicit `model` name if provided
//!   - else: registry's `find_by_modality(modality)` first match
//!
//! Modality:
//!   - `text` present, no `image_b64` -> Text
//!   - `image_b64` present, no `text` -> Image
//!   - both present or neither -> 400
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
    let (modality, input): (Modality, EmbedInput) = match (&req.text, &req.image_b64) {
        (Some(t), None) if !t.is_empty() => (Modality::Text, EmbedInput::Text(t.clone())),
        (None, Some(b)) if !b.is_empty() => {
            let bytes = BASE64_STANDARD
                .decode(b)
                .map_err(|e| bad_request(format!("invalid base64 in image_b64: {e}")))?;
            (Modality::Image, EmbedInput::Bytes(bytes))
        }
        (Some(_), Some(_)) => {
            return Err(bad_request(
                "only one of `text` or `image_b64` may be set".into(),
            ));
        }
        _ => {
            return Err(bad_request(
                "one of `text` or `image_b64` is required".into(),
            ));
        }
    };

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

enum EmbedInput {
    Text(String),
    Bytes(Vec<u8>),
}

fn bad_request(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg }))
}
