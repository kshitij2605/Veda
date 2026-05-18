//! POST /v1/llm/complete — proxies to the configured LLM endpoint.
//!
//! Several Niriksha surfaces need an LLM:
//!   * Atlas / Quality-Lab cluster labels (~50 prompts of <=10 tokens each)
//!   * Failure-mode auto-classification on eval misses
//!   * NL → Veda filter DSL drafting in the command palette
//!   * Faithfulness / answer-relevancy judge in eval runs
//!
//! Configuration is purely env-driven so an operator who doesn't want
//! an LLM gets a 503 here and the dashboard hides the related buttons:
//!
//!   VEDA_LLM_ENDPOINT=http://llm-host.internal.example:9000/v1/chat/completions
//!   VEDA_LLM_API_KEY=...
//!   VEDA_LLM_MODEL=Qwen/Qwen3.5-122B-A10B
//!
//! Wire: the client mirrors OpenAI `chat/completions`, so this also
//! works against vLLM, llama.cpp server, Ollama, etc.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use veda_core::llm_client::{
    ChatMessage, CompletionOptions, CompletionResult, LlmClient,
};

use super::collections::ErrorResponse;
use super::state::AppState;

#[derive(Debug, Deserialize)]
pub struct LlmRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub json_mode: bool,
}

#[derive(Debug, Serialize)]
pub struct LlmResponse {
    pub model: String,
    #[serde(flatten)]
    pub result: CompletionResult,
}

#[instrument(name = "llm_complete", skip(state, req), fields(n_messages = req.messages.len()))]
pub async fn complete(
    State(state): State<AppState>,
    Json(req): Json<LlmRequest>,
) -> Result<Json<LlmResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cfg = state.inner.llm_config.read().await.clone();
    let cfg = cfg.ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "VEDA_LLM_ENDPOINT not configured. Set the env var to enable LLM-backed features (cluster labels, failure classification, NL-to-DSL).".to_owned(),
            }),
        )
    })?;
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "messages must not be empty".to_owned(),
            }),
        ));
    }
    let model = cfg.model.clone();
    let client = LlmClient::new(cfg);
    let opts = CompletionOptions {
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        json_mode: req.json_mode,
    };
    match client.complete(&req.messages, &opts).await {
        Ok(result) => Ok(Json(LlmResponse { model, result })),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: format!("llm call failed: {e}"),
            }),
        )),
    }
}
