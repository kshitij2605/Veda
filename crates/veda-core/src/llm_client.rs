//! OpenAI-compatible chat-completion client.
//!
//! Veda doesn't host an LLM but the dashboard needs one for several
//! features: cluster labels in Atlas / Quality Lab, failure-mode
//! classification on eval misses, natural-language → Veda-filter
//! drafting, and faithfulness scoring in the eval pipeline.
//!
//! This client is a thin async wrapper over the `chat/completions`
//! HTTP API (the de-facto standard since OpenAI; supported by
//! vLLM, llama.cpp server, Ollama, Together, Anyscale, etc.). It is
//! deliberately *not* feature-rich — no streaming, no tool calls,
//! no function calling. The dashboard's use cases are short
//! one-shot prompts (≤2 KB in, ≤2 KB out) and that's all this
//! does.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::{Result, VedaError};

#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// e.g. `http://llm-host:9000/v1/chat/completions`
    pub endpoint: String,
    pub api_key: Option<String>,
    /// e.g. `Qwen/Qwen3.5-122B-A10B`
    pub model: String,
    /// Per-call timeout. The dashboard fires off small prompts; a
    /// 30 s ceiling is generous and protects the UI from hanging on
    /// a sick endpoint.
    pub timeout: Duration,
}

impl LlmConfig {
    /// Read configuration from environment. Returns `None` if no
    /// endpoint is configured — Veda runs fine without an LLM, the
    /// UI just hides the labeling features.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("VEDA_LLM_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())?;
        let model = std::env::var("VEDA_LLM_MODEL").unwrap_or_else(|_| {
            // Sensible default for OpenAI-compatible deployments
            // where the model name is part of the URL or single-tenant
            // on the endpoint.
            "default".to_owned()
        });
        let api_key = std::env::var("VEDA_LLM_API_KEY").ok().filter(|s| !s.is_empty());
        let timeout = std::env::var("VEDA_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));
        Some(Self {
            endpoint,
            api_key,
            model,
            timeout,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResult {
    pub content: String,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub elapsed_ms: f64,
}

/// Optional knobs for a single completion call.
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Force JSON-mode where supported (`{"type": "json_object"}`).
    pub json_mode: bool,
}

pub struct LlmClient {
    config: LlmConfig,
    http: reqwest::Client,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("reqwest client builds");
        Self { config, http }
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    #[instrument(
        name = "llm_complete",
        skip(self, messages, opts),
        fields(model = %self.config.model, n_messages = messages.len())
    )]
    pub async fn complete(
        &self,
        messages: &[ChatMessage],
        opts: &CompletionOptions,
    ) -> Result<CompletionResult> {
        let start = std::time::Instant::now();
        let body = ChatRequest {
            model: &self.config.model,
            messages,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            response_format: if opts.json_mode {
                Some(serde_json::json!({ "type": "json_object" }))
            } else {
                None
            },
        };
        let mut req = self.http.post(&self.config.endpoint).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| {
            VedaError::Query(format!("llm request failed: {e}"))
        })?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VedaError::Query(format!(
                "llm endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>(),
            )));
        }
        let chat: ChatResponse = resp.json().await.map_err(|e| {
            VedaError::Query(format!("llm response parse failed: {e}"))
        })?;
        let choice = chat
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| VedaError::Query("llm returned no choices".into()))?;
        let usage = chat.usage.unwrap_or_default();
        Ok(CompletionResult {
            content: choice.message.content,
            finish_reason: choice.finish_reason,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
