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
    /// `content` is optional because an assistant message that carries
    /// `tool_calls` may have a `null` content. Plain messages always set it,
    /// so they serialize unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Present on assistant messages that request tool invocations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on `role: "tool"` messages to bind a result to a prior call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Plain text message (the shape the dashboard's `complete()` uses).
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// A `role: "tool"` message carrying the result of a prior tool call.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_owned(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// OpenAI tool definition: `{"type":"function","function":{...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

impl ToolDef {
    /// Build a `function`-type tool from a name, description and a
    /// JSON-Schema object describing its parameters.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_owned(),
            function: FunctionDef {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    /// JSON-Schema object for the function's arguments.
    pub parameters: serde_json::Value,
}

/// A tool invocation requested by the assistant in a chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// `arguments` is a JSON-*encoded* string, not a JSON object.
    pub arguments: String,
}

impl FunctionCall {
    /// Parse the JSON-encoded `arguments` string into a `serde_json::Value`.
    pub fn parse_arguments(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments).map_err(|e| {
            VedaError::Query(format!("tool call arguments parse failed: {e}"))
        })
    }
}

/// Choice of which tool the model may call: `"auto"`, `"none"`, `"required"`.
pub type ToolChoice = String;

/// Result of a `complete_with_tools` call.
#[derive(Debug, Clone)]
pub struct ToolCompletion {
    /// Assistant text content. `None` when the model only emitted tool calls.
    pub content: Option<String>,
    /// Tool invocations requested by the model (empty when none).
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a str>,
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
            tools: None,
            tool_choice: None,
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
            content: choice.message.content.unwrap_or_default(),
            finish_reason: choice.finish_reason,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[instrument(
        name = "llm_complete_with_tools",
        skip(self, messages, tools, opts),
        fields(model = %self.config.model, n_messages = messages.len(), n_tools = tools.len())
    )]
    pub async fn complete_with_tools_impl(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        tool_choice: Option<&str>,
        opts: &CompletionOptions,
    ) -> Result<ToolCompletion> {
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
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
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
        Ok(Self::completion_from_response(chat))
    }

    /// Map a parsed chat response onto a `ToolCompletion`. Split out so the
    /// deserialization path is testable without a network call.
    fn completion_from_response(chat: ChatResponse) -> ToolCompletion {
        let finish_reason = chat.choices.first().and_then(|c| c.finish_reason.clone());
        let (content, tool_calls) = match chat.choices.into_iter().next() {
            Some(choice) => (
                choice.message.content,
                choice.message.tool_calls.unwrap_or_default(),
            ),
            None => (None, Vec::new()),
        };
        let usage = chat.usage.unwrap_or_default();
        ToolCompletion {
            content,
            tool_calls,
            finish_reason,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

/// Abstraction over a tool-calling chat model so callers (agents, tests)
/// can substitute a fake implementation. Implemented by [`LlmClient`].
#[allow(async_fn_in_trait)]
pub trait ChatModel {
    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        tool_choice: Option<&str>,
        opts: &CompletionOptions,
    ) -> Result<ToolCompletion>;
}

impl ChatModel for LlmClient {
    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        tool_choice: Option<&str>,
        opts: &CompletionOptions,
    ) -> Result<ToolCompletion> {
        self.complete_with_tools_impl(messages, tools, tool_choice, opts)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_with_tools_serializes_to_openai_shape() {
        let tools = vec![ToolDef::function(
            "search",
            "Search the collection",
            serde_json::json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"],
            }),
        )];
        let messages = vec![ChatMessage::text("user", "find cats")];
        let body = ChatRequest {
            model: "default",
            messages: &messages,
            temperature: None,
            max_tokens: None,
            response_format: None,
            tools: Some(&tools),
            tool_choice: Some("auto"),
        };
        let v = serde_json::to_value(&body).expect("serializes");
        assert_eq!(v["tool_choice"], "auto");
        let t = &v["tools"][0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "search");
        assert_eq!(t["function"]["description"], "Search the collection");
        assert_eq!(t["function"]["parameters"]["type"], "object");
        assert_eq!(
            t["function"]["parameters"]["properties"]["q"]["type"],
            "string"
        );
        // A plain user message still serializes with a string `content`
        // and without tool fields.
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "find cats");
        assert!(v["messages"][0].get("tool_calls").is_none());
        assert!(v["messages"][0].get("tool_call_id").is_none());
    }

    #[test]
    fn plain_message_serializes_unchanged() {
        // Guards the existing `complete()` wire format: {role, content}.
        let v = serde_json::to_value(ChatMessage::text("system", "be brief"))
            .expect("serializes");
        assert_eq!(v, serde_json::json!({"role":"system","content":"be brief"}));
    }

    #[test]
    fn response_with_tool_calls_parses() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"q\":\"x\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        });
        let chat: ChatResponse = serde_json::from_value(raw).expect("parses");
        let completion = LlmClient::completion_from_response(chat);
        assert!(completion.content.is_none());
        assert_eq!(completion.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(completion.tool_calls.len(), 1);
        let call = &completion.tool_calls[0];
        assert_eq!(call.id, "call_abc");
        assert_eq!(call.kind, "function");
        assert_eq!(call.function.name, "search");
        assert_eq!(call.function.arguments, "{\"q\":\"x\"}");
        let args = call.function.parse_arguments().expect("args parse");
        assert_eq!(args, serde_json::json!({ "q": "x" }));
        assert_eq!(completion.prompt_tokens, 10);
        assert_eq!(completion.total_tokens, 15);
    }

    #[test]
    fn response_with_plain_content_parses_with_no_tool_calls() {
        let raw = serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "hello there" },
                "finish_reason": "stop"
            }]
        });
        let chat: ChatResponse = serde_json::from_value(raw).expect("parses");
        let completion = LlmClient::completion_from_response(chat);
        assert_eq!(completion.content.as_deref(), Some("hello there"));
        assert!(completion.tool_calls.is_empty());
        assert_eq!(completion.finish_reason.as_deref(), Some("stop"));
    }
}
