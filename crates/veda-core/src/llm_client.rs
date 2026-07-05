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

use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::{Result, VedaError};

#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// e.g. `http://difgpu01:9000/v1/chat/completions`
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
            .filter(|s| !s.is_empty())
            .map(|e| normalize_endpoint(&e))?;
        let model = std::env::var("VEDA_LLM_MODEL").unwrap_or_else(|_| {
            // Sensible default for OpenAI-compatible deployments
            // where the model name is part of the URL or single-tenant
            // on the endpoint.
            "default".to_owned()
        });
        let api_key = std::env::var("VEDA_LLM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
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

    /// Read a DEDICATED answer-generation LLM config from `ANSWER_GEN_*`
    /// env vars — intentionally separate from `VEDA_LLM_*` (used by
    /// agents/triage/faithfulness-judging elsewhere), since eval-set
    /// generation may run against different, production-shared infra
    /// with its own concurrency budget (see `AppStateInner.answer_gen_semaphore`).
    /// Returns `None` if `ANSWER_GEN_BASE_URL` is unset — eval-set
    /// generation and the auto-optimization trigger both no-op cleanly.
    pub fn from_env_answer_gen() -> Option<Self> {
        let endpoint = std::env::var("ANSWER_GEN_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|e| normalize_endpoint(&e))?;
        let model = std::env::var("ANSWER_GEN_MODEL").unwrap_or_else(|_| "default".to_owned());
        let api_key = std::env::var("ANSWER_GEN_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let timeout = std::env::var("ANSWER_GEN_TIMEOUT_SECS")
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

/// Normalize an OpenAI-compatible chat endpoint so it always points at the
/// `chat/completions` path. A common misconfiguration is supplying only the
/// API base (e.g. `http://host:9000/v1`), which 404s because this client
/// POSTs the body to the endpoint verbatim. If the endpoint already ends in
/// `chat/completions` it is returned unchanged; otherwise `/chat/completions`
/// is appended (collapsing any trailing slash). Full, already-correct URLs
/// pass through untouched.
pub fn normalize_endpoint(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") || trimmed.ends_with("chat/completions") {
        trimmed.to_owned()
    } else {
        format!("{trimmed}/chat/completions")
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
        serde_json::from_str(&self.arguments)
            .map_err(|e| VedaError::Query(format!("tool call arguments parse failed: {e}")))
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
    /// Streaming flag. `None` (the default for `complete`/`complete_with_tools`)
    /// is skipped, so the non-streaming wire format is byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// `{"include_usage": true}` so the terminal SSE chunk carries token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
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

/// One incremental piece of a streamed completion. `content` is the new
/// text fragment (`choices[0].delta.content`); empty-content deltas (e.g.
/// the role-only opening chunk, or the terminal usage chunk) are still
/// yielded so callers see a faithful event sequence. `usage`, when set,
/// carries the final token tally from the terminal chunk
/// (`stream_options.include_usage`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamDelta {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Streaming SSE chunk shape: `{"choices":[{"delta":{"content":"..."}}], "usage": {...}}`.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDeltaWire,
}

#[derive(Debug, Deserialize, Default)]
struct StreamDeltaWire {
    #[serde(default)]
    content: Option<String>,
}

/// Parse the body of a single SSE `data:` line (the JSON after `data: `)
/// into a [`StreamDelta`]. Returns `Ok(None)` for the terminal `[DONE]`
/// sentinel. Split out so the parse is unit-testable without a network
/// call.
fn parse_stream_data(data: &str) -> Result<Option<StreamDelta>> {
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(None);
    }
    let chunk: StreamChunk = serde_json::from_str(data)
        .map_err(|e| VedaError::Query(format!("llm stream chunk parse failed: {e}")))?;
    let content = chunk
        .choices
        .first()
        .and_then(|c| c.delta.content.clone())
        .unwrap_or_default();
    let usage = chunk.usage.unwrap_or_default();
    Ok(Some(StreamDelta {
        content,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }))
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
            stream: None,
            stream_options: None,
        };
        let mut trace_headers = reqwest::header::HeaderMap::new();
        crate::trace_propagation::inject_current_context(&mut trace_headers);
        let mut req = self
            .http
            .post(&self.config.endpoint)
            .headers(trace_headers)
            .json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VedaError::Query(format!("llm request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VedaError::Query(format!(
                "llm endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>(),
            )));
        }
        let chat: ChatResponse = resp
            .json()
            .await
            .map_err(|e| VedaError::Query(format!("llm response parse failed: {e}")))?;
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
            stream: None,
            stream_options: None,
        };
        let mut trace_headers = reqwest::header::HeaderMap::new();
        crate::trace_propagation::inject_current_context(&mut trace_headers);
        let mut req = self
            .http
            .post(&self.config.endpoint)
            .headers(trace_headers)
            .json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VedaError::Query(format!("llm request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VedaError::Query(format!(
                "llm endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>(),
            )));
        }
        let chat: ChatResponse = resp
            .json()
            .await
            .map_err(|e| VedaError::Query(format!("llm response parse failed: {e}")))?;
        Ok(Self::completion_from_response(chat))
    }

    /// Stream a content-only completion (NO tools). POSTs with
    /// `stream: true` and parses the `text/event-stream` body into ordered
    /// [`StreamDelta`]s. Token usage, when the endpoint honors
    /// `stream_options.include_usage`, arrives on the terminal chunk's
    /// delta. The HTTP request is made eagerly (so endpoint/auth errors
    /// surface from this call); the returned stream then yields the body
    /// line by line.
    #[instrument(
        name = "llm_complete_stream",
        skip(self, messages, opts),
        fields(model = %self.config.model, n_messages = messages.len())
    )]
    pub async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        opts: &CompletionOptions,
    ) -> Result<impl Stream<Item = Result<StreamDelta>>> {
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
            stream: Some(true),
            stream_options: Some(serde_json::json!({ "include_usage": true })),
        };
        let mut trace_headers = reqwest::header::HeaderMap::new();
        crate::trace_propagation::inject_current_context(&mut trace_headers);
        let mut req = self
            .http
            .post(&self.config.endpoint)
            .headers(trace_headers)
            .json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VedaError::Query(format!("llm request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VedaError::Query(format!(
                "llm endpoint returned {status}: {}",
                text.chars().take(500).collect::<String>(),
            )));
        }

        // Reassemble SSE frames from the raw byte stream: accumulate a
        // text buffer, split on newlines, and parse each `data:` line.
        let byte_stream = resp.bytes_stream();
        let parsed = stream::unfold(
            (byte_stream, String::new(), false),
            |(mut bytes, mut buf, mut done)| async move {
                loop {
                    if done {
                        return None;
                    }
                    // Drain any complete lines already buffered.
                    if let Some(idx) = buf.find('\n') {
                        let line: String = buf.drain(..=idx).collect();
                        let line = line.trim_end_matches(['\r', '\n']);
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let data = match line.strip_prefix("data:") {
                            Some(d) => d.trim_start(),
                            None => continue, // ignore non-data SSE fields
                        };
                        match parse_stream_data(data) {
                            Ok(Some(delta)) => {
                                return Some((Ok(delta), (bytes, buf, done)));
                            }
                            Ok(None) => {
                                done = true; // [DONE]
                                continue;
                            }
                            Err(e) => {
                                return Some((Err(e), (bytes, buf, true)));
                            }
                        }
                    }
                    // Need more bytes.
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            buf.push_str(&String::from_utf8_lossy(&chunk));
                        }
                        Some(Err(e)) => {
                            return Some((
                                Err(VedaError::Query(format!("llm stream read failed: {e}"))),
                                (bytes, buf, true),
                            ));
                        }
                        None => return None, // stream ended
                    }
                }
            },
        );
        Ok(parsed)
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

/// A boxed stream of [`StreamDelta`]s — the return type of the
/// [`ChatModel::complete_stream`] trait method. Boxed (rather than
/// `impl Stream`) so the trait method can carry a default impl and so
/// callers can hold a single concrete type regardless of the
/// implementation.
pub type DeltaStream<'a> = std::pin::Pin<Box<dyn Stream<Item = Result<StreamDelta>> + Send + 'a>>;

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

    /// Stream a content-only completion as ordered [`StreamDelta`]s.
    ///
    /// The default implementation calls the blocking tool-less path
    /// (`complete_with_tools` with no tools) and yields the whole content
    /// as a single delta, so fakes and non-streaming models keep working
    /// without overriding this. [`LlmClient`] overrides it with real
    /// token-by-token SSE streaming.
    async fn complete_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        opts: &'a CompletionOptions,
    ) -> Result<DeltaStream<'a>> {
        let completion = self.complete_with_tools(messages, &[], None, opts).await?;
        let delta = StreamDelta {
            content: completion.content.unwrap_or_default(),
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            total_tokens: completion.total_tokens,
        };
        Ok(Box::pin(stream::once(async move { Ok(delta) })))
    }
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

    async fn complete_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        opts: &'a CompletionOptions,
    ) -> Result<DeltaStream<'a>> {
        let s = self.complete_stream(messages, opts).await?;
        Ok(Box::pin(s))
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
            stream: None,
            stream_options: None,
        };
        let v = serde_json::to_value(&body).expect("serializes");
        assert_eq!(v["tool_choice"], "auto");
        // The non-streaming request must NOT carry a `stream` field.
        assert!(
            v.get("stream").is_none(),
            "stream leaked into blocking req: {v}"
        );
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
        let v = serde_json::to_value(ChatMessage::text("system", "be brief")).expect("serializes");
        assert_eq!(v, serde_json::json!({"role":"system","content":"be brief"}));
    }

    #[test]
    fn normalize_endpoint_appends_chat_completions_to_api_base() {
        // The common misconfiguration: only the `/v1` base is supplied.
        assert_eq!(
            normalize_endpoint("http://host:9000/v1"),
            "http://host:9000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_endpoint_leaves_full_url_unchanged() {
        assert_eq!(
            normalize_endpoint("http://host:9000/v1/chat/completions"),
            "http://host:9000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_endpoint_handles_trailing_slash() {
        // Trailing slash on the base must not produce a double slash.
        assert_eq!(
            normalize_endpoint("http://host:9000/v1/"),
            "http://host:9000/v1/chat/completions"
        );
        // Trailing slash on an already-correct URL is collapsed, not doubled.
        assert_eq!(
            normalize_endpoint("http://host:9000/v1/chat/completions/"),
            "http://host:9000/v1/chat/completions"
        );
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
    fn parse_stream_data_extracts_content_delta() {
        let d = parse_stream_data(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#)
            .expect("parses")
            .expect("not [DONE]");
        assert_eq!(d.content, "Hel");
        assert_eq!(d.total_tokens, 0);
    }

    #[test]
    fn parse_stream_data_done_sentinel_is_none() {
        assert!(parse_stream_data("[DONE]").expect("ok").is_none());
    }

    #[test]
    fn parse_stream_data_terminal_usage_chunk() {
        // Final chunk: empty choices + a usage block (include_usage).
        let d = parse_stream_data(
            r#"{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}"#,
        )
        .expect("parses")
        .expect("not [DONE]");
        assert_eq!(d.content, "");
        assert_eq!(d.prompt_tokens, 11);
        assert_eq!(d.completion_tokens, 4);
        assert_eq!(d.total_tokens, 15);
    }

    /// A canned OpenAI `data:`-line stream parses into ordered content
    /// deltas, terminating cleanly on `[DONE]`.
    #[test]
    fn canned_sse_stream_parses_into_ordered_deltas() {
        let raw = "\
data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"content\":\", world\"}}]}\n\
\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\
\n\
data: [DONE]\n\
\n";
        let mut contents: Vec<String> = Vec::new();
        let mut final_total = 0u32;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let data = line.strip_prefix("data:").unwrap().trim_start();
            match parse_stream_data(data).expect("parses") {
                Some(d) => {
                    if !d.content.is_empty() {
                        contents.push(d.content);
                    }
                    if d.total_tokens > 0 {
                        final_total = d.total_tokens;
                    }
                }
                None => break,
            }
        }
        assert_eq!(contents, vec!["Hello".to_owned(), ", world".to_owned()]);
        assert_eq!(final_total, 5);
    }

    /// The default `ChatModel::complete_stream` impl falls back to the
    /// blocking path and yields the whole content as exactly one delta.
    #[tokio::test]
    async fn default_complete_stream_yields_full_content_as_one_delta() {
        struct Fake;
        impl ChatModel for Fake {
            async fn complete_with_tools(
                &self,
                _m: &[ChatMessage],
                _t: &[ToolDef],
                _tc: Option<&str>,
                _o: &CompletionOptions,
            ) -> Result<ToolCompletion> {
                Ok(ToolCompletion {
                    content: Some("the whole answer".to_owned()),
                    tool_calls: Vec::new(),
                    finish_reason: Some("stop".to_owned()),
                    prompt_tokens: 2,
                    completion_tokens: 3,
                    total_tokens: 5,
                })
            }
        }
        let fake = Fake;
        let msgs = vec![ChatMessage::text("user", "hi")];
        let opts = CompletionOptions::default();
        let stream = fake.complete_stream(&msgs, &opts).await.expect("stream");
        let deltas: Vec<StreamDelta> = stream.map(|r| r.expect("delta ok")).collect().await;
        assert_eq!(deltas.len(), 1, "default impl yields exactly one delta");
        assert_eq!(deltas[0].content, "the whole answer");
        assert_eq!(deltas[0].total_tokens, 5);
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

    #[test]
    fn from_env_answer_gen_reads_dedicated_prefix() {
        // SAFETY: tests run single-threaded within this module via serial env var mutation;
        // mirrors the existing `from_env_reads_...`-style tests in this same block.
        unsafe {
            std::env::remove_var("VEDA_LLM_ENDPOINT");
            std::env::set_var(
                "ANSWER_GEN_BASE_URL",
                "http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1",
            );
            std::env::set_var("ANSWER_GEN_API_KEY", "tensorflow");
            std::env::set_var("ANSWER_GEN_MODEL", "Qwen/Qwen3.5-122B-A10B");
        }
        let cfg = LlmConfig::from_env_answer_gen().expect("answer-gen config should be present");
        assert_eq!(
            cfg.endpoint,
            "http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1/chat/completions"
        );
        assert_eq!(cfg.model, "Qwen/Qwen3.5-122B-A10B");
        assert_eq!(cfg.api_key.as_deref(), Some("tensorflow"));
        unsafe {
            std::env::remove_var("ANSWER_GEN_BASE_URL");
            std::env::remove_var("ANSWER_GEN_API_KEY");
            std::env::remove_var("ANSWER_GEN_MODEL");
        }
    }

    #[test]
    fn from_env_answer_gen_none_when_unset() {
        unsafe {
            std::env::remove_var("ANSWER_GEN_BASE_URL");
        }
        assert!(LlmConfig::from_env_answer_gen().is_none());
    }
}
