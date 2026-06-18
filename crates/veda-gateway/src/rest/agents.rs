//! REST surface for Veda Agents.
//!
//! Wires the agent data model ([`crate::agent::model`]), persistence
//! ([`crate::agent::store`]), tool catalog ([`crate::agent::catalog`]),
//! and executor ([`crate::agent::executor`]) into the protected REST
//! router. Auth + tenant middleware apply to every route here.
//!
//! All agent / tool / conversation ids are tenant-scoped on the way in
//! (via [`crate::tenant::scope`]) and the prefix is stripped on the way
//! out, mirroring how collections are isolated, so tenants never see
//! one another's objects.
//!
//! Endpoints:
//!   GET    /v1/agents                         list
//!   POST   /v1/agents                         create/upsert (id optional)
//!   GET    /v1/agents/{id}                     fetch
//!   PUT    /v1/agents/{id}                     upsert
//!   DELETE /v1/agents/{id}                     delete
//!   GET    /v1/agent-tools                     list custom tools
//!   POST   /v1/agent-tools                     create/upsert
//!   GET    /v1/agent-tools/catalog             built-in tool specs
//!   GET    /v1/agent-tools/{id}                fetch
//!   PUT    /v1/agent-tools/{id}                upsert
//!   DELETE /v1/agent-tools/{id}                delete
//!   POST   /v1/agent-tools/{id}/execute        run a single built-in tool
//!   POST   /v1/agents/{id}/converse            chat with an agent
//!   GET    /v1/conversations                   list
//!   GET    /v1/conversations/{id}              fetch
//!   DELETE /v1/conversations/{id}              delete

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use tracing::Instrument;

use veda_core::llm_client::LlmClient;

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::agent::catalog::{ToolContext, ToolErrorKind, ToolSpec};
use crate::agent::executor::{self, DEFAULT_MAX_STEPS};
use crate::agent::grounding::{self, Citation, Grounding};
use crate::agent::model::{
    default_agent, Agent, AgentTool, Conversation, ConversationMessage, ConverseRequest,
    ConverseResponse,
};
use crate::agent::store;
use crate::auth::Claims;
use crate::tenant::{self, Tenant, TenantId};

type GatewayError = (StatusCode, Json<ErrorResponse>);

fn not_found(msg: impl Into<String>) -> GatewayError {
    (StatusCode::NOT_FOUND, Json(ErrorResponse { error: msg.into() }))
}
fn service_unavailable(msg: impl Into<String>) -> GatewayError {
    (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: msg.into() }))
}

/// Extract the optional authenticated subject for RLS: `None` when the
/// caller is an admin (or auth is disabled / no claims), else `Some(sub)`
/// so the tools enforce owner scoping.
pub(crate) fn rls_user(claims: &Option<Claims>) -> Option<String> {
    match claims {
        Some(c) if !crate::rls::is_admin(c) => Some(c.sub.clone()),
        _ => None,
    }
}

// ── Agents CRUD ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ListAgentsResponse {
    pub agents: Vec<Agent>,
}

pub async fn list_agents(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ListAgentsResponse> {
    let mut agents = store::list_agents(&state.inner.saved_objects).await;
    agents.retain(|a| tenant::strip_prefix(&tenant, &a.id).is_some());
    for a in &mut agents {
        if let Some(name) = tenant::strip_prefix(&tenant, &a.id) {
            a.id = name.to_owned();
        }
    }
    Json(ListAgentsResponse { agents })
}

/// Create or upsert an agent. If the body carries an empty id one is
/// generated. The user-facing id is returned (tenant prefix stripped).
pub async fn create_agent(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Json(mut agent): Json<Agent>,
) -> Result<Json<Agent>, GatewayError> {
    if agent.id.trim().is_empty() {
        agent.id = uuid::Uuid::new_v4().to_string();
    }
    save_and_return_agent(&state, &tenant, agent).await
}

pub async fn get_agent(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<Agent>, GatewayError> {
    let scoped = tenant::scope(&tenant, &id);
    let mut agent = store::get_agent(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| not_found(format!("agent '{id}' not found")))?;
    agent.id = id;
    Ok(Json(agent))
}

pub async fn put_agent(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(mut agent): Json<Agent>,
) -> Result<Json<Agent>, GatewayError> {
    agent.id = id;
    save_and_return_agent(&state, &tenant, agent).await
}

pub async fn delete_agent(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> StatusCode {
    let scoped = tenant::scope(&tenant, &id);
    store::delete_agent(&state.inner.saved_objects, &scoped).await;
    StatusCode::NO_CONTENT
}

/// Persist `agent` under the tenant-scoped id and return it with the
/// user-facing (unscoped) id.
async fn save_and_return_agent(
    state: &AppState,
    tenant: &TenantId,
    mut agent: Agent,
) -> Result<Json<Agent>, GatewayError> {
    let user_id = agent.id.clone();
    agent.id = tenant::scope(tenant, &user_id);
    store::save_agent(&state.inner.saved_objects, &agent).await;
    agent.id = user_id;
    Ok(Json(agent))
}

// ── AgentTool CRUD ───────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ListToolsResponse {
    pub tools: Vec<AgentTool>,
}

pub async fn list_tools(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ListToolsResponse> {
    let mut tools = store::list_tools(&state.inner.saved_objects).await;
    tools.retain(|t| tenant::strip_prefix(&tenant, &t.id).is_some());
    for t in &mut tools {
        if let Some(name) = tenant::strip_prefix(&tenant, &t.id) {
            t.id = name.to_owned();
        }
    }
    Json(ListToolsResponse { tools })
}

pub async fn create_tool(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Json(mut tool): Json<AgentTool>,
) -> Result<Json<AgentTool>, GatewayError> {
    if tool.id.trim().is_empty() {
        tool.id = uuid::Uuid::new_v4().to_string();
    }
    save_and_return_tool(&state, &tenant, tool).await
}

pub async fn get_tool(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<AgentTool>, GatewayError> {
    let scoped = tenant::scope(&tenant, &id);
    let mut tool = store::get_tool(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| not_found(format!("agent tool '{id}' not found")))?;
    tool.id = id;
    Ok(Json(tool))
}

pub async fn put_tool(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(mut tool): Json<AgentTool>,
) -> Result<Json<AgentTool>, GatewayError> {
    tool.id = id;
    save_and_return_tool(&state, &tenant, tool).await
}

pub async fn delete_tool(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> StatusCode {
    let scoped = tenant::scope(&tenant, &id);
    store::delete_tool(&state.inner.saved_objects, &scoped).await;
    StatusCode::NO_CONTENT
}

async fn save_and_return_tool(
    state: &AppState,
    tenant: &TenantId,
    mut tool: AgentTool,
) -> Result<Json<AgentTool>, GatewayError> {
    let user_id = tool.id.clone();
    tool.id = tenant::scope(tenant, &user_id);
    store::save_tool(&state.inner.saved_objects, &tool).await;
    tool.id = user_id;
    Ok(Json(tool))
}

// ── Built-in tool catalog + direct execute ───────────────────────────────

#[derive(Serialize)]
pub struct CatalogResponse {
    pub tools: Vec<ToolSpec>,
}

pub async fn catalog(State(state): State<AppState>) -> Json<CatalogResponse> {
    Json(CatalogResponse {
        tools: state.inner.tool_catalog.builtin_specs().to_vec(),
    })
}

/// Execute a single built-in tool directly. `{id}` is the catalog tool
/// id (e.g. `list_collections`). The body is either `{ "args": {...} }`
/// or the args object itself. Maps the tool's [`ToolErrorKind`] to an
/// HTTP status; success returns 200 with the tool content.
pub async fn execute_tool(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, GatewayError> {
    // Accept `{ "args": {...} }` or the args object directly.
    let args = match body {
        serde_json::Value::Object(ref m) if m.contains_key("args") => {
            body.get("args").cloned().unwrap_or(serde_json::Value::Null)
        }
        other => other,
    };

    let claims = claims.map(|axum::Extension(c)| c);
    let user = rls_user(&claims);
    let ctx = ToolContext::new(&state, &tenant).with_user(user.as_deref());

    let outcome = state.inner.tool_catalog.dispatch(&ctx, &id, args).await;
    if outcome.is_error {
        let status = match outcome.error_kind {
            Some(ToolErrorKind::InvalidArgs) => StatusCode::BAD_REQUEST,
            Some(ToolErrorKind::NotFound) => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let msg = outcome
            .content
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap_or("tool execution failed")
            .to_owned();
        return Err((status, Json(ErrorResponse { error: msg })));
    }
    Ok(Json(outcome.content))
}

// ── Shared agent-turn helper ─────────────────────────────────────────────

use crate::agent::executor::{RunMetrics, RunResult};

/// Failure modes of [`run_agent_turn`], mapped to HTTP/JSON-RPC by callers.
pub(crate) enum AgentTurnError {
    /// No LLM is configured (→ 503).
    NoLlm,
    /// The requested agent does not exist (→ 404). Carries the user id.
    NotFound(String),
}

/// Run a single agent turn: resolve the agent (tenant-scoped, with the
/// `veda-agent` default fallback), build the model honoring any per-agent
/// override, assemble the system prompt + tool context, and drive the
/// executor loop over `history` + `input`. Shared by `converse` and the
/// A2A `message/send` handler so the loop is not duplicated.
///
/// Returns the [`RunResult`] alongside the resolved [`Agent`] (callers may
/// want the scoped id for persistence). Conversation persistence is the
/// caller's responsibility.
pub(crate) async fn run_agent_turn(
    state: &AppState,
    tenant: &TenantId,
    user: Option<&str>,
    agent_id: &str,
    history: &[ConversationMessage],
    input: &str,
) -> Result<(Agent, RunResult, Vec<Citation>, Option<Grounding>), AgentTurnError> {
    // 1. LLM must be configured.
    let llm_cfg = state
        .inner
        .llm_config
        .read()
        .await
        .clone()
        .ok_or(AgentTurnError::NoLlm)?;

    // 2. Resolve the agent (tenant-scoped), falling back to the in-memory
    //    default agent only for the well-known `veda-agent` id.
    let scoped_agent_id = tenant::scope(tenant, agent_id);
    let agent = match store::get_agent(&state.inner.saved_objects, &scoped_agent_id).await {
        Some(a) => a,
        None if agent_id == "veda-agent" => default_agent(),
        None if agent_id == "veda-ops" => crate::agent::model::default_ops_agent(),
        None => return Err(AgentTurnError::NotFound(agent_id.to_owned())),
    };

    // 3. Build the model honoring any per-agent override. Capture the
    //    resolved model name first for metric labelling.
    let model_name = agent
        .model
        .as_ref()
        .and_then(|ov| ov.model.clone())
        .unwrap_or_else(|| llm_cfg.model.clone());
    // GenAI-semconv `gen_ai.system` value. Veda talks to an
    // OpenAI-compatible endpoint generically; infer a recognized provider
    // from the model name where possible, else fall back to `openai`.
    let provider = genai_system_for(&model_name);
    let client = build_client(llm_cfg, &agent);

    // 4. System prompt with the modality-awareness block.
    let summary = executor::collections_modality_summary(state, tenant).await;
    let sys = executor::build_system_prompt(&agent, &summary);

    // 5. Tool context with RLS user.
    let ctx = ToolContext::new(state, tenant).with_user(user);

    // 6. Run the tool-calling loop under a GenAI-semconv `invoke_agent`
    //    root span. The executor's per-call `chat` / `execute_tool` spans
    //    (and any tool retrieval spans) nest under this. Created here at the
    //    single shared invocation point so /converse, A2A, and triage all
    //    get the same span tree. Cheap when no subscriber records.
    use veda_core::genai_semconv as gen_ai;
    let root = tracing::info_span!(
        "invoke_agent",
        otel.name = gen_ai::invoke_agent_span_name(&agent.name),
        "gen_ai.operation.name" = gen_ai::OP_INVOKE_AGENT,
        "gen_ai.agent.name" = %agent.name,
    );
    let result = executor::run(
        &client,
        &state.inner.tool_catalog,
        &ctx,
        &agent,
        &sys,
        history,
        input,
        DEFAULT_MAX_STEPS,
        &provider,
        &model_name,
    )
    .instrument(root)
    .await;

    // 7. Emit APM metrics from this single shared point (covers /converse,
    //    A2A message/send, and ops triage).
    emit_run_metrics(&model_name, &result.metrics);

    // 8. Post-run grounding: extract citations from the trace and (best-
    //    effort, off the hot path) score faithfulness + answer-relevancy.
    //    A scoring failure/timeout must NEVER fail the turn — everything
    //    here is non-fatal and falls back to `None`/no-context.
    let citations = grounding::extract_citations(&result.steps);
    let grounding = if citations.is_empty() {
        None
    } else {
        let contexts: Vec<String> = citations
            .iter()
            .filter_map(|c| c.snippet.clone())
            .collect();
        // Resolve a text embedder for the fallback path (clone the config
        // off the lock; build the client off the lock — mirrors logs
        // categorize). `None` when the deployment can't embed text.
        let embedder = {
            use veda_core::embedding::Modality;
            let registry = state.inner.embedding_registry.read().await;
            registry
                .find_by_modality(Modality::Text)
                .first()
                .map(|c| veda_core::embedding_client::EmbeddingClient::new((*c).clone()))
        };
        let g = grounding::score_grounding(
            Some(&client),
            embedder.as_ref(),
            input,
            &result.response,
            &contexts,
        )
        .await;
        emit_grounding_metrics(&model_name, citations.len(), &g);
        Some(g)
    };

    Ok((agent, result, citations, grounding))
}

/// Emit grounding metrics for one scored run: a `veda_agent_citations`
/// count and, when faithfulness was scored, a `veda_agent_faithfulness`
/// gauge. Best-effort; never fails the turn.
fn emit_grounding_metrics(model: &str, citation_count: usize, g: &Grounding) {
    metrics::counter!(
        "veda_agent_citations",
        "model" => model.to_owned()
    )
    .increment(citation_count as u64);
    if let Some(f) = g.faithfulness {
        metrics::gauge!(
            "veda_agent_faithfulness",
            "model" => model.to_owned(),
            "method" => g.method.clone()
        )
        .set(f);
    }
}

/// Best-effort `gen_ai.system` (GenAI-semconv provider) value for a model
/// name. Veda always uses an OpenAI-compatible endpoint, so the wire
/// protocol is OpenAI's; we map well-known model-name prefixes to their
/// recognized provider id and default to `openai` otherwise.
fn genai_system_for(model: &str) -> String {
    let m = model.to_ascii_lowercase();
    if m.contains("claude") {
        "anthropic".to_owned()
    } else if m.contains("gemini") {
        "gcp.gemini".to_owned()
    } else if m.starts_with("qwen") || m.contains("/qwen") {
        "qwen".to_owned()
    } else {
        "openai".to_owned()
    }
}

/// Emit Prometheus token / duration / cost metrics for one agent run.
/// Tokens and run duration are always emitted; cost is emitted only when
/// the model has a known price (see [`veda_core::metrics::model_price`]) —
/// an unpriced model logs once at debug and never fails.
fn emit_run_metrics(model: &str, m: &RunMetrics) {
    use veda_core::metrics as vm;

    metrics::counter!(vm::LLM_TOKENS_TOTAL, "type" => "input", "model" => model.to_owned())
        .increment(m.prompt_tokens as u64);
    metrics::counter!(vm::LLM_TOKENS_TOTAL, "type" => "output", "model" => model.to_owned())
        .increment(m.completion_tokens as u64);
    metrics::histogram!(vm::AGENT_RUN_DURATION, "model" => model.to_owned())
        .record(m.total_ms / 1000.0);

    match vm::model_price(model) {
        Some(price) => {
            let micro = vm::cost_micro_usd(m.prompt_tokens, m.completion_tokens, price);
            metrics::counter!(vm::LLM_COST_USD_TOTAL, "model" => model.to_owned())
                .increment(micro);
        }
        None => {
            tracing::debug!(model, "no price configured for LLM model; skipping cost metric");
        }
    }
}

// ── Converse ─────────────────────────────────────────────────────────────

/// Build an [`LlmClient`] from the gateway's global config, applying the
/// agent's optional [`ModelOverride`] (endpoint / model).
fn build_client(
    base: veda_core::llm_client::LlmConfig,
    agent: &Agent,
) -> LlmClient {
    let mut cfg = base;
    if let Some(ov) = &agent.model {
        if let Some(ep) = &ov.endpoint {
            cfg.endpoint = ep.clone();
        }
        if let Some(m) = &ov.model {
            cfg.model = m.clone();
        }
    }
    LlmClient::new(cfg)
}

pub async fn converse(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    Json(req): Json<ConverseRequest>,
) -> Result<Json<ConverseResponse>, GatewayError> {
    // Load or create the conversation (converse-specific persistence).
    let scoped_agent_id = tenant::scope(&tenant, &id);
    let mut conversation = match &req.conversation_id {
        Some(cid) => {
            let scoped_cid = tenant::scope(&tenant, cid);
            store::get_conversation(&state.inner.saved_objects, &scoped_cid)
                .await
                .ok_or_else(|| not_found(format!("conversation '{cid}' not found")))?
        }
        None => Conversation {
            id: tenant::scope(&tenant, &uuid::Uuid::new_v4().to_string()),
            agent_id: scoped_agent_id.clone(),
            title: None,
            messages: Vec::new(),
            steps: Vec::new(),
        },
    };
    let history = conversation.messages.clone();

    // RLS user from the claims.
    let claims = claims.map(|axum::Extension(c)| c);
    let user = rls_user(&claims);

    // Attachments: image-as-auto-retrieval is deferred (phase 3). For now
    // we only note their presence in the user input so the model is aware.
    // TODO (phase 3): pass image attachments to the search tool as query_image_b64
    let user_input = match &req.attachments {
        Some(atts) if !atts.is_empty() => {
            format!("{}\n\n[the user attached {} image(s)]", req.input, atts.len())
        }
        _ => req.input.clone(),
    };

    // Run the turn via the shared helper (resolves agent, builds the model
    // + prompt + tool context, drives the executor loop).
    let (_agent, result, citations, grounding) = run_agent_turn(
        &state,
        &tenant,
        user.as_deref(),
        &id,
        &history,
        &user_input,
    )
    .await
    .map_err(|e| match e {
        AgentTurnError::NoLlm => service_unavailable(
            "VEDA_LLM_ENDPOINT not configured. Set the env var to enable agent converse.",
        ),
        AgentTurnError::NotFound(aid) => not_found(format!("agent '{aid}' not found")),
    })?;

    // Persist: user turn + new assistant/tool messages + steps.
    conversation.messages.push(ConversationMessage {
        role: "user".to_owned(),
        content: Some(user_input),
    });
    for m in &result.new_messages {
        conversation.messages.push(ConversationMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        });
    }
    conversation.steps.extend(result.steps.iter().cloned());
    store::save_conversation(&state.inner.saved_objects, &conversation).await;

    let user_facing_cid = tenant::strip_prefix(&tenant, &conversation.id)
        .unwrap_or(&conversation.id)
        .to_owned();

    Ok(Json(ConverseResponse {
        conversation_id: user_facing_cid,
        response: result.response,
        steps: result.steps,
        metrics: Some(result.metrics),
        citations,
        grounding,
    }))
}

// ── Conversations ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ListConversationsResponse {
    pub conversations: Vec<Conversation>,
}

pub async fn list_conversations(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ListConversationsResponse> {
    let mut conversations = store::list_conversations(&state.inner.saved_objects).await;
    conversations.retain(|c| tenant::strip_prefix(&tenant, &c.id).is_some());
    for c in &mut conversations {
        if let Some(name) = tenant::strip_prefix(&tenant, &c.id) {
            c.id = name.to_owned();
        }
        if let Some(name) = tenant::strip_prefix(&tenant, &c.agent_id) {
            c.agent_id = name.to_owned();
        }
    }
    Json(ListConversationsResponse { conversations })
}

pub async fn get_conversation(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<Conversation>, GatewayError> {
    let scoped = tenant::scope(&tenant, &id);
    let mut conv = store::get_conversation(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| not_found(format!("conversation '{id}' not found")))?;
    conv.id = id;
    if let Some(name) = tenant::strip_prefix(&tenant, &conv.agent_id) {
        conv.agent_id = name.to_owned();
    }
    Ok(Json(conv))
}

pub async fn delete_conversation(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> StatusCode {
    let scoped = tenant::scope(&tenant, &id);
    store::delete_conversation(&state.inner.saved_objects, &scoped).await;
    StatusCode::NO_CONTENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_exporter_prometheus::PrometheusBuilder;

    fn metrics(prompt: u32, completion: u32) -> RunMetrics {
        RunMetrics {
            total_ms: 12.0,
            llm_ms: 8.0,
            tool_ms: 3.0,
            steps: 2,
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    /// A priced model emits tokens + duration + cost series, rendered by
    /// the Prometheus exporter exactly as `/v1/admin/metrics` would.
    #[test]
    fn priced_model_emits_tokens_duration_and_cost() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            emit_run_metrics("gpt-4o-mini", &metrics(2000, 1000));
        });
        let out = handle.render();
        assert!(out.contains("veda_llm_tokens_total"), "missing tokens series:\n{out}");
        assert!(out.contains("veda_agent_run_duration_seconds"), "missing duration series:\n{out}");
        assert!(out.contains("veda_llm_cost_usd_total"), "missing cost series:\n{out}");
        assert!(out.contains("gpt-4o-mini"), "missing model label:\n{out}");
    }

    // ── Grounding wiring (post-run, the shape `converse`/triage return) ───
    //
    // `run_agent_turn` builds a real LlmClient (network) internally, so the
    // post-run grounding wiring is exercised here against the same
    // building blocks the handler uses: citations are extracted from the
    // trace and scored with a fake judge, then assembled into the response
    // shape. The full no-LLM converse path (503) is covered by the
    // gateway integration test `test_converse_without_llm_returns_503`.

    use crate::agent::grounding::{self, Grounding};
    use crate::agent::model::AgentStep;
    use veda_core::error::Result as LlmResult;
    use veda_core::llm_client::{
        ChatMessage, ChatModel, CompletionOptions, ToolCompletion, ToolDef,
    };

    struct FakeJudge(String);
    impl ChatModel for FakeJudge {
        async fn complete_with_tools(
            &self,
            _m: &[ChatMessage],
            _t: &[ToolDef],
            _tc: Option<&str>,
            _o: &CompletionOptions,
        ) -> LlmResult<ToolCompletion> {
            Ok(ToolCompletion {
                content: Some(self.0.clone()),
                tool_calls: Vec::new(),
                finish_reason: Some("stop".to_owned()),
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            })
        }
    }

    fn search_steps() -> Vec<AgentStep> {
        vec![
            AgentStep::ToolCall {
                tool_call_id: "c1".to_owned(),
                tool_id: "search".to_owned(),
                args: serde_json::json!({ "collection": "docs", "query": "hnsw" }),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: serde_json::json!({
                    "count": 1,
                    "hits": [{ "id": "id-1", "distance": 0.1, "fields": { "body": "HNSW is a graph index." } }]
                }),
                is_error: false,
                duration_ms: Some(1.0),
            },
            AgentStep::Reasoning { text: "Veda uses HNSW.".to_owned() },
        ]
    }

    /// When the agent used `search`, the converse-level surface carries
    /// citations + a (judged) grounding score — the shape `converse`
    /// builds from the post-run wiring.
    #[tokio::test]
    async fn converse_surface_has_citations_and_grounding_when_searched() {
        let steps = search_steps();
        let citations = grounding::extract_citations(&steps);
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].collection.as_deref(), Some("docs"));

        let contexts: Vec<String> =
            citations.iter().filter_map(|c| c.snippet.clone()).collect();
        let judge = FakeJudge("FAITHFULNESS: 0.9\nANSWER_RELEVANCY: 0.8".to_owned());
        let g = grounding::score_grounding(
            Some(&judge),
            None,
            "what index does veda use?",
            "Veda uses HNSW.",
            &contexts,
        )
        .await;

        let resp = ConverseResponse {
            conversation_id: "c".to_owned(),
            response: "Veda uses HNSW.".to_owned(),
            steps,
            metrics: None,
            citations,
            grounding: Some(g),
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["citations"][0]["chunk_id"], "id-1");
        assert_eq!(v["grounding"]["method"], "llm_judge");
        assert_eq!(v["grounding"]["faithfulness"], 0.9);
    }

    /// No search → no citations and grounding is `None`/`no_context`; the
    /// response omits the empty `citations`/null `grounding` (serde skip).
    #[tokio::test]
    async fn converse_surface_empty_citations_no_context_when_not_searched() {
        let steps = vec![AgentStep::Reasoning { text: "hi, no tools".to_owned() }];
        let citations = grounding::extract_citations(&steps);
        assert!(citations.is_empty());

        // Mirror the handler: no citations → grounding stays None.
        let grounding: Option<Grounding> = None;
        let resp = ConverseResponse {
            conversation_id: "c".to_owned(),
            response: "hi".to_owned(),
            steps,
            metrics: None,
            citations,
            grounding,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert!(v.get("citations").is_none(), "empty citations skipped: {v}");
        assert!(v.get("grounding").is_none(), "null grounding skipped: {v}");
    }

    #[test]
    fn grounding_metrics_emit_count_and_faithfulness_gauge() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            emit_grounding_metrics(
                "gpt-4o-mini",
                3,
                &Grounding {
                    faithfulness: Some(0.77),
                    answer_relevancy: Some(0.6),
                    method: "llm_judge".to_owned(),
                },
            );
        });
        let out = handle.render();
        assert!(out.contains("veda_agent_citations"), "missing citations series:\n{out}");
        assert!(out.contains("veda_agent_faithfulness"), "missing faithfulness series:\n{out}");
    }

    /// An unpriced model still emits tokens + duration, but no cost series
    /// and no error.
    #[test]
    fn unpriced_model_emits_tokens_but_no_cost() {
        // Ensure no env override prices this model.
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            emit_run_metrics("local-llama-7b-unpriced", &metrics(100, 50));
        });
        let out = handle.render();
        assert!(out.contains("veda_llm_tokens_total"), "missing tokens series:\n{out}");
        assert!(out.contains("veda_agent_run_duration_seconds"), "missing duration series:\n{out}");
        assert!(
            !out.contains("veda_llm_cost_usd_total"),
            "unpriced model must not emit a cost series:\n{out}"
        );
    }
}
