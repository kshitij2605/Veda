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
//!   POST   /v1/conversations/{id}/feedback     capture user feedback

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use veda_core::llm_client::LlmClient;

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::agent::catalog::{ToolContext, ToolErrorKind, ToolSpec};
use crate::agent::executor::{self, DEFAULT_MAX_STEPS};
use crate::agent::grounding::{self, Citation, Grounding};
use crate::agent::model::{
    Agent, AgentStep, AgentTool, Conversation, ConversationMessage, ConverseRequest,
    ConverseResponse, Feedback, default_agent,
};
use crate::agent::store;
use crate::auth::Claims;
use crate::tenant::{self, Tenant, TenantId};

type GatewayError = (StatusCode, Json<ErrorResponse>);

fn not_found(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse { error: msg.into() }),
    )
}
fn service_unavailable(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse { error: msg.into() }),
    )
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

/// Whether the caller proved `Role::Admin`, for schema-restricted fields.
///
/// Deliberately NOT derived from [`rls_user`] returning `None`. That
/// `None` means "admin **or** no claims at all", a conflation that is
/// harmless for RLS but would hand every unauthenticated caller admin
/// privilege here. Absent claims yield `false`, so this fails closed.
pub(crate) fn is_admin_caller(claims: &Option<Claims>) -> bool {
    claims.as_ref().is_some_and(crate::rls::is_admin)
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
    let ctx = ToolContext::new(&state, &tenant)
        .with_user(user.as_deref())
        .with_admin(is_admin_caller(&claims));

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
    conversation_id: Option<&str>,
) -> Result<
    (
        Agent,
        RunResult,
        Vec<Citation>,
        Option<Grounding>,
        bool,
        Option<crate::agent::guardrails::GuardrailReport>,
    ),
    AgentTurnError,
> {
    use crate::agent::guardrails::{self, GuardrailAction, GuardrailMode, GuardrailReport};
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
        None if agent_id == "veda-web" => crate::agent::model::default_web_agent(),
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

    // 3a. Input safety guardrail (opt-in). BEFORE the cache lookup and any
    //     LLM call: screen the user input for prompt-injection / jailbreak /
    //     unsafe content + PII. In Block mode a flagged input short-circuits
    //     with a safe refusal — NO LLM, NO cache serve. In Flag mode the
    //     verdict is attached and the turn proceeds. Best-effort / off-lock /
    //     fail-open (regex always applies); skipped entirely (byte-identical
    //     to today) when guardrails==false.
    let guard_mode = guardrails::mode_for(&agent);
    let input_verdict =
        guardrails::check_text(state, input, guard_mode, guardrails::DEFAULT_THRESHOLD).await;
    if guard_mode != GuardrailMode::Off && input_verdict.action == GuardrailAction::Block {
        metrics::counter!("veda_llm_guardrail_blocked_total", "model" => model_name.clone())
            .increment(1);
        let result = RunResult {
            response: guardrails::REFUSAL_MESSAGE.to_owned(),
            steps: Vec::new(),
            new_messages: Vec::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            hit_limit_or_error: false,
            metrics: RunMetrics::default(),
        };
        let report = GuardrailReport {
            input: input_verdict,
            output: guardrails::GuardrailVerdict::allow(),
        };
        return Ok((agent, result, Vec::new(), None, false, Some(report)));
    }
    if input_verdict.flagged {
        metrics::counter!("veda_llm_guardrail_flagged_total", "model" => model_name.clone())
            .increment(1);
    }

    // 3b. Semantic response cache (opt-in). BEFORE building the client or
    //     running the loop: if the agent opted in, look up a near-duplicate
    //     prompt and, on a hit, return the cached response WITHOUT calling
    //     the LLM. Best-effort / off-lock / fail-open — a lookup problem is
    //     a miss and the turn runs normally. Skipped entirely (and thus
    //     byte-identical to today) when cache==false.
    if crate::agent::cache::should_use_cache(&agent) {
        if let Some(hit) = crate::agent::cache::lookup(
            state,
            tenant,
            &model_name,
            input,
            crate::agent::cache::DEFAULT_SIMILARITY_THRESHOLD,
        )
        .await
        {
            metrics::counter!("veda_llm_cache_hits_total", "model" => model_name.clone())
                .increment(1);
            let mut result = RunResult {
                response: hit.response,
                steps: Vec::new(),
                new_messages: Vec::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                hit_limit_or_error: false,
                metrics: RunMetrics::default(),
            };
            // Output guardrail on the cached response (fresh-or-cached, before
            // returning). Block → redact to the refusal; Flag → annotate only.
            let report = finalize_output_guardrail(
                state,
                &model_name,
                guard_mode,
                input_verdict,
                &mut result.response,
            )
            .await;
            return Ok((agent, result, Vec::new(), None, true, report));
        }
        metrics::counter!("veda_llm_cache_misses_total", "model" => model_name.clone())
            .increment(1);
    }

    let client = build_client(llm_cfg, &agent);

    // 4. System prompt with the modality-awareness block. When the agent
    //    has opted into long-term memory, recall the most relevant past
    //    turns (across conversations) and prepend a bounded context block.
    //    Best-effort + off any lock: a recall failure yields an empty
    //    block and never fails the turn (mirrors logs categorize).
    let summary = executor::collections_modality_summary(state, tenant).await;
    let mut sys = executor::build_system_prompt(&agent, &summary);
    if crate::agent::memory::should_use_memory(&agent) {
        let turns = crate::agent::memory::retrieve_memory(
            state,
            tenant,
            agent_id,
            input,
            crate::agent::memory::DEFAULT_RECALL_K,
        )
        .await;
        if let Some(block) = crate::agent::memory::format_memory_block(
            &turns,
            crate::agent::memory::DEFAULT_MEMORY_BLOCK_CHARS,
        ) {
            sys = format!("{block}\n\n{sys}");
        }
    }

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
    let mut result = executor::run(
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
        let contexts: Vec<String> = citations.iter().filter_map(|c| c.snippet.clone()).collect();
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
        // Persist the faithfulness score for the quality-SLO engine. Brief
        // advisor lock, taken AFTER the scoring await above has completed
        // (never held across `score_grounding`), off the network path.
        if let Some(f) = g.faithfulness {
            let now = crate::advisor::engine::now_ms().max(0) as u64;
            let mut fr = state.advisor().faithfulness_results.lock().await;
            fr.record(agent_id, f, now);
            fr.save(
                &state
                    .inner
                    .data_dir
                    .join(crate::advisor::faithfulness::FAITHFULNESS_RESULTS_FILE),
            );
        }
        Some(g)
    };

    // 9. Write-back: when memory is on, persist the user input and the
    //    assistant response into the agent's long-term memory. Best-effort
    //    + off any lock; a write failure NEVER fails the turn (write_turn
    //    swallows its own errors). Skipped entirely for memory==false so
    //    the path is byte-identical to today.
    if crate::agent::memory::should_use_memory(&agent) {
        let ts = crate::advisor::engine::now_ms().max(0);
        let conversation_id = conversation_id.unwrap_or("");
        crate::agent::memory::write_turn(
            state,
            tenant,
            agent_id,
            conversation_id,
            "user",
            input,
            ts,
        )
        .await;
        if !result.response.trim().is_empty() {
            crate::agent::memory::write_turn(
                state,
                tenant,
                agent_id,
                conversation_id,
                "assistant",
                &result.response,
                ts,
            )
            .await;
        }
    }

    // 10. Cache write-back: when the cache is on and the run completed
    //     cleanly (not a model error / step-limit), store the prompt →
    //     response entry for future near-duplicate hits. Best-effort +
    //     off-lock; a store failure NEVER fails the turn. Skipped for a
    //     non-empty-response error path so we never cache a fallback.
    if crate::agent::cache::should_use_cache(&agent)
        && !result.hit_limit_or_error
        && !result.response.trim().is_empty()
    {
        let ts = crate::advisor::engine::now_ms().max(0);
        crate::agent::cache::store(state, tenant, &model_name, input, &result.response, ts).await;
    }

    // 11. Output safety guardrail (opt-in) on the FINAL response, after the
    //     cache write-back (so the real answer is cached) but before
    //     returning. Block → redact to the refusal; Flag → annotate only.
    //     Best-effort / off-lock / fail-open; no-op when guardrails==false.
    let report = finalize_output_guardrail(
        state,
        &model_name,
        guard_mode,
        input_verdict,
        &mut result.response,
    )
    .await;

    // 12. LLM trace store (opt-in). Persist this completed (non-cache-hit)
    //     run as an embedded, searchable trace AFTER the response +
    //     guardrails are finalized. Best-effort / off-lock / fail-open; a
    //     store failure NEVER fails the turn. Skipped entirely (and thus
    //     byte-identical to today) when trace==false. We don't store error /
    //     step-limit fallbacks (no useful run to search/cluster).
    if crate::agent::trace_store::should_store_trace(&agent) && !result.hit_limit_or_error {
        let fields = build_trace_fields(
            input,
            &result,
            &model_name,
            grounding.as_ref(),
            report.as_ref(),
            false,
        );
        crate::agent::trace_store::store_trace(state, tenant, fields).await;
    }

    Ok((agent, result, citations, grounding, false, report))
}

/// Assemble the per-run [`TraceFields`](crate::agent::trace_store::TraceFields)
/// from a finished run's signals. Pure (no I/O): comma-joins the tool ids
/// from the steps, maps the optional faithfulness to the `-1` sentinel when
/// unscored, computes the estimated cost from token counts + the model
/// price, and records whether any guardrail flagged the turn.
fn build_trace_fields(
    input: &str,
    result: &RunResult,
    model_name: &str,
    grounding: Option<&Grounding>,
    report: Option<&crate::agent::guardrails::GuardrailReport>,
    cache_hit: bool,
) -> crate::agent::trace_store::TraceFields {
    let tool_ids = tool_ids_from_steps(&result.steps);
    let faithfulness = grounding
        .and_then(|g| g.faithfulness)
        .unwrap_or(crate::agent::trace_store::FAITHFULNESS_NONE);
    let cost_usd = match veda_core::metrics::model_price(model_name) {
        Some(price) => veda_core::metrics::token_cost_usd(
            result.metrics.prompt_tokens,
            result.metrics.completion_tokens,
            price,
        ),
        None => 0.0,
    };
    let guardrail_flagged = report
        .map(|r| r.input.flagged || r.output.flagged)
        .unwrap_or(false);
    crate::agent::trace_store::TraceFields {
        prompt: input.to_owned(),
        response: result.response.clone(),
        tool_ids,
        faithfulness,
        cost_usd,
        cache_hit,
        guardrail_flagged,
        latency_ms: result.metrics.total_ms,
        ts: crate::advisor::engine::now_ms().max(0),
    }
}

/// Comma-join the distinct tool ids invoked across a run's steps, in first-
/// seen order. Empty string when no tools were called.
fn tool_ids_from_steps(steps: &[AgentStep]) -> String {
    let mut ids: Vec<&str> = Vec::new();
    for s in steps {
        if let AgentStep::ToolCall { tool_id, .. } = s
            && !ids.contains(&tool_id.as_str())
        {
            ids.push(tool_id.as_str());
        }
    }
    ids.join(",")
}

/// Run the OUTPUT guardrail on `response` and assemble the [`GuardrailReport`]
/// pairing the already-computed `input_verdict` with the fresh output verdict.
/// Mutates `response` to the safe refusal when the output verdict resolves to
/// [`GuardrailAction::Block`]. Returns `None` when guardrails are off (so the
/// path is byte-identical to today). Emits the blocked/flagged metrics.
/// Best-effort / off-lock / fail-open (regex always applies).
async fn finalize_output_guardrail(
    state: &AppState,
    model_name: &str,
    mode: crate::agent::guardrails::GuardrailMode,
    input_verdict: crate::agent::guardrails::GuardrailVerdict,
    response: &mut String,
) -> Option<crate::agent::guardrails::GuardrailReport> {
    use crate::agent::guardrails::{self, GuardrailAction, GuardrailMode, GuardrailReport};
    if mode == GuardrailMode::Off {
        return None;
    }
    let output_verdict =
        guardrails::check_text(state, response, mode, guardrails::DEFAULT_THRESHOLD).await;
    if output_verdict.action == GuardrailAction::Block {
        metrics::counter!("veda_llm_guardrail_blocked_total", "model" => model_name.to_owned())
            .increment(1);
        *response = guardrails::REFUSAL_MESSAGE.to_owned();
    } else if output_verdict.flagged {
        metrics::counter!("veda_llm_guardrail_flagged_total", "model" => model_name.to_owned())
            .increment(1);
    }
    Some(GuardrailReport {
        input: input_verdict,
        output: output_verdict,
    })
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

    // Time-to-first-token (streaming path only; `None` on the blocking path).
    if let Some(ttft_ms) = m.ttft_ms {
        metrics::histogram!(vm::LLM_TTFT_SECONDS, "model" => model.to_owned())
            .record(ttft_ms / 1000.0);
    }

    match vm::model_price(model) {
        Some(price) => {
            let micro = vm::cost_micro_usd(m.prompt_tokens, m.completion_tokens, price);
            metrics::counter!(vm::LLM_COST_USD_TOTAL, "model" => model.to_owned()).increment(micro);
        }
        None => {
            tracing::debug!(
                model,
                "no price configured for LLM model; skipping cost metric"
            );
        }
    }
}

// ── Converse ─────────────────────────────────────────────────────────────

/// Build an [`LlmClient`] from the gateway's global config, applying the
/// agent's optional [`ModelOverride`] (endpoint / model).
fn build_client(base: veda_core::llm_client::LlmConfig, agent: &Agent) -> LlmClient {
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
            feedback: Vec::new(),
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
            format!(
                "{}\n\n[the user attached {} image(s)]",
                req.input,
                atts.len()
            )
        }
        _ => req.input.clone(),
    };

    // Run the turn via the shared helper (resolves agent, builds the model
    // + prompt + tool context, drives the executor loop).
    let (_agent, result, citations, grounding, cache_hit, guardrails) = run_agent_turn(
        &state,
        &tenant,
        user.as_deref(),
        &id,
        &history,
        &user_input,
        Some(&conversation.id),
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
    // A cache hit (or a guardrail-blocked input) produces no `new_messages`
    // (the LLM was skipped); record the served answer / refusal as the
    // assistant turn so the transcript is complete.
    if result.new_messages.is_empty() && !result.response.trim().is_empty() {
        conversation.messages.push(ConversationMessage {
            role: "assistant".to_owned(),
            content: Some(result.response.clone()),
        });
    }
    conversation.steps.extend(result.steps.iter().cloned());
    store::save_conversation(&state.inner.saved_objects, &conversation).await;

    let user_facing_cid = tenant::strip_prefix(&tenant, &conversation.id)
        .unwrap_or(&conversation.id)
        .to_owned();

    // Roll the turn's per-tool-call reports up to the top level. Nested
    // inside `steps[].result.retrieval_guard` they are technically present
    // but not a surface an operator or client reads; design §7 promises the
    // field here, and §6's fail-open-on-`unscreened` decision is only
    // defensible if those counts are actually visible.
    let retrieval_guard = crate::agent::retrieval_guard::collect_from_steps(&result.steps);

    Ok(Json(ConverseResponse {
        conversation_id: user_facing_cid,
        response: result.response,
        steps: result.steps,
        metrics: Some(result.metrics),
        citations,
        grounding,
        cache_hit,
        guardrails,
        retrieval_guard,
    }))
}

// ── Converse (streaming, SSE) ────────────────────────────────────────────

/// Drive one streaming agent turn: resolve the agent + model + prompt +
/// tool context (mirroring [`run_agent_turn`]), run the executor's
/// [`executor::run_streaming`] loop feeding `event_tx`, then — post-run,
/// off the hot path — emit run/grounding metrics, persist the conversation
/// like `/converse`, and assemble the terminal [`StreamDonePayload`].
///
/// The caller has already verified an LLM is configured (the 503 path) and
/// resolved the `Conversation` shell; agent resolution failures here yield
/// a graceful `done` payload rather than an HTTP error (streaming has
/// begun). Best-effort grounding never fails the turn, mirroring the
/// blocking helper.
async fn run_streaming_turn(
    state: &AppState,
    tenant: &TenantId,
    user: Option<&str>,
    agent_id: &str,
    conversation: &mut Conversation,
    input: &str,
    event_tx: tokio::sync::mpsc::Sender<executor::StreamEvent>,
) -> StreamDonePayload {
    let user_facing_cid = tenant::strip_prefix(tenant, &conversation.id)
        .unwrap_or(&conversation.id)
        .to_owned();

    // LLM config (the handler already 503'd if absent; treat a race as a
    // graceful failure rather than a panic).
    let llm_cfg = match state.inner.llm_config.read().await.clone() {
        Some(c) => c,
        None => {
            return StreamDonePayload {
                conversation_id: user_facing_cid,
                response: "no LLM configured".to_owned(),
                metrics: None,
                citations: Vec::new(),
                grounding: None,
                cache_hit: false,
                retrieval_guard: None,
            };
        }
    };

    // Resolve the agent (same rules as run_agent_turn).
    let scoped_agent_id = tenant::scope(tenant, agent_id);
    let agent = match store::get_agent(&state.inner.saved_objects, &scoped_agent_id).await {
        Some(a) => a,
        None if agent_id == "veda-agent" => default_agent(),
        None if agent_id == "veda-ops" => crate::agent::model::default_ops_agent(),
        None if agent_id == "veda-web" => crate::agent::model::default_web_agent(),
        None => {
            return StreamDonePayload {
                conversation_id: user_facing_cid,
                response: format!("agent '{agent_id}' not found"),
                metrics: None,
                citations: Vec::new(),
                grounding: None,
                cache_hit: false,
                retrieval_guard: None,
            };
        }
    };

    let model_name = agent
        .model
        .as_ref()
        .and_then(|ov| ov.model.clone())
        .unwrap_or_else(|| llm_cfg.model.clone());
    let provider = genai_system_for(&model_name);

    // Input safety guardrail (opt-in), BEFORE the cache lookup / LLM loop.
    // In Block mode a flagged input short-circuits: emit the safe refusal as
    // a single `answer` delta and finish (no LLM, no cache serve). Flag mode
    // proceeds (the blocking helper attaches the verdict; the stream done
    // payload mirrors today's shape). Best-effort / off-lock / fail-open;
    // no-op when guardrails==false.
    let guard_mode = crate::agent::guardrails::mode_for(&agent);
    if guard_mode == crate::agent::guardrails::GuardrailMode::Block {
        let v = crate::agent::guardrails::check_text(
            state,
            input,
            guard_mode,
            crate::agent::guardrails::DEFAULT_THRESHOLD,
        )
        .await;
        if v.action == crate::agent::guardrails::GuardrailAction::Block {
            metrics::counter!("veda_llm_guardrail_blocked_total", "model" => model_name.clone())
                .increment(1);
            let refusal = crate::agent::guardrails::REFUSAL_MESSAGE.to_owned();
            let _ = event_tx
                .send(executor::StreamEvent::AnswerDelta {
                    text: refusal.clone(),
                })
                .await;
            conversation.messages.push(ConversationMessage {
                role: "user".to_owned(),
                content: Some(input.to_owned()),
            });
            conversation.messages.push(ConversationMessage {
                role: "assistant".to_owned(),
                content: Some(refusal.clone()),
            });
            store::save_conversation(&state.inner.saved_objects, conversation).await;
            return StreamDonePayload {
                conversation_id: user_facing_cid,
                response: refusal,
                metrics: Some(RunMetrics::default()),
                citations: Vec::new(),
                grounding: None,
                cache_hit: false,
                retrieval_guard: None,
            };
        }
    }

    // Semantic cache (opt-in): on a hit, emit the cached answer as a single
    // `answer` delta and finish — the LLM loop is skipped. Best-effort /
    // off-lock / fail-open (a miss runs normally). The user + assistant
    // turns are still persisted so the transcript is complete.
    if crate::agent::cache::should_use_cache(&agent) {
        if let Some(hit) = crate::agent::cache::lookup(
            state,
            tenant,
            &model_name,
            input,
            crate::agent::cache::DEFAULT_SIMILARITY_THRESHOLD,
        )
        .await
        {
            metrics::counter!("veda_llm_cache_hits_total", "model" => model_name.clone())
                .increment(1);
            let _ = event_tx
                .send(executor::StreamEvent::AnswerDelta {
                    text: hit.response.clone(),
                })
                .await;
            conversation.messages.push(ConversationMessage {
                role: "user".to_owned(),
                content: Some(input.to_owned()),
            });
            conversation.messages.push(ConversationMessage {
                role: "assistant".to_owned(),
                content: Some(hit.response.clone()),
            });
            store::save_conversation(&state.inner.saved_objects, conversation).await;
            return StreamDonePayload {
                conversation_id: user_facing_cid,
                response: hit.response,
                metrics: Some(RunMetrics::default()),
                citations: Vec::new(),
                grounding: None,
                cache_hit: true,
                retrieval_guard: None,
            };
        }
        metrics::counter!("veda_llm_cache_misses_total", "model" => model_name.clone())
            .increment(1);
    }

    let client = build_client(llm_cfg, &agent);

    // System prompt + optional memory recall (mirrors run_agent_turn).
    let summary = executor::collections_modality_summary(state, tenant).await;
    let mut sys = executor::build_system_prompt(&agent, &summary);
    if crate::agent::memory::should_use_memory(&agent) {
        let turns = crate::agent::memory::retrieve_memory(
            state,
            tenant,
            agent_id,
            input,
            crate::agent::memory::DEFAULT_RECALL_K,
        )
        .await;
        if let Some(block) = crate::agent::memory::format_memory_block(
            &turns,
            crate::agent::memory::DEFAULT_MEMORY_BLOCK_CHARS,
        ) {
            sys = format!("{block}\n\n{sys}");
        }
    }

    let ctx = ToolContext::new(state, tenant).with_user(user);

    use veda_core::genai_semconv as gen_ai;
    let root = tracing::info_span!(
        "invoke_agent",
        otel.name = gen_ai::invoke_agent_span_name(&agent.name),
        "gen_ai.operation.name" = gen_ai::OP_INVOKE_AGENT,
        "gen_ai.agent.name" = %agent.name,
    );
    let history = conversation.messages.clone();
    let result = executor::run_streaming(
        &client,
        &state.inner.tool_catalog,
        &ctx,
        &agent,
        &sys,
        &history,
        input,
        DEFAULT_MAX_STEPS,
        &provider,
        &model_name,
        event_tx,
    )
    .instrument(root)
    .await;

    emit_run_metrics(&model_name, &result.metrics);

    // Post-run grounding (same logic as run_agent_turn).
    let citations = grounding::extract_citations(&result.steps);
    let grounding = if citations.is_empty() {
        None
    } else {
        let contexts: Vec<String> = citations.iter().filter_map(|c| c.snippet.clone()).collect();
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
        if let Some(f) = g.faithfulness {
            let now = crate::advisor::engine::now_ms().max(0) as u64;
            let mut fr = state.advisor().faithfulness_results.lock().await;
            fr.record(agent_id, f, now);
            fr.save(
                &state
                    .inner
                    .data_dir
                    .join(crate::advisor::faithfulness::FAITHFULNESS_RESULTS_FILE),
            );
        }
        Some(g)
    };

    // Memory write-back (mirrors run_agent_turn). The conversation id is
    // owned here (the streaming handler created/loaded the Conversation).
    if crate::agent::memory::should_use_memory(&agent) {
        let ts = crate::advisor::engine::now_ms().max(0);
        let conversation_id = conversation.id.as_str();
        crate::agent::memory::write_turn(
            state,
            tenant,
            agent_id,
            conversation_id,
            "user",
            input,
            ts,
        )
        .await;
        if !result.response.trim().is_empty() {
            crate::agent::memory::write_turn(
                state,
                tenant,
                agent_id,
                conversation_id,
                "assistant",
                &result.response,
                ts,
            )
            .await;
        }
    }

    // Cache write-back (mirrors run_agent_turn): store on a clean run.
    if crate::agent::cache::should_use_cache(&agent)
        && !result.hit_limit_or_error
        && !result.response.trim().is_empty()
    {
        let ts = crate::advisor::engine::now_ms().max(0);
        crate::agent::cache::store(state, tenant, &model_name, input, &result.response, ts).await;
    }

    // Persist the conversation exactly like /converse.
    conversation.messages.push(ConversationMessage {
        role: "user".to_owned(),
        content: Some(input.to_owned()),
    });
    for m in &result.new_messages {
        conversation.messages.push(ConversationMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        });
    }
    conversation.steps.extend(result.steps.iter().cloned());
    store::save_conversation(&state.inner.saved_objects, conversation).await;

    // Same roll-up as the blocking `/converse` — see there for why the
    // report belongs at the top level and not only nested in the steps.
    let retrieval_guard = crate::agent::retrieval_guard::collect_from_steps(&result.steps);

    StreamDonePayload {
        conversation_id: user_facing_cid,
        response: result.response,
        metrics: Some(result.metrics),
        citations,
        grounding,
        cache_hit: false,
        retrieval_guard,
    }
}

/// Terminal `done` SSE frame payload: the same summary fields the blocking
/// `/converse` returns, sent once the streaming run completes.
#[derive(Serialize)]
struct StreamDonePayload {
    conversation_id: String,
    response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<RunMetrics>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    citations: Vec<Citation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding: Option<Grounding>,
    /// `true` when this answer was served from the semantic cache (the LLM
    /// loop was skipped). Defaults to `false`.
    #[serde(default)]
    cache_hit: bool,
    /// Turn-level retrieval-guardrail report, mirroring
    /// [`ConverseResponse::retrieval_guard`]: rolled up across the turn's
    /// retrieving tool calls, and omitted entirely when the turn retrieved
    /// nothing (including every early return below — a refusal, a cache
    /// serve and an unresolved agent all retrieve nothing).
    #[serde(skip_serializing_if = "Option::is_none")]
    retrieval_guard: Option<crate::agent::retrieval_guard::RetrievalGuardReport>,
}

/// `POST /v1/agents/{id}/converse/stream` — the streaming twin of
/// [`converse`]. Same auth / tenant / RLS. Streams the agent run live over
/// Server-Sent Events: a named `reasoning` / `tool_call` / `tool_result` /
/// `answer` event per [`executor::StreamEvent`], then a terminal `done`
/// event carrying `{ conversation_id, response, metrics, citations,
/// grounding }`. Persists the conversation exactly like `/converse`.
///
/// Returns HTTP 503 *before* streaming starts when no LLM is configured.
/// The blocking `/converse` and the A2A handler are untouched.
pub async fn converse_stream(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    Json(req): Json<ConverseRequest>,
) -> Response {
    // 503 before any streaming if no LLM is configured (mirrors converse).
    if state.inner.llm_config.read().await.is_none() {
        return service_unavailable(
            "VEDA_LLM_ENDPOINT not configured. Set the env var to enable agent converse.",
        )
        .into_response();
    }

    // Resolve the conversation up front (same persistence rules as converse).
    let scoped_agent_id = tenant::scope(&tenant, &id);
    let conversation = match &req.conversation_id {
        Some(cid) => {
            let scoped_cid = tenant::scope(&tenant, cid);
            match store::get_conversation(&state.inner.saved_objects, &scoped_cid).await {
                Some(c) => c,
                None => {
                    return not_found(format!("conversation '{cid}' not found")).into_response();
                }
            }
        }
        None => Conversation {
            id: tenant::scope(&tenant, &uuid::Uuid::new_v4().to_string()),
            agent_id: scoped_agent_id.clone(),
            title: None,
            messages: Vec::new(),
            steps: Vec::new(),
            feedback: Vec::new(),
        },
    };

    let claims = claims.map(|axum::Extension(c)| c);
    let user = rls_user(&claims);

    let user_input = match &req.attachments {
        Some(atts) if !atts.is_empty() => {
            format!(
                "{}\n\n[the user attached {} image(s)]",
                req.input,
                atts.len()
            )
        }
        _ => req.input.clone(),
    };

    // Channel carrying live executor events to the SSE forwarder, plus a
    // oneshot for the terminal `done` summary.
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<executor::StreamEvent>(64);
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<StreamDonePayload>();

    // Spawn the streaming run on an owned-state task (everything it needs is
    // reachable from the cloned AppState). It emits StreamEvents as it goes
    // and returns the final summary over the oneshot.
    let task_state = state.clone();
    let task_tenant = tenant.clone();
    let task_user = user.clone();
    let task_input = user_input.clone();
    let mut conversation = conversation;
    tokio::spawn(async move {
        let payload = run_streaming_turn(
            &task_state,
            &task_tenant,
            task_user.as_deref(),
            &id,
            &mut conversation,
            &task_input,
            event_tx,
        )
        .await;
        let _ = done_tx.send(payload);
    });

    Sse::new(build_sse_stream(event_rx, done_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// State threaded through the SSE [`futures::stream::unfold`]: live events
/// first, then the terminal `done` frame, then end.
enum SseState {
    /// Still forwarding live executor events; carries the receivers.
    Live(
        tokio::sync::mpsc::Receiver<executor::StreamEvent>,
        tokio::sync::oneshot::Receiver<StreamDonePayload>,
    ),
    /// Finished.
    End,
}

/// Map a live [`executor::StreamEvent`] to its named SSE [`Event`].
fn event_to_sse(ev: &executor::StreamEvent) -> Event {
    use executor::StreamEvent as E;
    let (name, data) = match ev {
        E::Reasoning { .. } => ("reasoning", serde_json::to_string(ev)),
        E::ToolCall { .. } => ("tool_call", serde_json::to_string(ev)),
        E::ToolResult { .. } => ("tool_result", serde_json::to_string(ev)),
        E::AnswerDelta { .. } => ("answer", serde_json::to_string(ev)),
    };
    Event::default()
        .event(name)
        .data(data.unwrap_or_else(|_| "{}".to_owned()))
}

/// Build the SSE item stream: forward each live event as a named SSE event
/// (`reasoning` / `tool_call` / `tool_result` / `answer`), then a terminal
/// `done` event carrying the summary payload.
fn build_sse_stream(
    event_rx: tokio::sync::mpsc::Receiver<executor::StreamEvent>,
    done_rx: tokio::sync::oneshot::Receiver<StreamDonePayload>,
) -> impl Stream<Item = std::result::Result<Event, std::convert::Infallible>> {
    futures::stream::unfold(SseState::Live(event_rx, done_rx), |state| async move {
        match state {
            SseState::Live(mut event_rx, done_rx) => match event_rx.recv().await {
                Some(ev) => Some((Ok(event_to_sse(&ev)), SseState::Live(event_rx, done_rx))),
                // Sender dropped → run finished; emit the done frame.
                None => emit_done(done_rx).await,
            },
            SseState::End => None,
        }
    })
}

/// Await the terminal summary and emit it as the `done` SSE event. A
/// dropped oneshot (run task panicked) yields a minimal error `done`.
async fn emit_done(
    done_rx: tokio::sync::oneshot::Receiver<StreamDonePayload>,
) -> Option<(
    std::result::Result<Event, std::convert::Infallible>,
    SseState,
)> {
    let payload = done_rx.await.unwrap_or_else(|_| StreamDonePayload {
        conversation_id: String::new(),
        response: "the agent run failed".to_owned(),
        metrics: None,
        citations: Vec::new(),
        grounding: None,
        cache_hit: false,
        retrieval_guard: None,
    });
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_owned());
    let event = Event::default().event("done").data(data);
    Some((Ok(event), SseState::End))
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

/// Request body for `POST /v1/conversations/{id}/feedback`.
#[derive(Deserialize)]
pub struct FeedbackRequest {
    /// `"up"` or `"down"`.
    pub rating: String,
    #[serde(default)]
    pub note: Option<String>,
}

fn bad_request(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg.into() }),
    )
}

/// Capture user feedback (thumbs up/down + optional note) on a conversation.
/// Tenant/RLS-scoped exactly like the other conversation endpoints: the id is
/// scoped on the way in, so a caller can only annotate their own tenant's
/// conversations. 404 for an unknown id. Increments
/// `veda_llm_feedback_total{rating}`. Feeds the eval flywheel later.
pub async fn submit_feedback(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(req): Json<FeedbackRequest>,
) -> Result<StatusCode, GatewayError> {
    let rating = req.rating.trim().to_ascii_lowercase();
    if rating != "up" && rating != "down" {
        return Err(bad_request("rating must be \"up\" or \"down\""));
    }
    let scoped = tenant::scope(&tenant, &id);
    let mut conv = store::get_conversation(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| not_found(format!("conversation '{id}' not found")))?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    conv.feedback.push(Feedback {
        rating: rating.clone(),
        note: req.note.filter(|n| !n.trim().is_empty()),
        ts,
    });
    store::save_conversation(&state.inner.saved_objects, &conv).await;
    metrics::counter!(veda_core::metrics::LLM_FEEDBACK_TOTAL, "rating" => rating).increment(1);
    Ok(StatusCode::NO_CONTENT)
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
            ttft_ms: None,
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
        assert!(
            out.contains("veda_llm_tokens_total"),
            "missing tokens series:\n{out}"
        );
        assert!(
            out.contains("veda_agent_run_duration_seconds"),
            "missing duration series:\n{out}"
        );
        assert!(
            out.contains("veda_llm_cost_usd_total"),
            "missing cost series:\n{out}"
        );
        assert!(out.contains("gpt-4o-mini"), "missing model label:\n{out}");
        // No TTFT on the blocking path (the helper leaves ttft_ms None).
        assert!(
            !out.contains("veda_llm_ttft_seconds"),
            "blocking metrics must not emit ttft:\n{out}"
        );
    }

    /// A streaming run (ttft_ms = Some) emits the TTFT histogram series.
    #[test]
    fn streaming_metrics_emit_ttft_histogram() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let mut m = metrics(2000, 1000);
        m.ttft_ms = Some(42.0);
        metrics::with_local_recorder(&recorder, || {
            emit_run_metrics("gpt-4o-mini", &m);
        });
        let out = handle.render();
        assert!(
            out.contains("veda_llm_ttft_seconds"),
            "missing ttft series:\n{out}"
        );
    }

    /// GAP-A: pre-registering the LLM metrics makes the cost / cache-hit /
    /// cache-miss counters render at 0 *before* any converse traffic — so a
    /// local (unpriced, no-cache) deployment still shows the series instead
    /// of an empty `/metrics`.
    #[test]
    fn pre_register_exports_cost_and_cache_counters_at_zero() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            veda_core::metrics::pre_register_llm_metrics();
        });
        let out = handle.render();
        assert!(
            out.contains("veda_llm_cost_usd_total"),
            "cost counter must export at 0:\n{out}"
        );
        assert!(
            out.contains("veda_llm_cache_hits_total"),
            "cache-hits counter must export at 0:\n{out}"
        );
        assert!(
            out.contains("veda_llm_cache_misses_total"),
            "cache-misses counter must export at 0:\n{out}"
        );
    }

    // ── Conversation feedback capture ────────────────────────────────────

    use crate::agent::model::Conversation;
    use crate::tenant::{Tenant, TenantId};

    /// Seed a conversation owned by `tenant` (user-facing id `cid`) and
    /// return the AppState (+ tempdir guard).
    async fn seed_conversation(cid: &str, tenant: &str) -> (AppState, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let scoped = crate::tenant::scope(&TenantId::parse(tenant).unwrap(), cid);
        let conv = Conversation {
            id: scoped,
            agent_id: crate::tenant::scope(&TenantId::parse(tenant).unwrap(), "veda-agent"),
            title: None,
            messages: Vec::new(),
            steps: Vec::new(),
            feedback: Vec::new(),
        };
        store::save_conversation(&state.inner.saved_objects, &conv).await;
        (state, tmp)
    }

    #[test]
    fn feedback_persists_on_conversation_and_increments_metric() {
        // Single-threaded runtime built explicitly so the whole async flow
        // runs inside `with_local_recorder` (which is a sync scope guard).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        let _guard = metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let (state, tmp) = seed_conversation("c1", "acme").await;
                let resp = submit_feedback(
                    State(state.clone()),
                    Tenant(TenantId::parse("acme").unwrap()),
                    Path("c1".to_owned()),
                    Json(FeedbackRequest {
                        rating: "up".to_owned(),
                        note: Some("great answer".to_owned()),
                    }),
                )
                .await
                .expect("feedback accepted");
                assert_eq!(resp, StatusCode::NO_CONTENT);

                // Persisted on the conversation.
                let scoped = crate::tenant::scope(&TenantId::parse("acme").unwrap(), "c1");
                let conv = store::get_conversation(&state.inner.saved_objects, &scoped)
                    .await
                    .expect("conversation present");
                assert_eq!(conv.feedback.len(), 1);
                assert_eq!(conv.feedback[0].rating, "up");
                assert_eq!(conv.feedback[0].note.as_deref(), Some("great answer"));
                // Return the tempdir guard so it outlives the block.
                tmp
            })
        });

        // Metric incremented with the rating label.
        let out = handle.render();
        assert!(
            out.contains("veda_llm_feedback_total"),
            "missing feedback metric:\n{out}"
        );
        assert!(
            out.contains("rating=\"up\""),
            "missing rating label:\n{out}"
        );
    }

    #[tokio::test]
    async fn feedback_unknown_conversation_is_404() {
        let (state, _tmp) = seed_conversation("c1", "acme").await;
        let err = submit_feedback(
            State(state),
            Tenant(TenantId::parse("acme").unwrap()),
            Path("does-not-exist".to_owned()),
            Json(FeedbackRequest {
                rating: "down".to_owned(),
                note: None,
            }),
        )
        .await
        .expect_err("unknown id should 404");
        assert_eq!(err.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn feedback_is_tenant_scoped() {
        // Conversation owned by "acme"; a "rival" caller must not reach it.
        let (state, _tmp) = seed_conversation("c1", "acme").await;
        let err = submit_feedback(
            State(state.clone()),
            Tenant(TenantId::parse("rival").unwrap()),
            Path("c1".to_owned()),
            Json(FeedbackRequest {
                rating: "up".to_owned(),
                note: None,
            }),
        )
        .await
        .expect_err("cross-tenant feedback should 404");
        assert_eq!(err.0, StatusCode::NOT_FOUND);

        // acme's conversation is untouched.
        let scoped = crate::tenant::scope(&TenantId::parse("acme").unwrap(), "c1");
        let conv = store::get_conversation(&state.inner.saved_objects, &scoped)
            .await
            .unwrap();
        assert!(conv.feedback.is_empty());
    }

    #[tokio::test]
    async fn feedback_rejects_invalid_rating() {
        let (state, _tmp) = seed_conversation("c1", "acme").await;
        let err = submit_feedback(
            State(state),
            Tenant(TenantId::parse("acme").unwrap()),
            Path("c1".to_owned()),
            Json(FeedbackRequest {
                rating: "meh".to_owned(),
                note: None,
            }),
        )
        .await
        .expect_err("invalid rating should 400");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
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
            AgentStep::Reasoning {
                text: "Veda uses HNSW.".to_owned(),
            },
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

        let contexts: Vec<String> = citations.iter().filter_map(|c| c.snippet.clone()).collect();
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
            cache_hit: false,
            guardrails: None,
            retrieval_guard: None,
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
        let steps = vec![AgentStep::Reasoning {
            text: "hi, no tools".to_owned(),
        }];
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
            cache_hit: false,
            guardrails: None,
            retrieval_guard: None,
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
        assert!(
            out.contains("veda_agent_citations"),
            "missing citations series:\n{out}"
        );
        assert!(
            out.contains("veda_agent_faithfulness"),
            "missing faithfulness series:\n{out}"
        );
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
        assert!(
            out.contains("veda_llm_tokens_total"),
            "missing tokens series:\n{out}"
        );
        assert!(
            out.contains("veda_agent_run_duration_seconds"),
            "missing duration series:\n{out}"
        );
        assert!(
            !out.contains("veda_llm_cost_usd_total"),
            "unpriced model must not emit a cost series:\n{out}"
        );
    }
}
