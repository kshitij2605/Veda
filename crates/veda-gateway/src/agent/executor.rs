//! Agent executor — the tool-calling (ReAct-style) loop.
//!
//! [`run`] drives a [`ChatModel`] through a bounded loop: the model is
//! given the agent's instructions plus a *modality-awareness* summary of
//! the available collections, the agent's whitelisted tools, and the
//! conversation so far. On each turn the model may either emit tool calls
//! (which are dispatched through the [`ToolCatalog`] and fed back as
//! `tool`-role messages) or a final text answer (which terminates the
//! loop). The loop is bounded by `max_steps` so a misbehaving model can
//! never spin forever, and every model / dispatch / parse failure is
//! turned into an error result rather than a panic.
//!
//! This module is the engine only — it has no REST surface. The REST
//! handler (a later task) calls [`run`] and persists the returned trace.

use serde::{Deserialize, Serialize};

use veda_core::llm_client::{
    ChatMessage, ChatModel, CompletionOptions, ToolDef,
};

use super::catalog::{ToolCatalog, ToolContext};
use super::model::{Agent, AgentStep, ConversationMessage};
use crate::rest::state::AppState;

/// Default ceiling on model turns. A turn = one `complete_with_tools`
/// call; a final-answer turn counts too, so the model gets at most this
/// many chances to act before the loop gives up gracefully.
pub const DEFAULT_MAX_STEPS: usize = 8;

/// Cap (in bytes) on a single tool result fed back to the model. Tool
/// outputs (e.g. large search hit sets) are truncated to this so a
/// verbose tool can't blow up the context window across iterations.
const TOOL_RESULT_CAP_BYTES: usize = 8 * 1024;

/// Run-level APM metrics accumulated across one [`run`] invocation: an
/// LLM-application analog of request timing. `total_ms` is the wall-clock
/// span of the whole loop; `llm_ms` / `tool_ms` are the summed latencies
/// of the model calls and tool dispatches respectively (their sum is `<=
/// total_ms` modulo bookkeeping overhead). Token counts mirror the ones
/// already reported by the model.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct RunMetrics {
    pub total_ms: f64,
    pub llm_ms: f64,
    pub tool_ms: f64,
    pub steps: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Outcome of an [`run`] invocation.
pub struct RunResult {
    /// The agent's final natural-language answer. Always present — even
    /// on the step-limit / error paths a graceful fallback string is set.
    pub response: String,
    /// The full reasoning/acting trace, in order. The REST layer persists
    /// this onto the conversation.
    pub steps: Vec<AgentStep>,
    /// New chat messages produced this turn (assistant + tool messages,
    /// not the user input or prior history). The REST layer can append
    /// these to a stored OpenAI-shape transcript if it keeps one.
    pub new_messages: Vec<ChatMessage>,
    /// Summed token usage reported by the model across all turns.
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// True when the loop terminated abnormally (model error or step
    /// limit) rather than on a clean final answer. `response` still
    /// carries a usable fallback message.
    pub hit_limit_or_error: bool,
    /// Run-level timing + token metrics for this turn.
    pub metrics: RunMetrics,
}

/// Build the system prompt: the agent's instructions followed by a
/// modality-awareness block summarizing the collections the agent can
/// search and which modalities each supports. Factored out so it is
/// unit-testable without a live model.
///
/// `modality_summary` is the pre-rendered collection/modality block (see
/// [`collections_modality_summary`]); keeping it an argument means this
/// helper needs no `AppState` and is trivially testable.
pub fn build_system_prompt(agent: &Agent, modality_summary: &str) -> String {
    let mut prompt = String::new();
    if !agent.instructions.trim().is_empty() {
        prompt.push_str(agent.instructions.trim());
        prompt.push_str("\n\n");
    }
    prompt.push_str(
        "You are an agent operating over Veda, a multimodal vector database. \
You can search not only text but also image, audio, video, and PDF/document \
corpora — embeddings from every modality share one unified vector space, so a \
text query can retrieve image/audio/video/document records and vice-versa. Use \
the provided tools to answer; do not invent collection names or results.",
    );
    if !modality_summary.trim().is_empty() {
        prompt.push_str("\n\nAvailable collections:\n");
        prompt.push_str(modality_summary.trim_end());
    } else {
        prompt.push_str("\n\nNo collections are currently available.");
    }
    prompt
}

/// Render a concise, line-per-collection summary of the tenant's
/// collections and the modalities each can be searched with, derived from
/// the live schemas + embedding registry. Async because it reads
/// `AppState`'s locks.
pub async fn collections_modality_summary(state: &AppState, tenant: &crate::tenant::TenantId) -> String {
    use veda_core::embedding::Modality;
    use veda_core::types::FieldType;

    let schemas = state.inner.schemas.read().await;
    let registry = state.inner.embedding_registry.read().await;

    // Modalities the deployment can embed *at all* (used as the fallback
    // for vector fields with no bound model).
    let mut deployment_modalities: Vec<Modality> = Vec::new();
    for m in [
        Modality::Text,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
        Modality::Pdf,
    ] {
        if !registry.find_by_modality(m).is_empty() {
            deployment_modalities.push(m);
        }
    }

    let mut lines: Vec<String> = Vec::new();
    let mut names: Vec<(&str, &veda_core::schema::CollectionSchema)> = schemas
        .iter()
        .filter_map(|(k, s)| crate::tenant::strip_prefix(tenant, k).map(|name| (name, s)))
        .collect();
    names.sort_by(|a, b| a.0.cmp(b.0));

    for (name, schema) in names {
        // Collect the modalities searchable across this collection's
        // vector fields.
        let mut mods: Vec<Modality> = Vec::new();
        let push = |m: Modality, acc: &mut Vec<Modality>| {
            if !acc.contains(&m) {
                acc.push(m);
            }
        };
        for f in schema.fields() {
            let model_id = match f.field_type() {
                FieldType::Vector { model_id, .. } => model_id.clone(),
                FieldType::MultiVector { .. } => None,
                _ => continue,
            };
            match model_id.as_deref().and_then(|id| registry.get(id)) {
                Some(cfg) => {
                    for m in &cfg.supported_modalities {
                        push(*m, &mut mods);
                    }
                }
                None => {
                    // Unbound vector field: infer from the field name, else
                    // fall back to what the deployment can embed.
                    if let Some(m) = Modality::from_field_name(f.name()) {
                        push(m, &mut mods);
                    } else {
                        for m in &deployment_modalities {
                            push(*m, &mut mods);
                        }
                    }
                }
            }
        }
        let modality_str = if mods.is_empty() {
            "no vector fields (scalar only)".to_owned()
        } else {
            mods.iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(format!("- {name} (searchable modalities: {modality_str})"));
    }

    lines.join("\n")
}

/// Truncate a string to at most `cap` bytes on a char boundary, appending
/// a marker when truncation occurred.
fn truncate_for_model(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_owned();
    }
    let mut end = cap;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &s[..end])
}

/// Run the tool-calling loop for `agent` against `model`, starting from
/// `history` plus a fresh `user_input` turn.
///
/// Generic over `M: ChatModel` (not `dyn`, since the trait is native
/// async-fn-in-trait) so tests can inject a fake model. Never panics on
/// model/dispatch/parse failures — they become error tool-results or a
/// graceful failure `RunResult`.
pub async fn run<M: ChatModel>(
    model: &M,
    catalog: &ToolCatalog,
    ctx: &ToolContext<'_>,
    agent: &Agent,
    system_prompt: &str,
    history: &[ConversationMessage],
    user_input: &str,
    max_steps: usize,
) -> RunResult {
    let max_steps = max_steps.max(1);

    let tools: Vec<ToolDef> = catalog
        .specs_for(&agent.tool_ids)
        .iter()
        .map(|s| s.to_openai_tooldef())
        .collect();

    // Assemble the OpenAI message list: system, then history, then the
    // new user turn.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(history.len() + 2);
    messages.push(ChatMessage::text("system", system_prompt));
    for m in history {
        if let Some(content) = &m.content {
            messages.push(ChatMessage::text(m.role.clone(), content.clone()));
        }
    }
    messages.push(ChatMessage::text("user", user_input));

    let mut steps: Vec<AgentStep> = Vec::new();
    let mut new_messages: Vec<ChatMessage> = Vec::new();
    let opts = CompletionOptions::default();

    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut total_tokens = 0u32;

    // APM timing accumulators. `run_start` measures the whole loop;
    // `llm_ms` / `tool_ms` sum per-call latencies.
    let run_start = std::time::Instant::now();
    let mut llm_ms = 0f64;
    let mut tool_ms = 0f64;
    let mut step_calls = 0u32;

    // Build the run-level metrics snapshot at each return point.
    macro_rules! run_metrics {
        () => {
            RunMetrics {
                total_ms: run_start.elapsed().as_secs_f64() * 1000.0,
                llm_ms,
                tool_ms,
                steps: step_calls,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }
        };
    }

    let mut last_content: Option<String> = None;

    for _ in 0..max_steps {
        step_calls = step_calls.saturating_add(1);
        let llm_start = std::time::Instant::now();
        let completion = match model
            .complete_with_tools(&messages, &tools, Some("auto"), &opts)
            .await
        {
            Ok(c) => {
                llm_ms += llm_start.elapsed().as_secs_f64() * 1000.0;
                c
            }
            Err(e) => {
                llm_ms += llm_start.elapsed().as_secs_f64() * 1000.0;
                let msg = format!("The model call failed: {e}");
                steps.push(AgentStep::Reasoning { text: msg.clone() });
                return RunResult {
                    response: msg,
                    steps,
                    new_messages,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    hit_limit_or_error: true,
                    metrics: run_metrics!(),
                };
            }
        };

        prompt_tokens = prompt_tokens.saturating_add(completion.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(completion.completion_tokens);
        total_tokens = total_tokens.saturating_add(completion.total_tokens);

        // No tool calls → this is the final answer.
        if completion.tool_calls.is_empty() {
            let answer = completion.content.unwrap_or_default();
            steps.push(AgentStep::Reasoning { text: answer.clone() });
            new_messages.push(ChatMessage::text("assistant", answer.clone()));
            return RunResult {
                response: answer,
                steps,
                new_messages,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                hit_limit_or_error: false,
                metrics: run_metrics!(),
            };
        }

        // The model wants to act. Record any interleaved reasoning text.
        if let Some(text) = &completion.content {
            if !text.trim().is_empty() {
                steps.push(AgentStep::Reasoning { text: text.clone() });
                last_content = Some(text.clone());
            }
        }

        // Push the assistant message carrying the tool_calls so the model
        // sees its own request alongside the results next turn.
        let assistant_msg = ChatMessage {
            role: "assistant".to_owned(),
            content: completion.content.clone(),
            tool_calls: Some(completion.tool_calls.clone()),
            tool_call_id: None,
        };
        messages.push(assistant_msg.clone());
        new_messages.push(assistant_msg);

        for call in &completion.tool_calls {
            let tool_id = &call.function.name;
            // Parse arguments; on failure feed an error result back rather
            // than aborting the whole run.
            match call.function.parse_arguments() {
                Ok(args) => {
                    steps.push(AgentStep::ToolCall {
                        tool_call_id: call.id.clone(),
                        tool_id: tool_id.clone(),
                        args: args.clone(),
                    });
                    let dispatch_start = std::time::Instant::now();
                    let outcome = catalog.dispatch(ctx, tool_id, args).await;
                    let dispatch_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;
                    tool_ms += dispatch_ms;
                    steps.push(AgentStep::ToolResult {
                        tool_call_id: call.id.clone(),
                        result: outcome.content.clone(),
                        is_error: outcome.is_error,
                        duration_ms: Some(dispatch_ms),
                    });
                    let result_str = outcome.content.to_string();
                    let tool_msg = ChatMessage::tool_result(
                        call.id.clone(),
                        truncate_for_model(&result_str, TOOL_RESULT_CAP_BYTES),
                    );
                    messages.push(tool_msg.clone());
                    new_messages.push(tool_msg);
                }
                Err(e) => {
                    let err = serde_json::json!({
                        "error": format!("could not parse tool arguments: {e}")
                    });
                    steps.push(AgentStep::ToolCall {
                        tool_call_id: call.id.clone(),
                        tool_id: tool_id.clone(),
                        args: serde_json::Value::Null,
                    });
                    steps.push(AgentStep::ToolResult {
                        tool_call_id: call.id.clone(),
                        result: err.clone(),
                        is_error: true,
                        duration_ms: None,
                    });
                    let tool_msg = ChatMessage::tool_result(call.id.clone(), err.to_string());
                    messages.push(tool_msg.clone());
                    new_messages.push(tool_msg);
                }
            }
        }
        // Loop: the model gets another turn to reason over the results.
    }

    // Step limit reached while still requesting tools. Return gracefully.
    let response = last_content.unwrap_or_else(|| {
        "I reached the maximum number of reasoning steps before producing a final \
answer. Here is what I gathered so far in the steps."
            .to_owned()
    });
    steps.push(AgentStep::Reasoning {
        text: format!("[step limit of {max_steps} reached]"),
    });
    RunResult {
        response,
        steps,
        new_messages,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        hit_limit_or_error: true,
        metrics: run_metrics!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use veda_core::error::Result as LlmResult;
    use veda_core::llm_client::{FunctionCall, ToolCall, ToolCompletion};
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::{DistanceMetric, FieldType};
    use veda_core::vector::DenseVector;

    use crate::agent::model::AgentStep;

    /// A scripted, network-free [`ChatModel`]. Pops one pre-built
    /// [`ToolCompletion`] from an internal queue per call. When `always`
    /// is set it returns that completion forever (for the max-steps test);
    /// otherwise it returns the queued items and then a default "done"
    /// answer once drained.
    struct FakeModel {
        queue: RefCell<VecDeque<ToolCompletion>>,
        always: Option<ToolCompletion>,
        calls: std::cell::Cell<usize>,
    }

    impl FakeModel {
        fn scripted(items: Vec<ToolCompletion>) -> Self {
            Self {
                queue: RefCell::new(items.into_iter().collect()),
                always: None,
                calls: std::cell::Cell::new(0),
            }
        }

        fn always(c: ToolCompletion) -> Self {
            Self {
                queue: RefCell::new(VecDeque::new()),
                always: Some(c),
                calls: std::cell::Cell::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.get()
        }
    }

    impl ChatModel for FakeModel {
        async fn complete_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDef],
            _tool_choice: Option<&str>,
            _opts: &CompletionOptions,
        ) -> LlmResult<ToolCompletion> {
            self.calls.set(self.calls.get() + 1);
            if let Some(c) = &self.always {
                return Ok(c.clone());
            }
            Ok(self.queue.borrow_mut().pop_front().unwrap_or_else(|| {
                ToolCompletion {
                    content: Some("done".to_owned()),
                    tool_calls: Vec::new(),
                    finish_reason: Some("stop".to_owned()),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                }
            }))
        }
    }

    fn tool_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: name.to_owned(),
                arguments: args.to_owned(),
            },
        }
    }

    fn completion_with_call(content: Option<&str>, call: ToolCall) -> ToolCompletion {
        ToolCompletion {
            content: content.map(str::to_owned),
            tool_calls: vec![call],
            finish_reason: Some("tool_calls".to_owned()),
            prompt_tokens: 5,
            completion_tokens: 5,
            total_tokens: 10,
        }
    }

    fn final_completion(text: &str) -> ToolCompletion {
        ToolCompletion {
            content: Some(text.to_owned()),
            tool_calls: Vec::new(),
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 3,
            completion_tokens: 7,
            total_tokens: 10,
        }
    }

    fn text_schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::vector("embedding", 4, DistanceMetric::Cosine))
            .add_field(FieldDefinition::new("title", FieldType::String).with_filterable(true))
            .shard_count(1)
            .build()
            .unwrap()
    }

    /// Build an AppState with one seeded text collection holding a single
    /// record (mirrors the catalog tests' harness).
    async fn seeded_state(
        tmp: &std::path::Path,
        user_collection: &str,
    ) -> (AppState, crate::tenant::TenantId) {
        let state = AppState::new(tmp);
        let tenant = crate::tenant::TenantId::default_tenant();
        let scoped = crate::tenant::scope(&tenant, user_collection);
        let schema = text_schema(&scoped);
        let col_dir = tmp.join(&scoped);
        std::fs::create_dir_all(&col_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(&state, &schema, &col_dir)
            .await
            .expect("materialize");

        let mut rec = Record::new();
        rec.set(
            "embedding",
            FieldValue::Vector(DenseVector::new(vec![1.0, 0.0, 0.0, 0.0])),
        );
        rec.set("title", FieldValue::String("hello world".to_owned()));
        let id = rec.id();
        {
            let mut managers = state.inner.segment_managers.write().await;
            let shard_map = managers.get_mut(&scoped).unwrap();
            let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
                &id,
                schema.shard_count(),
            );
            shard_map.get_mut(&shard_id).unwrap().insert(rec).unwrap();
        }
        (state, tenant)
    }

    fn test_agent(tool_ids: Vec<String>) -> Agent {
        Agent {
            id: "ag-test".to_owned(),
            name: "Test".to_owned(),
            description: String::new(),
            instructions: "Be concise and helpful.".to_owned(),
            tool_ids,
            model: None,
        }
    }

    #[tokio::test]
    async fn happy_path_dispatches_tool_then_returns_final_answer() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let agent = test_agent(vec!["list_collections".to_owned()]);

        let model = FakeModel::scripted(vec![
            completion_with_call(
                Some("Let me check the collections."),
                tool_call("call_1", "list_collections", "{}"),
            ),
            final_completion("There is one collection: docs."),
        ]);

        let result = run(
            &model, &cat, &ctx, &agent, "sys", &[], "what collections exist?", DEFAULT_MAX_STEPS,
        )
        .await;

        assert!(!result.hit_limit_or_error);
        assert_eq!(result.response, "There is one collection: docs.");
        assert_eq!(model.call_count(), 2);

        // Trace sequence: reasoning, tool_call(list_collections),
        // tool_result(ok), final reasoning.
        let kinds: Vec<&str> = result
            .steps
            .iter()
            .map(|s| match s {
                AgentStep::Reasoning { .. } => "reasoning",
                AgentStep::ToolCall { .. } => "tool_call",
                AgentStep::ToolResult { .. } => "tool_result",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["reasoning", "tool_call", "tool_result", "reasoning"]
        );

        // The dispatched tool result carries the real collection list.
        let tool_result = result
            .steps
            .iter()
            .find_map(|s| match s {
                AgentStep::ToolResult { result, is_error, .. } => Some((result, *is_error)),
                _ => None,
            })
            .unwrap();
        assert!(!tool_result.1, "tool should have succeeded");
        let cols = tool_result.0["collections"].as_array().unwrap();
        assert!(cols.iter().any(|c| c.as_str() == Some("docs")));

        // Token usage is summed across both turns.
        assert_eq!(result.total_tokens, 20);
    }

    #[tokio::test]
    async fn run_metrics_capture_timing_and_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let agent = test_agent(vec!["list_collections".to_owned()]);

        // [tool_call → final]: two LLM turns, one dispatch.
        let model = FakeModel::scripted(vec![
            completion_with_call(
                Some("Checking."),
                tool_call("call_1", "list_collections", "{}"),
            ),
            final_completion("Done."),
        ]);

        let result = run(
            &model, &cat, &ctx, &agent, "sys", &[], "what collections?", DEFAULT_MAX_STEPS,
        )
        .await;

        let m = result.metrics;
        // Two model turns happened.
        assert_eq!(m.steps, 2, "expected 2 model turns");
        // Trace has >= 2 steps (reasoning, tool_call, tool_result, reasoning).
        assert!(result.steps.len() >= 2);
        // LLM time was measured; tool time is non-negative; their sum fits
        // within the whole-run span (with generous slack for bookkeeping).
        assert!(m.llm_ms > 0.0, "llm_ms should be > 0, got {}", m.llm_ms);
        assert!(m.tool_ms >= 0.0, "tool_ms should be >= 0, got {}", m.tool_ms);
        assert!(
            m.llm_ms + m.tool_ms <= m.total_ms + 5.0,
            "llm_ms ({}) + tool_ms ({}) should be <= total_ms ({}) + slack",
            m.llm_ms, m.tool_ms, m.total_ms
        );
        // Tokens match the summed RunResult totals (5+5 + 3+7 = 20).
        assert_eq!(m.total_tokens, result.total_tokens);
        assert_eq!(m.total_tokens, 20);
        assert_eq!(m.prompt_tokens, 8);
        assert_eq!(m.completion_tokens, 12);

        // The dispatched tool-result step carries its own duration_ms.
        let durations: Vec<Option<f64>> = result
            .steps
            .iter()
            .filter_map(|s| match s {
                AgentStep::ToolResult { duration_ms, .. } => Some(*duration_ms),
                _ => None,
            })
            .collect();
        assert!(
            durations.iter().any(|d| d.is_some()),
            "a dispatched tool-result should carry duration_ms"
        );
    }

    #[tokio::test]
    async fn max_steps_guard_stops_and_returns_gracefully() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let agent = test_agent(vec!["list_collections".to_owned()]);

        // A model that ALWAYS asks for a tool — never produces a final
        // answer. The loop must stop at max_steps.
        let model = FakeModel::always(completion_with_call(
            Some("still working"),
            tool_call("call_loop", "list_collections", "{}"),
        ));

        let max_steps = 3;
        let result = run(
            &model, &cat, &ctx, &agent, "sys", &[], "loop forever", max_steps,
        )
        .await;

        assert!(result.hit_limit_or_error);
        // Exactly max_steps model calls, no more.
        assert_eq!(model.call_count(), max_steps);
        // Dispatched tool calls are bounded by max_steps too.
        let dispatched = result
            .steps
            .iter()
            .filter(|s| matches!(s, AgentStep::ToolCall { .. }))
            .count();
        assert!(dispatched <= max_steps, "dispatched {dispatched} > {max_steps}");
        assert!(!result.response.is_empty());
    }

    #[tokio::test]
    async fn tool_arg_parse_error_feeds_error_result_and_continues() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let agent = test_agent(vec!["describe_collection".to_owned()]);

        // First turn: a tool call whose arguments are NOT valid JSON.
        let model = FakeModel::scripted(vec![
            completion_with_call(
                None,
                tool_call("call_bad", "describe_collection", "{not json"),
            ),
            final_completion("Sorry, I could not parse that."),
        ]);

        let result = run(
            &model, &cat, &ctx, &agent, "sys", &[], "describe something", DEFAULT_MAX_STEPS,
        )
        .await;

        // No panic; loop continued to a graceful final answer.
        assert!(!result.hit_limit_or_error);
        assert_eq!(result.response, "Sorry, I could not parse that.");
        // A ToolResult step with is_error=true is recorded.
        let err_step = result.steps.iter().any(|s| {
            matches!(s, AgentStep::ToolResult { is_error: true, .. })
        });
        assert!(err_step, "expected an error tool-result step");
    }

    #[tokio::test]
    async fn model_error_returns_graceful_failure_not_panic() {
        // A model whose call errors out. (Reuse the real LlmClient would
        // hit the network; instead a tiny erroring fake.)
        struct ErrModel;
        impl ChatModel for ErrModel {
            async fn complete_with_tools(
                &self,
                _m: &[ChatMessage],
                _t: &[ToolDef],
                _tc: Option<&str>,
                _o: &CompletionOptions,
            ) -> LlmResult<ToolCompletion> {
                Err(veda_core::error::VedaError::Query("boom".to_owned()))
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let agent = test_agent(vec![]);

        let result = run(
            &ErrModel, &cat, &ctx, &agent, "sys", &[], "hello", DEFAULT_MAX_STEPS,
        )
        .await;
        assert!(result.hit_limit_or_error);
        assert!(result.response.contains("failed"));
    }

    #[tokio::test]
    async fn build_system_prompt_includes_instructions_and_modalities() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_state(tmp.path(), "docs").await;
        let agent = test_agent(vec![]);

        let summary = collections_modality_summary(&state, &tenant).await;
        // The seeded collection appears, with a Text modality (the
        // unbound `embedding` vector field has no model; "embedding" is
        // not a recognised modality name and the registry is empty, so it
        // falls back to scalar-only — assert the collection at least
        // shows up).
        assert!(summary.contains("docs"), "summary: {summary}");

        let prompt = build_system_prompt(&agent, &summary);
        assert!(prompt.contains("Be concise and helpful."));
        assert!(prompt.contains("docs"));
        assert!(prompt.contains("multimodal"));
    }

    #[tokio::test]
    async fn modality_summary_reflects_bound_image_model() {
        use veda_core::embedding::{EmbeddingModelConfig, Modality};

        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();

        // Register a multimodal model and a collection whose vector field
        // is bound to it.
        {
            let mut reg = state.inner.embedding_registry.write().await;
            reg.register(EmbeddingModelConfig {
                name: "mm".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text, Modality::Image],
                output_dimension: 4,
                max_input_size_bytes: 1024,
                timeout_seconds: 30,
                batch_size: 8,
            });
        }
        let scoped = crate::tenant::scope(&tenant, "media");
        let schema = CollectionSchema::builder(&scoped)
            .add_field(FieldDefinition::vector_with_model(
                "vec", 4, DistanceMetric::Cosine, "mm",
            ))
            .shard_count(1)
            .build()
            .unwrap();
        state
            .inner
            .schemas
            .write()
            .await
            .insert(scoped.clone(), schema);

        let summary = collections_modality_summary(&state, &tenant).await;
        assert!(summary.contains("media"), "summary: {summary}");
        assert!(summary.contains("Image"), "summary: {summary}");
        assert!(summary.contains("Text"), "summary: {summary}");
    }
}
