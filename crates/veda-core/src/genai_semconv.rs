//! OpenTelemetry **GenAI semantic-convention** attribute keys + span names.
//!
//! Centralizes the `gen_ai.*` attribute keys and span-name helpers so that
//! Veda's agent/RAG traces drop natively into Datadog / Grafana / Phoenix /
//! Langfuse. The GenAI semantic conventions are still **experimental**, so
//! keeping every spec-defined string in one module means a spec bump is a
//! one-file change rather than a scattered find-and-replace.
//!
//! See <https://opentelemetry.io/docs/specs/semconv/gen-ai/>.
//!
//! Spans created with the `tracing` crate carry these as **dotted field
//! names**; the `tracing_opentelemetry` layer maps them to OTel span
//! attributes verbatim, so e.g. `gen_ai.request.model = %model` becomes the
//! `gen_ai.request.model` attribute on the exported span.

// ── Operation / system identifiers ───────────────────────────────────────

/// `gen_ai.operation.name` — the kind of GenAI operation the span describes
/// (`chat`, `execute_tool`, `invoke_agent`, …).
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
/// `gen_ai.system` — the GenAI provider/system (e.g. `openai`, `anthropic`).
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";

// ── Request attributes ────────────────────────────────────────────────────

/// `gen_ai.request.model` — the model name requested for the operation.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

// ── Response / usage attributes ───────────────────────────────────────────

/// `gen_ai.usage.input_tokens` — prompt (input) tokens consumed.
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
/// `gen_ai.usage.output_tokens` — completion (output) tokens produced.
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
/// `gen_ai.response.finish_reasons` — array of finish reasons. Tracing has
/// no native array field, so we record a comma-joined string under this key
/// (acceptable for v1 — see `crate::agent` instrumentation).
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

// ── Tool attributes ───────────────────────────────────────────────────────

/// `gen_ai.tool.name` — the name of the tool being executed.
pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
/// `gen_ai.tool.call.id` — the model-assigned id of the tool call.
pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";

// ── Agent attributes ──────────────────────────────────────────────────────

/// `gen_ai.agent.name` — the human-readable name of the agent being invoked.
pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";

// ── Operation-name values ─────────────────────────────────────────────────

/// Value of `gen_ai.operation.name` for an LLM chat/completion call.
pub const OP_CHAT: &str = "chat";
/// Value of `gen_ai.operation.name` for a tool execution.
pub const OP_EXECUTE_TOOL: &str = "execute_tool";
/// Value of `gen_ai.operation.name` for an agent invocation.
pub const OP_INVOKE_AGENT: &str = "invoke_agent";

// ── Span-name helpers ─────────────────────────────────────────────────────
//
// The GenAI semconv recommends span names of the form `{operation} {target}`,
// e.g. `chat gpt-4`, `execute_tool search`, `invoke_agent veda-agent`.

/// Span name for a chat/completion call against `model`: `chat {model}`.
pub fn chat_span_name(model: &str) -> String {
    format!("{OP_CHAT} {model}")
}

/// Span name for a tool execution of `tool`: `execute_tool {tool}`.
pub fn execute_tool_span_name(tool: &str) -> String {
    format!("{OP_EXECUTE_TOOL} {tool}")
}

/// Span name for an agent invocation of `agent`: `invoke_agent {agent}`.
pub fn invoke_agent_span_name(agent: &str) -> String {
    format!("{OP_INVOKE_AGENT} {agent}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_keys_match_genai_semconv() {
        assert_eq!(GEN_AI_OPERATION_NAME, "gen_ai.operation.name");
        assert_eq!(GEN_AI_SYSTEM, "gen_ai.system");
        assert_eq!(GEN_AI_REQUEST_MODEL, "gen_ai.request.model");
        assert_eq!(GEN_AI_USAGE_INPUT_TOKENS, "gen_ai.usage.input_tokens");
        assert_eq!(GEN_AI_USAGE_OUTPUT_TOKENS, "gen_ai.usage.output_tokens");
        assert_eq!(
            GEN_AI_RESPONSE_FINISH_REASONS,
            "gen_ai.response.finish_reasons"
        );
        assert_eq!(GEN_AI_TOOL_NAME, "gen_ai.tool.name");
        assert_eq!(GEN_AI_TOOL_CALL_ID, "gen_ai.tool.call.id");
        assert_eq!(GEN_AI_AGENT_NAME, "gen_ai.agent.name");
    }

    #[test]
    fn span_name_helpers_format_operation_and_target() {
        assert_eq!(chat_span_name("gpt-4o"), "chat gpt-4o");
        assert_eq!(execute_tool_span_name("search"), "execute_tool search");
        assert_eq!(
            invoke_agent_span_name("veda-agent"),
            "invoke_agent veda-agent"
        );
    }
}
