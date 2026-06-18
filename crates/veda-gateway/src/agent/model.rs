//! Data model for Veda Agents.
//!
//! These are plain serde types — the persistence layer (`super::store`)
//! round-trips them through the gateway's existing `SavedObjectStore`,
//! and later tasks (tool catalog, executor, REST surface) build on top.
//! Nothing here calls an LLM or executes a tool.

use serde::{Deserialize, Serialize};

/// Per-agent override of the global `VEDA_LLM_*` configuration. `None`
/// fields fall back to the global config.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ModelOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// A configured agent: an instruction set plus the tools it may use.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Agent {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub instructions: String,
    /// Ids of tools (builtin or `AgentTool`) this agent is allowed to call.
    #[serde(default)]
    pub tool_ids: Vec<String>,
    /// Optional per-agent LLM override; `None` uses the global config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelOverride>,
    /// Opt-in vector-backed long-term memory. When `true`, the agent
    /// recalls semantically relevant past turns (across conversations)
    /// from Veda's own vector store and writes each turn back. Defaults
    /// to `false` (serde-default → existing agents are unchanged and
    /// behave byte-identically).
    #[serde(default)]
    pub memory: bool,
}

/// The default, always-available agent used when a converse request
/// targets `veda-agent` and no such agent has been explicitly created.
/// Held in memory only (never persisted unless a user saves one with
/// the same id). An empty `tool_ids` whitelist means "all builtins".
pub fn default_agent() -> Agent {
    Agent {
        id: "veda-agent".to_owned(),
        name: "Veda Agent".to_owned(),
        description: "General-purpose assistant over your Veda collections.".to_owned(),
        instructions: "You are Veda Agent, a helpful assistant that answers questions by \
searching and analyzing the user's multimodal vector collections. Prefer using the \
provided tools to ground your answers in real data; be concise and cite what you found."
            .to_owned(),
        tool_ids: Vec::new(),
        model: None,
        memory: false,
    }
}

/// The built-in operations-analyst agent used by AI-assisted ops triage
/// (`POST /v1/admin/insights/{id}/triage`) and reachable via the
/// well-known `veda-ops` id. Always available; held in memory only
/// (mirrors [`default_agent`]). Its whitelist is the read-only ops tools
/// plus the read-only data tools that help an investigation — never the
/// mutating `upsert_record`.
pub fn default_ops_agent() -> Agent {
    Agent {
        id: "veda-ops".to_owned(),
        name: "Veda Ops".to_owned(),
        description: "Operations assistant that explains Veda alerts and ops state.".to_owned(),
        instructions: "You are Veda's operations assistant. Given an alert/event, explain the \
likely root cause and concrete next steps, grounded ONLY in the tools (insights, logs, \
resources, SLO status, and read-only collection inspection). Investigate before concluding; \
be specific and brief."
            .to_owned(),
        tool_ids: vec![
            "get_insights".to_owned(),
            "get_logs".to_owned(),
            "get_resources".to_owned(),
            "get_slo_status".to_owned(),
            "list_collections".to_owned(),
            "describe_collection".to_owned(),
            "aggregate".to_owned(),
            "route".to_owned(),
        ],
        model: None,
        memory: false,
    }
}

/// A user-defined custom tool. `kind` + opaque `config` are interpreted
/// by the tool catalog in a later task; this is just the stored shape.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AgentTool {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub kind: String,
    #[serde(default)]
    pub config: serde_json::Value,
}

/// One step in an agent's reasoning/acting trace. Serializes to a
/// discriminated shape tagged by `kind`, e.g.
/// `{"kind":"tool_call","tool_call_id":"...","tool_id":"...","args":{}}`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentStep {
    Reasoning {
        text: String,
    },
    ToolCall {
        tool_call_id: String,
        tool_id: String,
        #[serde(default)]
        args: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        #[serde(default)]
        result: serde_json::Value,
        #[serde(default)]
        is_error: bool,
        /// Wall-clock latency of the tool dispatch in milliseconds.
        /// Additive + serde-back-compat: omitted when not captured.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<f64>,
    },
}

/// A stored chat turn. Kept intentionally minimal and compatible with an
/// OpenAI-style message (`role` + optional `content`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ConversationMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// A conversation with an agent: the visible message history plus the
/// full trace of steps taken across turns.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Conversation {
    pub id: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub messages: Vec<ConversationMessage>,
    #[serde(default)]
    pub steps: Vec<AgentStep>,
}

/// An attachment carried on a converse request (e.g. an image to caption
/// or search by). Minimal carrier; the executor interprets it later.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Attachment {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_b64: Option<String>,
}

/// A request to converse with an agent. `conversation_id` continues an
/// existing conversation; `None` starts a new one.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ConverseRequest {
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<Attachment>>,
}

/// The agent's reply plus the steps taken to produce it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ConverseResponse {
    pub conversation_id: String,
    pub response: String,
    #[serde(default)]
    pub steps: Vec<AgentStep>,
    /// Run-level APM metrics (timing + token usage) for this turn.
    /// Additive + serde-back-compat: omitted when unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<crate::agent::executor::RunMetrics>,
    /// Retrieved chunks the answer is grounded in (empty when the agent
    /// did not retrieve). Additive + serde-back-compat.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<crate::agent::grounding::Citation>,
    /// Faithfulness / answer-relevancy scores for this turn, when scored.
    /// Additive + serde-back-compat: omitted when unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding: Option<crate::agent::grounding::Grounding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_step_serde_omits_duration_when_none() {
        let step = AgentStep::ToolResult {
            tool_call_id: "c1".to_owned(),
            result: serde_json::json!({"ok": true}),
            is_error: false,
            duration_ms: None,
        };
        let v = serde_json::to_value(&step).unwrap();
        assert_eq!(v["kind"], "tool_result");
        assert!(v.get("duration_ms").is_none(), "duration_ms should be skipped: {v}");
        // Round-trips and back-compat: old payload without duration_ms parses.
        let back: AgentStep = serde_json::from_value(v).unwrap();
        assert_eq!(back, step);
    }

    #[test]
    fn tool_result_step_serde_includes_duration_when_some() {
        let step = AgentStep::ToolResult {
            tool_call_id: "c1".to_owned(),
            result: serde_json::Value::Null,
            is_error: false,
            duration_ms: Some(12.5),
        };
        let v = serde_json::to_value(&step).unwrap();
        assert_eq!(v["duration_ms"], 12.5);
        let back: AgentStep = serde_json::from_value(v).unwrap();
        assert_eq!(back, step);
    }

    #[test]
    fn tool_result_back_compat_missing_duration_field() {
        // A serialized step from before the timing field existed.
        let v = serde_json::json!({
            "kind": "tool_result",
            "tool_call_id": "c1",
            "result": {"x": 1},
            "is_error": false
        });
        let step: AgentStep = serde_json::from_value(v).unwrap();
        match step {
            AgentStep::ToolResult { duration_ms, .. } => assert!(duration_ms.is_none()),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn other_step_variants_unchanged() {
        let reasoning = AgentStep::Reasoning { text: "hi".to_owned() };
        assert_eq!(serde_json::to_value(&reasoning).unwrap()["kind"], "reasoning");
        let call = AgentStep::ToolCall {
            tool_call_id: "c1".to_owned(),
            tool_id: "t".to_owned(),
            args: serde_json::json!({}),
        };
        assert_eq!(serde_json::to_value(&call).unwrap()["kind"], "tool_call");
    }

    #[test]
    fn agent_memory_defaults_false_and_is_serde_back_compat() {
        // An agent JSON from before the field existed parses with
        // memory = false (additive + serde-default → no behavior change).
        let v = serde_json::json!({
            "id": "ag-1",
            "name": "Legacy",
            "instructions": "be brief",
            "tool_ids": []
        });
        let agent: Agent = serde_json::from_value(v).unwrap();
        assert!(!agent.memory, "memory must default to false");

        // A false flag is serialized (plain bool, not skipped) and an
        // explicit true round-trips.
        let mut a = default_agent();
        assert!(!a.memory);
        a.memory = true;
        let back: Agent = serde_json::from_value(serde_json::to_value(&a).unwrap()).unwrap();
        assert!(back.memory);
    }

    #[test]
    fn converse_response_omits_metrics_when_none() {
        let r = ConverseResponse {
            conversation_id: "c".to_owned(),
            response: "hi".to_owned(),
            steps: Vec::new(),
            metrics: None,
            citations: Vec::new(),
            grounding: None,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert!(v.get("metrics").is_none(), "metrics should be skipped: {v}");
        assert!(v.get("citations").is_none(), "empty citations should be skipped: {v}");
        assert!(v.get("grounding").is_none(), "grounding should be skipped: {v}");
    }
}
