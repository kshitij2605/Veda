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
}
