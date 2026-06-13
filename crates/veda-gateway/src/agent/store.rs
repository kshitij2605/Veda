//! Persistence helpers for Veda Agents.
//!
//! These are thin wrappers over the gateway's existing
//! [`SavedObjectStore`](crate::rest::saved_objects::SavedObjectStore):
//! each agent entity is stored as a `SavedObject` whose `type` tag is
//! one of the constants below, with the full struct serialized into the
//! object `body`. No new persistence file or layer is introduced.

use serde::Serialize;

use crate::rest::saved_objects::{SavedObject, SavedObjectStore};

use super::model::{Agent, AgentTool, Conversation};

/// `SavedObject.type` for an [`Agent`].
pub const TYPE_AGENT: &str = "agent";
/// `SavedObject.type` for an [`AgentTool`].
pub const TYPE_AGENT_TOOL: &str = "agent_tool";
/// `SavedObject.type` for a [`Conversation`].
pub const TYPE_CONVERSATION: &str = "agent_conversation";

/// Wrap a serializable entity into a `SavedObject` of the given type.
/// `body` failing to serialize is impossible for these structs, so we
/// fall back to `Value::Null` rather than panicking.
fn to_saved<T: Serialize>(id: String, name: String, kind: &str, value: &T) -> SavedObject {
    SavedObject {
        id,
        kind: kind.to_owned(),
        name,
        body: serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    }
}

// --- Agent ---------------------------------------------------------------

/// Insert or update an agent.
pub async fn save_agent(store: &SavedObjectStore, agent: &Agent) {
    let obj = to_saved(agent.id.clone(), agent.name.clone(), TYPE_AGENT, agent);
    store.put(obj).await;
}

/// Fetch an agent by id. Returns `None` if missing or if the stored row
/// is not a well-formed agent.
pub async fn get_agent(store: &SavedObjectStore, id: &str) -> Option<Agent> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_AGENT {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

/// List all agents, newest-first.
pub async fn list_agents(store: &SavedObjectStore) -> Vec<Agent> {
    store
        .list_by_type(TYPE_AGENT)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}

/// Delete an agent by id (idempotent).
pub async fn delete_agent(store: &SavedObjectStore, id: &str) -> bool {
    store.remove(id).await
}

// --- AgentTool -----------------------------------------------------------

/// Insert or update a custom tool definition.
pub async fn save_tool(store: &SavedObjectStore, tool: &AgentTool) {
    let obj = to_saved(tool.id.clone(), tool.name.clone(), TYPE_AGENT_TOOL, tool);
    store.put(obj).await;
}

/// Fetch a tool by id.
pub async fn get_tool(store: &SavedObjectStore, id: &str) -> Option<AgentTool> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_AGENT_TOOL {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

/// List all custom tools, newest-first.
pub async fn list_tools(store: &SavedObjectStore) -> Vec<AgentTool> {
    store
        .list_by_type(TYPE_AGENT_TOOL)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}

/// Delete a tool by id (idempotent).
pub async fn delete_tool(store: &SavedObjectStore, id: &str) -> bool {
    store.remove(id).await
}

// --- Conversation --------------------------------------------------------

/// Insert or update a conversation. A conversation has no name of its
/// own; we use its `title` if present, else its id, for the SavedObject
/// `name` (purely for human-readable listings).
pub async fn save_conversation(store: &SavedObjectStore, conv: &Conversation) {
    let name = conv.title.clone().unwrap_or_else(|| conv.id.clone());
    let obj = to_saved(conv.id.clone(), name, TYPE_CONVERSATION, conv);
    store.put(obj).await;
}

/// Fetch a conversation by id.
pub async fn get_conversation(store: &SavedObjectStore, id: &str) -> Option<Conversation> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_CONVERSATION {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

/// List all conversations, newest-first.
pub async fn list_conversations(store: &SavedObjectStore) -> Vec<Conversation> {
    store
        .list_by_type(TYPE_CONVERSATION)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}

/// Delete a conversation by id (idempotent).
pub async fn delete_conversation(store: &SavedObjectStore, id: &str) -> bool {
    store.remove(id).await
}
