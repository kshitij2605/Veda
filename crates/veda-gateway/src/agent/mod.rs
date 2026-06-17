//! Veda Agents — data model and persistence.
//!
//! This module defines the agent feature's data model ([`model`]) and
//! thin persistence helpers ([`store`]) over the gateway's existing
//! `SavedObjectStore`. The tool catalog ([`catalog`]) and executor
//! ([`executor`]) build on top; the REST surface is added in a later task.

pub mod catalog;
pub mod executor;
pub mod model;
pub mod store;
pub mod tools;

#[cfg(test)]
mod tests {
    use super::model::*;
    use super::store::*;
    use crate::rest::saved_objects::SavedObjectStore;

    fn new_store() -> (SavedObjectStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = SavedObjectStore::new(tmp.path());
        (store, tmp)
    }

    #[tokio::test]
    async fn agent_round_trip_create_get_list_delete() {
        let (store, _tmp) = new_store();
        let agent = Agent {
            id: "ag-1".to_owned(),
            name: "Researcher".to_owned(),
            description: "Searches collections".to_owned(),
            instructions: "Be concise.".to_owned(),
            tool_ids: vec!["search".to_owned(), "aggregate".to_owned()],
            model: Some(ModelOverride {
                endpoint: Some("http://llm:9000/v1/chat/completions".to_owned()),
                model: Some("Qwen3.5".to_owned()),
            }),
        };

        save_agent(&store, &agent).await;

        let loaded = get_agent(&store, "ag-1").await.expect("agent present");
        assert_eq!(loaded, agent);

        let listed = list_agents(&store).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0], agent);

        assert!(delete_agent(&store, "ag-1").await);
        assert!(get_agent(&store, "ag-1").await.is_none());
        assert!(list_agents(&store).await.is_empty());
    }

    #[tokio::test]
    async fn agent_tool_round_trip() {
        let (store, _tmp) = new_store();
        let tool = AgentTool {
            id: "tool-1".to_owned(),
            name: "Weather".to_owned(),
            description: "Fetch weather".to_owned(),
            kind: "http".to_owned(),
            config: serde_json::json!({ "url": "https://example.com", "method": "GET" }),
        };

        save_tool(&store, &tool).await;

        let loaded = get_tool(&store, "tool-1").await.expect("tool present");
        assert_eq!(loaded, tool);

        let listed = list_tools(&store).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0], tool);

        assert!(delete_tool(&store, "tool-1").await);
        assert!(get_tool(&store, "tool-1").await.is_none());
    }

    #[tokio::test]
    async fn conversation_round_trip_with_all_step_variants() {
        let (store, _tmp) = new_store();
        let conv = Conversation {
            id: "conv-1".to_owned(),
            agent_id: "ag-1".to_owned(),
            title: Some("First chat".to_owned()),
            messages: vec![
                ConversationMessage {
                    role: "user".to_owned(),
                    content: Some("find docs about HNSW".to_owned()),
                },
                ConversationMessage {
                    role: "assistant".to_owned(),
                    content: Some("Here are the results.".to_owned()),
                },
            ],
            steps: vec![
                AgentStep::Reasoning {
                    text: "I should search the collection.".to_owned(),
                },
                AgentStep::ToolCall {
                    tool_call_id: "call-1".to_owned(),
                    tool_id: "search".to_owned(),
                    args: serde_json::json!({ "query": "HNSW", "k": 5 }),
                },
                AgentStep::ToolResult {
                    tool_call_id: "call-1".to_owned(),
                    result: serde_json::json!({ "hits": 3 }),
                    is_error: false,
                    duration_ms: None,
                },
            ],
        };

        save_conversation(&store, &conv).await;

        let loaded = get_conversation(&store, "conv-1")
            .await
            .expect("conversation present");
        assert_eq!(loaded, conv);

        let listed = list_conversations(&store).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0], conv);

        assert!(delete_conversation(&store, "conv-1").await);
        assert!(get_conversation(&store, "conv-1").await.is_none());
    }

    #[tokio::test]
    async fn list_by_type_does_not_leak_across_types() {
        let (store, _tmp) = new_store();

        save_agent(
            &store,
            &Agent {
                id: "ag-1".to_owned(),
                name: "A".to_owned(),
                description: String::new(),
                instructions: String::new(),
                tool_ids: vec![],
                model: None,
            },
        )
        .await;
        save_tool(
            &store,
            &AgentTool {
                id: "tool-1".to_owned(),
                name: "T".to_owned(),
                description: String::new(),
                kind: "http".to_owned(),
                config: serde_json::Value::Null,
            },
        )
        .await;
        save_conversation(
            &store,
            &Conversation {
                id: "conv-1".to_owned(),
                agent_id: "ag-1".to_owned(),
                title: None,
                messages: vec![],
                steps: vec![],
            },
        )
        .await;

        assert_eq!(list_agents(&store).await.len(), 1);
        assert_eq!(list_tools(&store).await.len(), 1);
        assert_eq!(list_conversations(&store).await.len(), 1);

        // Cross-type fetches must miss.
        assert!(get_tool(&store, "ag-1").await.is_none());
        assert!(get_agent(&store, "tool-1").await.is_none());
        assert!(get_conversation(&store, "ag-1").await.is_none());
    }
}
