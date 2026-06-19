//! Vector-backed agent memory (Phase D).
//!
//! Opt-in semantic long-term memory: an agent with `memory == true`
//! recalls the most relevant past turns (across all of its conversations)
//! from Veda's own vector store and has each new turn written back. The
//! whole feature is **best-effort and non-fatal** — a memory failure
//! (missing collection, no embedder, embedding/search error) must NEVER
//! fail the agent turn, so every public entry point degrades to "empty"
//! / "skip" rather than propagating an error.
//!
//! Embedding is done **off any lock**, mirroring `logs/categorize.rs`:
//! we clone the embedder config out from under the registry read-lock and
//! build the [`EmbeddingClient`] before any `.await` on the network, so
//! the registry / advisor / segment locks are never held across an embed
//! or search await.
//!
//! Turns are embedded **explicitly on write** (we compute the vector with
//! the same text embedder the deployment uses and set it on the record),
//! then inserted directly into the memory collection's segment manager —
//! the same in-process insert path the `upsert_record` builtin uses. The
//! memory collection's vector field also carries an `auto_embed` config
//! over `text` so the schema is self-describing for other tooling, but
//! this path supplies the vector explicitly rather than routing through
//! the lock-holding `embed_insert` HTTP flow.

use serde::{Deserialize, Serialize};

use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::Record;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{AutoEmbedConfig, AutoEmbedModality, DistanceMetric, FieldType};
use veda_query::search::VectorSearchRequest;

use super::model::Agent;
use crate::rest::state::AppState;
use crate::tenant::{self, TenantId};

/// User-facing name of the per-tenant agent-memory collection. Tenant
/// scoping (`tenant::scope`) is applied on top, so each tenant gets its
/// own isolated `_agent_memory` collection.
pub const MEMORY_COLLECTION: &str = "_agent_memory";

/// The text field the turn is stored + embedded from.
const TEXT_FIELD: &str = "text";
/// The vector field auto-embedded over [`TEXT_FIELD`].
const EMBEDDING_FIELD: &str = "embedding";
/// The filterable agent id (memory is recalled per-agent).
const AGENT_ID_FIELD: &str = "agent_id";

/// Default number of past turns to recall.
pub const DEFAULT_RECALL_K: usize = 5;
/// Default character budget for the injected memory block.
pub const DEFAULT_MEMORY_BLOCK_CHARS: usize = 1500;

/// One recalled past turn.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MemoryTurn {
    pub text: String,
    pub role: String,
    pub conversation_id: String,
    pub ts: i64,
    pub score: f32,
}

/// Whether the given agent has opted into vector-backed memory. The
/// gating guard kept as a tiny named function so the wiring in
/// `run_agent_turn` is unit-testable without standing up an LLM.
#[inline]
pub fn should_use_memory(agent: &Agent) -> bool {
    agent.memory
}

/// Resolve a text embedder by cloning its config out from under the
/// registry read-lock (so we never hold the lock across the embed
/// await). `None` when the deployment cannot embed text.
async fn text_embedder(state: &AppState) -> Option<EmbeddingClient> {
    let registry = state.inner.embedding_registry.read().await;
    registry
        .find_by_modality(Modality::Text)
        .first()
        .map(|c| EmbeddingClient::new((*c).clone()))
}

/// Build the tenant-scoped memory collection schema for a given embedding
/// dimension. Pure — no I/O, so it is directly unit-testable.
pub fn memory_schema(scoped_name: &str, dimension: u32) -> Result<CollectionSchema, String> {
    CollectionSchema::builder(scoped_name)
        .add_field(FieldDefinition::new(TEXT_FIELD, FieldType::String))
        .add_field(FieldDefinition::new("role", FieldType::String))
        .add_field(FieldDefinition::new(AGENT_ID_FIELD, FieldType::String).with_filterable(true))
        .add_field(
            FieldDefinition::new("conversation_id", FieldType::String).with_filterable(true),
        )
        .add_field(FieldDefinition::new("ts", FieldType::Int64).with_filterable(true))
        .add_field(
            FieldDefinition::vector(EMBEDDING_FIELD, dimension, DistanceMetric::Cosine)
                .with_auto_embed(AutoEmbedConfig {
                    source: TEXT_FIELD.to_owned(),
                    modality: AutoEmbedModality::Text,
                }),
        )
        .build()
        .map_err(|e| e.to_string())
}

/// Build the JSON record map for one stored turn (the shape passed to
/// `parse_record_public`). Pure — the `embedding` is supplied explicitly.
fn turn_record_map(
    agent_id: &str,
    conversation_id: &str,
    role: &str,
    text: &str,
    ts: i64,
    embedding: Vec<f32>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    m.insert(TEXT_FIELD.to_owned(), serde_json::Value::String(text.to_owned()));
    m.insert("role".to_owned(), serde_json::Value::String(role.to_owned()));
    m.insert(
        AGENT_ID_FIELD.to_owned(),
        serde_json::Value::String(agent_id.to_owned()),
    );
    m.insert(
        "conversation_id".to_owned(),
        serde_json::Value::String(conversation_id.to_owned()),
    );
    m.insert("ts".to_owned(), serde_json::Value::Number(ts.into()));
    m.insert(
        EMBEDDING_FIELD.to_owned(),
        serde_json::Value::Array(
            embedding
                .into_iter()
                .map(|f| {
                    serde_json::Number::from_f64(f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect(),
        ),
    );
    m
}

/// Ensure the tenant's memory collection exists, creating it (schema
/// persisted to disk + materialized into in-process state) sized for the
/// given embedding `dimension` if absent. Best-effort: returns the scoped
/// collection name on success, `None` on any failure. Idempotent — an
/// already-present collection is left alone.
async fn ensure_memory_collection(
    state: &AppState,
    tenant: &TenantId,
    dimension: u32,
) -> Option<String> {
    let scoped = tenant::scope(tenant, MEMORY_COLLECTION);
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&scoped) {
            return Some(scoped);
        }
    }

    let schema = memory_schema(&scoped, dimension).ok()?;

    // Persist the schema marker so the collection survives a restart,
    // mirroring `create_collection_impl`.
    let collection_dir = state.inner.data_dir.join(&scoped);
    std::fs::create_dir_all(&collection_dir).ok()?;
    let schema_json = serde_json::to_vec_pretty(&schema).ok()?;
    std::fs::write(collection_dir.join("collection.json"), &schema_json).ok()?;

    // Materialize into in-process state (segment managers + executor +
    // schema). Reuses the shared path so the collection behaves like any
    // other (RLS-/tenant-scoped, searchable, recoverable).
    crate::rest::collections::materialize_collection_locally(state, &schema, &collection_dir)
        .await
        .ok()?;
    Some(scoped)
}

/// Recall the top-`k` semantically relevant past turns for `agent_id`.
///
/// Embeds `query` off-lock, vector-searches the tenant's memory
/// collection filtered to this agent, and returns the hits. Returns an
/// empty vec (never errors) when memory is absent, no embedder is
/// configured, or the search fails — memory is always best-effort.
pub async fn retrieve_memory(
    state: &AppState,
    tenant: &TenantId,
    agent_id: &str,
    query: &str,
    k: usize,
) -> Vec<MemoryTurn> {
    if query.trim().is_empty() || k == 0 {
        return Vec::new();
    }
    let scoped = tenant::scope(tenant, MEMORY_COLLECTION);

    // Memory collection must already exist; never create it on a read.
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Vec::new();
        }
    }

    // Embed the query off any lock.
    let Some(client) = text_embedder(state).await else {
        return Vec::new();
    };
    let vector = match client.embed_text(query).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "agent memory: query embed failed; skipping recall");
            return Vec::new();
        }
    };

    // Filter to this agent (the `agent_id` field is filterable). The id is
    // a UUID/slug from our own scoping, so a simple equality string is safe.
    // NB: the filter grammar's equality operator is a single `=` (see
    // veda-query FilterParser); `==` fails to parse and would silently
    // turn every recall into an empty result.
    let filter = format!("{AGENT_ID_FIELD} = \"{agent_id}\"");
    let request = VectorSearchRequest {
        field: EMBEDDING_FIELD.to_owned(),
        query_vector: vector,
        k,
        metric: DistanceMetric::Cosine,
        filter: Some(filter),
        filter_expr: None,
        truncate_to: None,
        read_ts: None,
    };

    let executors = state.inner.query_executors.read().await;
    let Some(executor) = executors.get(&scoped) else {
        return Vec::new();
    };
    let response = match executor.search(&request) {
        Ok((response, _)) => response,
        Err(e) => {
            tracing::debug!(error = %e, "agent memory: recall search failed; skipping");
            return Vec::new();
        }
    };
    drop(executors);

    // Hydrate stored fields via an O(1)-per-hit record lookup, mirroring
    // the `search` builtin.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&scoped);

    let mut out = Vec::new();
    for r in response.results {
        let Some(shard_map) = shard_map else { break };
        let Some(rec) = shard_map
            .values()
            .find_map(|sm| sm.lookup_record_at(r.id, now_secs))
        else {
            continue;
        };
        out.push(MemoryTurn {
            text: string_field(&rec, TEXT_FIELD),
            role: string_field(&rec, "role"),
            conversation_id: string_field(&rec, "conversation_id"),
            ts: int_field(&rec, "ts"),
            score: r.distance,
        });
    }
    out
}

fn string_field(rec: &Record, name: &str) -> String {
    match rec.get(name) {
        Some(veda_core::record::FieldValue::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn int_field(rec: &Record, name: &str) -> i64 {
    match rec.get(name) {
        Some(veda_core::record::FieldValue::Int64(v)) => *v,
        Some(veda_core::record::FieldValue::Int32(v)) => *v as i64,
        _ => 0,
    }
}

/// Format recalled turns into a bounded "Relevant earlier context:" block
/// for injection into the system prompt. Pure. Returns `None` when there
/// is nothing to inject; otherwise truncates the rendered block to
/// `max_chars` (on a char boundary).
pub fn format_memory_block(turns: &[MemoryTurn], max_chars: usize) -> Option<String> {
    if turns.is_empty() || max_chars == 0 {
        return None;
    }
    let mut block = String::from(
        "Relevant earlier context (recalled from this agent's long-term memory of past \
conversations; use only if helpful, do not assume it is current):\n",
    );
    for t in turns {
        let line = format!("- ({}) {}\n", t.role, t.text.trim());
        block.push_str(&line);
    }
    let block = block.trim_end().to_string();
    if block.chars().count() <= max_chars {
        return Some(block);
    }
    // Truncate to max_chars on a char boundary.
    let truncated: String = block.chars().take(max_chars).collect();
    Some(truncated)
}

/// Write one turn into the tenant's memory collection, auto-creating it on
/// first write. Best-effort and non-fatal: embeds off-lock, then inserts
/// the record (with the explicitly-computed vector) directly into the
/// segment manager — the same in-process insert path `upsert_record`
/// uses. Any failure is logged at debug and swallowed.
pub async fn write_turn(
    state: &AppState,
    tenant: &TenantId,
    agent_id: &str,
    conversation_id: &str,
    role: &str,
    text: &str,
    ts: i64,
) {
    if text.trim().is_empty() {
        return;
    }

    // Embed off any lock.
    let Some(client) = text_embedder(state).await else {
        return;
    };
    let embedding = match client.embed_text(text).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "agent memory: turn embed failed; skipping write");
            return;
        }
    };

    write_turn_embedded(state, tenant, agent_id, conversation_id, role, text, ts, embedding).await;
}

/// Write one already-embedded memory turn and make it **immediately
/// recallable**. Ensures the collection, parses the record, then inserts it
/// via the shared seal + executor-rebuild path so a subsequent vector recall
/// finds it (a bare `SegmentManager::insert` would leave it only in the
/// growing segment the executor never sees). Best-effort. The vector is
/// supplied explicitly so this is unit-testable without a live embedder.
async fn write_turn_embedded(
    state: &AppState,
    tenant: &TenantId,
    agent_id: &str,
    conversation_id: &str,
    role: &str,
    text: &str,
    ts: i64,
    embedding: Vec<f32>,
) {
    let dimension = embedding.len() as u32;

    let Some(scoped) = ensure_memory_collection(state, tenant, dimension).await else {
        tracing::debug!("agent memory: could not ensure memory collection; skipping write");
        return;
    };

    let schema = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(&scoped).cloned() {
            Some(s) => s,
            None => return,
        }
    };
    let raw = turn_record_map(agent_id, conversation_id, role, text, ts, embedding);
    let record = match crate::rest::records::parse_record_public(&raw, &schema) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "agent memory: record parse failed; skipping write");
            return;
        }
    };

    crate::agent::index::insert_and_index(state, &scoped, &schema, record, "agent memory").await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, text: &str) -> MemoryTurn {
        MemoryTurn {
            text: text.to_owned(),
            role: role.to_owned(),
            conversation_id: "c1".to_owned(),
            ts: 1,
            score: 0.1,
        }
    }

    #[test]
    fn format_block_none_when_empty() {
        assert!(format_memory_block(&[], 1500).is_none());
        assert!(format_memory_block(&[turn("user", "hi")], 0).is_none());
    }

    #[test]
    fn format_block_renders_role_and_text() {
        let turns = vec![turn("user", "what index does veda use?"), turn("assistant", "HNSW.")];
        let block = format_memory_block(&turns, 1500).expect("some block");
        assert!(block.starts_with("Relevant earlier context"));
        assert!(block.contains("(user) what index does veda use?"));
        assert!(block.contains("(assistant) HNSW."));
    }

    #[test]
    fn format_block_truncates_to_max_chars() {
        let turns = vec![turn("user", &"x".repeat(5000))];
        let block = format_memory_block(&turns, 100).expect("some block");
        assert_eq!(block.chars().count(), 100, "block must be truncated to max_chars");
    }

    #[test]
    fn should_use_memory_reflects_flag() {
        let mut a = crate::agent::model::default_agent();
        assert!(!should_use_memory(&a), "default agent has memory off");
        a.memory = true;
        assert!(should_use_memory(&a), "memory=true agent uses memory");
    }

    #[test]
    fn memory_turn_record_map_round_trips_through_schema() {
        // The insert record shape parses against the memory schema and
        // round-trips its fields (text/role/agent_id/conversation_id/ts +
        // the explicit embedding vector).
        let dim = 4u32;
        let schema = memory_schema("t1::_agent_memory", dim).expect("schema builds");
        let raw = turn_record_map("ag-1", "conv-7", "user", "remember HNSW", 12345, vec![0.1, 0.2, 0.3, 0.4]);
        let rec = crate::rest::records::parse_record_public(&raw, &schema).expect("record parses");

        assert_eq!(string_field(&rec, TEXT_FIELD), "remember HNSW");
        assert_eq!(string_field(&rec, "role"), "user");
        assert_eq!(string_field(&rec, AGENT_ID_FIELD), "ag-1");
        assert_eq!(string_field(&rec, "conversation_id"), "conv-7");
        assert_eq!(int_field(&rec, "ts"), 12345);
        match rec.get(EMBEDDING_FIELD) {
            Some(veda_core::record::FieldValue::Vector(v)) => assert_eq!(v.dimension(), 4),
            other => panic!("expected vector field, got {other:?}"),
        }
    }

    #[test]
    fn turn_record_map_with_empty_conversation_id_still_parses() {
        // The A2A / triage callers thread no conversation id (`None` →
        // empty string). The record must still parse — memory write is not
        // skipped for those callers.
        let dim = 4u32;
        let schema = memory_schema("t1::_agent_memory", dim).expect("schema builds");
        let raw = turn_record_map("ag-1", "", "user", "no conversation", 1, vec![0.1, 0.2, 0.3, 0.4]);
        let rec = crate::rest::records::parse_record_public(&raw, &schema).expect("record parses");
        assert_eq!(string_field(&rec, "conversation_id"), "");
    }

    // ── Best-effort / off-lock wiring against a bare AppState ────────────
    //
    // The test env has no live embedder registered, so both entry points
    // must degrade gracefully (never panic, never error): `retrieve_memory`
    // returns empty and `write_turn` is a no-op that does NOT create the
    // memory collection. This mirrors how the catalog/search tests handle
    // the absent embedder — the request SHAPE (schema + record map) is
    // exercised by the pure tests above; the live embed/search path needs a
    // running embedder and is validated against a deployment.

    #[tokio::test]
    async fn retrieve_returns_empty_without_embedder_or_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let turns = retrieve_memory(&state, &tenant, "ag-1", "anything", DEFAULT_RECALL_K).await;
        assert!(turns.is_empty(), "no embedder/collection → empty recall, never an error");
    }

    #[tokio::test]
    async fn write_is_noop_without_embedder() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        // Must not panic; best-effort no-op when nothing can embed.
        write_turn(&state, &tenant, "ag-1", "conv-1", "user", "remember me", 1).await;
        // No embedder → the collection is never created (write bails before
        // ensure_memory_collection).
        let scoped = tenant::scope(&tenant, MEMORY_COLLECTION);
        let schemas = state.inner.schemas.read().await;
        assert!(
            !schemas.contains_key(&scoped),
            "write_turn must not create the memory collection when it cannot embed",
        );
    }

    // Searchability regression: a written turn must be recallable by a
    // vector search for its own embedding immediately after write. Supplies
    // the vector via the `write_turn_embedded` seam (no live embedder).
    // Before the seal + executor-rebuild fix this returned 0 hits.
    #[tokio::test]
    async fn written_turn_is_immediately_searchable() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let agent_id = "ag-1";
        let embedding = vec![0.0f32, 1.0, 0.0, 0.0];

        write_turn_embedded(
            &state, &tenant, agent_id, "conv-1", "user", "remember HNSW", 1, embedding.clone(),
        )
        .await;

        let scoped = tenant::scope(&tenant, MEMORY_COLLECTION);
        assert!(state.inner.schemas.read().await.contains_key(&scoped));

        let request = VectorSearchRequest {
            field: EMBEDDING_FIELD.to_owned(),
            query_vector: embedding,
            k: DEFAULT_RECALL_K,
            metric: DistanceMetric::Cosine,
            filter: Some(format!("{AGENT_ID_FIELD} = \"{agent_id}\"")),
            filter_expr: None,
            truncate_to: None,
            read_ts: None,
        };
        let executors = state.inner.query_executors.read().await;
        let executor = executors.get(&scoped).expect("executor exists");
        let (response, _) = executor.search(&request).expect("search runs");
        assert_eq!(
            response.results.len(),
            1,
            "the written memory turn must be returned by a vector recall for its own embedding"
        );
    }

    #[tokio::test]
    async fn empty_query_and_empty_text_short_circuit() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        assert!(retrieve_memory(&state, &tenant, "ag-1", "   ", DEFAULT_RECALL_K).await.is_empty());
        // Empty text write is a no-op (returns before touching the embedder).
        write_turn(&state, &tenant, "ag-1", "c", "user", "   ", 1).await;
        let scoped = tenant::scope(&tenant, MEMORY_COLLECTION);
        assert!(!state.inner.schemas.read().await.contains_key(&scoped));
    }

    #[test]
    fn memory_schema_marks_agent_id_filterable_and_vector_dim() {
        let schema = memory_schema("t1::_agent_memory", 8).expect("schema builds");
        let agent_field = schema.get_field(AGENT_ID_FIELD).expect("agent_id field");
        assert!(agent_field.filterable(), "agent_id must be filterable for per-agent recall");
        let vec_field = schema.get_field(EMBEDDING_FIELD).expect("embedding field");
        match vec_field.field_type() {
            FieldType::Vector { dimension, auto_embed, .. } => {
                assert_eq!(*dimension, 8);
                assert!(auto_embed.is_some(), "embedding field must carry auto_embed config");
            }
            other => panic!("expected vector field type, got {other:?}"),
        }
    }
}
