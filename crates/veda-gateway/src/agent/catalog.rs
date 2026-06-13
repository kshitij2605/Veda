//! Shared tool catalog.
//!
//! A [`ToolCatalog`] holds the built-in tool specifications and routes
//! tool calls to their implementations. It is consumed by both the MCP
//! server ([`crate::rest::mcp`]) and — in a later task — the agent
//! executor, so the same tool logic and JSON-Schema definitions back
//! both surfaces.
//!
//! ## Surfaces a [`ToolSpec`] can render to
//!
//! - **MCP** `tools/list` entry: `{name, description, inputSchema}` via
//!   [`ToolSpec::to_mcp_entry`].
//! - **OpenAI** tool definition (`veda_core::llm_client::ToolDef`) via
//!   [`ToolSpec::to_openai_tooldef`] — used by the executor.
//!
//! ## Whitelisting
//!
//! [`ToolCatalog::specs_for`] selects a subset by id for an agent's
//! `tool_ids` whitelist. **An empty whitelist means "all builtins"** —
//! a freshly created agent with no explicit tool list can use everything.

use serde::{Deserialize, Serialize};
use veda_core::llm_client::ToolDef;

use crate::rest::state::AppState;

use super::tools::builtins;

/// A single tool's identity, human-readable docs, and JSON-Schema
/// description of its arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSpec {
    /// Stable identifier used for dispatch and in agent whitelists
    /// (e.g. `"search"`). For builtins this is also the MCP tool name.
    pub id: String,
    /// Display name. Equals `id` for builtins.
    pub name: String,
    /// Human-readable description shown to the model / MCP client.
    pub description: String,
    /// JSON Schema (`type: object`) describing the tool's arguments.
    pub input_schema: serde_json::Value,
    /// Coarse category, e.g. `"builtin"` or `"custom"`.
    pub kind: String,
}

impl ToolSpec {
    /// Render this spec as an MCP `tools/list` entry:
    /// `{ "name", "description", "inputSchema" }`.
    pub fn to_mcp_entry(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        })
    }

    /// Render this spec as an OpenAI-style tool definition for the
    /// chat-completions `tools` array. The function name is the tool
    /// `id` (what the executor dispatches on).
    pub fn to_openai_tooldef(&self) -> ToolDef {
        ToolDef::function(
            self.id.clone(),
            self.description.clone(),
            self.input_schema.clone(),
        )
    }
}

/// What a tool needs to execute: a handle to gateway state plus the
/// caller's tenant and (optionally) user identity for row-level
/// security. Borrow-friendly so callers thread their existing handles
/// without cloning the world.
pub struct ToolContext<'a> {
    pub state: &'a AppState,
    pub tenant: &'a crate::tenant::TenantId,
    /// Authenticated subject (`sub` claim) when RLS applies; `None`
    /// for unauthenticated / admin contexts.
    pub user: Option<&'a str>,
}

impl<'a> ToolContext<'a> {
    pub fn new(state: &'a AppState, tenant: &'a crate::tenant::TenantId) -> Self {
        Self {
            state,
            tenant,
            user: None,
        }
    }

    pub fn with_user(mut self, user: Option<&'a str>) -> Self {
        self.user = user;
        self
    }
}

/// Transport-agnostic category of a tool failure. Surfaces (MCP,
/// executor) map this to their own error representation — the catalog
/// never encodes protocol-specific codes itself.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// Caller supplied missing/ill-typed arguments.
    InvalidArgs,
    /// The requested collection/record does not exist (or is hidden).
    NotFound,
    /// An internal failure (embedding, search, insert, serialization…).
    Internal,
}

/// Structured result of a single tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolOutcome {
    /// Tool output payload (success) or an `{ "error": ... }` object
    /// (failure). Always JSON.
    pub content: serde_json::Value,
    /// True when the tool failed; `content` then carries the error.
    pub is_error: bool,
    /// Failure category (`None` on success). Lets transports choose a
    /// suitable error code without the catalog knowing about them.
    pub error_kind: Option<ToolErrorKind>,
}

impl ToolOutcome {
    pub fn ok(content: serde_json::Value) -> Self {
        Self {
            content,
            is_error: false,
            error_kind: None,
        }
    }

    fn error_with(message: impl Into<String>, kind: ToolErrorKind) -> Self {
        Self {
            content: serde_json::json!({ "error": message.into() }),
            is_error: true,
            error_kind: Some(kind),
        }
    }

    /// Missing or ill-typed arguments (e.g. `"query is required"`).
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::error_with(message, ToolErrorKind::InvalidArgs)
    }

    /// Requested collection/record not found (or hidden by RLS).
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::error_with(message, ToolErrorKind::NotFound)
    }

    /// Internal failure (embedding/search/insert/serialization).
    pub fn internal(message: impl Into<String>) -> Self {
        Self::error_with(message, ToolErrorKind::Internal)
    }

    /// Back-compat alias for [`ToolOutcome::internal`].
    pub fn error(message: impl Into<String>) -> Self {
        Self::internal(message)
    }
}

/// Catalog of the built-in tools. Stateless — the specs are static and
/// `dispatch` borrows everything it needs from the [`ToolContext`].
pub struct ToolCatalog {
    builtins: Vec<ToolSpec>,
}

impl Default for ToolCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCatalog {
    pub fn new() -> Self {
        Self {
            builtins: builtins::builtin_specs(),
        }
    }

    /// All built-in tool specs (currently 6).
    pub fn builtin_specs(&self) -> &[ToolSpec] {
        &self.builtins
    }

    /// Subset of builtins selected by id. An **empty** whitelist
    /// returns *all* builtins (the documented "no restriction" case);
    /// unknown ids are silently skipped.
    pub fn specs_for(&self, whitelist: &[String]) -> Vec<ToolSpec> {
        if whitelist.is_empty() {
            return self.builtins.clone();
        }
        self.builtins
            .iter()
            .filter(|s| whitelist.iter().any(|w| w == &s.id))
            .cloned()
            .collect()
    }

    /// Route a tool call to its built-in implementation by id. Unknown
    /// ids return an error [`ToolOutcome`] rather than panicking.
    pub async fn dispatch(
        &self,
        ctx: &ToolContext<'_>,
        tool_id: &str,
        args: serde_json::Value,
    ) -> ToolOutcome {
        match tool_id {
            "list_collections" => builtins::list_collections(ctx, args).await,
            "describe_collection" => builtins::describe_collection(ctx, args).await,
            "search" => builtins::search(ctx, args).await,
            "upsert_record" => builtins::upsert_record(ctx, args).await,
            "get_record" => builtins::get_record(ctx, args).await,
            "route" => builtins::route(ctx, args).await,
            other => ToolOutcome::error(format!("unknown tool: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::{DistanceMetric, FieldType};
    use veda_core::vector::DenseVector;

    fn text_schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::vector("embedding", 4, DistanceMetric::Cosine))
            .add_field(
                FieldDefinition::new("title", FieldType::String).with_filterable(true),
            )
            .shard_count(1)
            .build()
            .unwrap()
    }

    /// Build an AppState with one seeded text collection (scoped to the
    /// default tenant) holding a single record, and return the record's id.
    async fn seeded_state(
        tmp: &std::path::Path,
        user_collection: &str,
    ) -> (AppState, crate::tenant::TenantId, uuid::Uuid) {
        let state = AppState::new(tmp);
        let tenant = crate::tenant::TenantId::default_tenant();
        let scoped = crate::tenant::scope(&tenant, user_collection);
        let schema = text_schema(&scoped);
        let col_dir = tmp.join(&scoped);
        std::fs::create_dir_all(&col_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(&state, &schema, &col_dir)
            .await
            .expect("materialize");

        // Insert one record directly into the growing segment.
        let mut rec = Record::new();
        rec.set("embedding", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0, 0.0, 0.0])));
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
        (state, tenant, id)
    }

    #[test]
    fn builtin_specs_count_is_six() {
        let cat = ToolCatalog::new();
        assert_eq!(cat.builtin_specs().len(), 6);
        let ids: Vec<&str> = cat.builtin_specs().iter().map(|s| s.id.as_str()).collect();
        for expected in [
            "list_collections",
            "describe_collection",
            "search",
            "upsert_record",
            "get_record",
            "route",
        ] {
            assert!(ids.contains(&expected), "missing builtin {expected}");
        }
    }

    #[test]
    fn specs_for_empty_whitelist_returns_all() {
        let cat = ToolCatalog::new();
        assert_eq!(cat.specs_for(&[]).len(), 6);
    }

    #[test]
    fn specs_for_subset_selects_by_id() {
        let cat = ToolCatalog::new();
        let subset = cat.specs_for(&["search".to_owned(), "route".to_owned(), "nope".to_owned()]);
        let ids: Vec<&str> = subset.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(subset.len(), 2);
        assert!(ids.contains(&"search"));
        assert!(ids.contains(&"route"));
    }

    #[test]
    fn to_openai_tooldef_renders_name_desc_params() {
        let cat = ToolCatalog::new();
        let spec = cat
            .builtin_specs()
            .iter()
            .find(|s| s.id == "describe_collection")
            .unwrap();
        let td = spec.to_openai_tooldef();
        assert_eq!(td.kind, "function");
        assert_eq!(td.function.name, "describe_collection");
        assert_eq!(td.function.description, spec.description);
        assert_eq!(td.function.parameters, spec.input_schema);
    }

    #[test]
    fn to_mcp_entry_shape() {
        let cat = ToolCatalog::new();
        let spec = cat.builtin_specs().iter().find(|s| s.id == "search").unwrap();
        let entry = spec.to_mcp_entry();
        assert_eq!(entry["name"], "search");
        assert_eq!(entry["description"], serde_json::json!(spec.description));
        assert_eq!(entry["inputSchema"], spec.input_schema);
    }

    #[tokio::test]
    async fn dispatch_unknown_id_is_error_not_panic() {
        let cat = ToolCatalog::new();
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat.dispatch(&ctx, "does_not_exist", serde_json::json!({})).await;
        assert!(out.is_error);
        assert!(out.content["error"].as_str().unwrap().contains("unknown tool"));
    }

    #[tokio::test]
    async fn dispatch_list_collections() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat.dispatch(&ctx, "list_collections", serde_json::json!({})).await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        let cols = out.content["collections"].as_array().unwrap();
        assert!(cols.iter().any(|c| c.as_str() == Some("docs")));
        assert_eq!(out.content["count"], 1);
    }

    #[tokio::test]
    async fn dispatch_describe_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(&ctx, "describe_collection", serde_json::json!({ "name": "docs" }))
            .await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        assert_eq!(out.content["name"], "docs");
        assert_eq!(out.content["record_count"], 1);
        let fields = out.content["fields"].as_array().unwrap();
        assert!(fields.iter().any(|f| f["name"] == "embedding" && f["type"] == "vector"));
    }

    #[tokio::test]
    async fn dispatch_get_record() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &ctx,
                "get_record",
                serde_json::json!({ "collection": "docs", "id": id.to_string() }),
            )
            .await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        assert_eq!(out.content["id"], id.to_string());
        assert_eq!(out.content["fields"]["title"], "hello world");
    }

    #[tokio::test]
    async fn dispatch_get_record_missing_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let missing = uuid::Uuid::new_v4().to_string();
        let out = cat
            .dispatch(
                &ctx,
                "get_record",
                serde_json::json!({ "collection": "docs", "id": missing }),
            )
            .await;
        assert!(out.is_error);
    }

    fn rls_schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::vector("embedding", 4, DistanceMetric::Cosine))
            .add_field(FieldDefinition::new("owner", FieldType::String).with_filterable(true))
            .rls_owner_field(Some("owner".to_owned()))
            .shard_count(1)
            .build()
            .unwrap()
    }

    /// Seed an RLS-enabled collection with one record per owner; return
    /// the (alice_record_id, bob_record_id).
    async fn seeded_rls_state(
        tmp: &std::path::Path,
        user_collection: &str,
    ) -> (AppState, crate::tenant::TenantId, uuid::Uuid, uuid::Uuid) {
        let state = AppState::new(tmp);
        let tenant = crate::tenant::TenantId::default_tenant();
        let scoped = crate::tenant::scope(&tenant, user_collection);
        let schema = rls_schema(&scoped);
        let col_dir = tmp.join(&scoped);
        std::fs::create_dir_all(&col_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(&state, &schema, &col_dir)
            .await
            .expect("materialize");

        let mut ids = Vec::new();
        for owner in ["alice", "bob"] {
            let mut rec = Record::new();
            rec.set("embedding", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0, 0.0, 0.0])));
            rec.set("owner", FieldValue::String(owner.to_owned()));
            let id = rec.id();
            ids.push(id);
            let mut managers = state.inner.segment_managers.write().await;
            let shard_map = managers.get_mut(&scoped).unwrap();
            let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
                &id,
                schema.shard_count(),
            );
            shard_map.get_mut(&shard_id).unwrap().insert(rec).unwrap();
        }
        (state, tenant, ids[0], ids[1])
    }

    #[tokio::test]
    async fn get_record_enforces_rls_owner() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, alice_id, bob_id) = seeded_rls_state(tmp.path(), "secrets").await;
        let cat = ToolCatalog::new();

        // Scoped user "alice" sees her own record...
        let alice_ctx = ToolContext::new(&state, &tenant).with_user(Some("alice"));
        let out = cat
            .dispatch(
                &alice_ctx,
                "get_record",
                serde_json::json!({ "collection": "secrets", "id": alice_id.to_string() }),
            )
            .await;
        assert!(!out.is_error, "alice own record: {:?}", out.content);
        assert_eq!(out.content["fields"]["owner"], "alice");

        // ...but bob's record is hidden as not_found (no existence leak).
        let out = cat
            .dispatch(
                &alice_ctx,
                "get_record",
                serde_json::json!({ "collection": "secrets", "id": bob_id.to_string() }),
            )
            .await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::NotFound));

        // user=None (admin/system) bypasses RLS and can read either record.
        let admin_ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &admin_ctx,
                "get_record",
                serde_json::json!({ "collection": "secrets", "id": bob_id.to_string() }),
            )
            .await;
        assert!(!out.is_error, "admin bob record: {:?}", out.content);
        assert_eq!(out.content["fields"]["owner"], "bob");
    }

    #[tokio::test]
    async fn dispatch_route() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(&ctx, "route", serde_json::json!({ "query": "what is HNSW indexing?" }))
            .await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        assert!(out.content["strategy"].is_string());
        assert!(out.content["suggested_k"].is_number());
        assert!(out.content["reasoning"].is_string());
    }
}
