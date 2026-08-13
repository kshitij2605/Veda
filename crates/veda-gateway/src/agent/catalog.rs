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

/// Builtins that are **excluded** from the "empty whitelist == all"
/// expansion in [`ToolCatalog::specs_for`]. They are still real builtins
/// (present in [`ToolCatalog::builtin_specs`] and dispatchable), but an
/// agent must opt in to them explicitly by listing the id in its
/// `tool_ids` whitelist. The web tools reach the public internet, so
/// access is granted per-agent (see the built-in `veda-web` agent)
/// rather than handed to every default agent.
pub const OPT_IN_BUILTINS: &[&str] = &["web_search", "fetch_url"];

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
    /// Whether the tool only reads state (no mutations). Drives the MCP
    /// `readOnlyHint` annotation. Defaults to `false`.
    #[serde(default)]
    pub read_only: bool,
    /// Whether the tool may perform destructive/irreversible updates.
    /// Drives the MCP `destructiveHint` annotation. Defaults to `false`.
    #[serde(default)]
    pub destructive: bool,
    /// Short human-readable title for the tool (MCP `title` annotation).
    /// Falls back to `name` when absent.
    #[serde(default)]
    pub title: Option<String>,
}

impl ToolSpec {
    /// Render this spec as an MCP `tools/list` entry:
    /// `{ "name", "description", "inputSchema", "annotations" }`. The
    /// `annotations` object carries the MCP behavioural hints
    /// (`title`, `readOnlyHint`, `destructiveHint`).
    pub fn to_mcp_entry(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "title": self.title.clone().unwrap_or_else(|| self.name.clone()),
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
            },
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

    /// All built-in tool specs (currently 13).
    pub fn builtin_specs(&self) -> &[ToolSpec] {
        &self.builtins
    }

    /// Subset of builtins selected by id. An **empty** whitelist
    /// returns *all* builtins **except** the opt-in ones
    /// ([`OPT_IN_BUILTINS`], e.g. the web tools) — those must be named
    /// explicitly. Unknown ids are silently skipped.
    pub fn specs_for(&self, whitelist: &[String]) -> Vec<ToolSpec> {
        if whitelist.is_empty() {
            return self
                .builtins
                .iter()
                .filter(|s| !OPT_IN_BUILTINS.contains(&s.id.as_str()))
                .cloned()
                .collect();
        }
        self.builtins
            .iter()
            .filter(|s| whitelist.iter().any(|w| w == &s.id))
            .cloned()
            .collect()
    }

    /// Route a tool call to its built-in implementation by id. Unknown
    /// ids return an error [`ToolOutcome`] rather than panicking.
    async fn dispatch_inner(
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
            "aggregate" => builtins::aggregate(ctx, args).await,
            "filter_query" => builtins::filter_query(ctx, args).await,
            "get_insights" => super::tools::ops::get_insights(ctx, args).await,
            "get_logs" => super::tools::ops::get_logs(ctx, args).await,
            "get_resources" => super::tools::ops::get_resources(ctx, args).await,
            "get_slo_status" => super::tools::ops::get_slo_status(ctx, args).await,
            "correlate_incident" => super::tools::ops::correlate_incident(ctx, args).await,
            "web_search" => super::tools::web::web_search(ctx, args).await,
            "fetch_url" => super::tools::web::fetch_url(ctx, args).await,
            // A tool id that doesn't exist is the caller naming something
            // that isn't there, not an internal failure — `execute_tool`
            // maps the kind straight to a status, and the documented
            // contract for /v1/agent-tools/{id}/execute is
            // invalid->400, not-found->404, internal->500. `error()`
            // (an alias for `internal`) made it a 500. The message is
            // unchanged: MCP matches on "unknown tool" to return the
            // JSON-RPC "method not found" code.
            other => ToolOutcome::not_found(format!("unknown tool: {other}")),
        }
    }

    /// Dispatch a tool, then apply retrieval guardrails to its output.
    ///
    /// This is the single choke point every LLM-facing transport shares —
    /// `executor.rs` (converse / converse-stream / A2A), `rest/mcp.rs`
    /// (tools/call) and `rest/agents.rs` (`/agent-tools/{id}/execute`). Doing
    /// enforcement here rather than at each call site makes it structurally
    /// impossible to add a transport that bypasses it.
    ///
    /// Any tool whose success payload carries a `hits` or `records` array
    /// (the two shapes `search`/`filter_query` use) is screened; the
    /// convention is documented so a future content-returning tool inherits
    /// enforcement for free.
    pub async fn dispatch(
        &self,
        ctx: &ToolContext<'_>,
        tool_id: &str,
        args: serde_json::Value,
    ) -> ToolOutcome {
        let collection = args
            .get("collection")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let outcome = self.dispatch_inner(ctx, tool_id, args).await;
        if outcome.is_error {
            return outcome;
        }
        let vector_field = resolve_vector_field(ctx, collection.as_deref()).await;
        apply_retrieval_guard(outcome, vector_field.as_deref())
    }
}

/// Resolve the name of the collection's first vector field for a tool call's
/// `collection` argument, so [`apply_retrieval_guard`] can pass it into
/// `temporal_check`'s conflict detection.
///
/// Follows the same scope + alias-resolve idiom as the read paths (see
/// `rest::tuning::resolve_search_fields`) rather than trusting the caller's
/// raw name — this project has repeatedly shipped bugs where a handler read
/// the unresolved name and silently operated on an abandoned base
/// collection after a reindex swapped the alias.
///
/// Returns `None` on any failure (no `collection` arg, unknown collection,
/// no vector field): this must never error and never block the tool result,
/// it only means conflict detection is skipped while staleness/expiry still
/// run.
async fn resolve_vector_field(ctx: &ToolContext<'_>, collection: Option<&str>) -> Option<String> {
    let coll = collection?;
    let scoped = crate::tenant::scope(ctx.tenant, coll);
    let scoped = ctx.state.inner.aliases.resolve(&scoped).await;
    let schemas = ctx.state.inner.schemas.read().await;
    schemas
        .get(&scoped)
        .and_then(|schema| schema.vector_fields().first().map(|f| f.name().to_owned()))
}

/// Drop flagged hits from a tool payload and attach the report.
///
/// Returns the outcome unchanged when the payload has no `hits`/`records`
/// array — most tools (`upsert_record`, `get_resources`, …) are unaffected.
/// Checks `hits` (used by `search`) and `records` (used by `filter_query`)
/// because both are content-returning, LLM-facing arrays of retrieved
/// chunks; the kept array is written back under the same key it was found
/// under so each tool's existing response shape is preserved.
///
/// Dropping is on by default and is **not** gated on the agent's `guardrails`
/// flag — a per-agent opt-in would ship the default agent unprotected. The
/// one override is the operator kill switch `VEDA_RETRIEVAL_GUARD=off`
/// ([`crate::agent::retrieval_guard::enforcement_enabled`]), which degrades this to
/// annotate-only: hits are still classified and still reported, none are
/// removed.
fn apply_retrieval_guard(mut outcome: ToolOutcome, vector_field: Option<&str>) -> ToolOutcome {
    use crate::agent::retrieval_guard::{enforce_retrieved, enforcement_enabled, temporal_check};

    let Some(obj) = outcome.content.as_object_mut() else {
        return outcome;
    };
    let key = if obj.contains_key("hits") {
        "hits"
    } else if obj.contains_key("records") {
        "records"
    } else {
        return outcome;
    };
    let Some(hits) = obj.get_mut(key).and_then(|h| h.as_array_mut()) else {
        return outcome;
    };

    let taken = std::mem::take(hits);
    let (kept, mut report) = enforce_retrieved(taken, enforcement_enabled());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    report.temporal = temporal_check(&kept, now, vector_field);

    obj.insert("count".into(), serde_json::json!(kept.len()));
    obj.insert(key.to_owned(), serde_json::Value::Array(kept));
    obj.insert(
        "retrieval_guard".into(),
        serde_json::to_value(&report).unwrap_or(serde_json::Value::Null),
    );
    outcome
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
            .add_field(FieldDefinition::vector(
                "embedding",
                4,
                DistanceMetric::Cosine,
            ))
            .add_field(FieldDefinition::new("title", FieldType::String).with_filterable(true))
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
        (state, tenant, id)
    }

    /// Like [`seeded_state`], but the inserted record already carries a
    /// flagged guardrail verdict in the given `category`. The fields are set
    /// directly on the `Record` (mirroring `stamp_guardrail_fields`) and,
    /// deliberately, are NOT added to `text_schema`'s field list — on every
    /// real collection these are RESERVED fields, never schema-declared.
    async fn seeded_flagged_state_with_category(
        tmp: &std::path::Path,
        user_collection: &str,
        category: &str,
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

        let mut rec = Record::new();
        rec.set(
            "embedding",
            FieldValue::Vector(DenseVector::new(vec![1.0, 0.0, 0.0, 0.0])),
        );
        rec.set("title", FieldValue::String("flagged content".to_owned()));
        rec.set(
            veda_core::record::GUARDRAIL_FLAGGED_FIELD,
            FieldValue::Bool(true),
        );
        rec.set(
            veda_core::record::GUARDRAIL_SCORE_FIELD,
            FieldValue::Float64(0.9),
        );
        rec.set(
            veda_core::record::GUARDRAIL_CATEGORY_FIELD,
            FieldValue::String(category.to_owned()),
        );
        rec.set(
            veda_core::record::INDEXED_AT_FIELD,
            FieldValue::Int64(1_786_000_000),
        );
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
    fn builtin_specs_count_is_fifteen() {
        let cat = ToolCatalog::new();
        assert_eq!(cat.builtin_specs().len(), 15);
        let ids: Vec<&str> = cat.builtin_specs().iter().map(|s| s.id.as_str()).collect();
        for expected in [
            "list_collections",
            "describe_collection",
            "search",
            "upsert_record",
            "get_record",
            "route",
            "aggregate",
            "filter_query",
            "get_insights",
            "get_logs",
            "get_resources",
            "get_slo_status",
            "correlate_incident",
            "web_search",
            "fetch_url",
        ] {
            assert!(ids.contains(&expected), "missing builtin {expected}");
        }
    }

    #[test]
    fn specs_for_empty_whitelist_returns_all_non_opt_in() {
        let cat = ToolCatalog::new();
        // Empty whitelist == all builtins EXCEPT the opt-in web tools.
        let specs = cat.specs_for(&[]);
        assert_eq!(specs.len(), 13);
        let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        assert!(!ids.contains(&"web_search"), "web_search must be opt-in");
        assert!(!ids.contains(&"fetch_url"), "fetch_url must be opt-in");
    }

    #[test]
    fn specs_for_can_opt_in_to_web_tools() {
        let cat = ToolCatalog::new();
        let subset = cat.specs_for(&[
            "search".to_owned(),
            "web_search".to_owned(),
            "fetch_url".to_owned(),
        ]);
        let ids: Vec<&str> = subset.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(subset.len(), 3);
        assert!(ids.contains(&"web_search"));
        assert!(ids.contains(&"fetch_url"));
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
        let spec = cat
            .builtin_specs()
            .iter()
            .find(|s| s.id == "search")
            .unwrap();
        let entry = spec.to_mcp_entry();
        assert_eq!(entry["name"], "search");
        assert_eq!(entry["description"], serde_json::json!(spec.description));
        assert_eq!(entry["inputSchema"], spec.input_schema);
        // Annotations carry the MCP behavioural hints. `search` is a
        // read-only, non-destructive tool with a human title.
        let ann = &entry["annotations"];
        assert!(ann.is_object(), "annotations object present");
        assert_eq!(ann["readOnlyHint"], true);
        assert_eq!(ann["destructiveHint"], false);
        assert!(ann["title"].as_str().is_some());
    }

    #[test]
    fn to_mcp_entry_annotations_reflect_mutation() {
        let cat = ToolCatalog::new();
        // upsert_record is the one mutating/destructive builtin.
        let spec = cat
            .builtin_specs()
            .iter()
            .find(|s| s.id == "upsert_record")
            .unwrap();
        let ann = spec.to_mcp_entry()["annotations"].clone();
        assert_eq!(ann["readOnlyHint"], false);
        assert_eq!(ann["destructiveHint"], true);
    }

    #[test]
    fn read_tools_are_read_only_and_nondestructive() {
        let cat = ToolCatalog::new();
        for id in [
            "list_collections",
            "describe_collection",
            "search",
            "get_record",
            "route",
            "aggregate",
            "filter_query",
            "get_insights",
            "get_logs",
            "get_resources",
            "get_slo_status",
            "web_search",
            "fetch_url",
        ] {
            let spec = cat.builtin_specs().iter().find(|s| s.id == id).unwrap();
            assert!(spec.read_only, "{id} should be read_only");
            assert!(!spec.destructive, "{id} should be non-destructive");
            assert!(spec.title.is_some(), "{id} should have a title");
        }
    }

    #[tokio::test]
    async fn dispatch_unknown_id_is_error_not_panic() {
        let cat = ToolCatalog::new();
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(&ctx, "does_not_exist", serde_json::json!({}))
            .await;
        assert!(out.is_error);
        assert!(
            out.content["error"]
                .as_str()
                .unwrap()
                .contains("unknown tool")
        );
    }

    #[tokio::test]
    async fn dispatch_web_search_missing_query_is_invalid_args() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat_dispatch(&ctx, "web_search", serde_json::json!({})).await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::InvalidArgs));
    }

    #[tokio::test]
    async fn dispatch_fetch_url_missing_url_is_invalid_args() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat_dispatch(&ctx, "fetch_url", serde_json::json!({})).await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::InvalidArgs));
    }

    /// Helper: dispatch against a fresh catalog.
    async fn cat_dispatch(
        ctx: &ToolContext<'_>,
        tool_id: &str,
        args: serde_json::Value,
    ) -> ToolOutcome {
        ToolCatalog::new().dispatch(ctx, tool_id, args).await
    }

    #[tokio::test]
    async fn dispatch_list_collections() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(&ctx, "list_collections", serde_json::json!({}))
            .await;
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
            .dispatch(
                &ctx,
                "describe_collection",
                serde_json::json!({ "name": "docs" }),
            )
            .await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        assert_eq!(out.content["name"], "docs");
        assert_eq!(out.content["record_count"], 1);
        let fields = out.content["fields"].as_array().unwrap();
        assert!(
            fields
                .iter()
                .any(|f| f["name"] == "embedding" && f["type"] == "vector")
        );
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

    /// Critical gap this closes: `get_record` returns a bare `{id, fields}`
    /// object, which carries neither `hits` nor `records`, so the
    /// dispatch-level `apply_retrieval_guard` (which only screens those two
    /// array shapes) passes it through untouched. Without a check inside
    /// `get_record` itself, a model that saw a chunk's id reported in a
    /// `search`/`filter_query` response's `retrieval_guard.dropped_chunks`
    /// could turn around and fetch the withheld content verbatim via
    /// `get_record` — the array-shaped guard would be enforcement theater.
    #[tokio::test]
    async fn dispatch_get_record_withholds_flagged_content() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, id) =
            seeded_flagged_state_with_category(tmp.path(), "docs", "indirect_injection").await;
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
        assert_eq!(out.content["id"], id.to_string(), "{:?}", out.content);
        assert_eq!(out.content["withheld"], true, "{:?}", out.content);
        assert!(
            out.content.get("fields").is_none(),
            "flagged content must not leak through get_record: {:?}",
            out.content
        );
        assert_eq!(
            out.content["retrieval_guard"]["dropped"], 1,
            "{:?}",
            out.content
        );
    }

    /// 2026-08-13 ruling: PII is exempt from dropping, and `get_record` must
    /// apply the SAME category policy as the array-shaped guard
    /// (`apply_retrieval_guard`/`enforce_retrieved`) — a PII-flagged record
    /// fetched directly by id must come back with its content intact, not
    /// withheld.
    #[tokio::test]
    async fn dispatch_get_record_does_not_withhold_pii() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, id) =
            seeded_flagged_state_with_category(tmp.path(), "docs", "pii").await;
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
        assert_eq!(out.content["id"], id.to_string(), "{:?}", out.content);
        assert!(
            out.content.get("withheld").is_none(),
            "PII must not be withheld: {:?}",
            out.content
        );
        assert_eq!(
            out.content["fields"]["title"], "flagged content",
            "{:?}",
            out.content
        );
    }

    fn rls_schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::vector(
                "embedding",
                4,
                DistanceMetric::Cosine,
            ))
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
            rec.set(
                "embedding",
                FieldValue::Vector(DenseVector::new(vec![1.0, 0.0, 0.0, 0.0])),
            );
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
            .dispatch(
                &ctx,
                "route",
                serde_json::json!({ "query": "what is HNSW indexing?" }),
            )
            .await;
        assert!(!out.is_error, "outcome: {:?}", out.content);
        assert!(out.content["strategy"].is_string());
        assert!(out.content["suggested_k"].is_number());
        assert!(out.content["reasoning"].is_string());
    }

    // ── aggregate tool ──────────────────────────────────────────────────────

    fn agg_schema(name: &str) -> CollectionSchema {
        CollectionSchema::builder(name)
            .add_field(FieldDefinition::new("category", FieldType::String).with_filterable(true))
            .add_field(FieldDefinition::new("price", FieldType::Float64).with_filterable(true))
            .shard_count(1)
            .build()
            .unwrap()
    }

    /// Seed: electronics(100,200), books(10,20,30) — mirrors the REST
    /// aggregate integration test fixture.
    async fn seeded_agg_state(tmp: &std::path::Path) -> (AppState, crate::tenant::TenantId) {
        let state = AppState::new(tmp);
        let tenant = crate::tenant::TenantId::default_tenant();
        let scoped = crate::tenant::scope(&tenant, "agg");
        let schema = agg_schema(&scoped);
        let col_dir = tmp.join(&scoped);
        std::fs::create_dir_all(&col_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(&state, &schema, &col_dir)
            .await
            .expect("materialize");

        let rows = [
            ("electronics", 100.0),
            ("electronics", 200.0),
            ("books", 10.0),
            ("books", 20.0),
            ("books", 30.0),
        ];
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(&scoped).unwrap();
        for (cat, price) in rows {
            let mut rec = Record::new();
            rec.set("category", FieldValue::String(cat.to_owned()));
            rec.set("price", FieldValue::Float64(price));
            let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
                &rec.id(),
                schema.shard_count(),
            );
            shard_map.get_mut(&shard_id).unwrap().insert(rec).unwrap();
        }
        drop(managers);
        (state, tenant)
    }

    #[tokio::test]
    async fn dispatch_aggregate_count_and_sum() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_agg_state(tmp.path()).await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);

        // count, no group -> single bucket of 5.
        let out = cat
            .dispatch(
                &ctx,
                "aggregate",
                serde_json::json!({
                    "collection": "agg", "field": "price", "metric": "count"
                }),
            )
            .await;
        assert!(!out.is_error, "count: {:?}", out.content);
        let groups = out.content["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0]["value"], 5.0);

        // sum, no group -> 360.
        let out = cat
            .dispatch(
                &ctx,
                "aggregate",
                serde_json::json!({
                    "collection": "agg", "field": "price", "metric": "sum"
                }),
            )
            .await;
        assert!(!out.is_error, "sum: {:?}", out.content);
        assert_eq!(out.content["groups"][0]["value"], 360.0);
    }

    #[tokio::test]
    async fn dispatch_aggregate_avg_group_by() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_agg_state(tmp.path()).await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);

        let out = cat
            .dispatch(
                &ctx,
                "aggregate",
                serde_json::json!({
                    "collection": "agg", "field": "price", "metric": "avg", "group_by": "category"
                }),
            )
            .await;
        assert!(!out.is_error, "avg group_by: {:?}", out.content);
        let groups = out.content["groups"].as_array().unwrap();
        // Sorted by key: books then electronics.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0]["key"], "books");
        assert_eq!(groups[0]["value"], 20.0);
        assert_eq!(groups[1]["key"], "electronics");
        assert_eq!(groups[1]["value"], 150.0);
    }

    #[tokio::test]
    async fn dispatch_aggregate_bad_metric_is_invalid_args() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_agg_state(tmp.path()).await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &ctx,
                "aggregate",
                serde_json::json!({
                    "collection": "agg", "field": "price", "metric": "median"
                }),
            )
            .await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::InvalidArgs));
    }

    // ── filter_query tool ───────────────────────────────────────────────────

    #[tokio::test]
    async fn dispatch_filter_query_returns_subset() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_agg_state(tmp.path()).await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &ctx,
                "filter_query",
                serde_json::json!({
                    "collection": "agg", "filter": "category = 'books'"
                }),
            )
            .await;
        assert!(!out.is_error, "filter: {:?}", out.content);
        assert_eq!(out.content["count"], 3);
        let recs = out.content["records"].as_array().unwrap();
        assert!(recs.iter().all(|r| r["fields"]["category"] == "books"));
    }

    #[tokio::test]
    async fn dispatch_filter_query_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant) = seeded_agg_state(tmp.path()).await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &ctx,
                "filter_query",
                serde_json::json!({
                    "collection": "agg", "filter": "price >= 0", "limit": 2
                }),
            )
            .await;
        assert!(!out.is_error, "{:?}", out.content);
        assert_eq!(out.content["count"], 2);
    }

    #[tokio::test]
    async fn filter_query_enforces_rls_owner() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _alice_id, _bob_id) = seeded_rls_state(tmp.path(), "secrets").await;
        let cat = ToolCatalog::new();

        // alice sees only her own row.
        let alice_ctx = ToolContext::new(&state, &tenant).with_user(Some("alice"));
        let out = cat
            .dispatch(
                &alice_ctx,
                "filter_query",
                serde_json::json!({
                    "collection": "secrets", "filter": "owner != ''"
                }),
            )
            .await;
        assert!(!out.is_error, "alice: {:?}", out.content);
        assert_eq!(out.content["count"], 1);
        assert_eq!(out.content["records"][0]["fields"]["owner"], "alice");

        // admin (user=None) sees both.
        let admin_ctx = ToolContext::new(&state, &tenant);
        let out = cat
            .dispatch(
                &admin_ctx,
                "filter_query",
                serde_json::json!({
                    "collection": "secrets", "filter": "owner != ''"
                }),
            )
            .await;
        assert!(!out.is_error, "admin: {:?}", out.content);
        assert_eq!(out.content["count"], 2);
    }

    // ── multimodal search helpers ───────────────────────────────────────────

    #[tokio::test]
    async fn search_requires_exactly_one_query() {
        let tmp = tempfile::tempdir().unwrap();
        let (state, tenant, _id) = seeded_state(tmp.path(), "docs").await;
        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);

        // Neither.
        let out = cat
            .dispatch(&ctx, "search", serde_json::json!({ "collection": "docs" }))
            .await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::InvalidArgs));

        // Both.
        let out = cat
            .dispatch(
                &ctx,
                "search",
                serde_json::json!({
                    "collection": "docs", "query": "hi", "query_image_b64": "AAAA"
                }),
            )
            .await;
        assert!(out.is_error);
        assert_eq!(out.error_kind, Some(ToolErrorKind::InvalidArgs));
    }

    #[test]
    fn search_spec_is_multimodal() {
        let cat = ToolCatalog::new();
        let spec = cat
            .builtin_specs()
            .iter()
            .find(|s| s.id == "search")
            .unwrap();
        let props = &spec.input_schema["properties"];
        assert!(props["query"].is_object());
        assert!(props["query_image_b64"].is_object());
        assert!(props["mode"].is_object());
        // collection is the only hard requirement now (query XOR image is
        // enforced in the impl, not the schema).
        let required = spec.input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "collection");
    }

    #[test]
    fn build_query_body_maps_dense_modes() {
        use crate::agent::tools::builtins::{SearchMode, build_query_body};
        for mode in [SearchMode::Auto, SearchMode::Dense] {
            let qb = build_query_body(mode, "embedding", vec![1.0, 0.0], false).unwrap();
            let v = qb.vector.expect("dense leg set");
            assert_eq!(v.field, "embedding");
            assert_eq!(v.values, vec![1.0, 0.0]);
            assert!(qb.sparse_vector.is_none());
            assert!(qb.hybrid.is_none());
            assert!(qb.colpali.is_none());
        }
    }

    #[test]
    fn build_query_body_sparse_unsupported_message() {
        use crate::agent::tools::builtins::{SearchMode, build_query_body};
        // No sparse field on the collection -> clear "requires" error.
        let err = build_query_body(SearchMode::Sparse, "embedding", vec![1.0], false).unwrap_err();
        assert!(err.contains("sparse"), "msg: {err}");
        // Other non-dense modes also error (not executable in MVP).
        assert!(build_query_body(SearchMode::Hybrid, "embedding", vec![1.0], true).is_err());
        assert!(build_query_body(SearchMode::Colpali, "embedding", vec![1.0], true).is_err());
    }

    #[test]
    fn search_mode_parse_rejects_unknown() {
        use crate::agent::tools::builtins::SearchMode;
        assert_eq!(SearchMode::parse("dense").unwrap(), SearchMode::Dense);
        assert_eq!(SearchMode::parse("auto").unwrap(), SearchMode::Auto);
        assert!(SearchMode::parse("bogus").is_err());
    }

    #[test]
    fn select_query_model_picks_by_modality() {
        use crate::agent::tools::builtins::select_query_model;
        use veda_core::embedding::{EmbeddingModelConfig, EmbeddingModelRegistry, Modality};

        let mut reg = EmbeddingModelRegistry::new();
        // text-only model
        reg.register(EmbeddingModelConfig {
            name: "txt".to_owned(),
            endpoint: String::new(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 1024,
            timeout_seconds: 30,
            batch_size: 8,
        });
        // multimodal model (text + image)
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

        // Image query -> first image-capable model (the multimodal one).
        let m = select_query_model(&reg, None, Modality::Image).unwrap();
        assert_eq!(m.name, "mm");

        // Text query with a field-bound model -> that model.
        let m = select_query_model(&reg, Some("txt"), Modality::Text).unwrap();
        assert_eq!(m.name, "txt");

        // Text query, no binding -> first text-capable model.
        let m = select_query_model(&reg, None, Modality::Text).unwrap();
        assert_eq!(m.name, "txt");

        // No image model -> error.
        let mut text_only = EmbeddingModelRegistry::new();
        text_only.register(EmbeddingModelConfig {
            name: "txt".to_owned(),
            endpoint: String::new(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 4,
            max_input_size_bytes: 1024,
            timeout_seconds: 30,
            batch_size: 8,
        });
        assert!(select_query_model(&text_only, None, Modality::Image).is_err());
    }

    // ── apply_retrieval_guard (sync, in-crate) ──────────────────────────

    fn synthetic_hit(id: &str, flagged: bool, indexed_at: i64) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "distance": 0.1,
            "fields": {
                "content": "text",
                "_guardrail_flagged": flagged,
                "_guardrail_score": if flagged { 0.9 } else { 0.1 },
                "_indexed_at": indexed_at,
            }
        })
    }

    #[test]
    fn apply_retrieval_guard_drops_flagged_hit_and_updates_count() {
        let outcome = ToolOutcome::ok(serde_json::json!({
            "hits": [synthetic_hit("a", false, 1), synthetic_hit("b", true, 1)],
            "count": 2,
        }));
        let out = apply_retrieval_guard(outcome, None);
        assert!(!out.is_error);
        let hits = out.content["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1, "{:?}", out.content);
        assert_eq!(hits[0]["id"], "a");
        assert_eq!(out.content["count"], 1, "{:?}", out.content);
        assert_eq!(
            out.content["retrieval_guard"]["dropped"], 1,
            "{:?}",
            out.content
        );
        assert_eq!(
            out.content["retrieval_guard"]["kept"], 1,
            "{:?}",
            out.content
        );
    }

    fn synthetic_hit_with_category(
        id: &str,
        flagged: bool,
        category: &str,
        indexed_at: i64,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "distance": 0.1,
            "fields": {
                "content": "text",
                "_guardrail_flagged": flagged,
                "_guardrail_score": if flagged { 0.9 } else { 0.1 },
                "_guardrail_category": category,
                "_indexed_at": indexed_at,
            }
        })
    }

    /// 2026-08-13 ruling: PII must be detected and reported, but never
    /// dropped, at the dispatch-level choke point too — not just inside
    /// `retrieval_guard::enforce_retrieved`'s own unit tests.
    #[test]
    fn apply_retrieval_guard_keeps_pii_and_reports_flagged_not_dropped() {
        let outcome = ToolOutcome::ok(serde_json::json!({
            "hits": [
                synthetic_hit("a", false, 1),
                synthetic_hit_with_category("b", true, "pii", 1),
            ],
            "count": 2,
        }));
        let out = apply_retrieval_guard(outcome, None);
        assert!(!out.is_error);
        let hits = out.content["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 2, "PII must not be dropped: {:?}", out.content);
        assert_eq!(out.content["count"], 2, "{:?}", out.content);
        assert_eq!(
            out.content["retrieval_guard"]["dropped"], 0,
            "{:?}",
            out.content
        );
        assert_eq!(
            out.content["retrieval_guard"]["flagged_not_dropped"], 1,
            "{:?}",
            out.content
        );
    }

    #[test]
    fn apply_retrieval_guard_screens_records_key_too() {
        // `filter_query`'s success payload uses `records`, not `hits`.
        let outcome = ToolOutcome::ok(serde_json::json!({
            "records": [synthetic_hit("a", false, 1), synthetic_hit("b", true, 1)],
            "count": 2,
        }));
        let out = apply_retrieval_guard(outcome, None);
        let records = out.content["records"].as_array().unwrap();
        assert_eq!(records.len(), 1, "{:?}", out.content);
        assert!(out.content.get("hits").is_none(), "{:?}", out.content);
        assert_eq!(
            out.content["retrieval_guard"]["dropped"], 1,
            "{:?}",
            out.content
        );
    }

    #[test]
    fn apply_retrieval_guard_leaves_payloads_without_hits_or_records_untouched() {
        let outcome = ToolOutcome::ok(serde_json::json!({ "inserted": 3 }));
        let out = apply_retrieval_guard(outcome.clone(), None);
        assert_eq!(
            out, outcome,
            "payload with neither key must pass through unchanged"
        );
    }

    #[test]
    fn apply_retrieval_guard_error_outcomes_are_not_reachable_here() {
        // apply_retrieval_guard itself has no error short-circuit (that
        // lives in `dispatch`); confirm it still degrades harmlessly if
        // ever handed content shaped like an error (no hits/records key).
        let outcome = ToolOutcome::invalid_args("bad args");
        let out = apply_retrieval_guard(outcome.clone(), None);
        assert_eq!(out, outcome);
    }
}
