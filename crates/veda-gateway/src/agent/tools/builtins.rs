//! Built-in tool implementations.
//!
//! Each builtin is a `ToolSpec` (id, docs, JSON-Schema) plus an async
//! impl `(&ToolContext, args) -> ToolOutcome`. The logic here is the
//! single source of truth for both the MCP server and the agent
//! executor — the MCP endpoint renders/dispatches a subset of these via
//! the catalog rather than carrying its own copies.

use base64::prelude::*;
use veda_core::embedding::{EmbeddingModelConfig, Modality};
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::Record;
use veda_core::types::{DistanceMetric, FieldType};
use veda_query::adaptive::route_query;
use veda_query::filter::ast::FilterExpr;
use veda_query::filter::parser::FilterParser;
use veda_query::filter::record_eval::evaluate_record;
use veda_query::search::VectorSearchRequest;

use crate::agent::catalog::{ToolContext, ToolOutcome, ToolSpec};
use crate::agent::retrieval_guard::enforcement_enabled;
use crate::rest::aggregate::{AggErrorKind, AggregateQuery, aggregate_core};
use crate::rest::records::{field_value_to_json, parse_record_public, record_passes_rls_owner};

/// The full set of built-in tool specs (id == MCP/OpenAI tool name).
/// Includes the read-only ops/observability tools (see
/// [`super::ops::ops_specs`]) that back the `veda-ops` agent.
pub fn builtin_specs() -> Vec<ToolSpec> {
    let mut specs = vec![
        ToolSpec {
            id: "list_collections".to_owned(),
            name: "list_collections".to_owned(),
            description: "List all Veda collections by name.".to_owned(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("List collections".to_owned()),
        },
        ToolSpec {
            id: "describe_collection".to_owned(),
            name: "describe_collection".to_owned(),
            description: "Return the schema and total record count for a Veda collection."
                .to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Collection name" }
                },
                "required": ["name"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Describe collection".to_owned()),
        },
        ToolSpec {
            id: "search".to_owned(),
            name: "search".to_owned(),
            description: "Cross-modal search over a Veda collection. Supply EITHER a text `query` OR a base64-encoded `query_image_b64` image as the retrieval query (exactly one). The query is auto-embedded into Veda's unified multimodal vector space using the collection's text model (for text) or the registry's image model (for images), so an image query can retrieve text/image/document records and vice-versa. Returns the top-k nearest records by vector similarity.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection to search" },
                    "query": { "type": "string", "description": "Natural-language query text. Provide this OR query_image_b64, not both." },
                    "query_image_b64": { "type": "string", "description": "Base64-encoded image bytes used as the retrieval query (cross-modal). Provide this OR query, not both." },
                    "mode": { "type": "string", "enum": ["auto", "dense", "hybrid", "sparse", "colpali"], "default": "auto", "description": "Retrieval mode. 'auto' (default) maps to dense for the MVP." },
                    "field": { "type": "string", "description": "Vector field name (optional; defaults to the first vector field on the collection)" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                    "filter": { "type": "string", "description": "Optional SQL-like filter expression, e.g. \"category = 'electronics' AND price < 500\"" },
                    "fields": { "type": "array", "items": { "type": "string" }, "description": "Optional list of field names to return for each hit; omitted = all fields." }
                },
                "required": ["collection"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Multimodal search".to_owned()),
        },
        ToolSpec {
            id: "upsert_record".to_owned(),
            name: "upsert_record".to_owned(),
            description: "Insert one record into a Veda collection. Fields must match the collection's schema. Vector fields can be supplied as float arrays, or omitted and computed via auto-embedding (call embed-insert instead for that flow).".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string" },
                    "record": { "type": "object", "description": "Field name → value map." }
                },
                "required": ["collection", "record"]
            }),
            kind: "builtin".to_owned(),
            read_only: false,
            destructive: true,
            title: Some("Insert record".to_owned()),
        },
        ToolSpec {
            id: "get_record".to_owned(),
            name: "get_record".to_owned(),
            description: "Fetch a single record by its UUID from a Veda collection. Returns the record's fields, or an error if it does not exist (or has expired via TTL).".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection to read from" },
                    "id": { "type": "string", "description": "Record UUID" }
                },
                "required": ["collection", "id"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Get record".to_owned()),
        },
        ToolSpec {
            id: "route".to_owned(),
            name: "route".to_owned(),
            description: "Analyse a natural-language query and recommend a retrieval strategy (sparse/dense/hybrid), whether to rerank, and a suggested k. Heuristic hint only.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language query to analyse" }
                },
                "required": ["query"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Route query".to_owned()),
        },
        ToolSpec {
            id: "aggregate".to_owned(),
            name: "aggregate".to_owned(),
            description: "Compute a scalar OLAP metric (count/sum/avg/min/max) over a field in a Veda collection, optionally grouped by another field and/or filtered. Returns one value per group.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection to aggregate over" },
                    "field": { "type": "string", "description": "Field to aggregate. Must be numeric for sum/avg/min/max; any field for count." },
                    "metric": { "type": "string", "enum": ["count", "sum", "avg", "min", "max"], "description": "Aggregation metric" },
                    "group_by": { "type": "string", "description": "Optional field to group results by" },
                    "filter": { "type": "string", "description": "Optional SQL-like filter expression" }
                },
                "required": ["collection", "field", "metric"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Aggregate".to_owned()),
        },
        ToolSpec {
            id: "filter_query".to_owned(),
            name: "filter_query".to_owned(),
            description: "Return records matching a SQL-like filter expression from a Veda collection (no vector search). Row-level security and TTL are enforced. Returns up to `limit` records with their fields.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection to scan" },
                    "filter": { "type": "string", "description": "SQL-like filter expression, e.g. \"price < 500 AND category = 'books'\"" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 100 }
                },
                "required": ["collection", "filter"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Filter query".to_owned()),
        },
    ];
    specs.extend(super::ops::ops_specs());
    specs.extend(super::web::web_specs());
    specs
}

// ── Implementations ─────────────────────────────────────────────────────────

pub async fn list_collections(ctx: &ToolContext<'_>, _args: serde_json::Value) -> ToolOutcome {
    let schemas = ctx.state.inner.schemas.read().await;
    let names: Vec<String> = schemas
        .keys()
        .filter_map(|k| crate::tenant::strip_prefix(ctx.tenant, k).map(str::to_owned))
        .collect();
    ToolOutcome::ok(serde_json::json!({ "collections": names, "count": names.len() }))
}

pub async fn describe_collection(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_name) = args["name"].as_str() else {
        return ToolOutcome::invalid_args("name is required");
    };
    let name = crate::tenant::scope(ctx.tenant, user_name);
    let schema = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(&name).cloned() {
            Some(s) => s,
            None => return ToolOutcome::not_found(format!("collection '{user_name}' not found")),
        }
    };
    let managers = ctx.state.inner.segment_managers.read().await;
    let total: u64 = managers
        .get(&name)
        .map(|m| m.values().map(|sm| sm.total_records()).sum())
        .unwrap_or(0);

    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|f| {
            let kind = match f.field_type() {
                FieldType::Vector { .. } => "vector",
                FieldType::SparseVector { .. } => "sparse_vector",
                FieldType::MultiVector { .. } => "multi_vector",
                FieldType::String => "string",
                FieldType::Int32 => "int32",
                FieldType::Int64 => "int64",
                FieldType::Float32 => "float32",
                FieldType::Float64 => "float64",
                FieldType::Bool => "bool",
                FieldType::Timestamp => "timestamp",
                FieldType::Blob => "blob",
                FieldType::Json => "json",
                FieldType::Uuid => "uuid",
                FieldType::Array(_) => "array",
            };
            let mut out = serde_json::json!({
                "name": f.name(),
                "type": kind,
                "filterable": f.filterable(),
            });
            // Field meaning, when the schema carries any. Flattened in
            // beside the physical facts rather than nested under an
            // "annotations" key: the model reads this as one description
            // of the field, and an extra level of nesting is one more
            // thing for it to get wrong. Absent keys stay absent, so an
            // unannotated collection renders exactly as it did before.
            let ann = f.annotations();
            if let Some(obj) = out.as_object_mut() {
                if let Some(d) = &ann.description {
                    obj.insert("description".into(), serde_json::json!(d));
                }
                if let Some(s) = &ann.semantic_type {
                    obj.insert("semantic_type".into(), serde_json::json!(s));
                }
                if let Some(v) = &ann.value_domain {
                    obj.insert("value_domain".into(), serde_json::json!(v));
                }
                if f.is_withheld_from_non_admin() {
                    // Say so explicitly. A model that knows a field is
                    // withheld can answer "I can't see that" instead of
                    // burning steps searching for a column that will
                    // never appear in any result.
                    obj.insert("restricted".into(), serde_json::json!(true));
                }
            }
            out
        })
        .collect();

    ToolOutcome::ok(serde_json::json!({
        "name": user_name,
        "fields": fields,
        "record_count": total,
        "shard_count": schema.shard_count(),
    }))
}

/// Retrieval mode for the multimodal `search` tool. Parsed from the
/// `mode` argument; `auto` is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Auto,
    Dense,
    Hybrid,
    Sparse,
    Colpali,
}

impl SearchMode {
    /// Parse the `mode` argument. Unknown values are an error.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "hybrid" => Ok(Self::Hybrid),
            "sparse" => Ok(Self::Sparse),
            "colpali" => Ok(Self::Colpali),
            other => Err(format!(
                "invalid mode '{other}'; expected one of auto|dense|hybrid|sparse|colpali"
            )),
        }
    }
}

/// Whether the collection/embedder for `field` can produce a sparse
/// query representation. bge-m3-style models that declare a
/// `SparseVector` field on the schema support it.
fn supports_sparse(schema: &veda_core::schema::CollectionSchema) -> bool {
    schema
        .fields()
        .iter()
        .any(|f| matches!(f.field_type(), FieldType::SparseVector { .. }))
}

/// Map a [`SearchMode`] + a single dense query `vector` to the REST
/// [`QueryBody`] the search executor consumes. The single-vector
/// embedding path produces a dense vector, so only `dense`/`auto` yield
/// an executable query for the MVP; the other modes are recognised and
/// validated here but reported as unsupported (clear message) until the
/// matching query-representation embedders are wired.
///
/// Returns the populated [`QueryBody`] (only its `vector` leg is set) or
/// an error message suitable for `invalid_args`.
pub fn build_query_body(
    mode: SearchMode,
    field: &str,
    vector: Vec<f32>,
    sparse_supported: bool,
) -> Result<crate::rest::search::QueryBody, String> {
    use crate::rest::search::{QueryBody, SingleVectorQuery};
    let dense = || QueryBody {
        vector: Some(SingleVectorQuery {
            field: field.to_owned(),
            values: vector.clone(),
        }),
        sparse_vector: None,
        colpali: None,
        multi_vector: None,
        bm25: None,
        match_phrase: None,
        hybrid: None,
        fusion: None,
        fusion_params: None,
    };
    match mode {
        // TODO: wire adaptive router for auto
        SearchMode::Auto | SearchMode::Dense => Ok(dense()),
        SearchMode::Sparse => {
            if !sparse_supported {
                return Err(
                    "mode 'sparse' requires a collection with a sparse-vector field and a \
                     sparse-capable embedder"
                        .to_owned(),
                );
            }
            Err(
                "mode 'sparse' is not yet supported via the agent search tool (the query \
                 embedder produces a dense vector); use mode 'dense' or 'auto'"
                    .to_owned(),
            )
        }
        SearchMode::Hybrid => Err(
            "mode 'hybrid' is not yet supported via the agent search tool; use 'dense' or 'auto'"
                .to_owned(),
        ),
        SearchMode::Colpali => Err(
            "mode 'colpali' is not yet supported via the agent search tool; use 'dense' or 'auto'"
                .to_owned(),
        ),
    }
}

/// Select the embedding model for a query of the given modality.
///
/// For a TEXT query we prefer the model bound to the target vector field
/// (matching how records were embedded); falling back to any
/// text-capable model. For an IMAGE query we always pick an image-capable
/// model from the registry (the unified multimodal space), reusing the
/// same `find_by_modality` path as `/v1/embed`.
pub fn select_query_model(
    registry: &veda_core::embedding::EmbeddingModelRegistry,
    bound_model: Option<&str>,
    modality: Modality,
) -> Result<EmbeddingModelConfig, String> {
    if modality == Modality::Text
        && let Some(name) = bound_model
    {
        return registry
            .get(name)
            .cloned()
            .ok_or_else(|| format!("bound model '{name}' not in registry"));
    }
    registry
        .find_by_modality(modality)
        .first()
        .map(|c| (*c).clone())
        .ok_or_else(|| format!("no embedding model configured for modality '{modality}'"))
}

/// Field names this caller may not read, from an already-resolved schema.
///
/// Empty for an Admin caller and for any schema carrying no sensitivity
/// annotation. Takes the schema rather than re-resolving it, so callers
/// that already hold one do not pay a second alias lookup.
fn withheld_fields(
    ctx: &ToolContext<'_>,
    schema: &veda_core::schema::CollectionSchema,
) -> Vec<String> {
    if ctx.is_admin {
        return Vec::new();
    }
    schema
        .withheld_fields_for_non_admin()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// The first restricted field named in an optional filter expression.
///
/// Thin wrapper over [`crate::agent::field_guard::filter_mentions_withheld`]
/// that tolerates the `None` filter case, which is the common one.
fn withheld_field_in_filter(expr: &Option<FilterExpr>, withheld: &[String]) -> Option<String> {
    if withheld.is_empty() {
        return None;
    }
    expr.as_ref()
        .and_then(|e| crate::agent::field_guard::filter_mentions_withheld(e, withheld))
}

/// Build the AST-level filter for an agent `search`, enforcing RLS.
///
/// Parses the optional user-supplied `filter` string, then — for a scoped
/// (`Some`) caller on an RLS collection (`owner_field` is `Some`) — ANDs a
/// trusted `owner_field == user` predicate via [`crate::rls::inject_owner_filter`]
/// (`is_admin = false`). A `None` user (admin/system) or a collection without
/// an owner field bypasses injection and yields just the parsed user filter.
///
/// The owner predicate is injected at the AST level (never via a filter
/// string) so a quote in a username cannot break out and bypass RLS — the
/// same security-correct path as REST `/search`.
fn search_rls_filter_expr(
    user_filter: Option<&str>,
    owner_field: Option<&str>,
    user: Option<&str>,
) -> Result<Option<FilterExpr>, String> {
    let parsed = match user_filter {
        Some(s) => Some(FilterParser::parse(s).map_err(|e| format!("invalid filter: {e}"))?),
        None => None,
    };
    match (owner_field, user) {
        (Some(field), Some(u)) => Ok(crate::rls::inject_owner_filter(parsed, field, u, false)),
        _ => Ok(parsed),
    }
}

pub async fn search(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);
    let collection = scoped.as_str();

    // Exactly one of query / query_image_b64.
    let text_query = args["query"].as_str().filter(|s| !s.is_empty());
    let image_b64 = args["query_image_b64"].as_str().filter(|s| !s.is_empty());
    let (modality, query_text, query_bytes): (Modality, Option<&str>, Option<Vec<u8>>) =
        match (text_query, image_b64) {
            (Some(q), None) => (Modality::Text, Some(q), None),
            (None, Some(b)) => match BASE64_STANDARD.decode(b) {
                Ok(bytes) => (Modality::Image, None, Some(bytes)),
                Err(e) => {
                    return ToolOutcome::invalid_args(format!(
                        "invalid base64 in query_image_b64: {e}"
                    ));
                }
            },
            (Some(_), Some(_)) => {
                return ToolOutcome::invalid_args(
                    "provide exactly one of `query` or `query_image_b64`, not both",
                );
            }
            (None, None) => {
                return ToolOutcome::invalid_args(
                    "one of `query` or `query_image_b64` is required",
                );
            }
        };

    let mode = match SearchMode::parse(args["mode"].as_str().unwrap_or("auto")) {
        Ok(m) => m,
        Err(e) => return ToolOutcome::invalid_args(e),
    };
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let filter = args["filter"].as_str().map(|s| s.to_owned());
    // Optional field projection: narrow each hit's `fields` map to these
    // names. An empty array is treated as `None` (return all fields).
    let keep_fields: Option<Vec<String>> = args["fields"].as_array().and_then(|arr| {
        let names: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        if names.is_empty() { None } else { Some(names) }
    });

    let schema = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(collection).cloned() {
            Some(s) => s,
            None => return ToolOutcome::not_found(format!("collection '{collection}' not found")),
        }
    };

    // RLS: inject an owner-equality predicate at the AST level for a scoped
    // (`Some`) caller on an RLS collection; `None` (admin/system) bypasses.
    // Mirrors REST `/search` — the executor prefers `filter_expr` over the
    // raw string `filter`, so this is the security-correct path.
    let filter_expr =
        match search_rls_filter_expr(filter.as_deref(), schema.rls_owner_field(), ctx.user) {
            Ok(e) => e,
            Err(e) => return ToolOutcome::invalid_args(e),
        };

    // A restricted field is an oracle even when it is never returned: the
    // caller cannot see `salary`, but `salary > 100000` reads it off the
    // result count one comparison at a time. Redaction cannot close that,
    // so the query is refused instead.
    if let Some(bad) = withheld_field_in_filter(&filter_expr, &withheld_fields(ctx, &schema)) {
        return ToolOutcome::invalid_args(crate::agent::field_guard::restricted_field_error(&bad));
    }

    // Resolve target vector field.
    let vec_fields = schema.vector_fields();
    let field_name = if let Some(f) = args["field"].as_str() {
        f.to_owned()
    } else if let Some(first) = vec_fields.first() {
        first.name().to_owned()
    } else {
        return ToolOutcome::invalid_args(format!(
            "collection '{collection}' has no vector fields"
        ));
    };
    let Some(field_def) = schema.get_field(&field_name) else {
        return ToolOutcome::invalid_args(format!("field '{field_name}' not found"));
    };
    let metric = match field_def.field_type() {
        FieldType::Vector { metric, .. } => *metric,
        _ => DistanceMetric::Cosine,
    };
    let bound_model = match field_def.field_type() {
        FieldType::Vector { model_id, .. } => model_id.clone(),
        _ => None,
    };

    // Select the model for this query's modality (image → image model,
    // text → field-bound/text model) — the unified cross-modal space.
    let model = {
        let registry = ctx.state.inner.embedding_registry.read().await;
        match select_query_model(&registry, bound_model.as_deref(), modality) {
            Ok(m) => m,
            Err(e) => return ToolOutcome::internal(e),
        }
    };

    // Embed the query: text via embed_text, image bytes via embed_bytes.
    let client = EmbeddingClient::new(model);
    let vector = match (query_text, query_bytes) {
        (Some(q), _) => match client.embed_text(q).await {
            Ok(v) => v,
            Err(e) => return ToolOutcome::internal(format!("embedding failed: {e}")),
        },
        (None, Some(bytes)) => match client.embed_bytes(&bytes, modality).await {
            Ok(v) => v,
            Err(e) => return ToolOutcome::internal(format!("embedding failed: {e}")),
        },
        (None, None) => unreachable!("modality was resolved above"),
    };

    // Map mode → QueryBody. Only the dense leg is executable in the MVP.
    let query_body = match build_query_body(mode, &field_name, vector, supports_sparse(&schema)) {
        Ok(qb) => qb,
        Err(e) => return ToolOutcome::invalid_args(e),
    };
    let Some(vec_query) = query_body.vector else {
        // build_query_body only returns Ok for dense/auto today.
        return ToolOutcome::internal("unexpected non-dense query body");
    };

    let executors = ctx.state.inner.query_executors.read().await;
    let Some(executor) = executors.get(collection) else {
        return ToolOutcome::internal("no query executor");
    };
    let request = VectorSearchRequest {
        field: field_name.clone(),
        query_vector: vec_query.values,
        k: limit,
        metric,
        filter,
        filter_expr,
        truncate_to: None,
        read_ts: None,
        ef_search: None,
    };
    let response = match executor.search(&request) {
        Ok((response, _)) => response,
        Err(e) => return ToolOutcome::internal(format!("search failed: {e}")),
    };

    // Hydrate each hit's record fields (O(1) UUID→Record lookup per hit,
    // skipping vector fields), then apply the optional `fields` projection.
    // Mirrors REST `/search` field hydration so the agent surface matches.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let vector_field_names: std::collections::HashSet<&str> = schema
        .fields()
        .iter()
        .filter(|f| {
            matches!(
                f.field_type(),
                FieldType::Vector { .. }
                    | FieldType::MultiVector { .. }
                    | FieldType::SparseVector { .. }
            )
        })
        .map(|f| f.name())
        .collect();
    let managers = ctx.state.inner.segment_managers.read().await;
    let shard_map = managers.get(collection);

    let hits: Vec<serde_json::Value> = response
        .results
        .into_iter()
        .map(|r| {
            let mut fields = serde_json::Map::new();
            if let Some(shard_map) = shard_map {
                let rec = shard_map
                    .values()
                    .find_map(|sm| sm.lookup_record_at(r.id, now_secs));
                if let Some(rec) = rec {
                    for field_def in schema.fields() {
                        let name = field_def.name();
                        if vector_field_names.contains(name) {
                            continue;
                        }
                        if let Some(value) = rec.get(name) {
                            fields.insert(name.to_string(), field_value_to_json(value));
                        }
                    }
                    // The retrieval-guardrail/TTL fields are RESERVED, not
                    // schema-declared (segment.rs's WAL writer persists them
                    // "even when not declared in the schema") — no
                    // `CollectionSchema` a real `/v1/collections` call
                    // produces ever lists them, so the loop above never
                    // hydrates them and every hit classifies `Unscreened`.
                    // Fetch them directly so the dispatch-level guard
                    // (`apply_retrieval_guard` in `agent/catalog.rs`) has
                    // something to classify on every real collection, not
                    // only ones a test happens to declare them on.
                    for reserved in RESERVED_GUARD_FIELDS {
                        if let Some(value) = rec.get(reserved) {
                            fields.insert((*reserved).to_owned(), field_value_to_json(value));
                        }
                    }
                }
            }
            project_fields(&mut fields, &keep_fields);
            serde_json::json!({
                "id": r.id.to_string(),
                "distance": r.distance,
                "fields": fields,
            })
        })
        .collect();
    ToolOutcome::ok(serde_json::json!({ "hits": hits, "count": hits.len() }))
}

pub async fn upsert_record(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);
    let collection = scoped.as_str();
    let Some(raw) = args["record"].as_object() else {
        return ToolOutcome::invalid_args("record must be an object");
    };

    let schema = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(collection).cloned() {
            Some(s) => s,
            None => return ToolOutcome::not_found(format!("collection '{collection}' not found")),
        }
    };
    let record: Record = match parse_record_public(raw, &schema) {
        Ok(r) => r,
        Err(e) => return ToolOutcome::invalid_args(format!("parse_record: {e}")),
    };

    let mut managers = ctx.state.inner.segment_managers.write().await;
    let Some(shard_map) = managers.get_mut(collection) else {
        return ToolOutcome::internal("segment managers missing");
    };
    let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
        &record.id(),
        schema.shard_count(),
    );
    let Some(sm) = shard_map.get_mut(&shard_id) else {
        return ToolOutcome::internal(format!("shard {shard_id} not found"));
    };
    if let Err(e) = sm.insert(record) {
        return ToolOutcome::internal(format!("insert failed: {e}"));
    }
    let id = sm.growing_segment().count(); // approximate
    ToolOutcome::ok(serde_json::json!({ "ok": true, "doc_id_estimate": id }))
}

pub async fn get_record(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let Some(id_str) = args["id"].as_str() else {
        return ToolOutcome::invalid_args("id is required");
    };
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);
    let collection = scoped.as_str();
    let record_id: veda_core::types::RecordId = match id_str.parse() {
        Ok(u) => u,
        Err(_) => return ToolOutcome::invalid_args(format!("invalid UUID: '{id_str}'")),
    };

    // Look up the collection's RLS owner field. For a scoped (`Some`)
    // user we require the record's owner to match; `None` (admin/system)
    // bypasses. Reuses the same predicate as the REST read path.
    let rls_field = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(collection) {
            Some(s) => s.rls_owner_field().map(str::to_owned),
            None => {
                return ToolOutcome::not_found(format!("collection '{user_collection}' not found"));
            }
        }
    };
    let rls_owner = rls_field.as_deref().and(ctx.user);

    let managers = ctx.state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return ToolOutcome::not_found(format!("collection '{user_collection}' not found"));
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if let Some(rec) = gs.get_record(doc_id)
                && rec.id() == record_id
                && !gs.is_deleted(doc_id)
            {
                if rec.is_expired_at(now_secs) {
                    return ToolOutcome::not_found("record not found");
                }
                if !crate::rest::records::record_passes_rls_owner(
                    rec,
                    rls_field.as_deref(),
                    rls_owner,
                ) {
                    return ToolOutcome::not_found("record not found");
                }
                return guarded_record_outcome(rec);
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id)
                    && rec.id() == record_id
                {
                    if rec.is_expired_at(now_secs) {
                        return ToolOutcome::not_found("record not found");
                    }
                    if !crate::rest::records::record_passes_rls_owner(
                        &rec,
                        rls_field.as_deref(),
                        rls_owner,
                    ) {
                        return ToolOutcome::not_found("record not found");
                    }
                    return guarded_record_outcome(&rec);
                }
            }
        }
    }

    ToolOutcome::not_found(format!("record '{record_id}' not found"))
}

/// Apply the same category-aware drop policy `apply_retrieval_guard`
/// (`agent/catalog.rs`) applies to `hits`/`records` arrays to a single
/// fetched record, withholding its content when hostile-flagged
/// (PII-flagged is exempt and comes back intact — see
/// `retrieval_guard::category_is_hostile`).
///
/// `get_record` is a targeted fetch-by-id, so its bare `{id, fields}`
/// payload never carries a `hits`/`records` array and the dispatch-level
/// guard passes it through untouched. Without a check here, a model that
/// saw a chunk's id reported in a `search`/`filter_query` response's
/// `retrieval_guard.dropped_chunks` (reported there precisely so the drop
/// is never silent) could call `get_record` on that same id and retrieve
/// the withheld content verbatim — the array-shaped guard would have been
/// enforcement theater. `enforce_retrieved` is reused rather than
/// reimplemented so the exact same classification logic applies here as
/// everywhere else.
fn guarded_record_outcome(rec: &Record) -> ToolOutcome {
    use crate::agent::retrieval_guard::enforce_retrieved;

    let hit = record_json(rec);
    let (mut kept, report) = enforce_retrieved(vec![hit], enforcement_enabled());
    match kept.pop() {
        Some(hit) => ToolOutcome::ok(hit),
        None => ToolOutcome::ok(serde_json::json!({
            "id": rec.id().to_string(),
            "withheld": true,
            "retrieval_guard": serde_json::to_value(&report).unwrap_or(serde_json::Value::Null),
        })),
    }
}

pub async fn route(_ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(query) = args["query"].as_str() else {
        return ToolOutcome::invalid_args("query is required");
    };
    let rec = route_query(query);
    match serde_json::to_value(&rec) {
        Ok(v) => ToolOutcome::ok(v),
        Err(e) => ToolOutcome::internal(format!("serialize route: {e}")),
    }
}

pub async fn aggregate(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let Some(field) = args["field"].as_str() else {
        return ToolOutcome::invalid_args("field is required");
    };
    let Some(metric) = args["metric"].as_str() else {
        return ToolOutcome::invalid_args("metric is required");
    };
    if !matches!(metric, "count" | "sum" | "avg" | "min" | "max") {
        return ToolOutcome::invalid_args(format!(
            "invalid metric '{metric}'; expected count|sum|avg|min|max"
        ));
    }
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);

    // Aggregation reaches a restricted field's values without ever
    // returning the field, so redaction at the payload level is useless
    // here and the request is refused instead. Three separate routes:
    //   - `field` with min/max returns the value directly;
    //   - `group_by` emits every distinct key VERBATIM, which is the
    //     whole column one row at a time;
    //   - `filter` is the same oracle `search`/`filter_query` refuse.
    // The same reasoning that made this tool pass `true` to
    // `aggregate_core` for hostile-content withholding applies again.
    {
        let resolved = ctx.state.inner.aliases.resolve(&scoped).await;
        let schemas = ctx.state.inner.schemas.read().await;
        if let Some(schema) = schemas.get(&resolved) {
            let withheld = withheld_fields(ctx, schema);
            if !withheld.is_empty() {
                let group_by = args["group_by"].as_str();
                let named = [Some(field), group_by]
                    .into_iter()
                    .flatten()
                    .find(|n| withheld.iter().any(|w| w == n))
                    .map(str::to_owned);
                if let Some(bad) = named {
                    return ToolOutcome::invalid_args(
                        crate::agent::field_guard::restricted_field_error(&bad),
                    );
                }
                if let Some(f) = args["filter"].as_str() {
                    match FilterParser::parse(f) {
                        Ok(expr) => {
                            if let Some(bad) = crate::agent::field_guard::filter_mentions_withheld(
                                &expr, &withheld,
                            ) {
                                return ToolOutcome::invalid_args(
                                    crate::agent::field_guard::restricted_field_error(&bad),
                                );
                            }
                        }
                        Err(e) => {
                            return ToolOutcome::invalid_args(format!("invalid filter: {e}"));
                        }
                    }
                }
            }
        }
    }

    let query = AggregateQuery {
        field: field.to_owned(),
        metric: metric.to_owned(),
        group_by: args["group_by"].as_str().map(str::to_owned),
        filter: args["filter"].as_str().map(str::to_owned),
    };
    // `true` = withhold hostile-flagged records from the aggregation. This
    // tool is model-reachable on every LLM transport, and `group_by` on a
    // text field emits each group's key VERBATIM — so without this,
    // `group_by=content&filter=_guardrail_flagged = true` returns the full
    // text of every chunk the rest of the feature withholds, and
    // `dropped_chunks[].id` tells a model exactly what to ask for. The raw
    // REST `/aggregate` keeps its annotate-don't-drop behaviour (it passes
    // `false`) — see `aggregate_core`.
    match aggregate_core(ctx.state, scoped.as_str(), &query, enforcement_enabled()).await {
        Ok(resp) => {
            let groups: Vec<serde_json::Value> = resp
                .groups
                .into_iter()
                .map(|g| serde_json::json!({ "key": g.key, "value": g.value }))
                .collect();
            // Report the skip rather than silently returning short counts —
            // the same "a drop is never silent" rule the hit-shaped guard
            // follows.
            ToolOutcome::ok(serde_json::json!({
                "groups": groups,
                "retrieval_guard": { "withheld_records": resp.withheld.unwrap_or(0) },
            }))
        }
        Err(e) => match e.kind {
            AggErrorKind::NotFound => ToolOutcome::not_found(e.message),
            AggErrorKind::BadRequest => ToolOutcome::invalid_args(e.message),
            AggErrorKind::Internal => ToolOutcome::internal(e.message),
        },
    }
}

pub async fn filter_query(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let Some(filter) = args["filter"].as_str() else {
        return ToolOutcome::invalid_args("filter is required");
    };
    let limit = args["limit"].as_u64().unwrap_or(100) as usize;
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);
    let collection = scoped.as_str();

    let filter_expr = match FilterParser::parse(filter) {
        Ok(e) => e,
        Err(e) => return ToolOutcome::invalid_args(format!("invalid filter: {e}")),
    };

    // RLS: a scoped (`Some`) user only sees rows they own when the
    // collection declares an owner field; `None` (admin) bypasses.
    let (rls_field, withheld) = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(collection) {
            Some(s) => (
                s.rls_owner_field().map(str::to_owned),
                withheld_fields(ctx, s),
            ),
            None => {
                return ToolOutcome::not_found(format!("collection '{user_collection}' not found"));
            }
        }
    };

    // Same oracle argument as in `search`: a predicate on a restricted
    // field leaks its value through the result count without ever
    // returning it, so refuse rather than redact.
    if let Some(bad) = crate::agent::field_guard::filter_mentions_withheld(&filter_expr, &withheld)
    {
        return ToolOutcome::invalid_args(crate::agent::field_guard::restricted_field_error(&bad));
    }
    let rls_owner = rls_field.as_deref().and(ctx.user);

    let managers = ctx.state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return ToolOutcome::not_found(format!("collection '{user_collection}' not found"));
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut records = Vec::new();
    'outer: for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                if rec.is_expired_at(now_secs) {
                    continue;
                }
                if !record_passes_rls_owner(rec, rls_field.as_deref(), rls_owner) {
                    continue;
                }
                if !evaluate_record(&filter_expr, rec) {
                    continue;
                }
                records.push(record_json(rec));
                if records.len() >= limit {
                    break 'outer;
                }
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if rec.is_expired_at(now_secs) {
                        continue;
                    }
                    if !record_passes_rls_owner(&rec, rls_field.as_deref(), rls_owner) {
                        continue;
                    }
                    if !evaluate_record(&filter_expr, &rec) {
                        continue;
                    }
                    records.push(record_json(&rec));
                    if records.len() >= limit {
                        break 'outer;
                    }
                }
            }
        }
    }

    let count = records.len();
    ToolOutcome::ok(serde_json::json!({ "records": records, "count": count }))
}

/// The retrieval-guardrail/TTL fields, reserved rather than
/// schema-declared. Mirrors the identical inline list in
/// `veda-storage/src/segment.rs`'s WAL writer (the other place that must
/// not rely on schema declaration to see them).
const RESERVED_GUARD_FIELDS: [&str; 5] = [
    veda_core::record::EXPIRES_AT_FIELD,
    veda_core::record::GUARDRAIL_FLAGGED_FIELD,
    veda_core::record::GUARDRAIL_SCORE_FIELD,
    veda_core::record::GUARDRAIL_CATEGORY_FIELD,
    veda_core::record::INDEXED_AT_FIELD,
];

/// Narrow a hit's `fields` map to only the requested `keep` names.
///
/// Pure and testable. `None` (or, per the caller, an empty request array)
/// leaves the map untouched — all fields are returned. A requested name
/// that is absent from the map is simply skipped (no error). Field order
/// follows the `keep` list. Only the record `fields` map is affected;
/// the hit's `id`/`distance` are never touched (they live outside this map).
///
/// The reserved guardrail/TTL fields (see [`RESERVED_GUARD_FIELDS`])
/// survive projection even when not requested: they are what
/// `apply_retrieval_guard` (`agent/catalog.rs`) reads to classify and drop
/// a hit at the tool-dispatch choke point. Without this, `"fields":
/// ["content"]` would silently strip the verdict before enforcement ever
/// saw it — a one-argument, model-reachable bypass.
fn project_fields(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    keep: &Option<Vec<String>>,
) {
    let Some(keep) = keep else { return };
    let mut projected = serde_json::Map::new();
    for name in keep {
        if let Some(value) = fields.remove(name) {
            projected.insert(name.clone(), value);
        }
    }
    for reserved in RESERVED_GUARD_FIELDS {
        if let Some(value) = fields.remove(reserved) {
            projected.insert(reserved.to_owned(), value);
        }
    }
    *fields = projected;
}

/// Render a field value for an LLM, summarising embeddings instead of
/// dumping them.
///
/// `field_value_to_json` serialises a `Vector` as its full component
/// list, which is right for REST — a client asking for a record may
/// genuinely want the embedding — and wrong for a model. A 1024-dim
/// vector is ~1k JSON numbers, and the tools return one per record.
///
/// This is not only a token-cost problem, it measurably degrades
/// answers. In the 2026-08-20 annotation measurement, `filter_query`
/// on a 5-record collection returned `count: 2` correctly, but both
/// records arrived behind ~2k float components; the model reported only
/// the first supplier and in one run invented a second name outright.
/// The retrieval was perfect and the answer was wrong purely because
/// the useful fields were buried.
///
/// A model can do nothing with raw components, so they are replaced by
/// a shape summary that still says the field exists and is populated.
fn agent_field_value(value: &veda_core::record::FieldValue) -> serde_json::Value {
    use veda_core::record::FieldValue as FV;
    match value {
        FV::Vector(v) => serde_json::json!({ "vector": { "dim": v.as_slice().len() } }),
        FV::SparseVector(sv) => {
            serde_json::json!({ "sparse_vector": { "nnz": sv.indices().len() } })
        }
        FV::MultiVector(rows) => serde_json::json!({
            "multi_vector": {
                "vectors": rows.len(),
                "dim": rows.first().map(|r| r.len()).unwrap_or(0),
            }
        }),
        other => field_value_to_json(other),
    }
}

/// Serialise a record to `{ "id", "fields" }` — matching the REST
/// `GetRecordResponse` shape so the agent/MCP surfaces stay consistent,
/// except that embeddings are summarised (see [`agent_field_value`]).
fn record_json(rec: &Record) -> serde_json::Value {
    let mut fields = serde_json::Map::new();
    for (key, value) in rec.fields() {
        fields.insert(key.clone(), agent_field_value(value));
    }
    serde_json::json!({ "id": rec.id().to_string(), "fields": fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_query::filter::ast::{FilterExpr, FilterValue};

    fn owner_eq(user: &str) -> FilterExpr {
        FilterExpr::Eq {
            field: "owner".to_owned(),
            value: FilterValue::String(user.to_owned()),
        }
    }

    #[test]
    fn rls_injects_owner_when_no_user_filter() {
        let f = search_rls_filter_expr(None, Some("owner"), Some("alice")).unwrap();
        assert_eq!(f, Some(owner_eq("alice")));
    }

    #[test]
    fn rls_ands_owner_onto_user_filter() {
        let f = search_rls_filter_expr(Some("status = \"active\""), Some("owner"), Some("alice"))
            .unwrap();
        let user_f = FilterExpr::Eq {
            field: "status".to_owned(),
            value: FilterValue::String("active".to_owned()),
        };
        assert_eq!(
            f,
            Some(FilterExpr::And(
                Box::new(user_f),
                Box::new(owner_eq("alice")),
            ))
        );
    }

    #[test]
    fn rls_none_user_bypasses_injection() {
        // Admin/system caller: parsed user filter passes through unchanged.
        let f = search_rls_filter_expr(Some("status = \"active\""), Some("owner"), None).unwrap();
        assert_eq!(
            f,
            Some(FilterExpr::Eq {
                field: "status".to_owned(),
                value: FilterValue::String("active".to_owned()),
            })
        );
        let f2 = search_rls_filter_expr(None, Some("owner"), None).unwrap();
        assert_eq!(f2, None);
    }

    #[test]
    fn rls_no_owner_field_no_injection() {
        let f = search_rls_filter_expr(None, None, Some("alice")).unwrap();
        assert_eq!(f, None);
        let f2 = search_rls_filter_expr(Some("status = \"active\""), None, Some("alice")).unwrap();
        assert_eq!(
            f2,
            Some(FilterExpr::Eq {
                field: "status".to_owned(),
                value: FilterValue::String("active".to_owned()),
            })
        );
    }

    #[test]
    fn rls_invalid_user_filter_errors() {
        let r = search_rls_filter_expr(Some("= = ="), Some("owner"), Some("alice"));
        assert!(r.is_err());
    }

    fn sample_fields() -> serde_json::Map<String, serde_json::Value> {
        serde_json::json!({
            "title": "Hello",
            "body": "World",
            "author": "alice",
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn a_records_embedding_is_summarised_not_dumped_to_the_model() {
        // `search` already skipped vector fields when hydrating hits;
        // `filter_query` and `get_record` did not, because both go
        // through `record_json`. On a 5-record collection that meant two
        // matching records arrived behind ~2k float components and the
        // model answered from the first one only. Retrieval was correct
        // and the answer was wrong purely from burial.
        use veda_core::record::{FieldValue, Record};
        use veda_core::vector::DenseVector;

        let mut rec = Record::new();
        rec.set("name", FieldValue::String("Nordwind".to_owned()));
        rec.set(
            "vec",
            FieldValue::Vector(DenseVector::new(vec![0.5f32; 1024])),
        );

        let json = record_json(&rec);
        let fields = json["fields"].as_object().unwrap();

        assert_eq!(fields["name"], "Nordwind", "real fields must survive");
        assert!(
            !fields["vec"].is_array(),
            "the raw component list must not reach the model: {}",
            fields["vec"]
        );
        assert_eq!(
            fields["vec"]["vector"]["dim"], 1024,
            "but the model should still be told the field exists and its shape"
        );

        // The whole payload must stay small. The pre-fix version of this
        // record serialised to roughly 8 KB.
        let len = serde_json::to_string(&json).unwrap().len();
        assert!(len < 400, "record json unexpectedly large: {len} bytes");
    }

    #[test]
    fn project_fields_keeps_only_requested() {
        let mut fields = sample_fields();
        project_fields(&mut fields, &Some(vec!["title".to_owned()]));
        assert_eq!(fields.len(), 1);
        assert_eq!(fields.get("title").and_then(|v| v.as_str()), Some("Hello"));
        assert!(!fields.contains_key("body"));
        assert!(!fields.contains_key("author"));
    }

    #[test]
    fn project_fields_none_keeps_all() {
        let mut fields = sample_fields();
        project_fields(&mut fields, &None);
        assert_eq!(fields.len(), 3);
        assert!(fields.contains_key("title"));
        assert!(fields.contains_key("body"));
        assert!(fields.contains_key("author"));
    }

    #[test]
    fn project_fields_absent_name_omitted_no_error() {
        let mut fields = sample_fields();
        project_fields(
            &mut fields,
            &Some(vec!["title".to_owned(), "missing".to_owned()]),
        );
        assert_eq!(fields.len(), 1);
        assert_eq!(fields.get("title").and_then(|v| v.as_str()), Some("Hello"));
        assert!(!fields.contains_key("missing"));
    }

    #[test]
    fn search_spec_advertises_fields_param() {
        let spec = builtin_specs()
            .into_iter()
            .find(|s| s.id == "search")
            .expect("search spec present");
        let props = &spec.input_schema["properties"];
        // existing params preserved + the new `fields` projection param.
        for p in [
            "collection",
            "query",
            "query_image_b64",
            "mode",
            "field",
            "limit",
            "filter",
            "fields",
        ] {
            assert!(props.get(p).is_some(), "search schema missing '{p}'");
        }
        assert_eq!(props["fields"]["type"], "array");
        assert_eq!(props["fields"]["items"]["type"], "string");
    }
}
