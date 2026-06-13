//! Built-in tool implementations.
//!
//! Each builtin is a `ToolSpec` (id, docs, JSON-Schema) plus an async
//! impl `(&ToolContext, args) -> ToolOutcome`. The logic here is the
//! single source of truth for both the MCP server and the agent
//! executor — the MCP endpoint renders/dispatches a subset of these via
//! the catalog rather than carrying its own copies.

use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::Record;
use veda_core::types::{DistanceMetric, FieldType};
use veda_query::adaptive::route_query;
use veda_query::search::VectorSearchRequest;

use crate::agent::catalog::{ToolContext, ToolOutcome, ToolSpec};
use crate::rest::records::{field_value_to_json, parse_record_public};

/// The full set of built-in tool specs (id == MCP/OpenAI tool name).
pub fn builtin_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            id: "list_collections".to_owned(),
            name: "list_collections".to_owned(),
            description: "List all Veda collections by name.".to_owned(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            kind: "builtin".to_owned(),
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
        },
        ToolSpec {
            id: "search".to_owned(),
            name: "search".to_owned(),
            description: "Text search over a Veda collection. The query is auto-embedded using the collection's configured text embedding model. Returns the top-k nearest records by vector similarity.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection to search" },
                    "query": { "type": "string", "description": "Natural-language query text" },
                    "field": { "type": "string", "description": "Vector field name (optional; defaults to the first vector field on the collection)" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                    "filter": { "type": "string", "description": "Optional SQL-like filter expression, e.g. \"category = 'electronics' AND price < 500\"" }
                },
                "required": ["collection", "query"]
            }),
            kind: "builtin".to_owned(),
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
        },
    ]
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
            serde_json::json!({
                "name": f.name(),
                "type": kind,
                "filterable": f.filterable(),
            })
        })
        .collect();

    ToolOutcome::ok(serde_json::json!({
        "name": user_name,
        "fields": fields,
        "record_count": total,
        "shard_count": schema.shard_count(),
    }))
}

pub async fn search(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(user_collection) = args["collection"].as_str() else {
        return ToolOutcome::invalid_args("collection is required");
    };
    let scoped = crate::tenant::scope(ctx.tenant, user_collection);
    let collection = scoped.as_str();
    let Some(query) = args["query"].as_str() else {
        return ToolOutcome::invalid_args("query is required");
    };
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let filter = args["filter"].as_str().map(|s| s.to_owned());

    let schema = {
        let schemas = ctx.state.inner.schemas.read().await;
        match schemas.get(collection).cloned() {
            Some(s) => s,
            None => return ToolOutcome::not_found(format!("collection '{collection}' not found")),
        }
    };

    // Resolve target vector field.
    let vec_fields = schema.vector_fields();
    let field_name = if let Some(f) = args["field"].as_str() {
        f.to_owned()
    } else if let Some(first) = vec_fields.first() {
        first.name().to_owned()
    } else {
        return ToolOutcome::invalid_args(format!("collection '{collection}' has no vector fields"));
    };
    let Some(field_def) = schema.get_field(&field_name) else {
        return ToolOutcome::invalid_args(format!("field '{field_name}' not found"));
    };
    let metric = match field_def.field_type() {
        FieldType::Vector { metric, .. } => *metric,
        _ => DistanceMetric::Cosine,
    };

    // Find a text-supporting embedding model. Prefer the one bound to
    // this field if any; otherwise auto-select.
    let bound_model = match field_def.field_type() {
        FieldType::Vector { model_id, .. } => model_id.clone(),
        _ => None,
    };
    let model = {
        let registry = ctx.state.inner.embedding_registry.read().await;
        if let Some(name) = bound_model {
            match registry.get(&name).cloned() {
                Some(m) => m,
                None => {
                    return ToolOutcome::internal(format!("bound model '{name}' not in registry"))
                }
            }
        } else {
            match registry
                .find_by_modality(Modality::Text)
                .first()
                .map(|c| (*c).clone())
            {
                Some(m) => m,
                None => return ToolOutcome::internal("no text embedding model configured"),
            }
        }
    };

    let client = EmbeddingClient::new(model);
    let vector = match client.embed_text(query) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::internal(format!("embedding failed: {e}")),
    };

    let executors = ctx.state.inner.query_executors.read().await;
    let Some(executor) = executors.get(collection) else {
        return ToolOutcome::internal("no query executor");
    };
    let request = VectorSearchRequest {
        field: field_name.clone(),
        query_vector: vector,
        k: limit,
        metric,
        filter,
        filter_expr: None,
        truncate_to: None,
        read_ts: None,
    };
    let response = match executor.search(&request) {
        Ok((response, _)) => response,
        Err(e) => return ToolOutcome::internal(format!("search failed: {e}")),
    };

    let hits: Vec<serde_json::Value> = response
        .results
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id.to_string(),
                "distance": r.distance,
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
            None => return ToolOutcome::not_found(format!("collection '{user_collection}' not found")),
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
                if !crate::rest::records::record_passes_rls_owner(rec, rls_field.as_deref(), rls_owner) {
                    return ToolOutcome::not_found("record not found");
                }
                return ToolOutcome::ok(record_json(rec));
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
                    if !crate::rest::records::record_passes_rls_owner(&rec, rls_field.as_deref(), rls_owner) {
                        return ToolOutcome::not_found("record not found");
                    }
                    return ToolOutcome::ok(record_json(&rec));
                }
            }
        }
    }

    ToolOutcome::not_found(format!("record '{record_id}' not found"))
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

/// Serialise a record to `{ "id", "fields" }` — matching the REST
/// `GetRecordResponse` shape so the agent/MCP surfaces stay consistent.
fn record_json(rec: &Record) -> serde_json::Value {
    let mut fields = serde_json::Map::new();
    for (key, value) in rec.fields() {
        fields.insert(key.clone(), field_value_to_json(value));
    }
    serde_json::json!({ "id": rec.id().to_string(), "fields": fields })
}
