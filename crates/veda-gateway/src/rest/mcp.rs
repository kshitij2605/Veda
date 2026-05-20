//! Model Context Protocol (MCP) server endpoint.
//!
//! Implements the JSON-RPC 2.0 surface that AI agents (Claude Desktop,
//! VS Code MCP clients, etc.) use to discover and call Veda's tools. The
//! transport is plain HTTP — each request is one JSON-RPC message.
//!
//! Spec: <https://spec.modelcontextprotocol.io/>
//!
//! ## Tools exposed
//!
//! - `list_collections` — list all collections by name.
//! - `describe_collection` — get schema + record count for a named collection.
//! - `search` — text query auto-embedded via configured text model, returns
//!   top-k hits with id + distance.
//! - `upsert_record` — insert one record (caller supplies fields as JSON).
//!
//! ## Notes
//!
//! - Auth: when JWT auth is enabled, the MCP endpoint requires the same
//!   `Authorization: Bearer ...` header as other protected routes; auth is
//!   handled by the gateway-wide middleware.
//! - Errors follow JSON-RPC 2.0: `{ "jsonrpc": "2.0", "id": ..., "error":
//!   { "code": -32600|-32601|-32602|-32603, "message": ... } }`.

use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::Record;
use veda_core::types::{DistanceMetric, FieldType};
use veda_query::search::VectorSearchRequest;

use super::state::AppState;

const PROTOCOL_VERSION: &str = "2025-03-26";

// ---------------------------------------------------------------------------
// JSON-RPC envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

fn ok(id: serde_json::Value, result: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    }
}

fn err(id: serde_json::Value, code: i32, message: impl Into<String>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn handle_mcp(
    State(state): State<AppState>,
    crate::tenant::Tenant(tenant): crate::tenant::Tenant,
    Json(req): Json<JsonRpcRequest>,
) -> (StatusCode, Json<JsonRpcResponse>) {
    let id = req.id.unwrap_or(serde_json::Value::Null);
    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(),
        "tools/list" => handle_tools_list(),
        "tools/call" => handle_tools_call(&state, &tenant, req.params).await,
        "ping" => Ok(serde_json::json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    };

    let body = match resp {
        Ok(result) => ok(id, result),
        Err((code, msg)) => err(id, code, msg),
    };
    (StatusCode::OK, Json(body))
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

fn handle_initialize() -> Result<serde_json::Value, (i32, String)> {
    Ok(serde_json::json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {
            "name": "veda",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "capabilities": {
            "tools": { "listChanged": false }
        }
    }))
}

// ---------------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------------

fn handle_tools_list() -> Result<serde_json::Value, (i32, String)> {
    Ok(serde_json::json!({
        "tools": [
            {
                "name": "list_collections",
                "description": "List all Veda collections by name.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "describe_collection",
                "description": "Return the schema and total record count for a Veda collection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Collection name" }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "search",
                "description": "Text search over a Veda collection. The query is auto-embedded using the collection's configured text embedding model. Returns the top-k nearest records by vector similarity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": { "type": "string", "description": "Collection to search" },
                        "query": { "type": "string", "description": "Natural-language query text" },
                        "field": { "type": "string", "description": "Vector field name (optional; defaults to the first vector field on the collection)" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                        "filter": { "type": "string", "description": "Optional SQL-like filter expression, e.g. \"category = 'electronics' AND price < 500\"" }
                    },
                    "required": ["collection", "query"]
                }
            },
            {
                "name": "upsert_record",
                "description": "Insert one record into a Veda collection. Fields must match the collection's schema. Vector fields can be supplied as float arrays, or omitted and computed via auto-embedding (call embed-insert instead for that flow).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection": { "type": "string" },
                        "record": { "type": "object", "description": "Field name → value map." }
                    },
                    "required": ["collection", "record"]
                }
            }
        ]
    }))
}

// ---------------------------------------------------------------------------
// tools/call dispatcher
// ---------------------------------------------------------------------------

async fn handle_tools_call(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    params: Option<serde_json::Value>,
) -> Result<serde_json::Value, (i32, String)> {
    let params = params.ok_or((-32602, "missing params".into()))?;
    let tool_name = params["name"].as_str().ok_or((-32602, "missing tool name".into()))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let content = match tool_name {
        "list_collections" => tool_list_collections(state, tenant).await?,
        "describe_collection" => tool_describe_collection(state, tenant, &args).await?,
        "search" => tool_search(state, tenant, &args).await?,
        "upsert_record" => tool_upsert_record(state, tenant, &args).await?,
        other => return Err((-32601, format!("unknown tool: {other}"))),
    };

    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&content).unwrap_or_default() }],
        "isError": false
    }))
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

async fn tool_list_collections(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
) -> Result<serde_json::Value, (i32, String)> {
    let schemas = state.inner.schemas.read().await;
    let names: Vec<String> = schemas
        .keys()
        .filter_map(|k| crate::tenant::strip_prefix(tenant, k).map(str::to_owned))
        .collect();
    Ok(serde_json::json!({ "collections": names, "count": names.len() }))
}

async fn tool_describe_collection(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i32, String)> {
    let user_name = args["name"]
        .as_str()
        .ok_or((-32602, "name is required".into()))?;
    let name = crate::tenant::scope(tenant, user_name);
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas
            .get(&name)
            .cloned()
            .ok_or((-32602, format!("collection '{user_name}' not found")))?
    };
    let managers = state.inner.segment_managers.read().await;
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

    Ok(serde_json::json!({
        "name": user_name,
        "fields": fields,
        "record_count": total,
        "shard_count": schema.shard_count(),
    }))
}

async fn tool_search(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i32, String)> {
    let user_collection = args["collection"]
        .as_str()
        .ok_or((-32602, "collection is required".into()))?;
    let collection = crate::tenant::scope(tenant, user_collection);
    let collection = collection.as_str();
    let query = args["query"]
        .as_str()
        .ok_or((-32602, "query is required".into()))?;
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let filter = args["filter"].as_str().map(|s| s.to_owned());

    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas
            .get(collection)
            .cloned()
            .ok_or((-32602, format!("collection '{collection}' not found")))?
    };

    // Resolve target vector field.
    let vec_fields = schema.vector_fields();
    let field_name = if let Some(f) = args["field"].as_str() {
        f.to_owned()
    } else if let Some(first) = vec_fields.first() {
        first.name().to_owned()
    } else {
        return Err((-32602, format!("collection '{collection}' has no vector fields")));
    };
    let field_def = schema
        .get_field(&field_name)
        .ok_or((-32602, format!("field '{field_name}' not found")))?;
    let metric = match field_def.field_type() {
        FieldType::Vector { metric, .. } => *metric,
        _ => DistanceMetric::Cosine,
    };

    // Find a text-supporting embedding model. Prefer the one bound to this
    // field if any; otherwise auto-select.
    let bound_model = match field_def.field_type() {
        FieldType::Vector { model_id, .. } => model_id.clone(),
        _ => None,
    };
    let model = {
        let registry = state.inner.embedding_registry.read().await;
        if let Some(name) = bound_model {
            registry
                .get(&name)
                .cloned()
                .ok_or((-32603, format!("bound model '{name}' not in registry")))?
        } else {
            registry
                .find_by_modality(Modality::Text)
                .first()
                .map(|c| (*c).clone())
                .ok_or((-32603, "no text embedding model configured".into()))?
        }
    };

    let client = EmbeddingClient::new(model);
    let vector = client
        .embed_text(query)
        .map_err(|e| (-32603, format!("embedding failed: {e}")))?;

    let executors = state.inner.query_executors.read().await;
    let executor = executors
        .get(collection)
        .ok_or((-32603, "no query executor".into()))?;
    let request = VectorSearchRequest {
        field: field_name.clone(),
        query_vector: vector,
        k: limit,
        metric,
        filter,
        truncate_to: None,
        read_ts: None,
    };
    let (response, _) = executor
        .search(&request)
        .map_err(|e| (-32603, format!("search failed: {e}")))?;

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
    Ok(serde_json::json!({ "hits": hits, "count": hits.len() }))
}

async fn tool_upsert_record(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    args: &serde_json::Value,
) -> Result<serde_json::Value, (i32, String)> {
    let user_collection = args["collection"]
        .as_str()
        .ok_or((-32602, "collection is required".into()))?;
    let collection = crate::tenant::scope(tenant, user_collection);
    let collection = collection.as_str();
    let raw = args["record"]
        .as_object()
        .ok_or((-32602, "record must be an object".into()))?;

    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas
            .get(collection)
            .cloned()
            .ok_or((-32602, format!("collection '{collection}' not found")))?
    };
    let record: Record = super::records::parse_record_public(raw, &schema)
        .map_err(|e| (-32602, format!("parse_record: {e}")))?;

    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers
        .get_mut(collection)
        .ok_or((-32603, "segment managers missing".into()))?;
    let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
        &record.id(),
        schema.shard_count(),
    );
    let sm = shard_map
        .get_mut(&shard_id)
        .ok_or((-32603, format!("shard {shard_id} not found")))?;
    sm.insert(record)
        .map_err(|e| (-32603, format!("insert failed: {e}")))?;
    let id = sm.growing_segment().count(); // approximate

    Ok(serde_json::json!({ "ok": true, "doc_id_estimate": id }))
}
