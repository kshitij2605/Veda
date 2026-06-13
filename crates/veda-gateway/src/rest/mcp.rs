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
//! The full built-in catalog ([`crate::agent::catalog::ToolCatalog::builtin_specs`])
//! is surfaced — `tools/list` renders every builtin (including its MCP
//! `annotations`) and `tools/call` dispatches any builtin id:
//!
//! - `list_collections`, `describe_collection`, `get_record`, `route`,
//!   `aggregate`, `filter_query` — read-only tools.
//! - `search` — cross-modal (text/image) query auto-embedded into Veda's
//!   unified vector space, returns top-k hits with id + distance.
//! - `upsert_record` — insert one record (mutating).
//!
//! ## Notes
//!
//! - Auth: when JWT auth is enabled, the MCP endpoint requires the same
//!   `Authorization: Bearer ...` header as other protected routes; auth is
//!   handled by the gateway-wide middleware.
//! - Errors follow JSON-RPC 2.0: `{ "jsonrpc": "2.0", "id": ..., "error":
//!   { "code": -32600|-32601|-32602|-32603, "message": ... } }`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::agent::catalog::{ToolContext, ToolErrorKind};

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
        "tools/list" => handle_tools_list(&state),
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

fn handle_tools_list(state: &AppState) -> Result<serde_json::Value, (i32, String)> {
    // Expose the full builtin catalog over MCP — each spec renders its
    // own `{name, description, inputSchema, annotations}` entry.
    let tools: Vec<serde_json::Value> = state
        .inner
        .tool_catalog
        .builtin_specs()
        .iter()
        .map(|s| s.to_mcp_entry())
        .collect();
    Ok(serde_json::json!({ "tools": tools }))
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

    // Any builtin the catalog knows is callable. An unknown id is
    // dispatched too and surfaces as an `Internal` error outcome
    // (`"unknown tool: ..."`), which we map to JSON-RPC -32601 below.
    let ctx = ToolContext::new(state, tenant);
    let outcome = state.inner.tool_catalog.dispatch(&ctx, tool_name, args).await;
    if outcome.is_error {
        let msg = outcome.content["error"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| outcome.content.to_string());
        // Unknown tool ids -> "Method not found" (-32601); argument
        // validation failures -> "Invalid params" (-32602); not-found and
        // other internal failures -> "Internal error" (-32603).
        let code = if msg.starts_with("unknown tool") {
            -32601
        } else {
            match outcome.error_kind {
                Some(ToolErrorKind::InvalidArgs) => -32602,
                Some(ToolErrorKind::NotFound) | Some(ToolErrorKind::Internal) | None => -32603,
            }
        };
        return Err((code, msg));
    }

    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&outcome.content).unwrap_or_default() }],
        "isError": false
    }))
}
