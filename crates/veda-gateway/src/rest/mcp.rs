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

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::agent::catalog::ToolContext;

use super::state::AppState;

const PROTOCOL_VERSION: &str = "2025-03-26";

/// Tool ids exposed over the MCP transport. The catalog holds more
/// builtins (e.g. `get_record`, `route`); MCP intentionally surfaces
/// only this original set so external clients see an unchanged tool
/// list.
const MCP_EXPOSED_TOOLS: [&str; 4] = [
    "list_collections",
    "describe_collection",
    "search",
    "upsert_record",
];

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
    let catalog = &state.inner.tool_catalog;
    let exposed: Vec<String> = MCP_EXPOSED_TOOLS.iter().map(|s| (*s).to_owned()).collect();
    // `specs_for` preserves builtin order, which matches the original
    // hard-coded ordering of the four MCP tools.
    let tools: Vec<serde_json::Value> = catalog
        .specs_for(&exposed)
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

    // Only the original four tools are reachable over MCP, even though
    // the catalog holds more builtins.
    if !MCP_EXPOSED_TOOLS.contains(&tool_name) {
        return Err((-32601, format!("unknown tool: {tool_name}")));
    }

    let ctx = ToolContext::new(state, tenant);
    let outcome = state.inner.tool_catalog.dispatch(&ctx, tool_name, args).await;
    if outcome.is_error {
        let msg = outcome.content["error"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| outcome.content.to_string());
        return Err((-32603, msg));
    }

    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&outcome.content).unwrap_or_default() }],
        "isError": false
    }))
}
