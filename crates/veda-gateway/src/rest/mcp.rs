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
//! - *Protocol* errors (bad JSON-RPC, unknown method, unknown tool name)
//!   follow JSON-RPC 2.0: `{ "jsonrpc": "2.0", "id": ..., "error": { "code":
//!   -32600|-32601|-32602|-32700, "message": ... } }`.
//! - *Tool-execution* errors are NOT JSON-RPC errors: per MCP they are a
//!   successful `result` carrying a `CallToolResult` with `isError: true`,
//!   so the calling model can read the message and self-correct.

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

use crate::agent::catalog::ToolContext;

use super::state::AppState;

const PROTOCOL_VERSION: &str = "2025-06-18";

/// Protocol versions this server can speak. The first entry is preferred.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26"];

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

/// MCP Streamable HTTP POST endpoint (stateless).
///
/// Honors the `MCP-Protocol-Version` header leniently: absent → assume a
/// supported version (proceed); present and unsupported → HTTP 400. The
/// success/error JSON-RPC envelope is returned as `application/json` (the
/// spec allows a plain JSON response when there is no SSE stream). Veda is
/// stateless, so no `Mcp-Session-Id` is issued and any client-sent session id
/// is ignored.
///
/// `HeaderMap` is placed before the `Json` body extractor: in axum the body
/// extractor consumes the request and must come last.
pub async fn handle_mcp(
    State(state): State<AppState>,
    crate::tenant::Tenant(tenant): crate::tenant::Tenant,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> axum::response::Response {
    // Lenient MCP-Protocol-Version check: only reject a header that is
    // *present and unsupported*.
    if let Some(v) = headers.get("mcp-protocol-version") {
        let ok_version = v
            .to_str()
            .ok()
            .map(|s| SUPPORTED_PROTOCOL_VERSIONS.contains(&s))
            .unwrap_or(false);
        if !ok_version {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "unsupported MCP-Protocol-Version",
                    "supported": SUPPORTED_PROTOCOL_VERSIONS,
                })),
            )
                .into_response();
        }
    }

    let id = req.id.unwrap_or(serde_json::Value::Null);
    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(req.params),
        "tools/list" => handle_tools_list(&state),
        "tools/call" => handle_tools_call(&state, &tenant, req.params).await,
        "ping" => Ok(serde_json::json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    };

    let body = match resp {
        Ok(result) => ok(id, result),
        Err((code, msg)) => err(id, code, msg),
    };
    (StatusCode::OK, Json(body)).into_response()
}

/// MCP Streamable HTTP GET endpoint.
///
/// Clients issue `GET /v1/mcp` to open a server→client SSE stream. Veda has
/// no server-initiated messages, so this is unsupported: return 405 with an
/// `Allow: POST` header rather than hanging on an empty stream.
pub async fn handle_mcp_get() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [(axum::http::header::ALLOW, "POST")],
        "GET not supported: Veda's MCP endpoint has no server-initiated stream; use POST.",
    )
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

fn handle_initialize(
    params: Option<serde_json::Value>,
) -> Result<serde_json::Value, (i32, String)> {
    // Negotiate the protocol version: echo the client's requested version
    // if we support it, otherwise fall back to our preferred version.
    let requested = params.as_ref().and_then(|p| p["protocolVersion"].as_str());
    let negotiated = match requested {
        Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v,
        _ => PROTOCOL_VERSION,
    };
    Ok(serde_json::json!({
        "protocolVersion": negotiated,
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
    let tool_name = params["name"]
        .as_str()
        .ok_or((-32602, "missing tool name".into()))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    // Any builtin the catalog knows is callable. An unknown id is
    // dispatched too and surfaces as an error outcome
    // (`"unknown tool: ..."`), which we map to the JSON-RPC -32601
    // protocol error below.
    let ctx = ToolContext::new(state, tenant);
    let outcome = state
        .inner
        .tool_catalog
        .dispatch(&ctx, tool_name, args)
        .await;
    if outcome.is_error {
        let msg = outcome.content["error"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| outcome.content.to_string());
        // An *unknown tool* is a protocol error ("Method not found",
        // -32601) — the tool genuinely doesn't exist. Every other error
        // is a *tool-execution* failure: per MCP, return a successful
        // JSON-RPC result with `isError: true` (a `CallToolResult`) so the
        // calling LLM can read the message and self-correct.
        if msg.starts_with("unknown tool") {
            return Err((-32601, msg));
        }
        // `error_kind` (InvalidArgs | NotFound | Internal) is carried by
        // the outcome but does not change the wire shape — all become an
        // `isError` result.
        return Ok(serde_json::json!({
            "content": [{ "type": "text", "text": msg }],
            "isError": true
        }));
    }

    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&outcome.content).unwrap_or_default() }],
        "structuredContent": outcome.content,
        "isError": false
    }))
}
