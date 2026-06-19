//! Agent-to-Agent (A2A) endpoint.
//!
//! Exposes a Veda agent as an A2A peer so external orchestrators can
//! discover and consume it. This is a pragmatic, **synchronous-only**
//! subset of the A2A protocol (matching Elastic's sync-only A2A): an
//! [AgentCard] for discovery plus a JSON-RPC 2.0 `message/send` for a
//! one-shot turn. Streaming and push notifications are advertised as
//! unsupported.
//!
//! The actual agent turn reuses the same plumbing as `converse` via
//! [`super::agents::run_agent_turn`] — no LLM loop is duplicated here.
//!
//! Endpoints (protected router; auth + tenant middleware apply):
//!   GET  /v1/a2a/{agent_id}/card   discovery AgentCard
//!   POST /v1/a2a/{agent_id}        JSON-RPC 2.0 (`message/send`)

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use super::agents::{run_agent_turn, AgentTurnError};
use super::collections::ErrorResponse;
use super::state::AppState;
use crate::agent::model::default_agent;
use crate::auth::Claims;
use crate::tenant::{self, Tenant};

/// Resolve the agent (tenant-scoped, with the `veda-agent` default
/// fallback). Mirrors the resolution in `converse` / `run_agent_turn`.
async fn resolve_agent(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    agent_id: &str,
) -> Option<crate::agent::model::Agent> {
    let scoped = tenant::scope(tenant, agent_id);
    match crate::agent::store::get_agent(&state.inner.saved_objects, &scoped).await {
        Some(a) => Some(a),
        None if agent_id == "veda-agent" => Some(default_agent()),
        None if agent_id == "veda-web" => Some(crate::agent::model::default_web_agent()),
        None => None,
    }
}

// ── AgentCard (discovery) ─────────────────────────────────────────────────

/// `GET /v1/a2a/{agent_id}/card` — return an A2A AgentCard describing the
/// agent's identity, the A2A endpoint URL, capabilities (sync-only), and
/// the skills derived from the agent's whitelisted tools.
pub async fn agent_card(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(agent_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let agent = resolve_agent(&state, &tenant, &agent_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("agent '{agent_id}' not found"),
            }),
        )
    })?;

    // Skills are the agent's tool specs (empty whitelist = all builtins),
    // mapped one-to-one onto A2A skills.
    let skills: Vec<serde_json::Value> = state
        .inner
        .tool_catalog
        .specs_for(&agent.tool_ids)
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "name": s.title.clone().unwrap_or_else(|| s.name.clone()),
                "description": s.description,
                "tags": ["tool"],
            })
        })
        .collect();

    let card = serde_json::json!({
        "name": agent.name,
        "description": agent.description,
        "version": env!("CARGO_PKG_VERSION"),
        "url": format!("/v1/a2a/{agent_id}"),
        "capabilities": {
            "streaming": false,
            "pushNotifications": false,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
    });
    Ok(Json(card))
}

// ── JSON-RPC message/send ─────────────────────────────────────────────────

/// `POST /v1/a2a/{agent_id}` — JSON-RPC 2.0. The only supported method is
/// `message/send`: run a one-shot agent turn over the user message's text
/// parts and return the agent's reply as an A2A Message.
///
/// - No LLM configured → HTTP 503 (consistent with `converse`).
/// - Unknown agent (non-default) → HTTP 404.
/// - Unknown JSON-RPC method → JSON-RPC error `-32601`.
/// - Malformed params → JSON-RPC error `-32602`.
pub async fn message_send(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(agent_id): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    Json(req): Json<serde_json::Value>,
) -> axum::response::Response {
    let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

    if method != "message/send" {
        return Json(jsonrpc_error(
            id,
            -32601,
            format!("method not found: {method}"),
        ))
        .into_response();
    }

    // Extract and concatenate the text parts of the user message.
    let input = match extract_text(&req) {
        Some(text) => text,
        None => {
            return Json(jsonrpc_error(
                id,
                -32602,
                "invalid params: message.parts must contain at least one text part",
            ))
            .into_response();
        }
    };

    let claims = claims.map(|axum::Extension(c)| c);
    let user = super::agents::rls_user(&claims);

    // A2A does not persist a Veda conversation, so no conversation id is
    // threaded into agent memory here.
    match run_agent_turn(&state, &tenant, user.as_deref(), &agent_id, &[], &input, None).await {
        Ok((_agent, result, citations, grounding, _cache_hit, _guardrails)) => {
            let message = serde_json::json!({
                "role": "agent",
                "parts": [ { "kind": "text", "text": result.response } ],
                "kind": "message",
                "metadata": {
                    "steps": result.steps,
                    "citations": citations,
                    "grounding": grounding,
                },
            });
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": message,
            }))
            .into_response()
        }
        Err(AgentTurnError::NoLlm) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "VEDA_LLM_ENDPOINT not configured. Set the env var to enable A2A."
                    .to_owned(),
            }),
        )
            .into_response(),
        Err(AgentTurnError::NotFound(aid)) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("agent '{aid}' not found"),
            }),
        )
            .into_response(),
    }
}

/// Concatenate the `text` of every text part in `params.message.parts`.
/// Returns `None` if there is no message or no text part at all.
fn extract_text(req: &serde_json::Value) -> Option<String> {
    let parts = req
        .get("params")?
        .get("message")?
        .get("parts")?
        .as_array()?;
    let mut texts: Vec<&str> = Vec::new();
    for p in parts {
        if p.get("kind").and_then(|k| k.as_str()) == Some("text") {
            if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                texts.push(t);
            }
        }
    }
    if texts.is_empty() {
        return None;
    }
    Some(texts.join(""))
}

fn jsonrpc_error(id: serde_json::Value, code: i32, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() },
    })
}
