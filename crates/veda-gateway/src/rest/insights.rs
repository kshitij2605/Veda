//! Admin REST API for the ops advisor ("Insights").
//!
//! All endpoints are admin-only — registered under the same protected
//! router as `/v1/admin/jobs`, with an additional `rbac::require_admin`
//! route layer so only `Role::Admin` callers reach them.
//!
//!   GET    /v1/admin/insights                 — open events (filterable)
//!   GET    /v1/admin/insights/history         — closed events (bounded)
//!   GET    /v1/admin/insights/rules           — rule catalog + effective config
//!   PUT    /v1/admin/insights/rules/{id}      — merge partial RuleConfig
//!   GET    /v1/admin/insights/channels        — channel config, secrets REDACTED
//!   PUT    /v1/admin/insights/channels        — set channel config (write-only secrets)
//!   POST   /v1/admin/insights/run             — force one advisor tick
//!   POST   /v1/admin/insights/{event_id}/ack  — acknowledge an open event
//!
//! Secrets (webhook/slack URLs, PagerDuty routing key, SMTP password)
//! are NEVER returned by the GET endpoints — [`redact_channels`] maps the
//! stored [`ChannelsConfig`] to a secret-free view.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::advisor::anomaly::anomaly_rule_ids;
use crate::advisor::config::{ChannelsConfig, RuleConfig};
use crate::advisor::engine::{engine_rule_ids, run_tick, save_config};
use crate::advisor::event::{OpsEvent, Severity};
use crate::advisor::rules::catalog;
use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;

// ---------------------------------------------------------------------
// GET /v1/admin/insights  — open events
// ---------------------------------------------------------------------

/// Optional filters for the open-events listing.
#[derive(Debug, Deserialize)]
pub struct InsightsQuery {
    /// `low` | `medium` | `high` — only events at this severity.
    pub severity: Option<String>,
    /// Scope kind: `global` | `collection` | `shard` | `node`.
    pub scope: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EventsResponse {
    pub events: Vec<OpsEvent>,
}

fn parse_severity(s: &str) -> Option<Severity> {
    match s.to_ascii_lowercase().as_str() {
        "low" => Some(Severity::Low),
        "medium" => Some(Severity::Medium),
        "high" => Some(Severity::High),
        _ => None,
    }
}

/// Scope kind string for filtering, matching the serde tag.
fn scope_kind(ev: &OpsEvent) -> &'static str {
    use crate::advisor::event::Scope;
    match ev.scope {
        Scope::Global => "global",
        Scope::Collection(_) => "collection",
        Scope::Shard { .. } => "shard",
        Scope::Node(_) => "node",
        Scope::Slo(_) => "slo",
    }
}

/// `GET /v1/admin/insights` — open events, severity-desc, optionally
/// filtered by `?severity=` and/or `?scope=`.
pub async fn list_insights(
    State(state): State<AppState>,
    Query(q): Query<InsightsQuery>,
) -> Json<EventsResponse> {
    let want_sev = q.severity.as_deref().and_then(parse_severity);
    let want_scope = q.scope.as_deref().map(|s| s.to_ascii_lowercase());

    let events = state.advisor().events.lock().await.open_events();
    let events = events
        .into_iter()
        .filter(|e| want_sev.is_none_or(|s| e.severity == s))
        .filter(|e| {
            want_scope
                .as_deref()
                .is_none_or(|k| scope_kind(e) == k)
        })
        .collect();
    Json(EventsResponse { events })
}

// ---------------------------------------------------------------------
// GET /v1/admin/insights/history — closed events
// ---------------------------------------------------------------------

/// `GET /v1/admin/insights/history` — the bounded ring of closed events,
/// newest-first.
pub async fn list_history(State(state): State<AppState>) -> Json<EventsResponse> {
    let events: Vec<OpsEvent> = state
        .advisor()
        .events
        .lock()
        .await
        .history()
        .iter()
        .rev()
        .cloned()
        .collect();
    Json(EventsResponse { events })
}

// ---------------------------------------------------------------------
// GET /v1/admin/insights/rules — catalog + effective config
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct RuleInfo {
    pub id: String,
    /// Natural severity for threshold rules; `None` for anomaly
    /// detectors (which pick severity by deviation magnitude).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_severity: Option<Severity>,
    /// `true` for the statistical anomaly detectors.
    pub anomaly: bool,
    pub effective_config: RuleConfig,
}

#[derive(Debug, Serialize)]
pub struct RulesResponse {
    pub rules: Vec<RuleInfo>,
}

/// `GET /v1/admin/insights/rules` — every threshold rule (with its
/// natural severity) plus every anomaly detector, each with the config
/// that would currently apply (`AdvisorConfig::for_rule`).
pub async fn list_rules(State(state): State<AppState>) -> Json<RulesResponse> {
    let cfg = state.advisor().config.read().await.clone();
    let mut rules = Vec::new();
    for rule in catalog() {
        rules.push(RuleInfo {
            id: rule.id().to_owned(),
            default_severity: Some(rule.default_severity()),
            anomaly: false,
            effective_config: cfg.for_rule(rule.id()),
        });
    }
    for id in anomaly_rule_ids() {
        rules.push(RuleInfo {
            id: (*id).to_owned(),
            default_severity: None,
            anomaly: true,
            effective_config: cfg.for_rule(id),
        });
    }
    // Engine-level findings (e.g. `new_error_category`) — fixed Medium
    // severity, tunable/mutable via the same per-rule config.
    for id in engine_rule_ids() {
        rules.push(RuleInfo {
            id: (*id).to_owned(),
            default_severity: Some(Severity::Medium),
            anomaly: false,
            effective_config: cfg.for_rule(id),
        });
    }
    Json(RulesResponse { rules })
}

// ---------------------------------------------------------------------
// PUT /v1/admin/insights/rules/{id} — merge partial RuleConfig
// ---------------------------------------------------------------------

/// Partial rule-config patch. Absent fields leave the stored value
/// unchanged; present fields overwrite it.
#[derive(Debug, Deserialize)]
pub struct RuleConfigPatch {
    pub enabled: Option<bool>,
    pub muted: Option<bool>,
    pub severity_override: Option<Option<Severity>>,
    pub thresholds: Option<serde_json::Value>,
    pub auto_remediate: Option<bool>,
}

/// `PUT /v1/admin/insights/rules/{id}` — merge the patch into
/// `config.rules[id]` (creating it from defaults if absent), persist the
/// whole [`AdvisorConfig`] to disk, and return the new effective config.
pub async fn put_rule(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(patch): Json<RuleConfigPatch>,
) -> Json<RuleConfig> {
    let data_dir = state.inner.data_dir.clone();
    let advisor = state.advisor();
    let mut cfg = advisor.config.write().await;

    let entry = cfg.rules.entry(id).or_default();
    if let Some(v) = patch.enabled {
        entry.enabled = v;
    }
    if let Some(v) = patch.muted {
        entry.muted = v;
    }
    if let Some(v) = patch.severity_override {
        entry.severity_override = v;
    }
    if let Some(v) = patch.thresholds {
        entry.thresholds = v;
    }
    if let Some(v) = patch.auto_remediate {
        entry.auto_remediate = v;
    }
    let effective = entry.clone();

    save_config(&data_dir, &cfg);
    Json(effective)
}

// ---------------------------------------------------------------------
// channels — GET (redacted) / PUT (write-only secrets)
// ---------------------------------------------------------------------

/// A secret-free view of one channel: whether it is configured + enabled,
/// its severity floor, and any non-secret addressing bits.
#[derive(Debug, Serialize)]
pub struct ChannelView {
    pub configured: bool,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_severity: Option<Severity>,
    /// Non-secret SMTP addressing for the email channel (host/port/from/
    /// recipient count) — password + username are never returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<EmailView>,
}

#[derive(Debug, Serialize)]
pub struct EmailView {
    pub smtp_host: String,
    pub smtp_port: u16,
    pub from: String,
    pub recipients: usize,
}

#[derive(Debug, Serialize)]
pub struct ChannelsView {
    pub webhook: ChannelView,
    pub slack: ChannelView,
    pub pagerduty: ChannelView,
    pub email: ChannelView,
}

/// Map stored channel config to a secret-free view. URLs, the PagerDuty
/// routing key, and the SMTP password/username are intentionally omitted.
fn redact_channels(c: &ChannelsConfig) -> ChannelsView {
    ChannelsView {
        webhook: match &c.webhook {
            Some(w) => ChannelView {
                configured: true,
                enabled: w.enabled,
                min_severity: Some(w.min_severity),
                email: None,
            },
            None => ChannelView {
                configured: false,
                enabled: false,
                min_severity: None,
                email: None,
            },
        },
        slack: match &c.slack {
            Some(s) => ChannelView {
                configured: true,
                enabled: s.enabled,
                min_severity: Some(s.min_severity),
                email: None,
            },
            None => ChannelView {
                configured: false,
                enabled: false,
                min_severity: None,
                email: None,
            },
        },
        pagerduty: match &c.pagerduty {
            Some(p) => ChannelView {
                configured: true,
                enabled: p.enabled,
                min_severity: Some(p.min_severity),
                email: None,
            },
            None => ChannelView {
                configured: false,
                enabled: false,
                min_severity: None,
                email: None,
            },
        },
        email: match &c.email {
            Some(e) => ChannelView {
                configured: true,
                enabled: e.enabled,
                min_severity: Some(e.min_severity),
                email: Some(EmailView {
                    smtp_host: e.smtp_host.clone(),
                    smtp_port: e.smtp_port,
                    from: e.from.clone(),
                    recipients: e.to.len(),
                }),
            },
            None => ChannelView {
                configured: false,
                enabled: false,
                min_severity: None,
                email: None,
            },
        },
    }
}

/// `GET /v1/admin/insights/channels` — channel config with all secrets
/// redacted (no URLs, routing key, or password).
pub async fn get_channels(State(state): State<AppState>) -> Json<ChannelsView> {
    let cfg = state.advisor().config.read().await;
    Json(redact_channels(&cfg.channels))
}

/// `PUT /v1/admin/insights/channels` — replace the channel config
/// (accepts secrets write-only), persist, and return the redacted view.
pub async fn put_channels(
    State(state): State<AppState>,
    Json(channels): Json<ChannelsConfig>,
) -> Json<ChannelsView> {
    let data_dir = state.inner.data_dir.clone();
    let advisor = state.advisor();
    let mut cfg = advisor.config.write().await;
    cfg.channels = channels;
    let view = redact_channels(&cfg.channels);
    save_config(&data_dir, &cfg);
    Json(view)
}

// ---------------------------------------------------------------------
// POST /v1/admin/insights/run — force one tick
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub opened: usize,
    pub closed: usize,
}

/// `POST /v1/admin/insights/run` — force a single advisor tick and
/// report how many events opened/closed this round.
pub async fn run_now(State(state): State<AppState>) -> Json<RunResponse> {
    let r = run_tick(&state).await;
    Json(RunResponse {
        opened: r.opened.len(),
        closed: r.closed.len(),
    })
}

// ---------------------------------------------------------------------
// POST /v1/admin/insights/{event_id}/ack
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AckResponse {
    pub acknowledged: bool,
}

/// `POST /v1/admin/insights/{event_id}/ack` — acknowledge an open event,
/// persisting the change. 404 when no open event has that id.
pub async fn ack_event(
    State(state): State<AppState>,
    Path(event_id): Path<String>,
) -> Result<Json<AckResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut events = state.advisor().events.lock().await;
    if events.acknowledge(&event_id) {
        events.save();
        Ok(Json(AckResponse { acknowledged: true }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("no open event with id '{event_id}'"),
            }),
        ))
    }
}
