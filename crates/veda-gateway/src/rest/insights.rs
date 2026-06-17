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
use crate::agent::model::AgentStep;
use crate::rest::agents::{run_agent_turn, AgentTurnError};
use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;
use crate::tenant::TenantId;

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
        Scope::Synthetic(_) => "synthetic",
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

// ---------------------------------------------------------------------
// POST /v1/admin/insights/{event_id}/triage — AI-assisted triage
// ---------------------------------------------------------------------

/// Triage result: the agent's explanation plus the full reasoning/acting
/// trace it produced (same step shape as `/v1/agents/{id}/converse`).
#[derive(Debug, Serialize)]
pub struct TriageResponse {
    pub event_id: String,
    pub explanation: String,
    pub steps: Vec<AgentStep>,
    /// Run-level APM metrics (timing + token usage) for the triage turn.
    pub metrics: crate::agent::executor::RunMetrics,
}

/// Find an event by id: open events first, then closed history.
async fn find_event(state: &AppState, event_id: &str) -> Option<OpsEvent> {
    let store = state.advisor().events.lock().await;
    if let Some(ev) = store.open_events().into_iter().find(|e| e.id == event_id) {
        return Some(ev);
    }
    store.history().iter().rev().find(|e| e.id == event_id).cloned()
}

/// Build the triage prompt embedding the event context and asking the
/// ops agent to investigate via tools and produce an explanation + next
/// steps. Pure + testable (no `AppState`).
pub fn build_triage_input(ev: &OpsEvent) -> String {
    format!(
        "An ops alert fired in Veda. Investigate it using your tools (insights, logs, \
resources, SLO status) and explain the most likely ROOT CAUSE and the concrete NEXT STEPS \
the operator should take. Be specific and brief.\n\n\
Alert:\n\
- rule_id: {rule}\n\
- severity: {sev:?}\n\
- scope: {scope}\n\
- title: {title}\n\
- detail: {detail}\n\
- recommendation: {rec}\n\
- opened_at_ms: {opened}",
        rule = ev.rule_id,
        sev = ev.severity,
        scope = ev.scope.key(),
        title = ev.title,
        detail = ev.detail,
        rec = ev.recommendation,
        opened = ev.opened_at_ms,
    )
}

/// `POST /v1/admin/insights/{event_id}/triage` — run the built-in
/// `veda-ops` agent to explain why an alert fired.
///
/// 404 when no event (open or history) has that id; 503 when no LLM is
/// configured (mirrors `/converse`). On success returns the agent's
/// explanation plus its step trace.
pub async fn triage_event(
    State(state): State<AppState>,
    Path(event_id): Path<String>,
) -> Result<Json<TriageResponse>, (StatusCode, Json<ErrorResponse>)> {
    let ev = find_event(&state, &event_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("no event with id '{event_id}'"),
            }),
        )
    })?;

    let prompt = build_triage_input(&ev);

    // Admin caller (no RLS user). Resolve the in-memory `veda-ops` agent
    // and drive the executor via the shared agent-turn helper.
    let tenant = TenantId::default_tenant();
    let (_agent, result) = run_agent_turn(&state, &tenant, None, "veda-ops", &[], &prompt)
        .await
        .map_err(|e| match e {
            AgentTurnError::NoLlm => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "VEDA_LLM_ENDPOINT not configured. Set the env var to enable triage."
                        .to_owned(),
                }),
            ),
            AgentTurnError::NotFound(aid) => (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("agent '{aid}' not found"),
                }),
            ),
        })?;

    Ok(Json(TriageResponse {
        event_id,
        explanation: result.response,
        steps: result.steps,
        metrics: result.metrics,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use veda_core::error::Result as LlmResult;
    use veda_core::llm_client::{
        ChatMessage, ChatModel, CompletionOptions, FunctionCall, ToolCall, ToolCompletion, ToolDef,
    };

    use crate::advisor::event::{RemediationAction, Scope};
    use crate::agent::catalog::{ToolCatalog, ToolContext};
    use crate::agent::executor::{self, DEFAULT_MAX_STEPS};
    use crate::agent::model::default_ops_agent;

    /// Scripted, network-free [`ChatModel`] (mirrors the executor's test
    /// fake): pops queued completions, then a default final answer.
    struct FakeModel {
        queue: RefCell<VecDeque<ToolCompletion>>,
    }
    impl FakeModel {
        fn scripted(items: Vec<ToolCompletion>) -> Self {
            Self {
                queue: RefCell::new(items.into_iter().collect()),
            }
        }
    }
    impl ChatModel for FakeModel {
        async fn complete_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDef],
            _tool_choice: Option<&str>,
            _opts: &CompletionOptions,
        ) -> LlmResult<ToolCompletion> {
            Ok(self.queue.borrow_mut().pop_front().unwrap_or(ToolCompletion {
                content: Some("done".to_owned()),
                tool_calls: Vec::new(),
                finish_reason: Some("stop".to_owned()),
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }))
        }
    }

    fn ev(rule: &str) -> OpsEvent {
        OpsEvent {
            id: OpsEvent::make_id(rule, &Scope::Global),
            rule_id: rule.to_owned(),
            scope: Scope::Global,
            severity: Severity::High,
            status: crate::advisor::event::EventStatus::Open,
            title: "errors spiking".to_owned(),
            detail: "5xx ratio over threshold".to_owned(),
            recommendation: "inspect logs".to_owned(),
            runbook: None,
            remediation: RemediationAction::None,
            opened_at_ms: 1000,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }

    #[test]
    fn build_triage_input_embeds_event_context() {
        let prompt = build_triage_input(&ev("high_error_rate"));
        assert!(prompt.contains("high_error_rate"));
        assert!(prompt.contains("errors spiking"));
        assert!(prompt.contains("5xx ratio over threshold"));
        assert!(prompt.contains("ROOT CAUSE"));
    }

    /// The veda-ops agent, driven by a fake LLM that calls an ops tool
    /// then answers, produces a non-empty explanation + step trace. This
    /// exercises the prompt + the `veda-ops` whitelist + the ops tools +
    /// the executor (the same chain the triage handler runs, minus the
    /// LlmClient construction that the 503/converse paths cover).
    #[tokio::test]
    async fn triage_agent_runs_tool_then_explains() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();

        // Seed one open advisor event so get_insights returns it.
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![crate::advisor::rules::RuleHit {
                    rule_id: "high_error_rate".to_owned(),
                    scope: Scope::Global,
                    severity: Severity::High,
                    title: "errors spiking".to_owned(),
                    detail: "5xx ratio over threshold".to_owned(),
                    recommendation: "inspect logs".to_owned(),
                    runbook: None,
                    remediation: RemediationAction::None,
                }],
                1000,
            );
        }

        let agent = default_ops_agent();
        let event = ev("high_error_rate");
        let prompt = build_triage_input(&event);

        let model = FakeModel::scripted(vec![
            ToolCompletion {
                content: Some("Let me check open alerts.".to_owned()),
                tool_calls: vec![ToolCall {
                    id: "c1".to_owned(),
                    kind: "function".to_owned(),
                    function: FunctionCall {
                        name: "get_insights".to_owned(),
                        arguments: "{}".to_owned(),
                    },
                }],
                finish_reason: Some("tool_calls".to_owned()),
                prompt_tokens: 5,
                completion_tokens: 5,
                total_tokens: 10,
            },
            ToolCompletion {
                content: Some("Root cause: error rate spiked; next: inspect logs.".to_owned()),
                tool_calls: Vec::new(),
                finish_reason: Some("stop".to_owned()),
                prompt_tokens: 3,
                completion_tokens: 7,
                total_tokens: 10,
            },
        ]);

        let cat = ToolCatalog::new();
        let ctx = ToolContext::new(&state, &tenant);
        let sys = executor::build_system_prompt(&agent, "");
        let result = executor::run(
            &model,
            &cat,
            &ctx,
            &agent,
            &sys,
            &[],
            &prompt,
            DEFAULT_MAX_STEPS,
            "openai",
            "test-model",
        )
        .await;

        assert!(!result.hit_limit_or_error);
        assert_eq!(
            result.response,
            "Root cause: error rate spiked; next: inspect logs."
        );
        assert!(!result.steps.is_empty(), "expected a non-empty step trace");
        // The get_insights tool was dispatched and returned our seeded event.
        let saw_tool = result.steps.iter().any(|s| matches!(
            s,
            AgentStep::ToolResult { result, is_error: false, .. }
                if result["events"][0]["rule_id"] == "high_error_rate"
        ));
        assert!(saw_tool, "expected get_insights to surface the seeded event: {:?}", result.steps);
    }

    /// The handler finds the seeded event, then — with no LLM configured —
    /// returns 503 (deterministic, mirrors `/converse`'s no-LLM path).
    #[tokio::test]
    async fn triage_handler_no_llm_returns_503() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        // Seed one open event so lookup succeeds and we reach the LLM check.
        let id = OpsEvent::make_id("high_error_rate", &Scope::Global);
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![crate::advisor::rules::RuleHit {
                    rule_id: "high_error_rate".to_owned(),
                    scope: Scope::Global,
                    severity: Severity::High,
                    title: "errors spiking".to_owned(),
                    detail: "5xx ratio over threshold".to_owned(),
                    recommendation: "inspect logs".to_owned(),
                    runbook: None,
                    remediation: RemediationAction::None,
                }],
                1000,
            );
        }
        let res = triage_event(
            axum::extract::State(state),
            axum::extract::Path(id),
        )
        .await;
        let err = res.err().expect("no LLM → error");
        assert_eq!(err.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    /// Unknown event id → 404 (before the LLM check).
    #[tokio::test]
    async fn triage_handler_unknown_event_returns_404() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let res = triage_event(
            axum::extract::State(state),
            axum::extract::Path("nope:global".to_owned()),
        )
        .await;
        let err = res.err().expect("missing event → error");
        assert_eq!(err.0, StatusCode::NOT_FOUND);
    }
}
