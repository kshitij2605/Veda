//! Read-only ops/observability tools for the agent ToolCatalog.
//!
//! These back the built-in `veda-ops` agent (AI-assisted ops triage):
//! each surfaces a slice of the ops-advisor / Log Insights / resource
//! state as compact JSON, reaching [`AppState`] through the
//! [`ToolContext`] exactly like the `search` builtin. All four are
//! read-only and never mutate state.
//!
//!   * `get_insights`     — recent advisor events (open / history)
//!   * `get_logs`         — recent captured log records
//!   * `get_resources`    — latest node resource snapshot
//!   * `get_slo_status`   — SLO status for one id or all SLOs

use crate::advisor::engine::now_ms;
use crate::advisor::event::{OpsEvent, Severity};
use crate::agent::catalog::{ToolContext, ToolOutcome, ToolSpec};

/// The read-only ops tool specs (id == OpsenAI/MCP tool name). Appended to
/// the builtin catalog so `tool_specs` / `specs_for` can whitelist them.
pub fn ops_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            id: "get_insights".to_owned(),
            name: "get_insights".to_owned(),
            description: "List recent Veda ops-advisor events (alerts). `status` selects open events (default) or closed `history`; optional `severity` (low|medium|high) and `limit` filter the result. Each event carries rule_id, severity, status, scope, title, detail, recommendation and opened_at.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "status": { "type": "string", "enum": ["open", "history"], "default": "open", "description": "open events (default) or closed history" },
                    "severity": { "type": "string", "enum": ["low", "medium", "high"], "description": "Only events at this severity" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 50 }
                }
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Get insights".to_owned()),
        },
        ToolSpec {
            id: "get_logs".to_owned(),
            name: "get_logs".to_owned(),
            description: "Return the most recent captured log records (newest-first) from the in-process Log Insights buffer. Optional `level` keeps records at or above that severity (ERROR|WARN|INFO|DEBUG|TRACE) and `q` is a case-insensitive substring over message/target. Each record has ts, level, target, message and trace_id.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "level": { "type": "string", "description": "Minimum severity, e.g. WARN" },
                    "q": { "type": "string", "description": "Case-insensitive substring filter over message/target" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 100 }
                }
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Get logs".to_owned()),
        },
        ToolSpec {
            id: "get_resources".to_owned(),
            name: "get_resources".to_owned(),
            description: "Return the latest local-node resource snapshot: disk (total/free/used%), cpu%, load averages, open file descriptors + limit, memory (total/available) and process RSS. Sourced from the ops-advisor's snapshot history; the same data as GET /v1/admin/resources.".to_owned(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Get resources".to_owned()),
        },
        ToolSpec {
            id: "get_slo_status".to_owned(),
            name: "get_slo_status".to_owned(),
            description: "Return live SLO status (SLI %, error-budget remaining, fast/slow burn rates, compliance/breach flags) for one SLO `id`, or for every defined SLO when `id` is omitted.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "SLO id; omit for all SLOs" }
                }
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Get SLO status".to_owned()),
        },
        ToolSpec {
            id: "correlate_incident".to_owned(),
            name: "correlate_incident".to_owned(),
            description: "Build the cross-signal root-cause bundle for an ops event `event_id`: auto-joins the correlated WARN/ERROR logs, slow/errored search queries, the nearest resource snapshot and the related incident within the event's incident window. Use this to ground a root-cause explanation in the signals that surrounded the alert. Returns null when no event has that id.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "event_id": { "type": "string", "description": "The ops event id to correlate" }
                },
                "required": ["event_id"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Correlate incident".to_owned()),
        },
    ]
}

// ── Implementations ─────────────────────────────────────────────────────────

fn parse_severity(s: &str) -> Option<Severity> {
    match s.to_ascii_lowercase().as_str() {
        "low" => Some(Severity::Low),
        "medium" => Some(Severity::Medium),
        "high" => Some(Severity::High),
        _ => None,
    }
}

/// Compact JSON projection of an advisor event for the agent.
fn event_json(ev: &OpsEvent) -> serde_json::Value {
    serde_json::json!({
        "id": ev.id,
        "rule_id": ev.rule_id,
        "severity": ev.severity,
        "status": ev.status,
        "scope": ev.scope,
        "title": ev.title,
        "detail": ev.detail,
        "recommendation": ev.recommendation,
        "opened_at_ms": ev.opened_at_ms,
    })
}

/// `get_insights` — recent advisor events (open or history), optionally
/// filtered by severity and capped at `limit`.
pub async fn get_insights(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let status = args["status"].as_str().unwrap_or("open");
    let want_sev = args["severity"].as_str().and_then(parse_severity);
    let limit = args["limit"].as_u64().unwrap_or(50).clamp(1, 200) as usize;

    let store = ctx.state.advisor().events.lock().await;
    let events: Vec<OpsEvent> = match status {
        "history" => store.history().iter().rev().cloned().collect(),
        _ => store.open_events(),
    };
    drop(store);

    let out: Vec<serde_json::Value> = events
        .iter()
        .filter(|e| want_sev.is_none_or(|s| e.severity == s))
        .take(limit)
        .map(event_json)
        .collect();
    ToolOutcome::ok(serde_json::json!({ "events": out, "count": out.len() }))
}

/// `get_logs` — recent captured log records from the Log Insights buffer.
pub async fn get_logs(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let limit = args["limit"].as_u64().unwrap_or(100).clamp(1, 500) as usize;
    let level = args["level"].as_str().filter(|s| !s.is_empty());
    let q = args["q"].as_str().filter(|s| !s.is_empty());

    let records = ctx.state.log_buffer().recent(limit, level, q);
    let logs: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            serde_json::json!({
                "ts_ms": r.ts_ms,
                "level": r.level,
                "target": r.target,
                "message": r.message,
                "trace_id": r.trace_id,
            })
        })
        .collect();
    ToolOutcome::ok(serde_json::json!({ "logs": logs, "count": logs.len() }))
}

/// `get_resources` — the latest node resource snapshot (disk/cpu/load/fds/
/// memory/rss). Mirrors GET /v1/admin/resources; cpu% is over the
/// advisor's previous→latest snapshots (null with fewer than two).
pub async fn get_resources(ctx: &ToolContext<'_>, _args: serde_json::Value) -> ToolOutcome {
    use crate::advisor::snapshot::{collect, cpu_pct};

    let history = ctx.state.advisor().history.lock().await;
    let (snap, pct);
    if let Some(latest) = history.latest() {
        pct = history.previous().and_then(|prev| cpu_pct(prev, latest));
        snap = latest.clone();
        drop(history);
    } else {
        drop(history);
        // Advisor hasn't ticked yet: collect a fresh snapshot (no previous
        // → cpu% null), matching the resources endpoint's fallback.
        snap = collect(ctx.state, now_ms()).await;
        pct = None;
    }

    ToolOutcome::ok(serde_json::json!({
        "disk": {
            "total_bytes": snap.disk_total_bytes,
            "free_bytes": snap.disk_free_bytes,
            "used_pct": snap.disk_used_pct,
        },
        "cpu_pct": pct,
        "load": { "load1": snap.load1, "load5": snap.load5, "load15": snap.load15 },
        "num_cpus": snap.num_cpus,
        "fds": { "open": snap.open_fds, "limit": snap.fd_limit },
        "memory": { "total_bytes": snap.mem_total_bytes, "available_bytes": snap.mem_available_bytes },
        "rss_bytes": snap.rss_bytes,
    }))
}

/// `get_slo_status` — live status for one SLO id (404-style not_found if
/// absent) or every defined SLO.
pub async fn get_slo_status(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let advisor = ctx.state.advisor();
    let now = now_ms().max(0) as u64;

    if let Some(id) = args["id"].as_str() {
        let def = match advisor.slos.read().await.get(id).cloned() {
            Some(d) => d,
            None => return ToolOutcome::not_found(format!("no SLO with id '{id}'")),
        };
        let status = advisor.slo_state.lock().await.status(&def, now);
        return ToolOutcome::ok(serde_json::json!({ "status": status }));
    }

    let defs = advisor.slos.read().await.list().to_vec();
    let state = advisor.slo_state.lock().await;
    let statuses: Vec<_> = defs.iter().map(|d| state.status(d, now)).collect();
    drop(state);
    ToolOutcome::ok(serde_json::json!({ "statuses": statuses, "count": statuses.len() }))
}

/// `correlate_incident` — the cross-signal root-cause bundle for an event.
/// Returns the bundle JSON, or `{ "bundle": null }` when no event has that
/// id, so the triage agent can pull joined logs/queries/resources/incident
/// context itself.
pub async fn correlate_incident(ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(event_id) = args["event_id"].as_str().filter(|s| !s.is_empty()) else {
        return ToolOutcome::invalid_args("event_id is required");
    };
    match crate::advisor::rca::correlate_incident(ctx.state, event_id).await {
        Some(bundle) => match serde_json::to_value(&bundle) {
            Ok(v) => ToolOutcome::ok(v),
            Err(e) => ToolOutcome::internal(format!("serialize rca bundle: {e}")),
        },
        None => ToolOutcome::ok(serde_json::json!({ "bundle": null })),
    }
}

/// Whether `id` is one of the read-only ops tools (used by the
/// `veda-ops` agent's whitelist).
pub fn is_ops_tool(id: &str) -> bool {
    matches!(
        id,
        "get_insights"
            | "get_logs"
            | "get_resources"
            | "get_slo_status"
            | "correlate_incident"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::event::{RemediationAction, Scope};
    use crate::advisor::rules::RuleHit;
    use crate::advisor::slo::{default_fast_burn, default_slow_burn, SloDef, SloKind};
    use crate::agent::catalog::ToolContext;
    use crate::logs::buffer::LogRecord;
    use crate::rest::state::AppState;

    fn hit(rule: &str, scope: Scope, sev: Severity) -> RuleHit {
        RuleHit {
            rule_id: rule.to_owned(),
            scope,
            severity: sev,
            title: format!("{rule} fired"),
            detail: "root cause text".to_owned(),
            recommendation: "do the thing".to_owned(),
            runbook: None,
            remediation: RemediationAction::None,
        }
    }

    #[tokio::test]
    async fn get_insights_returns_open_events() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![hit("high_error_rate", Scope::Global, Severity::High)],
                1000,
            );
        }
        let ctx = ToolContext::new(&state, &tenant);
        let out = get_insights(&ctx, serde_json::json!({})).await;
        assert!(!out.is_error, "{:?}", out.content);
        let events = out.content["events"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["rule_id"], "high_error_rate");
        assert_eq!(events[0]["severity"], "high");
        assert_eq!(events[0]["title"], "high_error_rate fired");
        assert_eq!(out.content["count"], 1);
    }

    #[tokio::test]
    async fn get_insights_severity_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![
                    hit("a", Scope::Node("n1".to_owned()), Severity::Low),
                    hit("b", Scope::Node("n2".to_owned()), Severity::High),
                ],
                1000,
            );
        }
        let ctx = ToolContext::new(&state, &tenant);
        let out = get_insights(&ctx, serde_json::json!({ "severity": "high" })).await;
        let events = out.content["events"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["rule_id"], "b");
    }

    #[tokio::test]
    async fn get_logs_returns_recent_records() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        state.log_buffer().push(LogRecord {
            ts_ms: 42,
            level: "ERROR".to_owned(),
            target: "veda_gateway::search".to_owned(),
            message: "boom".to_owned(),
            fields: serde_json::Map::new(),
            trace_id: Some("abc".to_owned()),
            span_id: None,
        });
        let ctx = ToolContext::new(&state, &tenant);
        let out = get_logs(&ctx, serde_json::json!({})).await;
        assert!(!out.is_error, "{:?}", out.content);
        let logs = out.content["logs"].as_array().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0]["level"], "ERROR");
        assert_eq!(logs[0]["message"], "boom");
        assert_eq!(logs[0]["trace_id"], "abc");
    }

    #[tokio::test]
    async fn get_logs_level_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        for (lvl, msg) in [("INFO", "noise"), ("ERROR", "real problem")] {
            state.log_buffer().push(LogRecord {
                ts_ms: 0,
                level: lvl.to_owned(),
                target: "t".to_owned(),
                message: msg.to_owned(),
                fields: serde_json::Map::new(),
                trace_id: None,
                span_id: None,
            });
        }
        let ctx = ToolContext::new(&state, &tenant);
        let out = get_logs(&ctx, serde_json::json!({ "level": "WARN" })).await;
        let logs = out.content["logs"].as_array().unwrap();
        assert_eq!(logs.len(), 1, "only ERROR is >= WARN");
        assert_eq!(logs[0]["message"], "real problem");
    }

    #[tokio::test]
    async fn get_resources_returns_snapshot_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        // Empty history → fresh collect; the shape is always present.
        let out = get_resources(&ctx, serde_json::json!({})).await;
        assert!(!out.is_error, "{:?}", out.content);
        assert!(out.content["disk"].is_object());
        assert!(out.content["load"].is_object());
        assert!(out.content["memory"].is_object());
        assert!(out.content["fds"].is_object());
        assert!(out.content.get("cpu_pct").is_some());
        assert!(out.content["num_cpus"].is_number());
    }

    fn slo(id: &str) -> SloDef {
        SloDef {
            id: id.to_owned(),
            name: format!("{id} slo"),
            kind: SloKind::Availability,
            target_pct: 99.0,
            window_secs: 3600,
            fast_burn: default_fast_burn(),
            slow_burn: default_slow_burn(),
        }
    }

    #[tokio::test]
    async fn correlate_incident_returns_bundle_for_seeded_event() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        // Seed one open event so the gatherer finds it.
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![hit("high_error_rate", Scope::Global, Severity::High)],
                crate::advisor::engine::now_ms(),
            );
        }
        // Seed an in-window ERROR log carrying a trace id.
        state.log_buffer().push(LogRecord {
            ts_ms: crate::advisor::engine::now_ms(),
            level: "ERROR".to_owned(),
            target: "veda_gateway::search".to_owned(),
            message: "boom".to_owned(),
            fields: serde_json::Map::new(),
            trace_id: Some("trace-xyz".to_owned()),
            span_id: None,
        });

        let ctx = ToolContext::new(&state, &tenant);
        let id = OpsEvent::make_id("high_error_rate", &Scope::Global);
        let out = correlate_incident(&ctx, serde_json::json!({ "event_id": id })).await;
        assert!(!out.is_error, "{:?}", out.content);
        // Bundle shape: event_id + the joined-signal arrays.
        assert_eq!(out.content["event_id"], "high_error_rate:global");
        assert!(out.content["logs"].is_array());
        assert!(out.content["slow_queries"].is_array());
        assert!(out.content["trace_ids"].is_array());
        // The seeded ERROR log was joined and its trace id collected.
        assert_eq!(out.content["logs"][0]["message"], "boom");
        assert_eq!(out.content["trace_ids"][0], "trace-xyz");
    }

    #[tokio::test]
    async fn correlate_incident_missing_event_returns_null_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out =
            correlate_incident(&ctx, serde_json::json!({ "event_id": "nope:global" })).await;
        assert!(!out.is_error);
        assert!(out.content["bundle"].is_null());
    }

    #[tokio::test]
    async fn correlate_incident_requires_event_id() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        let ctx = ToolContext::new(&state, &tenant);
        let out = correlate_incident(&ctx, serde_json::json!({})).await;
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn get_slo_status_one_and_all() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = crate::tenant::TenantId::default_tenant();
        {
            let mut reg = state.advisor().slos.write().await;
            reg.upsert(slo("api"));
            reg.upsert(slo("search"));
        }
        let ctx = ToolContext::new(&state, &tenant);

        // One id.
        let out = get_slo_status(&ctx, serde_json::json!({ "id": "api" })).await;
        assert!(!out.is_error, "{:?}", out.content);
        assert_eq!(out.content["status"]["slo_id"], "api");
        assert_eq!(out.content["status"]["target_pct"], 99.0);

        // All.
        let out = get_slo_status(&ctx, serde_json::json!({})).await;
        assert_eq!(out.content["count"], 2);
        let statuses = out.content["statuses"].as_array().unwrap();
        assert_eq!(statuses.len(), 2);

        // Unknown id → not_found.
        let out = get_slo_status(&ctx, serde_json::json!({ "id": "nope" })).await;
        assert!(out.is_error);
    }
}
