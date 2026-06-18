//! Cross-signal root-cause analysis (RCA) bundle.
//!
//! Given an ops [`OpsEvent`], this module *joins* the correlated signals
//! observed in the incident window — captured logs, slow/errored search
//! queries, the nearest resource snapshot, and the related incident — into
//! a single compact [`RcaBundle`]. The bundle grounds the `veda-ops` triage
//! agent's explanation in the actual signals that surrounded the alert.
//!
//! The join itself is a **pure** function ([`correlate`]) over already
//! gathered inputs, so it is fully unit-testable with constructed records.
//! [`correlate_incident`] is the thin async gatherer: it pulls the four
//! signal sources off `AppState` (each under a brief lock — never held
//! across an await) then calls [`correlate`].
//!
//! Correlation key: the time window `[opened_at - Δ, opened_at + Δ]` plus
//! the `trace_id` shared between [`LogRecord`] and [`QueryLogEntry`]. The
//! distinct set of in-window trace ids is the correlation set; logs and
//! queries that carry one of those ids are the cross-signal evidence.

use serde::Serialize;

use crate::advisor::event::OpsEvent;
use crate::advisor::incidents::Incident;
use crate::advisor::snapshot::OpsSnapshot;
use crate::logs::buffer::LogRecord;
use crate::rest::query_log::QueryLogEntry;

/// Default half-window (Δ) around an event's `opened_at_ms`, in ms (5 min).
pub const DEFAULT_WINDOW_MS: i64 = 5 * 60 * 1000;
/// Cap on logs / queries projected into a bundle (keeps it bounded).
const MAX_LINES: usize = 50;
/// A query slower than this (ms) is treated as "slow" even if it succeeded.
const SLOW_QUERY_MS: u64 = 1000;

/// Compact projection of one correlated log record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogLine {
    pub ts_ms: i64,
    pub level: String,
    pub target: String,
    pub message: String,
    pub trace_id: Option<String>,
}

/// Compact projection of one slow/errored query log entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QueryLine {
    pub ts_ms: u64,
    pub took_ms: u64,
    pub status: String,
    pub error_kind: Option<String>,
    pub query_type: String,
    pub collection: String,
    pub index_type: Option<String>,
    pub trace_id: Option<String>,
}

/// Compact projection of the resource snapshot nearest the incident.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResourceState {
    pub taken_at_ms: i64,
    pub disk_used_pct: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub query_p95_ms: f64,
    pub error_log_count: u64,
}

/// Compact reference to the incident that contains the event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IncidentRef {
    pub id: String,
    pub representative: String,
    pub count: u32,
}

/// The cross-signal root-cause bundle for one ops event.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RcaBundle {
    pub event_id: String,
    /// Half-window Δ (ms) applied around `opened_at_ms`.
    pub window_ms: i64,
    pub logs: Vec<LogLine>,
    pub slow_queries: Vec<QueryLine>,
    /// Distinct trace ids appearing across the correlated logs + queries.
    pub trace_ids: Vec<String>,
    pub resource_snapshot: Option<ResourceState>,
    pub related_incident: Option<IncidentRef>,
}

/// Whether `ts_ms` falls within `[opened - window, opened + window]`.
fn in_window(ts_ms: i64, opened_at_ms: i64, window_ms: i64) -> bool {
    ts_ms >= opened_at_ms - window_ms && ts_ms <= opened_at_ms + window_ms
}

/// Pure cross-signal join. Given the event and the already-gathered
/// signals, produce the bundle:
///
/// * **logs** — WARN/ERROR records inside the window, newest-first, capped.
/// * **slow_queries** — queries inside the window that either failed
///   (`status == "error"`) or were slow (`took_ms >= SLOW_QUERY_MS`),
///   newest-first, capped.
/// * **trace_ids** — the distinct set of trace ids carried by the
///   correlated logs + queries (the cross-signal correlation set). Logs and
///   queries sharing a trace id are surfaced first within each list.
/// * **resource_snapshot** — the snapshot whose `taken_at_ms` is nearest
///   `opened_at_ms`.
/// * **related_incident** — the incident whose `member_event_ids` contains
///   `event.id`.
///
/// Deterministic and I/O-free.
pub fn correlate(
    event: &OpsEvent,
    logs: &[LogRecord],
    queries: &[QueryLogEntry],
    snapshots: &[OpsSnapshot],
    incidents: &[Incident],
    window_ms: i64,
) -> RcaBundle {
    let opened = event.opened_at_ms;

    // --- in-window WARN/ERROR logs ----------------------------------------
    let mut logs: Vec<&LogRecord> = logs
        .iter()
        .filter(|r| in_window(r.ts_ms, opened, window_ms))
        .filter(|r| {
            let lvl = r.level.to_ascii_uppercase();
            lvl == "WARN" || lvl == "ERROR"
        })
        .collect();

    // --- in-window slow / errored queries ---------------------------------
    let mut queries: Vec<&QueryLogEntry> = queries
        .iter()
        .filter(|e| in_window(e.ts_ms as i64, opened, window_ms))
        .filter(|e| e.status == "error" || e.took_ms >= SLOW_QUERY_MS)
        .collect();

    // --- correlation set: distinct trace ids across both signals ----------
    let mut trace_ids: Vec<String> = Vec::new();
    let push_trace = |t: &Option<String>, ids: &mut Vec<String>| {
        if let Some(t) = t
            && !t.is_empty()
            && !ids.contains(t)
        {
            ids.push(t.clone());
        }
    };
    for r in &logs {
        push_trace(&r.trace_id, &mut trace_ids);
    }
    for q in &queries {
        push_trace(&q.trace_id, &mut trace_ids);
    }

    // Prioritise records sharing a trace id (the cross-signal evidence),
    // then by recency. Stable so same-priority records keep relative order.
    let shares_trace = |t: &Option<String>| t.as_ref().is_some_and(|t| trace_ids.contains(t));
    logs.sort_by(|a, b| {
        shares_trace(&b.trace_id)
            .cmp(&shares_trace(&a.trace_id))
            .then(b.ts_ms.cmp(&a.ts_ms))
    });
    queries.sort_by(|a, b| {
        shares_trace(&b.trace_id)
            .cmp(&shares_trace(&a.trace_id))
            .then(b.ts_ms.cmp(&a.ts_ms))
    });

    let logs: Vec<LogLine> = logs
        .into_iter()
        .take(MAX_LINES)
        .map(|r| LogLine {
            ts_ms: r.ts_ms,
            level: r.level.clone(),
            target: r.target.clone(),
            message: r.message.clone(),
            trace_id: r.trace_id.clone(),
        })
        .collect();

    let slow_queries: Vec<QueryLine> = queries
        .into_iter()
        .take(MAX_LINES)
        .map(|e| QueryLine {
            ts_ms: e.ts_ms,
            took_ms: e.took_ms,
            status: e.status.clone(),
            error_kind: e.error_kind.clone(),
            query_type: e.query_type.clone(),
            collection: e.collection.clone(),
            index_type: e.index_type.clone(),
            trace_id: e.trace_id.clone(),
        })
        .collect();

    // --- nearest snapshot --------------------------------------------------
    let resource_snapshot = snapshots
        .iter()
        .min_by_key(|s| (s.taken_at_ms - opened).abs())
        .map(|s| ResourceState {
            taken_at_ms: s.taken_at_ms,
            disk_used_pct: s.disk_used_pct,
            rss_bytes: s.rss_bytes,
            query_p95_ms: s.query.p95_ms,
            error_log_count: s.error_log_count,
        });

    // --- related incident --------------------------------------------------
    let related_incident = incidents
        .iter()
        .find(|i| i.member_event_ids.iter().any(|id| id == &event.id))
        .map(|i| IncidentRef {
            id: i.id.clone(),
            representative: i.representative.clone(),
            count: i.count,
        });

    RcaBundle {
        event_id: event.id.clone(),
        window_ms,
        logs,
        slow_queries,
        trace_ids,
        resource_snapshot,
        related_incident,
    }
}

/// Gather the four signal sources off `AppState` and run the pure
/// [`correlate`] join for the event identified by `event_id`. Returns
/// `None` when no event (open or history) has that id.
///
/// **Lock discipline:** every advisor / state lock is taken *briefly* to
/// clone the needed data out, then dropped before the next one. No lock is
/// held across the `read_window_entries` blocking work (which runs on a
/// `spawn_blocking`) or across any await. The query-log scan covers the
/// tenant's collections only (schema keys scoped to the default tenant).
pub async fn correlate_incident(state: &AppState, event_id: &str) -> Option<RcaBundle> {
    // 1. Find the event (open first, then history). Brief lock.
    let event = {
        let store = state.advisor().events.lock().await;
        store
            .open_events()
            .into_iter()
            .find(|e| e.id == event_id)
            .or_else(|| store.history().iter().rev().find(|e| e.id == event_id).cloned())
    }?;

    let window_ms = DEFAULT_WINDOW_MS;

    // 2. Logs from the in-process buffer (its own internal lock).
    let logs = state
        .log_buffer()
        .recent(crate::logs::buffer::DEFAULT_CAPACITY, Some("WARN"), None);

    // 3. Query log across the (default) tenant's collections. The scoped
    //    schema keys are exactly the on-disk query-log directory names.
    let tenant = crate::tenant::TenantId::default_tenant();
    let collections: Vec<String> = {
        let schemas = state.inner.schemas.read().await;
        schemas
            .keys()
            .filter(|k| crate::tenant::strip_prefix(&tenant, k).is_some())
            .cloned()
            .collect()
    };
    let data_dir = state.inner.data_dir.clone();
    // Read window: cover the full [opened-Δ, opened+Δ] relative to now.
    let now = crate::advisor::engine::now_ms();
    let span_ago_ms = (now - (event.opened_at_ms - window_ms)).max(window_ms);
    let window_secs = (span_ago_ms / 1000).max(1) as u64;
    let queries: Vec<QueryLogEntry> = tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        for c in &collections {
            out.extend(crate::rest::query_log::read_window_entries(
                &data_dir,
                c,
                window_secs,
            ));
        }
        out
    })
    .await
    .unwrap_or_default();

    // 4. Snapshots — clone the history out under a brief lock.
    let snapshots: Vec<OpsSnapshot> = {
        let history = state.advisor().history.lock().await;
        history.iter().cloned().collect()
    };

    // 5. Incidents — latest correlation pass, brief lock.
    let incidents: Vec<Incident> = {
        let store = state.advisor().incidents.lock().await;
        store.latest().to_vec()
    };

    Some(correlate(
        &event,
        &logs,
        &queries,
        &snapshots,
        &incidents,
        window_ms,
    ))
}

use crate::rest::state::AppState;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::event::{EventStatus, RemediationAction, Scope, Severity};
    use crate::advisor::snapshot::{
        GlobalSnapshot, HealthSnapshot, NodeSnapshot, QuerySnapshot,
    };
    use std::collections::HashMap;

    fn event(id_rule: &str, opened: i64) -> OpsEvent {
        OpsEvent {
            id: OpsEvent::make_id(id_rule, &Scope::Global),
            rule_id: id_rule.to_owned(),
            scope: Scope::Global,
            severity: Severity::High,
            status: EventStatus::Open,
            title: "errors spiking".to_owned(),
            detail: "5xx ratio over threshold".to_owned(),
            recommendation: "inspect logs".to_owned(),
            runbook: None,
            remediation: RemediationAction::None,
            opened_at_ms: opened,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }

    fn log(ts_ms: i64, level: &str, msg: &str, trace: Option<&str>) -> LogRecord {
        LogRecord {
            ts_ms,
            level: level.to_owned(),
            target: "veda_gateway::search".to_owned(),
            message: msg.to_owned(),
            fields: serde_json::Map::new(),
            trace_id: trace.map(str::to_owned),
            span_id: None,
        }
    }

    fn query(ts_ms: u64, took_ms: u64, status: &str, trace: Option<&str>) -> QueryLogEntry {
        QueryLogEntry {
            ts_ms,
            top_k: 10,
            doc_ids: Vec::new(),
            took_ms,
            trace_id: trace.map(str::to_owned),
            collection: "docs".to_owned(),
            query_type: "dense".to_owned(),
            has_filter: false,
            reranked: false,
            result_count: 0,
            status: status.to_owned(),
            error_kind: if status == "error" {
                Some("internal".to_owned())
            } else {
                None
            },
            index_type: Some("hnsw".to_owned()),
            shard_ids: Vec::new(),
        }
    }

    fn snapshot(taken_at_ms: i64) -> OpsSnapshot {
        OpsSnapshot {
            taken_at_ms,
            global: GlobalSnapshot {
                active_collections: 0,
                total_records: 0,
            },
            collections: Vec::new(),
            nodes: Vec::<NodeSnapshot>::new(),
            replica_safe_ts: None,
            health: HealthSnapshot {
                minio: "healthy".to_owned(),
                nats: "healthy".to_owned(),
                embedding: "unconfigured".to_owned(),
                multimodal_embedding: "unconfigured".to_owned(),
            },
            query: QuerySnapshot {
                window_secs: 60,
                count: 0,
                p50_ms: 0.0,
                p95_ms: 42.0,
                slow_count: 0,
                recent_latencies_ms: Vec::new(),
            },
            rss_bytes: Some(1234),
            disk_total_bytes: None,
            disk_free_bytes: None,
            disk_used_pct: Some(50.0),
            cpu_total_jiffies: None,
            cpu_busy_jiffies: None,
            num_cpus: 1,
            load1: None,
            load5: None,
            load15: None,
            open_fds: None,
            fd_limit: None,
            mem_total_bytes: None,
            mem_available_bytes: None,
            counters: HashMap::new(),
            error_log_count: 7,
            warn_log_count: 9,
            slo_statuses: Vec::new(),
            synthetic_summaries: Vec::new(),
            index_health: Vec::new(),
            forecasts: Vec::new(),
            change_points: Vec::new(),
        }
    }

    fn incident(member_ids: &[&str]) -> Incident {
        Incident {
            id: "inc_1".to_owned(),
            representative: "errors spiking".to_owned(),
            member_event_ids: member_ids.iter().map(|s| s.to_string()).collect(),
            rule_ids: vec!["high_error_rate".to_owned()],
            count: member_ids.len() as u32,
            max_severity: Severity::High,
            first_seen_ms: 0,
            last_seen_ms: 0,
            draft: None,
        }
    }

    // Tests use an explicit small window (1s) so the in/out-of-window
    // boundaries are unambiguous, independent of DEFAULT_WINDOW_MS.
    const TEST_WINDOW_MS: i64 = 1_000;

    #[test]
    fn correlate_joins_only_in_window_logs_and_slow_or_errored_queries() {
        let ev = event("high_error_rate", 10_000);
        let logs = vec![
            log(9_500, "ERROR", "in window err", Some("t1")),
            log(10_500, "WARN", "in window warn", None),
            log(9_800, "INFO", "in window info (excluded by level)", None),
            log(100, "ERROR", "out of window (excluded)", None),
        ];
        let queries = vec![
            query(9_900, 50, "error", Some("t1")), // errored, in window
            query(10_100, 2000, "ok", None),       // slow, in window
            query(10_200, 10, "ok", None),         // fast ok → excluded
            query(50, 5000, "error", None),        // out of window → excluded
        ];
        let b = correlate(&ev, &logs, &queries, &[], &[], TEST_WINDOW_MS);

        // Only WARN/ERROR in-window logs (2 of 4).
        assert_eq!(b.logs.len(), 2, "{:?}", b.logs);
        assert!(b.logs.iter().all(|l| l.level == "ERROR" || l.level == "WARN"));
        // Only slow/errored in-window queries (2 of 4).
        assert_eq!(b.slow_queries.len(), 2, "{:?}", b.slow_queries);
        assert!(b
            .slow_queries
            .iter()
            .all(|q| q.status == "error" || q.took_ms >= SLOW_QUERY_MS));
    }

    #[test]
    fn shared_trace_id_set_is_collected_and_prioritised() {
        let ev = event("high_error_rate", 10_000);
        // A log and a query share trace t1; another log has t2.
        let logs = vec![
            log(10_400, "WARN", "no trace", None),
            log(9_600, "ERROR", "shared trace", Some("t1")),
            log(9_700, "ERROR", "other trace", Some("t2")),
        ];
        let queries = vec![query(9_900, 50, "error", Some("t1"))];
        let b = correlate(&ev, &logs, &queries, &[], &[], DEFAULT_WINDOW_MS);

        // Distinct trace ids across both signals: t1 then t2 (insertion order).
        assert_eq!(b.trace_ids, vec!["t1".to_owned(), "t2".to_owned()]);
        // Trace-bearing logs are prioritised ahead of the no-trace one.
        assert!(b.logs[0].trace_id.is_some(), "{:?}", b.logs);
        assert!(b.logs.last().unwrap().trace_id.is_none());
    }

    #[test]
    fn nearest_snapshot_is_chosen() {
        let ev = event("high_error_rate", 10_000);
        let snaps = vec![snapshot(2_000), snapshot(9_800), snapshot(30_000)];
        let b = correlate(&ev, &[], &[], &snaps, &[], DEFAULT_WINDOW_MS);
        let rs = b.resource_snapshot.expect("a nearest snapshot");
        assert_eq!(rs.taken_at_ms, 9_800);
        assert_eq!(rs.query_p95_ms, 42.0);
        assert_eq!(rs.error_log_count, 7);
        assert_eq!(rs.disk_used_pct, Some(50.0));
    }

    #[test]
    fn related_incident_is_matched_by_member_id() {
        let ev = event("high_error_rate", 10_000);
        let incidents = vec![
            incident(&["other:global"]),
            incident(&[&ev.id, "x:global"]),
        ];
        let b = correlate(&ev, &[], &[], &[], &incidents, DEFAULT_WINDOW_MS);
        let inc = b.related_incident.expect("a related incident");
        assert_eq!(inc.id, "inc_1");
        assert_eq!(inc.count, 2);
    }

    #[test]
    fn empty_when_nothing_in_window() {
        let ev = event("high_error_rate", 10_000);
        let logs = vec![log(100, "ERROR", "far before", None)];
        let queries = vec![query(50, 9000, "error", None)];
        let b = correlate(&ev, &logs, &queries, &[], &[], TEST_WINDOW_MS);
        assert!(b.logs.is_empty());
        assert!(b.slow_queries.is_empty());
        assert!(b.trace_ids.is_empty());
        assert!(b.resource_snapshot.is_none());
        assert!(b.related_incident.is_none());
        assert_eq!(b.event_id, ev.id);
        assert_eq!(b.window_ms, TEST_WINDOW_MS);
    }
}
