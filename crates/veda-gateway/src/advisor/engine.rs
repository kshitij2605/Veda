//! The ops-advisor engine: shared state + the background tick that ties
//! the advisor modules together.
//!
//! [`AdvisorState`] hangs off [`AppState`](crate::rest::state::AppState)
//! and owns the mutable advisor runtime: the bounded
//! [`SnapshotHistory`], the persisted [`EventStore`], the editable
//! [`AdvisorConfig`], and the per-scope remediation [`RateLimiter`].
//!
//! Each [`run_tick`] collects a fresh [`OpsSnapshot`], pushes it onto the
//! history, then runs the synchronous [`reconcile_hits`] core under the
//! advisor locks and finally notifies + remediates *after* releasing
//! those locks:
//!
//! 1. `run_rules` + `detect_anomalies` → combined hits,
//! 2. `EventStore::reconcile` → open/close transitions, persist,
//! 3. notify on the transitions (failures logged, never fatal),
//! 4. opt-in auto-remediation on the opened events, annotating outcomes.
//!
//! The tick never blocks the runtime: collection + notify + remediation
//! are all async and isolated, and a per-tick panic/failure cannot take
//! down the loop (each step swallows its own errors via the underlying
//! modules' `tracing::warn!` paths). Crucially, **no advisor lock is held
//! across the notify/remediation network awaits** — steps 3–4 run with
//! the `history`/`events` guards already dropped, so a slow notification
//! endpoint cannot freeze `GET /v1/admin/insights` or the next tick.

use std::path::Path;

use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

use crate::advisor::config::AdvisorConfig;
use crate::advisor::event::{EventStore, ReconcileResult};
use crate::advisor::notify::Notifier;
use crate::advisor::remediation::{run_remediation, LiveDispatcher, RateLimiter};
use crate::advisor::rules::{catalog, run_rules, Rule, RuleHit};
use crate::advisor::event::{RemediationAction, Scope, Severity};
use crate::advisor::snapshot::{collect, OpsSnapshot, SnapshotHistory, DEFAULT_HISTORY_CAP};
use crate::advisor::anomaly::detect_anomalies;
use crate::logs::{LogCategory, categorize::CategorizeOptions};
use crate::rest::state::AppState;

/// Config file name under `data_dir`.
const CONFIG_FILE: &str = "_advisor_config.json";
/// Default tick interval (seconds) when the env override is unset.
const DEFAULT_INTERVAL_SECS: u64 = 30;

/// Engine-level rule ids — findings produced directly by [`run_tick`]
/// (not by a snapshot [`Rule`] or an anomaly detector), exposed so the
/// rules-listing REST surface can show + tune them via config.
const ENGINE_RULE_IDS: &[&str] = &["new_error_category"];

/// The engine-level rule ids (stable, unique).
pub fn engine_rule_ids() -> &'static [&'static str] {
    ENGINE_RULE_IDS
}

/// Mutable runtime state for the ops advisor, shared via `AppState`.
pub struct AdvisorState {
    pub history: Mutex<SnapshotHistory>,
    pub events: Mutex<EventStore>,
    pub config: RwLock<AdvisorConfig>,
    pub limiter: Mutex<RateLimiter>,
    /// Shared async HTTP client, reused by the per-tick [`Notifier`].
    pub http: reqwest::Client,
}

impl AdvisorState {
    /// Construct from `data_dir`, loading the persisted config (default
    /// if absent) and event store.
    pub fn load(data_dir: &Path) -> Self {
        Self {
            history: Mutex::new(SnapshotHistory::new(DEFAULT_HISTORY_CAP)),
            events: Mutex::new(EventStore::load(data_dir)),
            config: RwLock::new(load_config(data_dir)),
            limiter: Mutex::new(RateLimiter::default()),
            http: reqwest::Client::new(),
        }
    }
}

/// Read the advisor config from `{data_dir}/_advisor_config.json`. A
/// missing or unparseable file yields [`AdvisorConfig::default`] (logged).
pub fn load_config(data_dir: &Path) -> AdvisorConfig {
    let path = data_dir.join(CONFIG_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return AdvisorConfig::default(),
    };
    match serde_json::from_slice(&bytes) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "advisor_config: parse failed, using defaults");
            AdvisorConfig::default()
        }
    }
}

/// Persist `cfg` to `{data_dir}/_advisor_config.json` (pretty JSON).
/// Failure is logged, not surfaced — the in-memory copy is authoritative.
pub fn save_config(data_dir: &Path, cfg: &AdvisorConfig) {
    let path = data_dir.join(CONFIG_FILE);
    match serde_json::to_vec_pretty(cfg) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                tracing::warn!(error = %e, path = %path.display(), "advisor_config: write failed");
            }
        }
        Err(e) => tracing::warn!(error = %e, "advisor_config: serialize failed"),
    }
}

/// The *synchronous* core of the advisor pipeline over a single
/// (already-collected) snapshot.
///
/// The caller pushes the snapshot onto the history *before* calling this
/// (so the history's `latest()` is the just-collected snapshot, matching
/// how the rate-based rules expect to read it). This runs the threshold
/// rules + anomaly detectors against `snap` and the rest of the history,
/// reconciles the combined hits into open/close transitions, and persists
/// the event store. It performs **no network I/O** — notification and
/// auto-remediation are orchestrated by [`run_tick`] *after* the advisor
/// locks have been released, so a slow endpoint can never stall a held
/// lock.
///
/// Returns the [`ReconcileResult`] (opened/closed) for this round.
pub fn reconcile_hits(
    snap: &OpsSnapshot,
    history: &SnapshotHistory,
    events: &mut EventStore,
    cfg: &AdvisorConfig,
    catalog: &[Box<dyn Rule>],
    extra_hits: Vec<RuleHit>,
    now_ms: i64,
) -> ReconcileResult {
    // 1. Combine threshold-rule hits with anomaly hits and any
    //    engine-level hits computed by the caller outside the locks
    //    (e.g. `new_error_category`, which needs an async categorize).
    let mut hits = run_rules(catalog, snap, history, cfg);
    hits.extend(detect_anomalies(history, cfg));
    hits.extend(extra_hits);

    // 2. Reconcile into open/close transitions, then persist.
    let result = events.reconcile(hits, now_ms);
    events.save();
    result
}

/// Detect *newly-appeared* error/warn log categories.
///
/// Emits a `new_error_category` [`RuleHit`] (Medium, `Scope::Global`)
/// for each category whose `max_severity` is WARN/ERROR **and** whose
/// `first_seen_ms` falls within the last `window_ms` of `now_ms` — i.e.
/// a category that first showed up during this tick window. Respects the
/// rule's `enabled`/`muted` config (returns empty when disabled/muted).
pub fn detect_new_error_categories(
    categories: &[LogCategory],
    cfg: &AdvisorConfig,
    now_ms: i64,
    window_ms: i64,
) -> Vec<RuleHit> {
    let rc = cfg.for_rule("new_error_category");
    if !rc.enabled || rc.muted {
        return Vec::new();
    }
    let cutoff = now_ms - window_ms;
    let mut hits = Vec::new();
    for cat in categories {
        let sev = cat.max_severity.to_ascii_uppercase();
        if sev != "WARN" && sev != "ERROR" {
            continue; // only error-ish categories are alert-worthy.
        }
        if cat.first_seen_ms < cutoff {
            continue; // pre-existing category — not new.
        }
        let mut severity = Severity::Medium;
        if let Some(over) = rc.severity_override {
            severity = over;
        }
        let example = cat.examples.first().cloned().unwrap_or_default();
        hits.push(RuleHit {
            rule_id: "new_error_category".into(),
            scope: Scope::Global,
            severity,
            title: format!("New {sev} log category"),
            detail: format!(
                "A new {sev} log category appeared: \"{}\" ({} occurrence(s); e.g. \"{}\").",
                cat.representative, cat.count, example
            ),
            recommendation: "Inspect this newly-appeared error category (/v1/admin/logs/categories) — it may be a fresh failure mode introduced by a recent change.".into(),
            runbook: Some("GET /v1/admin/logs/categories".into()),
            remediation: RemediationAction::None,
        });
    }
    hits
}

/// The configured advisor tick interval in seconds (env override,
/// default [`DEFAULT_INTERVAL_SECS`]). Used both by the loop and as the
/// "new category" detection window for `new_error_category`.
fn interval_secs() -> u64 {
    std::env::var("VEDA_ADVISOR_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_INTERVAL_SECS)
}

/// Current wall-clock time in epoch milliseconds.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Run exactly one advisor tick against `state`: collect a snapshot,
/// push it onto the history, reconcile, then notify + remediate. Returns
/// the [`ReconcileResult`] so callers (e.g. the force-run REST endpoint)
/// can report how many events opened/closed.
///
/// **Lock discipline — no advisor lock is ever held across a network
/// `.await`.** The tick runs in three phases, the network awaits living
/// strictly in phase 2 where neither `history` nor `events` is locked:
///
/// - **Phase 0 (no advisor lock):** `log_categorizer().categorize(...)`
///   — async, may embed over the network — runs first and produces the
///   `new_error_category` hits, exactly like the notify phase keeps
///   network awaits out of any held advisor lock.
/// - **Phase 1 (locked, synchronous):** lock `history`, push the
///   snapshot; lock `events`; [`reconcile_hits`] (run rules + anomalies,
///   reconcile, save). The opened/closed edges are cloned out, then both
///   guards are dropped at the end of the explicit scope block — before
///   any network I/O. `collect` above touches `AppState` only.
/// - **Phase 2 (no `events`/`history` lock):** build the [`Notifier`] +
///   [`LiveDispatcher`], `notify_transitions(...).await`, then
///   `run_remediation(...).await`. The `limiter` lock may be held here
///   (no REST handler takes it), but `events`/`history` are not.
/// - **Phase 3 (re-lock `events` briefly):** if remediation produced
///   outcomes, re-lock `events`, annotate each, save, drop the guard.
pub async fn run_tick(state: &AppState) -> ReconcileResult {
    let now = now_ms();
    let advisor = state.advisor();

    // Collect a fresh snapshot (touches AppState, not advisor locks).
    let snap = collect(state, now).await;

    // Per-tick config snapshot (released immediately).
    let cfg = advisor.config.read().await.clone();
    let rule_catalog = catalog();

    // --- Phase 0: semantic log categorization (NO advisor lock held).
    // `categorize` is async and may embed over the network, so it must
    // run before we take the history/events locks (mirroring how the
    // notify/remediation phase keeps network awaits lock-free). It only
    // touches `AppState` (the embedding-registry RwLock + log buffer),
    // never an advisor lock. Any failure / missing embedder simply
    // yields no categories → no `new_error_category` hits; never panics.
    let new_category_hits = {
        let window_ms = (interval_secs() as i64) * 1000;
        let registry = state.embedding_registry().read().await;
        let categories = state
            .log_categorizer()
            .categorize(state.log_buffer(), &registry, CategorizeOptions::default())
            .await;
        drop(registry);
        detect_new_error_categories(&categories, &cfg, now, window_ms)
    };

    // --- Phase 1: reconcile under the history+events locks (NO network).
    // The guards live only inside this block and are dropped before any
    // notify/remediation await below.
    let result = {
        let mut history = advisor.history.lock().await;
        history.push(snap.clone());
        let mut events = advisor.events.lock().await;
        reconcile_hits(
            &snap,
            &history,
            &mut events,
            &cfg,
            &rule_catalog,
            new_category_hits,
            now,
        )
    };

    // --- Phase 2: notify + remediate with NO events/history lock held.
    let notifier = Notifier::new(cfg.channels.clone());
    let dispatcher = LiveDispatcher::new(state.clone());
    notifier
        .notify_transitions(&result.opened, &result.closed)
        .await;
    let outcomes = {
        let mut limiter = advisor.limiter.lock().await;
        run_remediation(&result.opened, &cfg, &dispatcher, &mut limiter, now).await
    };

    // --- Phase 3: persist remediation annotations under a brief events lock.
    if !outcomes.is_empty() {
        let mut events = advisor.events.lock().await;
        for o in &outcomes {
            events.annotate_remediation(&o.event_id, &o.note, o.ok, o.at_ms);
        }
        events.save();
    }

    result
}

/// Spawn the background advisor tick loop. Ticks every
/// `VEDA_ADVISOR_INTERVAL_SECS` seconds (default 30), with
/// [`MissedTickBehavior::Delay`] so a slow tick never piles up backlog.
/// A failing/slow `run_tick` cannot crash the loop.
pub fn spawn_advisor(state: AppState) -> JoinHandle<()> {
    let secs = interval_secs();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(secs));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let r = run_tick(&state).await;
            if !r.opened.is_empty() || !r.closed.is_empty() {
                tracing::info!(
                    opened = r.opened.len(),
                    closed = r.closed.len(),
                    "advisor: tick reconciled events"
                );
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::config::RuleConfig;
    use crate::advisor::event::EventStatus;
    use crate::advisor::snapshot::{
        CollectionSnapshot, GlobalSnapshot, HealthSnapshot, NodeSnapshot, OpsSnapshot,
        QuerySnapshot,
    };
    use std::collections::HashMap;

    fn healthy_health() -> HealthSnapshot {
        HealthSnapshot {
            minio: "healthy".into(),
            nats: "healthy".into(),
            embedding: "healthy".into(),
            multimodal_embedding: "healthy".into(),
        }
    }

    fn base_snapshot(now_ms: i64) -> OpsSnapshot {
        OpsSnapshot {
            taken_at_ms: now_ms,
            global: GlobalSnapshot {
                active_collections: 0,
                total_records: 0,
            },
            collections: Vec::new(),
            nodes: vec![NodeSnapshot {
                id: "node-0".into(),
                role: "leader".into(),
                status: "healthy".into(),
                shard_count: 1,
            }],
            replica_safe_ts: None,
            health: healthy_health(),
            query: QuerySnapshot {
                window_secs: 300,
                count: 0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                slow_count: 0,
            },
            rss_bytes: None,
            disk_total_bytes: None,
            disk_free_bytes: None,
            disk_used_pct: None,
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
            error_log_count: 0,
            warn_log_count: 0,
        }
    }

    fn category(id: &str, sev: &str, first_seen_ms: i64) -> LogCategory {
        LogCategory {
            id: id.into(),
            representative: format!("template {id}"),
            count: 3,
            max_severity: sev.into(),
            examples: vec![format!("example for {id}")],
            first_seen_ms,
            last_seen_ms: first_seen_ms + 10,
        }
    }

    /// A collection with a big sealed-segment backlog → trips
    /// `compaction_backlog`.
    fn tripping_collection() -> CollectionSnapshot {
        CollectionSnapshot {
            name: "docs".into(),
            shard_count: 1,
            total_segments: 26,
            sealed_segments: 25,
            growing_records: 10,
            total_records: 10,
            compaction_in_progress: false,
        }
    }

    #[test]
    fn reconcile_opens_then_closes_event() {
        let tmp = tempfile::tempdir().unwrap();
        let mut events = EventStore::new(tmp.path());
        let cfg = AdvisorConfig::default();
        let cat = catalog();

        // --- round 1: a tripping snapshot opens compaction_backlog ---
        let mut tripping = base_snapshot(1000);
        tripping.collections.push(tripping_collection());
        let mut history = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        history.push(tripping.clone());

        let r1 = reconcile_hits(&tripping, &history, &mut events, &cfg, &cat, Vec::new(), 1000);
        assert_eq!(r1.opened.len(), 1, "expected one opened event");
        assert_eq!(r1.opened[0].rule_id, "compaction_backlog");
        assert_eq!(events.open_events().len(), 1);
        assert_eq!(events.open_events()[0].status, EventStatus::Open);

        // --- round 2: a clean snapshot closes it ---
        let clean = base_snapshot(2000);
        history.push(clean.clone());
        let r2 = reconcile_hits(&clean, &history, &mut events, &cfg, &cat, Vec::new(), 2000);
        assert!(r2.opened.is_empty());
        assert_eq!(r2.closed.len(), 1, "expected one closed event");
        assert_eq!(r2.closed[0].rule_id, "compaction_backlog");
        assert!(events.open_events().is_empty());
        assert_eq!(events.history().len(), 1);
    }

    #[test]
    fn reconcile_respects_muted_rule() {
        let tmp = tempfile::tempdir().unwrap();
        let mut events = EventStore::new(tmp.path());
        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(
            "compaction_backlog".into(),
            RuleConfig {
                muted: true,
                ..Default::default()
            },
        );
        let cat = catalog();

        let mut tripping = base_snapshot(1000);
        tripping.collections.push(tripping_collection());
        let mut history = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        history.push(tripping.clone());

        let r = reconcile_hits(&tripping, &history, &mut events, &cfg, &cat, Vec::new(), 1000);
        assert!(r.opened.is_empty(), "muted rule must not open an event");
        assert!(events.open_events().is_empty());
    }

    #[test]
    fn new_error_category_fires_for_recent_error_category() {
        let now = 100_000i64;
        let window = 30_000i64; // 30s tick window.
        let cats = vec![
            category("err1", "ERROR", now - 5_000), // within window → fire
            category("warn1", "WARN", now - 1_000),  // within window → fire
        ];
        let hits =
            detect_new_error_categories(&cats, &AdvisorConfig::default(), now, window);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.rule_id == "new_error_category"));
        assert!(hits.iter().all(|h| h.scope == Scope::Global));
        assert!(hits.iter().all(|h| h.severity == Severity::Medium));
        assert!(hits.iter().all(|h| !h.recommendation.is_empty()));
    }

    #[test]
    fn new_error_category_ignores_old_category() {
        let now = 100_000i64;
        let window = 30_000i64;
        // first_seen well before the window → not new.
        let cats = vec![category("err_old", "ERROR", now - 90_000)];
        let hits =
            detect_new_error_categories(&cats, &AdvisorConfig::default(), now, window);
        assert!(hits.is_empty());
    }

    #[test]
    fn new_error_category_ignores_info_category() {
        let now = 100_000i64;
        let window = 30_000i64;
        // recent, but INFO severity → not alert-worthy.
        let cats = vec![category("info1", "INFO", now - 1_000)];
        let hits =
            detect_new_error_categories(&cats, &AdvisorConfig::default(), now, window);
        assert!(hits.is_empty());
    }

    #[test]
    fn new_error_category_respects_disabled_and_muted() {
        let now = 100_000i64;
        let window = 30_000i64;
        let cats = vec![category("err1", "ERROR", now - 1_000)];

        let mut disabled = AdvisorConfig::default();
        disabled.rules.insert(
            "new_error_category".into(),
            RuleConfig { enabled: false, ..Default::default() },
        );
        assert!(detect_new_error_categories(&cats, &disabled, now, window).is_empty());

        let mut muted = AdvisorConfig::default();
        muted.rules.insert(
            "new_error_category".into(),
            RuleConfig { muted: true, ..Default::default() },
        );
        assert!(detect_new_error_categories(&cats, &muted, now, window).is_empty());
    }

    #[test]
    fn engine_rule_ids_contains_new_error_category() {
        assert!(engine_rule_ids().contains(&"new_error_category"));
    }

    #[test]
    fn config_save_load_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        // Missing file → default.
        let cfg0 = load_config(tmp.path());
        assert!(cfg0.rules.is_empty());

        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(
            "compaction_backlog".into(),
            RuleConfig {
                muted: true,
                ..Default::default()
            },
        );
        save_config(tmp.path(), &cfg);

        let loaded = load_config(tmp.path());
        assert!(loaded.for_rule("compaction_backlog").muted);
    }
}
