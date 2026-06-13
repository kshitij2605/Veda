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
use crate::advisor::rules::{catalog, run_rules, Rule};
use crate::advisor::snapshot::{collect, OpsSnapshot, SnapshotHistory, DEFAULT_HISTORY_CAP};
use crate::advisor::anomaly::detect_anomalies;
use crate::rest::state::AppState;

/// Config file name under `data_dir`.
const CONFIG_FILE: &str = "_advisor_config.json";
/// Default tick interval (seconds) when the env override is unset.
const DEFAULT_INTERVAL_SECS: u64 = 30;

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
    now_ms: i64,
) -> ReconcileResult {
    // 1. Combine threshold-rule hits with anomaly hits.
    let mut hits = run_rules(catalog, snap, history, cfg);
    hits.extend(detect_anomalies(history, cfg));

    // 2. Reconcile into open/close transitions, then persist.
    let result = events.reconcile(hits, now_ms);
    events.save();
    result
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

    // --- Phase 1: reconcile under the history+events locks (NO network).
    // The guards live only inside this block and are dropped before any
    // notify/remediation await below.
    let result = {
        let mut history = advisor.history.lock().await;
        history.push(snap.clone());
        let mut events = advisor.events.lock().await;
        reconcile_hits(&snap, &history, &mut events, &cfg, &rule_catalog, now)
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
    let interval_secs = std::env::var("VEDA_ADVISOR_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_INTERVAL_SECS);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
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
            counters: HashMap::new(),
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

        let r1 = reconcile_hits(&tripping, &history, &mut events, &cfg, &cat, 1000);
        assert_eq!(r1.opened.len(), 1, "expected one opened event");
        assert_eq!(r1.opened[0].rule_id, "compaction_backlog");
        assert_eq!(events.open_events().len(), 1);
        assert_eq!(events.open_events()[0].status, EventStatus::Open);

        // --- round 2: a clean snapshot closes it ---
        let clean = base_snapshot(2000);
        history.push(clean.clone());
        let r2 = reconcile_hits(&clean, &history, &mut events, &cfg, &cat, 2000);
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

        let r = reconcile_hits(&tripping, &history, &mut events, &cfg, &cat, 1000);
        assert!(r.opened.is_empty(), "muted rule must not open an event");
        assert!(events.open_events().is_empty());
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
