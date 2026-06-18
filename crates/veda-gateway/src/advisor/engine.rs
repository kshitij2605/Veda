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
use crate::advisor::slo::{SloKind, SloRegistry, SloState, SloStatus};
use crate::advisor::synthetic::{run_probe, SyntheticRegistry, SyntheticResults};
use crate::advisor::recall_probe::RecallResults;
use crate::advisor::faithfulness::FaithfulnessResults;
use crate::advisor::incidents::IncidentStore;
use crate::advisor::snapshot::{collect, OpsSnapshot, SnapshotHistory, DEFAULT_HISTORY_CAP};
use crate::advisor::anomaly::detect_anomalies;
use crate::logs::{LogCategory, categorize::CategorizeOptions};
use crate::rest::state::AppState;

/// Config file name under `data_dir`.
const CONFIG_FILE: &str = "_advisor_config.json";
/// SLO registry file name under `data_dir`.
const SLOS_FILE: &str = "_slos.json";
/// SLO rolling-counter state file name under `data_dir`.
const SLO_STATE_FILE: &str = "_slo_state.json";
/// Synthetic probe registry file name under `data_dir`.
const SYNTHETICS_FILE: &str = "_synthetics.json";
/// Synthetic probe results file name under `data_dir`.
const SYNTHETIC_RESULTS_FILE: &str = "_synthetic_results.json";
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
    /// User-defined SLO definitions, persisted to `{data_dir}/_slos.json`.
    pub slos: RwLock<SloRegistry>,
    /// Bucketed rolling SLI counters, persisted to
    /// `{data_dir}/_slo_state.json`.
    pub slo_state: Mutex<SloState>,
    /// User-defined synthetic probes, persisted to
    /// `{data_dir}/_synthetics.json`.
    pub synthetics: RwLock<SyntheticRegistry>,
    /// Per-probe bounded result rings, persisted to
    /// `{data_dir}/_synthetic_results.json`.
    pub synthetic_results: Mutex<SyntheticResults>,
    /// Per-collection proxy-recall samples, persisted to
    /// `{data_dir}/_recall_results.json`. Populated by the dedicated
    /// off-tick recall-probe task ([`crate::advisor::run_recall_probes`]),
    /// READ (cheaply) by [`crate::advisor::snapshot::collect`] to build
    /// `index_health`. The probe NEVER runs inside `run_tick`.
    pub recall_results: Mutex<RecallResults>,
    /// Bounded faithfulness-score series, persisted to
    /// `{data_dir}/_faithfulness_results.json`. Written (brief lock, off
    /// the network path) by `run_agent_turn` when a turn scored
    /// faithfulness; READ by [`run_tick`] to feed the
    /// [`SloKind::Faithfulness`] SLI. Mirrors [`RecallResults`].
    pub faithfulness_results: Mutex<FaithfulnessResults>,
    /// Latest semantic incident clusters + bounded history, persisted to
    /// `{data_dir}/_incidents.json`. Populated by the dedicated off-tick
    /// incident-correlation task ([`crate::advisor::run_incident_correlation`])
    /// — the embedding network calls NEVER run inside `run_tick`.
    pub incidents: Mutex<IncidentStore>,
    /// Shared async HTTP client, reused by the per-tick [`Notifier`].
    pub http: reqwest::Client,
}

impl AdvisorState {
    /// Construct from `data_dir`, loading the persisted config (default
    /// if absent), event store, SLO registry, and SLO state.
    pub fn load(data_dir: &Path) -> Self {
        Self {
            history: Mutex::new(SnapshotHistory::new(DEFAULT_HISTORY_CAP)),
            events: Mutex::new(EventStore::load(data_dir)),
            config: RwLock::new(load_config(data_dir)),
            limiter: Mutex::new(RateLimiter::default()),
            slos: RwLock::new(SloRegistry::load(&data_dir.join(SLOS_FILE))),
            slo_state: Mutex::new(SloState::load(&data_dir.join(SLO_STATE_FILE))),
            synthetics: RwLock::new(SyntheticRegistry::load(&data_dir.join(SYNTHETICS_FILE))),
            synthetic_results: Mutex::new(SyntheticResults::load(
                &data_dir.join(SYNTHETIC_RESULTS_FILE),
            )),
            recall_results: Mutex::new(crate::advisor::recall_probe::RecallResults::load(
                &data_dir.join(crate::advisor::recall_probe::RECALL_RESULTS_FILE),
            )),
            faithfulness_results: Mutex::new(crate::advisor::faithfulness::FaithfulnessResults::load(
                &data_dir.join(crate::advisor::faithfulness::FAITHFULNESS_RESULTS_FILE),
            )),
            incidents: Mutex::new(crate::advisor::incidents::IncidentStore::load(
                &data_dir.join(crate::advisor::incidents::INCIDENTS_FILE),
            )),
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

/// Compute `(good, total)` SLI observations for one SLO over the tick
/// interval, from the new snapshot `snap` and the `prev`ious snapshot.
///
/// * **Availability / ErrorRate**: counter deltas — `total =
///   Δ veda_requests_total`, `good = total − Δ veda_errors_total`
///   (floored at 0). Same SLI; both frame successes as "good".
/// * **Latency{threshold_ms}**: over the new snapshot's bounded
///   `recent_latencies_ms` — `total = len`, `good = count(≤ threshold)`.
/// * **Synthetic**: `(0, 0)` here — synthetic observations come from the
///   probe result ring, not the snapshot, so the caller handles them
///   separately (see [`run_tick`], which counts results via
///   [`SyntheticResults::since`]).
///
/// Returns `(0, 0)` (a no-op the caller skips) when the counters are
/// absent or went backwards, or the latency window is empty.
fn slo_observation(kind: &SloKind, snap: &OpsSnapshot, prev: Option<&OpsSnapshot>) -> (u64, u64) {
    match kind {
        SloKind::Availability | SloKind::ErrorRate => {
            let Some(prev) = prev else { return (0, 0) };
            let counter = |s: &OpsSnapshot, k: &str| s.counters.get(k).copied().unwrap_or(0.0);
            let d_req = counter(snap, "veda_requests_total") - counter(prev, "veda_requests_total");
            let d_err = counter(snap, "veda_errors_total") - counter(prev, "veda_errors_total");
            if d_req < 0.0 || d_err < 0.0 {
                return (0, 0); // counter reset — can't interpret.
            }
            let total = d_req as u64;
            let bad = (d_err as u64).min(total);
            (total - bad, total)
        }
        SloKind::Latency { threshold_ms } => {
            let lats = &snap.query.recent_latencies_ms;
            let total = lats.len() as u64;
            let good = lats.iter().filter(|&&ms| ms <= *threshold_ms).count() as u64;
            (good, total)
        }
        // Synthetic / Recall / Faithfulness observations come from the
        // advisor's result stores, not the snapshot, so the caller gathers
        // them separately (see `run_tick`).
        SloKind::Synthetic { .. } | SloKind::Recall { .. } | SloKind::Faithfulness { .. } => {
            (0, 0)
        }
    }
}

/// Run every synthetic probe whose `interval_secs` has elapsed since its
/// last result (or that has never run), record each [`ProbeResult`], and
/// persist the result rings.
///
/// **Lock discipline:** the synthetics/results locks are held only
/// *briefly* and never across a probe `.await`. We snapshot the due-probe
/// list under a short read lock + a short results lock (to read each
/// probe's last timestamp), drop both, await the probes, then re-lock the
/// results mutex briefly to record + save. A slow probe therefore never
/// freezes the synthetics registry or the next tick.
///
/// Cadence is bounded below by the advisor tick interval (the only
/// in-process timer): a probe with a sub-tick interval simply runs every
/// tick. See [`crate::advisor::synthetic`].
pub async fn run_due_probes(state: &AppState, now_ms: u64) {
    let advisor = state.advisor();

    // Snapshot the full probe list under a short read lock, then drop it.
    let probes = {
        let reg = advisor.synthetics.read().await;
        if reg.list().is_empty() {
            return;
        }
        reg.list().to_vec()
    };

    // Read each probe's last timestamp under a short results lock, drop it.
    let due: Vec<_> = {
        let results = advisor.synthetic_results.lock().await;
        probes
            .into_iter()
            .filter(|p| {
                let last_ts = results
                    .recent(&p.id)
                    .last()
                    .map(|r| r.ts_ms)
                    .unwrap_or(0);
                let interval_ms = p.interval_secs.saturating_mul(1_000);
                last_ts == 0 || now_ms.saturating_sub(last_ts) >= interval_ms
            })
            .collect()
    };
    if due.is_empty() {
        return;
    }

    // Await all due probes with NO advisor lock held.
    let mut outcomes = Vec::with_capacity(due.len());
    for probe in &due {
        let result = run_probe(state, probe).await;
        outcomes.push((probe.id.clone(), result));
    }

    // Re-lock briefly to record + persist.
    let mut results = advisor.synthetic_results.lock().await;
    for (id, result) in outcomes {
        results.record(&id, result);
    }
    results.save(&state.inner.data_dir.join(SYNTHETIC_RESULTS_FILE));
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
/// - **Phase 0 (no advisor lock held across awaits):**
///   [`run_due_probes`] runs any due synthetic self-checks (its
///   synthetics/results locks are taken only briefly, never across a probe
///   await), then `log_categorizer().categorize(...)` — async, may embed
///   over the network — produces the `new_error_category` hits, keeping
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
    let mut snap = collect(state, now).await;

    // Per-tick config snapshot (released immediately).
    let cfg = advisor.config.read().await.clone();
    let rule_catalog = catalog();

    // --- Phase 0a: run any due synthetic self-check probes (locks held
    // only briefly, never across a probe await). Records this tick's
    // ProbeResults so the SLO/summary steps below see them.
    run_due_probes(state, now.max(0) as u64).await;

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

    // --- Phase 1: record SLO observations + reconcile under the advisor
    // locks (NO network). All guards live only inside this block and are
    // dropped before any notify/remediation await below.
    let result = {
        // SLO recording: brief slos/slo_state locks. Use the history's
        // current latest() as the PREVIOUS snapshot (before pushing the
        // new one) so counter deltas span exactly this tick interval.
        let now_u = now.max(0) as u64;

        // PREVIOUS snapshot (history's current latest, before we push the
        // new one). Its `taken_at_ms` is the previous tick time, used to
        // count synthetic results recorded since then for the Synthetic SLI.
        let prev = {
            let history = advisor.history.lock().await;
            history.latest().cloned()
        };
        let prev_tick_ms = prev.as_ref().map(|p| p.taken_at_ms.max(0) as u64);

        // Synthetic summaries: snapshot probe defs (brief read lock) +
        // results (brief lock), derive one summary per probe. Recorded onto
        // the snapshot before reconcile so `synthetic_failing` can read it.
        {
            let probes = {
                let reg = advisor.synthetics.read().await;
                reg.list().to_vec()
            };
            if !probes.is_empty() {
                let results = advisor.synthetic_results.lock().await;
                snap.synthetic_summaries = probes
                    .iter()
                    .map(|p| results.summary(&p.id, &p.name))
                    .collect();
            }
        }

        {
            let defs = advisor.slos.read().await;
            let defs = defs.list();
            if !defs.is_empty() {
                // For Synthetic SLOs we count probe results recorded since
                // the previous tick; gather those counts under a brief
                // results lock (dropped before taking slo_state).
                let synth_obs: Vec<(String, (u64, u64))> = {
                    let results = advisor.synthetic_results.lock().await;
                    defs.iter()
                        .filter_map(|d| match &d.kind {
                            SloKind::Synthetic { probe_id } => {
                                let since = prev_tick_ms.unwrap_or(0);
                                let rs = results.since(probe_id, since);
                                let total = rs.len() as u64;
                                let good = rs.iter().filter(|r| r.ok).count() as u64;
                                Some((d.id.clone(), (good, total)))
                            }
                            _ => None,
                        })
                        .collect()
                };

                // Recall (proxy) observations: read each Recall SLO's
                // latest sample under a brief recall_results lock, scale to
                // good/1000 so `sli_pct = recall%`. Skip when no sample.
                let recall_obs: Vec<(String, (u64, u64))> = {
                    let results = advisor.recall_results.lock().await;
                    defs.iter()
                        .filter_map(|d| match &d.kind {
                            SloKind::Recall { collection, .. } => results
                                .latest_recall(collection)
                                .map(|s| {
                                    let good = (s.recall_at_k * 1000.0).round() as u64;
                                    (d.id.clone(), (good.min(1000), 1000u64))
                                }),
                            _ => None,
                        })
                        .collect()
                };

                // Faithfulness observations: every score recorded since the
                // previous tick, each scaled to good/100 so the accumulated
                // `sli_pct` is the mean faithfulness %. Skip when no samples.
                let faith_obs: Vec<(String, (u64, u64))> = {
                    let results = advisor.faithfulness_results.lock().await;
                    let since = prev_tick_ms.unwrap_or(0);
                    defs.iter()
                        .filter_map(|d| match &d.kind {
                            SloKind::Faithfulness { agent } => {
                                let scores = results.since(agent.as_deref(), since);
                                if scores.is_empty() {
                                    return None;
                                }
                                let mut good = 0u64;
                                let mut total = 0u64;
                                for f in scores {
                                    good += (f * 100.0).round().clamp(0.0, 100.0) as u64;
                                    total += 100;
                                }
                                Some((d.id.clone(), (good, total)))
                            }
                            _ => None,
                        })
                        .collect()
                };

                let mut slo_state = advisor.slo_state.lock().await;
                for d in defs {
                    let (good, total) = match &d.kind {
                        SloKind::Synthetic { .. } => synth_obs
                            .iter()
                            .find(|(id, _)| id == &d.id)
                            .map(|(_, gt)| *gt)
                            .unwrap_or((0, 0)),
                        SloKind::Recall { .. } => recall_obs
                            .iter()
                            .find(|(id, _)| id == &d.id)
                            .map(|(_, gt)| *gt)
                            .unwrap_or((0, 0)),
                        SloKind::Faithfulness { .. } => faith_obs
                            .iter()
                            .find(|(id, _)| id == &d.id)
                            .map(|(_, gt)| *gt)
                            .unwrap_or((0, 0)),
                        _ => slo_observation(&d.kind, &snap, prev.as_ref()),
                    };
                    if total > 0 {
                        slo_state.record(&d.id, good, total, now_u);
                    }
                }
                let statuses: Vec<SloStatus> =
                    defs.iter().map(|d| slo_state.status(d, now_u)).collect();
                slo_state.save(&state.inner.data_dir.join(SLO_STATE_FILE));
                drop(slo_state);
                snap.slo_statuses = statuses;
            }
        }

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
                recent_latencies_ms: Vec::new(),
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
            slo_statuses: Vec::new(),
            synthetic_summaries: Vec::new(),
            index_health: Vec::new(),
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

    fn latency_slo(id: &str, threshold_ms: f64) -> crate::advisor::slo::SloDef {
        crate::advisor::slo::SloDef {
            id: id.into(),
            name: format!("{id} latency"),
            kind: SloKind::Latency { threshold_ms },
            target_pct: 99.0,
            // Tiny windows so a single tick of observations fills them.
            window_secs: 300,
            fast_burn: crate::advisor::slo::default_fast_burn(),
            slow_burn: crate::advisor::slo::default_slow_burn(),
        }
    }

    #[test]
    fn slo_observation_latency_counts_under_threshold() {
        let mut snap = base_snapshot(1000);
        snap.query.recent_latencies_ms = vec![10.0, 20.0, 2000.0, 50.0]; // 1 over 100ms
        let (good, total) = slo_observation(&SloKind::Latency { threshold_ms: 100.0 }, &snap, None);
        assert_eq!((good, total), (3, 4));
    }

    #[test]
    fn slo_observation_availability_uses_counter_deltas() {
        let mut prev = base_snapshot(1000);
        prev.counters.insert("veda_requests_total".into(), 100.0);
        prev.counters.insert("veda_errors_total".into(), 0.0);
        let mut now = base_snapshot(2000);
        now.counters.insert("veda_requests_total".into(), 200.0);
        now.counters.insert("veda_errors_total".into(), 30.0);
        // Δreq=100, Δerr=30 → good 70 / total 100.
        let (good, total) = slo_observation(&SloKind::Availability, &now, Some(&prev));
        assert_eq!((good, total), (70, 100));
        // No previous snapshot → no-op.
        assert_eq!(slo_observation(&SloKind::Availability, &now, None), (0, 0));
    }

    #[test]
    fn slo_fast_burn_rule_opens_event_for_breaching_latency_slo() {
        use crate::advisor::slo::SloState;

        let def = latency_slo("api", 100.0);
        let mut slo_state = SloState::new(60);
        let now_ms_u = 10_000_000u64;
        // 900 over threshold / 1000 → 90% bad, far above the budget;
        // burn = 0.9 / 0.01 = 90x ≥ fast threshold 14.4 → breaching_fast.
        slo_state.record(&def.id, 100, 1000, now_ms_u);
        let status = slo_state.status(&def, now_ms_u);
        assert!(status.breaching_fast, "expected fast breach, got {status:?}");

        let mut snap = base_snapshot(10_000_000);
        snap.slo_statuses = vec![status];

        let tmp = tempfile::tempdir().unwrap();
        let mut events = EventStore::new(tmp.path());
        let cfg = AdvisorConfig::default();
        let cat = catalog();
        let mut history = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        history.push(snap.clone());

        let r = reconcile_hits(&snap, &history, &mut events, &cfg, &cat, Vec::new(), 10_000_000);
        let fast = r
            .opened
            .iter()
            .find(|e| e.rule_id == "slo_burn_rate_fast")
            .expect("slo_burn_rate_fast must open an event");
        assert_eq!(fast.scope, Scope::Slo("api".into()));
        assert!(events.open_events().iter().any(|e| e.id == "slo_burn_rate_fast:slo:api"));
    }

    #[test]
    fn slo_rules_silent_for_healthy_slo() {
        use crate::advisor::slo::SloState;

        let def = latency_slo("api", 100.0);
        let mut slo_state = SloState::new(60);
        let now_ms_u = 10_000_000u64;
        // All under threshold → 100% good → no burn, no breach.
        slo_state.record(&def.id, 1000, 1000, now_ms_u);
        let status = slo_state.status(&def, now_ms_u);
        assert!(!status.breaching_fast);
        assert!(!status.breaching_slow);
        assert!(!status.budget_exhausted);

        let mut snap = base_snapshot(10_000_000);
        snap.slo_statuses = vec![status];
        let history = {
            let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
            h.push(snap.clone());
            h
        };
        let cfg = AdvisorConfig::default();
        let cat = catalog();
        let hits = run_rules(&cat, &snap, &history, &cfg);
        assert!(
            !hits.iter().any(|h| h.rule_id.starts_with("slo_")),
            "healthy SLO must trip no slo_* rule: {hits:?}"
        );
    }

    fn synthetic_slo(id: &str, probe_id: &str) -> crate::advisor::slo::SloDef {
        crate::advisor::slo::SloDef {
            id: id.into(),
            name: format!("{id} synthetic"),
            kind: SloKind::Synthetic {
                probe_id: probe_id.into(),
            },
            target_pct: 99.0,
            window_secs: 300,
            fast_burn: crate::advisor::slo::default_fast_burn(),
            slow_burn: crate::advisor::slo::default_slow_burn(),
        }
    }

    fn probe_result(ts_ms: u64, ok: bool) -> crate::advisor::synthetic::ProbeResult {
        crate::advisor::synthetic::ProbeResult {
            ts_ms,
            ok,
            latency_ms: 1.0,
            error: if ok { None } else { Some("boom".into()) },
        }
    }

    #[test]
    fn synthetic_results_feed_the_synthetic_sli() {
        use crate::advisor::synthetic::SyntheticResults;

        // Inject mostly-failing probe results, count them since the prev
        // tick, and record into the SLO counter exactly as run_tick does.
        let mut results = SyntheticResults::new();
        let now_u = 10_000_000u64;
        for i in 0..10 {
            // 8 of 10 fail.
            results.record("p", probe_result(now_u - 1000 + i, i >= 8));
        }
        let def = synthetic_slo("s", "p");
        let since = now_u - 2000; // before all results
        let rs = results.since("p", since);
        let total = rs.len() as u64;
        let good = rs.iter().filter(|r| r.ok).count() as u64;
        assert_eq!((good, total), (2, 10));

        let mut slo_state = SloState::new(60);
        slo_state.record(&def.id, good, total, now_u);
        let status = slo_state.status(&def, now_u);
        // SLI 20% vs target 99% → non-compliant + budget exhausted + burning.
        assert!((status.sli_pct.unwrap() - 20.0).abs() < 1e-9);
        assert!(!status.compliant);
        assert!(status.budget_exhausted);
        assert!(status.breaching_fast);
    }

    #[test]
    fn synthetic_failing_rule_opens_event_for_failing_probe() {
        use crate::advisor::synthetic::SyntheticResults;

        // A probe with a trailing run of failures → consecutive_failures >= K.
        let mut results = SyntheticResults::new();
        results.record("health", probe_result(1, true));
        for i in 0..4 {
            results.record("health", probe_result(2 + i, false)); // 4 trailing fails
        }
        let summary = results.summary("health", "health probe");
        assert_eq!(summary.consecutive_failures, 4);

        let mut snap = base_snapshot(10_000_000);
        snap.synthetic_summaries = vec![summary];

        let tmp = tempfile::tempdir().unwrap();
        let mut events = EventStore::new(tmp.path());
        let cfg = AdvisorConfig::default();
        let cat = catalog();
        let mut history = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        history.push(snap.clone());

        let r = reconcile_hits(&snap, &history, &mut events, &cfg, &cat, Vec::new(), 10_000_000);
        let hit = r
            .opened
            .iter()
            .find(|e| e.rule_id == "synthetic_failing")
            .expect("synthetic_failing must open an event");
        assert_eq!(hit.scope, Scope::Synthetic("health".into()));
        assert!(events
            .open_events()
            .iter()
            .any(|e| e.id == "synthetic_failing:synthetic:health"));
    }

    #[test]
    fn synthetic_failing_rule_silent_for_healthy_probe() {
        use crate::advisor::synthetic::SyntheticResults;

        let mut results = SyntheticResults::new();
        for i in 0..5 {
            results.record("health", probe_result(i, true)); // all ok
        }
        let summary = results.summary("health", "health probe");
        assert_eq!(summary.consecutive_failures, 0);

        let mut snap = base_snapshot(10_000_000);
        snap.synthetic_summaries = vec![summary];
        let history = {
            let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
            h.push(snap.clone());
            h
        };
        let hits = run_rules(&catalog(), &snap, &history, &AdvisorConfig::default());
        assert!(
            !hits.iter().any(|h| h.rule_id == "synthetic_failing"),
            "healthy probe must trip no synthetic_failing: {hits:?}"
        );
    }

    fn recall_slo(id: &str, collection: &str) -> crate::advisor::slo::SloDef {
        crate::advisor::slo::SloDef {
            id: id.into(),
            name: format!("{id} recall"),
            kind: SloKind::Recall {
                collection: collection.into(),
                k: Some(10),
            },
            target_pct: 90.0,
            window_secs: 300,
            fast_burn: crate::advisor::slo::default_fast_burn(),
            slow_burn: crate::advisor::slo::default_slow_burn(),
        }
    }

    fn faith_slo(id: &str, agent: Option<&str>) -> crate::advisor::slo::SloDef {
        crate::advisor::slo::SloDef {
            id: id.into(),
            name: format!("{id} faithfulness"),
            kind: SloKind::Faithfulness {
                agent: agent.map(|a| a.to_owned()),
            },
            target_pct: 90.0,
            window_secs: 300,
            fast_burn: crate::advisor::slo::default_fast_burn(),
            slow_burn: crate::advisor::slo::default_slow_burn(),
        }
    }

    #[test]
    fn recall_sample_feeds_recall_sli_as_proxy() {
        use crate::advisor::slo::SloState;

        // recall 0.7 → good 700 / total 1000 → sli 70% vs target 90 →
        // non-compliant + budget exhausted (consumed = 0.30/0.10 = 3.0×).
        // The slo_budget_exhausted rule fires on this unchanged.
        let def = recall_slo("docs", "docs");
        let now_u = 10_000_000u64;
        let recall_at_k: f64 = 0.7;
        let good = (recall_at_k * 1000.0).round() as u64;

        let mut slo_state = SloState::new(60);
        slo_state.record(&def.id, good, 1000, now_u);
        let status = slo_state.status(&def, now_u);
        assert!((status.sli_pct.unwrap() - 70.0).abs() < 1e-9);
        assert!(!status.compliant);
        assert!(status.budget_exhausted);
        // burn = 0.30 / 0.10 = 3.0× (below the 14.4 fast threshold but a
        // sustained over-budget burn the slow window catches).
        assert!((status.burn_rate_fast.unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn healthy_recall_sample_is_compliant() {
        use crate::advisor::slo::SloState;
        let def = recall_slo("docs", "docs");
        let now_u = 10_000_000u64;
        let good = (0.99_f64 * 1000.0).round() as u64; // 990
        let mut slo_state = SloState::new(60);
        slo_state.record(&def.id, good, 1000, now_u);
        let status = slo_state.status(&def, now_u);
        assert!((status.sli_pct.unwrap() - 99.0).abs() < 1e-9);
        assert!(status.compliant);
        assert!(!status.breaching_fast);
        assert!(!status.budget_exhausted);
    }

    #[test]
    fn faithfulness_samples_feed_mean_faithfulness_sli() {
        use crate::advisor::faithfulness::FaithfulnessResults;
        use crate::advisor::slo::SloState;

        // Seeded scores with mean 0.6 → accumulating good/100 each → sli
        // ≈ 60%. Mirrors how run_tick gathers faith_obs.
        let mut fr = FaithfulnessResults::new();
        let now_u = 10_000_000u64;
        for (i, f) in [0.5, 0.6, 0.7].iter().enumerate() {
            fr.record("veda-agent", *f, now_u - 100 + i as u64);
        }
        let def = faith_slo("agent-faith", Some("veda-agent"));
        let since = now_u - 1000;
        let scores = fr.since(Some("veda-agent"), since);
        assert_eq!(scores.len(), 3);

        let mut good = 0u64;
        let mut total = 0u64;
        for f in scores {
            good += (f * 100.0).round().clamp(0.0, 100.0) as u64;
            total += 100;
        }
        // (50 + 60 + 70) / 300 = 60%.
        assert_eq!((good, total), (180, 300));

        let mut slo_state = SloState::new(60);
        slo_state.record(&def.id, good, total, now_u);
        let status = slo_state.status(&def, now_u);
        assert!((status.sli_pct.unwrap() - 60.0).abs() < 1e-9);
        assert!(!status.compliant, "60% mean faithfulness vs target 90");
    }

    #[tokio::test]
    async fn run_tick_records_recall_and_faithfulness_slis() {
        // End-to-end through run_tick: seed a Recall SLO + a recall sample
        // (0.7) and a Faithfulness SLO + scores (mean 0.6); after a tick the
        // slo_state status should reflect ~70% and ~60% SLIs.
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let advisor = state.advisor();
        let now_u = now_ms().max(0) as u64;

        // Register the two quality SLOs.
        {
            let mut slos = advisor.slos.write().await;
            slos.upsert(recall_slo("docs-recall", "docs"));
            slos.upsert(faith_slo("agent-faith", Some("veda-agent")));
        }
        // Seed a recall sample for "docs".
        {
            let mut rr = advisor.recall_results.lock().await;
            rr.record(crate::advisor::recall_probe::RecallSample {
                collection: "docs".into(),
                index_type: "hnsw".into(),
                recall_at_k: 0.7,
                k: 10,
                sampled_queries: 20,
                p95_ms: None,
                params: Default::default(),
                ts_ms: now_u,
                is_proxy: true,
            });
        }
        // Seed faithfulness scores after the (empty) previous-tick boundary.
        {
            let mut fr = advisor.faithfulness_results.lock().await;
            for f in [0.5, 0.6, 0.7] {
                fr.record("veda-agent", f, now_u);
            }
        }

        run_tick(&state).await;

        let slo_state = advisor.slo_state.lock().await;
        let now_after = now_ms().max(0) as u64;
        let r_status = slo_state.status(&recall_slo("docs-recall", "docs"), now_after);
        assert!(
            (r_status.sli_pct.unwrap() - 70.0).abs() < 1e-6,
            "recall sli {:?}",
            r_status.sli_pct
        );
        let f_status = slo_state.status(&faith_slo("agent-faith", Some("veda-agent")), now_after);
        assert!(
            (f_status.sli_pct.unwrap() - 60.0).abs() < 1e-6,
            "faith sli {:?}",
            f_status.sli_pct
        );
    }

    #[tokio::test]
    async fn run_tick_skips_quality_slis_with_no_sample() {
        // A Recall SLO with no sample + a Faithfulness SLO with no scores →
        // the tick records nothing → status has no data (None sli, compliant).
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let advisor = state.advisor();
        {
            let mut slos = advisor.slos.write().await;
            slos.upsert(recall_slo("docs-recall", "missing-coll"));
            slos.upsert(faith_slo("agent-faith", Some("nobody")));
        }
        run_tick(&state).await;
        let slo_state = advisor.slo_state.lock().await;
        let now_after = now_ms().max(0) as u64;
        let r = slo_state.status(&recall_slo("docs-recall", "missing-coll"), now_after);
        assert_eq!(r.sli_pct, None);
        assert!(r.compliant);
        let f = slo_state.status(&faith_slo("agent-faith", Some("nobody")), now_after);
        assert_eq!(f.sli_pct, None);
        assert!(f.compliant);
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
