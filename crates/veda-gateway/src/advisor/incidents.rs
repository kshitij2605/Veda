//! Semantic incident correlation & dedup for ops-advisor events.
//!
//! Recurring / duplicate [`OpsEvent`]s are noisy: the same underlying
//! failure shows up as many events (differently worded, across rounds, or
//! across scopes). [`cluster_incidents`] groups them into a small set of
//! [`Incident`]s so the operator sees one row per distinct problem.
//!
//! Two clustering paths exist, mirroring
//! [`crate::logs::categorize`]:
//!
//!   * **Embedding path** — when an embedding for each event is supplied,
//!     events are greedily clustered by cosine similarity of their
//!     `title + " " + detail` embedding against running-mean centroids
//!     ([`cluster_incidents`] with `embeddings = Some(..)`). Two
//!     differently-worded "disk full" events collapse into one incident.
//!   * **Fallback path** — when no embedder is configured (`embeddings =
//!     None`), events are grouped by `(rule_id, scope.key())`. No network.
//!
//! [`compute_incidents`] is the async orchestrator: it clones the open +
//! recent events out of the [`EventStore`] under a brief lock, drops the
//! lock, embeds the distinct `title+detail` strings OFF any advisor lock
//! (cache-free, but the embedding network await never runs while a lock is
//! held), then clusters. The pure [`cluster_incidents`] is unit-testable
//! without any embedder.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;

use crate::advisor::event::{OpsEvent, Severity};
use crate::logs::categorize::cosine;
use crate::rest::state::AppState;

/// Default cosine threshold for the embedding clustering path. Matches
/// the log categorizer's tuned value.
pub const DEFAULT_THRESHOLD: f32 = 0.82;
/// Maximum number of distinct event texts embedded per correlation pass.
const MAX_TEXTS: usize = 200;
/// Per-correlation cap on retained incident snapshots (bounded history).
pub const MAX_HISTORY: usize = 100;
/// Incident-correlation results file name under `data_dir`.
pub const INCIDENTS_FILE: &str = "_incidents.json";
/// Default seconds between incident-correlation passes (env override).
pub const DEFAULT_INCIDENT_INTERVAL_SECS: u64 = 300;

/// A cluster of semantically-similar (or same-rule/scope) ops events —
/// one distinct incident the operator should act on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Incident {
    /// Stable-ish id derived from the representative event id.
    pub id: String,
    /// Representative title (highest-severity, then earliest member).
    pub representative: String,
    /// Member event ids collapsed into this incident.
    pub member_event_ids: Vec<String>,
    /// Distinct rule ids contributing to this incident.
    pub rule_ids: Vec<String>,
    /// Number of member events (alert-noise collapsed).
    pub count: u32,
    /// Highest severity across members.
    pub max_severity: Severity,
    /// Earliest `opened_at_ms` across members.
    pub first_seen_ms: i64,
    /// Latest `opened_at_ms` across members.
    pub last_seen_ms: i64,
}

/// Mutable accumulator while clustering events into an incident.
struct Accum {
    /// Running-mean centroid (empty in the fallback path).
    centroid: Vec<f32>,
    /// The member chosen as representative so far (highest sev, then
    /// earliest opened).
    rep_event: OpsEvent,
    member_event_ids: Vec<String>,
    rule_ids: Vec<String>,
    count: u32,
    max_severity: Severity,
    first_seen_ms: i64,
    last_seen_ms: i64,
}

impl Accum {
    fn new(ev: &OpsEvent, centroid: Vec<f32>) -> Self {
        Accum {
            centroid,
            rep_event: ev.clone(),
            member_event_ids: vec![ev.id.clone()],
            rule_ids: vec![ev.rule_id.clone()],
            count: 1,
            max_severity: ev.severity,
            first_seen_ms: ev.opened_at_ms,
            last_seen_ms: ev.opened_at_ms,
        }
    }

    /// Fold another event into this incident, updating the representative
    /// when the new event is more severe, or equally severe but opened
    /// earlier.
    fn add(&mut self, ev: &OpsEvent) {
        self.member_event_ids.push(ev.id.clone());
        if !self.rule_ids.contains(&ev.rule_id) {
            self.rule_ids.push(ev.rule_id.clone());
        }
        self.count += 1;
        if ev.severity > self.max_severity {
            self.max_severity = ev.severity;
        }
        self.first_seen_ms = self.first_seen_ms.min(ev.opened_at_ms);
        self.last_seen_ms = self.last_seen_ms.max(ev.opened_at_ms);
        // Representative: highest severity wins; tie → earliest opened.
        let better = ev.severity > self.rep_event.severity
            || (ev.severity == self.rep_event.severity
                && ev.opened_at_ms < self.rep_event.opened_at_ms);
        if better {
            self.rep_event = ev.clone();
        }
    }

    fn finish(self) -> Incident {
        Incident {
            id: format!("inc_{}", self.rep_event.id),
            representative: self.rep_event.title,
            member_event_ids: self.member_event_ids,
            rule_ids: self.rule_ids,
            count: self.count,
            max_severity: self.max_severity,
            first_seen_ms: self.first_seen_ms,
            last_seen_ms: self.last_seen_ms,
        }
    }
}

/// Cluster `events` into [`Incident`]s, collapsing recurring/duplicate
/// alerts.
///
/// * **Embedding path** (`embeddings = Some(embs)`, `embs[i]` the embedding
///   of `events[i]`'s `title + " " + detail`): greedy clustering by cosine
///   against running-mean centroids, exactly like
///   [`crate::logs::categorize::cluster_embedded`]. An empty embedding for
///   an event makes it its own incident (it can never reach the threshold).
/// * **Fallback path** (`embeddings = None`): group by
///   `(rule_id, scope.key())`. No similarity math, fully deterministic.
///
/// The representative is the highest-severity member (ties broken by
/// earliest `opened_at_ms`). Deterministic in input order. Pure — given
/// the embeddings, it does no I/O, so it is directly unit-testable.
pub fn cluster_incidents(
    events: &[OpsEvent],
    embeddings: Option<&[Vec<f32>]>,
    threshold: f32,
) -> Vec<Incident> {
    match embeddings {
        Some(embs) if embs.len() == events.len() => {
            cluster_by_embedding(events, embs, threshold)
        }
        // No embeddings (or a length mismatch we won't trust) → fallback.
        _ => cluster_by_rule_scope(events),
    }
}

/// Embedding path: greedy cosine clustering against running-mean centroids.
fn cluster_by_embedding(events: &[OpsEvent], embs: &[Vec<f32>], threshold: f32) -> Vec<Incident> {
    let mut accs: Vec<Accum> = Vec::new();
    for (ev, emb) in events.iter().zip(embs.iter()) {
        let mut best: Option<usize> = None;
        let mut best_sim = threshold;
        if !emb.is_empty() {
            for (i, acc) in accs.iter().enumerate() {
                let sim = cosine(&acc.centroid, emb);
                if sim >= best_sim {
                    best_sim = sim;
                    best = Some(i);
                }
            }
        }
        match best {
            Some(i) => {
                let acc = &mut accs[i];
                // Running-mean centroid update.
                let n = acc.count as f32;
                for k in 0..acc.centroid.len().min(emb.len()) {
                    acc.centroid[k] = (acc.centroid[k] * n + emb[k]) / (n + 1.0);
                }
                acc.add(ev);
            }
            None => accs.push(Accum::new(ev, emb.clone())),
        }
    }
    accs.into_iter().map(Accum::finish).collect()
}

/// Fallback path: group by `(rule_id, scope.key())`, deterministic in
/// first-seen (input) order.
fn cluster_by_rule_scope(events: &[OpsEvent]) -> Vec<Incident> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Accum> = HashMap::new();
    for ev in events {
        let key = format!("{}|{}", ev.rule_id, ev.scope.key());
        match map.get_mut(&key) {
            Some(acc) => acc.add(ev),
            None => {
                order.push(key.clone());
                map.insert(key, Accum::new(ev, Vec::new()));
            }
        }
    }
    order
        .into_iter()
        .filter_map(|k| map.remove(&k))
        .map(Accum::finish)
        .collect()
}

/// Sort incidents most-urgent-first: severity desc, then count desc, then
/// id for determinism.
fn sort_incidents(incidents: &mut [Incident]) {
    incidents.sort_by(|a, b| {
        b.max_severity
            .cmp(&a.max_severity)
            .then(b.count.cmp(&a.count))
            .then(a.id.cmp(&b.id))
    });
}

/// Recompute incidents from the advisor's current open + recent (history)
/// events.
///
/// **Lock discipline:** the `events` lock is taken only *briefly* to clone
/// the events out, then DROPPED before any embedding `.await`. The
/// embedding-registry read lock (an `AppState` lock, never an advisor
/// lock) is likewise dropped before the network embed calls. No advisor
/// lock is ever held across an embed await. Falls back to the
/// rule/scope grouping when no text embedder is configured or any embed
/// fails.
pub async fn compute_incidents(state: &AppState) -> Vec<Incident> {
    let advisor = state.advisor();

    // Brief events lock: clone open + history events out, then drop it.
    let events: Vec<OpsEvent> = {
        let store = advisor.events.lock().await;
        let mut all = store.open_events();
        all.extend(store.history().iter().cloned());
        all
    };
    if events.is_empty() {
        return Vec::new();
    }

    // Resolve a text embedding model config (clone it so we don't hold the
    // registry borrow across the awaits). None → fallback path.
    let text_model = {
        let registry = state.embedding_registry().read().await;
        registry
            .find_by_modality(Modality::Text)
            .first()
            .map(|c| (*c).clone())
    };

    let embeddings: Option<Vec<Vec<f32>>> = match text_model {
        Some(model) => embed_events(&events, model).await,
        None => None,
    };

    let mut incidents = cluster_incidents(&events, embeddings.as_deref(), DEFAULT_THRESHOLD);
    sort_incidents(&mut incidents);
    incidents
}

/// Embed each event's `title + " " + detail` (deduping identical texts via
/// a small in-pass cache). Returns `None` on the first embedding error so
/// the caller falls back to the rule/scope grouping. NO advisor lock is
/// held here.
async fn embed_events(
    events: &[OpsEvent],
    model: veda_core::embedding::EmbeddingModelConfig,
) -> Option<Vec<Vec<f32>>> {
    let client = EmbeddingClient::new(model);

    // Embed distinct texts only (most events repeat across rounds/history).
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(events.len());
    for ev in events {
        let text = format!("{} {}", ev.title, ev.detail);
        if let Some(emb) = cache.get(&text) {
            out.push(emb.clone());
            continue;
        }
        if cache.len() >= MAX_TEXTS {
            // Too many distinct texts → bail to fallback rather than fan
            // out an unbounded number of network calls.
            return None;
        }
        match client.embed_text(&text).await {
            Ok(emb) => {
                cache.insert(text, emb.clone());
                out.push(emb);
            }
            Err(_) => return None,
        }
    }
    Some(out)
}

/// Latest incident clusters + a small bounded history, persisted to
/// `{data_dir}/_incidents.json`. Mirrors
/// [`crate::advisor::recall_probe::RecallResults`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncidentStore {
    /// The most recent correlation pass's incidents.
    latest: Vec<Incident>,
    /// Newest-last ring of prior passes' incidents (one entry per
    /// recorded pass), bounded to [`MAX_HISTORY`].
    history: Vec<Vec<Incident>>,
}

impl IncidentStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the latest incidents, pushing the previous set onto the
    /// bounded history ring.
    pub fn record(&mut self, incidents: Vec<Incident>) {
        if !self.latest.is_empty() {
            self.history.push(std::mem::take(&mut self.latest));
            if self.history.len() > MAX_HISTORY {
                let excess = self.history.len() - MAX_HISTORY;
                self.history.drain(0..excess);
            }
        }
        self.latest = incidents;
    }

    /// The latest correlation pass's incidents.
    pub fn latest(&self) -> &[Incident] {
        &self.latest
    }

    /// Persist to `path` as pretty JSON. Failure is logged, not surfaced.
    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    tracing::warn!(error = %e, path = %path.display(), "incidents: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "incidents: serialize failed"),
        }
    }

    /// Load from `path`. Missing/unparseable → empty (logged on parse error).
    pub fn load(path: &Path) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Self::new(),
        };
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "incidents: parse failed, starting empty");
                Self::new()
            }
        }
    }
}

/// The incident-correlation cadence in seconds (env override, default
/// [`DEFAULT_INCIDENT_INTERVAL_SECS`]).
pub fn incident_interval_secs() -> u64 {
    std::env::var("VEDA_INCIDENT_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_INCIDENT_INTERVAL_SECS)
}

/// Recompute incidents from the current events and store + persist them.
///
/// Invoked exclusively from the dedicated [`spawn_incident_correlator`]
/// task, **never** from `run_tick`, so the embedding network calls never
/// run inside the advisor's locked tick. [`compute_incidents`] keeps every
/// embed await off the advisor lock; this function only re-locks the
/// `incidents` mutex briefly to record + save.
pub async fn run_incident_correlation(state: &AppState) {
    let incidents = compute_incidents(state).await;
    let advisor = state.advisor();
    let mut store = advisor.incidents.lock().await;
    store.record(incidents);
    store.save(&state.inner.data_dir.join(INCIDENTS_FILE));
}

/// Spawn the background incident-correlation loop. Ticks every
/// [`incident_interval_secs`] (default 300s) — a *separate* tokio task from
/// [`crate::advisor::spawn_advisor`], so the embedding network calls run
/// OFF the advisor tick. A failing/slow pass cannot crash the loop.
pub fn spawn_incident_correlator(state: AppState) -> tokio::task::JoinHandle<()> {
    let secs = incident_interval_secs();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(secs));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            run_incident_correlation(&state).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::event::{EventStatus, RemediationAction, Scope};

    fn ev(id_rule: &str, scope: Scope, sev: Severity, title: &str, opened: i64) -> OpsEvent {
        OpsEvent {
            id: OpsEvent::make_id(id_rule, &scope),
            rule_id: id_rule.to_owned(),
            scope,
            severity: sev,
            status: EventStatus::Open,
            title: title.to_owned(),
            detail: title.to_owned(),
            recommendation: String::new(),
            runbook: None,
            remediation: RemediationAction::None,
            opened_at_ms: opened,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }

    #[test]
    fn embedding_path_clusters_similar_events_into_one_incident() {
        // Two differently-worded "disk full" events with near-identical
        // embeddings → ONE incident; a distinct event stays separate.
        let events = vec![
            ev("disk_a", Scope::Node("n1".into()), Severity::Medium, "Disk almost full on n1", 100),
            ev("disk_b", Scope::Node("n2".into()), Severity::High, "Low free space on node n2", 200),
            ev("latency", Scope::Global, Severity::Low, "Query latency elevated", 300),
        ];
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0], // ~identical to the first → merges
            vec![0.0, 1.0, 0.0],   // orthogonal → separate
        ];
        let incidents = cluster_incidents(&events, Some(&embs), 0.9);
        assert_eq!(incidents.len(), 2, "expected 2 incidents, got {incidents:?}");

        let disk = incidents.iter().find(|i| i.count == 2).expect("a 2-member incident");
        assert_eq!(disk.count, 2);
        assert_eq!(disk.member_event_ids.len(), 2);
        // Representative is the highest-severity member (the High n2 event).
        assert_eq!(disk.max_severity, Severity::High);
        assert_eq!(disk.representative, "Low free space on node n2");
        assert_eq!(disk.first_seen_ms, 100);
        assert_eq!(disk.last_seen_ms, 200);
        // Both rule ids surfaced.
        assert!(disk.rule_ids.contains(&"disk_a".to_owned()));
        assert!(disk.rule_ids.contains(&"disk_b".to_owned()));

        let solo = incidents.iter().find(|i| i.count == 1).expect("a 1-member incident");
        assert_eq!(solo.representative, "Query latency elevated");
    }

    #[test]
    fn embedding_path_keeps_distinct_events_separate() {
        let events = vec![
            ev("a", Scope::Global, Severity::High, "thing a", 1),
            ev("b", Scope::Global, Severity::High, "thing b", 2),
        ];
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let incidents = cluster_incidents(&events, Some(&embs), 0.82);
        assert_eq!(incidents.len(), 2);
    }

    #[test]
    fn fallback_path_groups_by_rule_and_scope() {
        // No embeddings → group by (rule_id, scope.key()). Same rule on the
        // same scope (a recurring alert) collapses; different scope splits.
        let events = vec![
            ev("compaction_backlog", Scope::Collection("docs".into()), Severity::Medium, "Compaction backlog on docs", 10),
            ev("compaction_backlog", Scope::Collection("docs".into()), Severity::High, "Compaction backlog on docs (worse)", 20),
            ev("compaction_backlog", Scope::Collection("imgs".into()), Severity::Low, "Compaction backlog on imgs", 30),
        ];
        let incidents = cluster_incidents(&events, None, DEFAULT_THRESHOLD);
        assert_eq!(incidents.len(), 2, "docs collapses, imgs separate");

        let docs = incidents.iter().find(|i| i.count == 2).expect("docs incident");
        assert_eq!(docs.count, 2);
        assert_eq!(docs.max_severity, Severity::High);
        assert_eq!(docs.first_seen_ms, 10);
        assert_eq!(docs.last_seen_ms, 20);
        // Tie-break representative: equal severity isn't the case here, the
        // High event wins outright.
        assert_eq!(docs.representative, "Compaction backlog on docs (worse)");

        let imgs = incidents.iter().find(|i| i.count == 1).expect("imgs incident");
        assert_eq!(imgs.representative, "Compaction backlog on imgs");
    }

    #[test]
    fn representative_breaks_severity_ties_by_earliest() {
        // Two equal-severity members → earliest opened is representative.
        let events = vec![
            ev("a", Scope::Node("n1".into()), Severity::High, "later opened", 500),
            ev("b", Scope::Node("n2".into()), Severity::High, "earlier opened", 100),
        ];
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let incidents = cluster_incidents(&events, Some(&embs), 0.5);
        assert_eq!(incidents.len(), 1);
        assert_eq!(incidents[0].representative, "earlier opened");
        assert_eq!(incidents[0].count, 2);
    }

    #[test]
    fn empty_events_yield_no_incidents() {
        assert!(cluster_incidents(&[], None, DEFAULT_THRESHOLD).is_empty());
        assert!(cluster_incidents(&[], Some(&[]), DEFAULT_THRESHOLD).is_empty());
    }

    #[test]
    fn length_mismatch_falls_back_to_rule_scope() {
        // embeddings shorter than events → treated as the fallback path.
        let events = vec![
            ev("r", Scope::Global, Severity::High, "a", 1),
            ev("r", Scope::Global, Severity::High, "b", 2),
        ];
        let incidents = cluster_incidents(&events, Some(&[vec![1.0, 0.0]]), 0.5);
        // Fallback groups by (rule_id, scope) → both share r:global → 1.
        assert_eq!(incidents.len(), 1);
        assert_eq!(incidents[0].count, 2);
    }

    // ---- IncidentStore --------------------------------------------------

    fn incident(id: &str, count: u32) -> Incident {
        Incident {
            id: id.into(),
            representative: format!("rep {id}"),
            member_event_ids: vec![id.into()],
            rule_ids: vec!["r".into()],
            count,
            max_severity: Severity::High,
            first_seen_ms: 1,
            last_seen_ms: 2,
        }
    }

    #[test]
    fn store_record_replaces_latest_and_rings_history() {
        let mut store = IncidentStore::new();
        assert!(store.latest().is_empty());
        store.record(vec![incident("a", 1)]);
        assert_eq!(store.latest().len(), 1);
        store.record(vec![incident("b", 2), incident("c", 3)]);
        assert_eq!(store.latest().len(), 2);
        assert_eq!(store.latest()[0].id, "b");
    }

    #[test]
    fn store_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(INCIDENTS_FILE);
        let mut store = IncidentStore::new();
        store.record(vec![incident("a", 4)]);
        store.save(&path);

        let loaded = IncidentStore::load(&path);
        assert_eq!(loaded.latest().len(), 1);
        assert_eq!(loaded.latest()[0].id, "a");
        assert_eq!(loaded.latest()[0].count, 4);
    }

    #[test]
    fn store_load_missing_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(IncidentStore::load(&dir.path().join("nope.json")).latest().is_empty());
    }

    #[tokio::test]
    async fn compute_incidents_falls_back_without_embedder() {
        // No embedder configured → rule/scope fallback over seeded events.
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![
                    crate::advisor::rules::RuleHit {
                        rule_id: "compaction_backlog".into(),
                        scope: Scope::Collection("docs".into()),
                        severity: Severity::Medium,
                        title: "Compaction backlog on docs".into(),
                        detail: "lots of sealed segments".into(),
                        recommendation: "compact".into(),
                        runbook: None,
                        remediation: RemediationAction::None,
                    },
                    crate::advisor::rules::RuleHit {
                        rule_id: "disk_pressure".into(),
                        scope: Scope::Node("n1".into()),
                        severity: Severity::High,
                        title: "Disk pressure on n1".into(),
                        detail: "free space low".into(),
                        recommendation: "free disk".into(),
                        runbook: None,
                        remediation: RemediationAction::None,
                    },
                ],
                1000,
            );
        }
        let incidents = compute_incidents(&state).await;
        assert_eq!(incidents.len(), 2, "two distinct rule/scope incidents");
        // Sorted most-severe first.
        assert_eq!(incidents[0].max_severity, Severity::High);
    }

    #[tokio::test]
    async fn compute_incidents_empty_on_fresh_state() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        assert!(compute_incidents(&state).await.is_empty());
    }

    #[tokio::test]
    async fn run_incident_correlation_stores_results() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        {
            let mut store = state.advisor().events.lock().await;
            store.reconcile(
                vec![crate::advisor::rules::RuleHit {
                    rule_id: "r".into(),
                    scope: Scope::Global,
                    severity: Severity::High,
                    title: "boom".into(),
                    detail: "it broke".into(),
                    recommendation: "fix".into(),
                    runbook: None,
                    remediation: RemediationAction::None,
                }],
                1000,
            );
        }
        run_incident_correlation(&state).await;
        let store = state.advisor().incidents.lock().await;
        assert_eq!(store.latest().len(), 1);
        assert_eq!(store.latest()[0].count, 1);
    }
}
