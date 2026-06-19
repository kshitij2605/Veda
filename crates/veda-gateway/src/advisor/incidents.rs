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
use crate::advisor::notify::Notifier;
use crate::advisor::rca::{correlate_incident, RcaBundle};
use crate::logs::categorize::cosine;
use crate::rest::insights::rca_summary;
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
/// Default novelty threshold: an incident is novel when its score
/// (`1 - max_cosine_to_history`) is at or above this — i.e. its nearest
/// historical incident is below 0.75 cosine.
pub const NOVELTY_THRESHOLD: f64 = 0.25;
/// Minimum number of historical incident embeddings before novelty is
/// flagged. A fresh deployment must accumulate this much history first, so
/// the first incidents are not all spuriously flagged "unprecedented".
pub const NOVELTY_WARMUP: usize = 5;
/// Maximum number of incident embeddings retained for novelty comparison.
pub const MAX_EMBED_HISTORY: usize = 500;

/// A cluster of semantically-similar (or same-rule/scope) ops events —
/// one distinct incident the operator should act on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// LLM-drafted root-cause paragraph + suggested runbook step,
    /// generated once (best-effort, off-tick) when the incident first
    /// forms. `None` until drafted (or when no LLM is configured).
    #[serde(default)]
    pub draft: Option<String>,
    /// `true` when this incident's representative embedding is far from
    /// every prior incident in the embedding history — a genuinely
    /// unprecedented pattern. Only ever set on the embedding path, after a
    /// warm-up of [`NOVELTY_WARMUP`] historical incidents; never set when
    /// no embedder is configured (no false novelty).
    #[serde(default)]
    pub is_novel: bool,
    /// Novelty score in `0..=1`: `1 - max_cosine_to_history` (higher =
    /// more novel). `None` when novelty was not scored (no embedder, or
    /// still in warm-up).
    #[serde(default)]
    pub novelty_score: Option<f64>,
}

impl Incident {
    /// The representative *event* id, recovered from the incident id
    /// (`inc_{event_id}`). Used to fetch the RCA bundle for drafting.
    pub fn representative_event_id(&self) -> &str {
        self.id.strip_prefix("inc_").unwrap_or(&self.id)
    }
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
            draft: None,
            is_novel: false,
            novelty_score: None,
        }
    }

    /// The incident's representative embedding (the cluster centroid), used
    /// for novelty scoring against the embedding history. Empty in the
    /// fallback (no-embedder) path.
    fn representative_embedding(&self) -> Vec<f32> {
        self.centroid.clone()
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
    cluster_incidents_with_embeddings(events, embeddings, threshold).0
}

/// Like [`cluster_incidents`], but also returns each incident's
/// representative embedding (its final cluster centroid), in the same order
/// as the returned incidents. On the fallback (no-embedder) path the
/// embeddings are all empty. Reusing these centroids avoids re-embedding
/// for novelty scoring.
pub fn cluster_incidents_with_embeddings(
    events: &[OpsEvent],
    embeddings: Option<&[Vec<f32>]>,
    threshold: f32,
) -> (Vec<Incident>, Vec<Vec<f32>>) {
    let accs = match embeddings {
        Some(embs) if embs.len() == events.len() => cluster_by_embedding(events, embs, threshold),
        // No embeddings (or a length mismatch we won't trust) → fallback.
        _ => cluster_by_rule_scope(events),
    };
    let embs: Vec<Vec<f32>> = accs.iter().map(Accum::representative_embedding).collect();
    let incidents: Vec<Incident> = accs.into_iter().map(Accum::finish).collect();
    (incidents, embs)
}

/// Embedding path: greedy cosine clustering against running-mean centroids.
fn cluster_by_embedding(events: &[OpsEvent], embs: &[Vec<f32>], threshold: f32) -> Vec<Accum> {
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
    accs
}

/// Fallback path: group by `(rule_id, scope.key())`, deterministic in
/// first-seen (input) order.
fn cluster_by_rule_scope(events: &[OpsEvent]) -> Vec<Accum> {
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
        .collect()
}

/// Sort incidents most-urgent-first (severity desc, then count desc, then
/// id for determinism) while keeping the parallel `embeddings` vector
/// aligned by incident.
fn sort_incidents_with_embeddings(incidents: &mut Vec<Incident>, embeddings: &mut Vec<Vec<f32>>) {
    let mut paired: Vec<(Incident, Vec<f32>)> = incidents
        .drain(..)
        .zip(embeddings.drain(..))
        .collect();
    paired.sort_by(|a, b| {
        b.0.max_severity
            .cmp(&a.0.max_severity)
            .then(b.0.count.cmp(&a.0.count))
            .then(a.0.id.cmp(&b.0.id))
    });
    for (inc, emb) in paired {
        incidents.push(inc);
        embeddings.push(emb);
    }
}

/// Score how *novel* `current` is against a history of prior incident
/// embeddings. Pure — no I/O.
///
/// `max_sim` is the maximum cosine similarity of `current` to any history
/// vector (`0.0` when the history is empty, so a first-ever incident scores
/// as maximally novel — warm-up is applied by the caller, not here). The
/// novelty `score = 1 - max_sim` (clamped to `0..=1`), and `is_novel` is
/// `score >= novelty_threshold`. An empty `current` (no embedder) is never
/// novel.
pub fn score_novelty(current: &[f32], history: &[Vec<f32>], novelty_threshold: f64) -> (bool, f64) {
    if current.is_empty() {
        return (false, 0.0);
    }
    let mut max_sim = 0.0f64;
    for h in history {
        if h.is_empty() {
            continue;
        }
        let sim = cosine(current, h) as f64;
        if sim > max_sim {
            max_sim = sim;
        }
    }
    let score = (1.0 - max_sim).clamp(0.0, 1.0);
    (score >= novelty_threshold, score)
}

/// Annotate each incident with novelty, in place, reusing the parallel
/// `embeddings` (the clustering centroids — no re-embedding). For each
/// incident with a non-empty embedding, set `novelty_score`; flag
/// `is_novel` only when the `history` is warmed up (`>= NOVELTY_WARMUP`)
/// AND the score clears the threshold. Incidents with an empty embedding
/// (fallback / no-embedder path) are left untouched — never flagged.
/// Pure (no I/O); the caller supplies the history snapshot.
pub fn annotate_novelty(
    incidents: &mut [Incident],
    embeddings: &[Vec<f32>],
    history: &[Vec<f32>],
    threshold: f64,
) {
    let warmed_up = history.len() >= NOVELTY_WARMUP;
    for (inc, emb) in incidents.iter_mut().zip(embeddings.iter()) {
        if emb.is_empty() {
            continue;
        }
        let (is_novel, score) = score_novelty(emb, history, threshold);
        inc.novelty_score = Some(score);
        inc.is_novel = warmed_up && is_novel;
    }
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
    compute_incidents_with_embeddings(state).await.0
}

/// Like [`compute_incidents`], but also returns each incident's
/// representative embedding (cluster centroid), in incident order, so the
/// caller can score novelty without re-embedding. On the fallback path the
/// embeddings are all empty (no embedder → no novelty).
pub async fn compute_incidents_with_embeddings(state: &AppState) -> (Vec<Incident>, Vec<Vec<f32>>) {
    let advisor = state.advisor();

    // Brief events lock: clone open + history events out, then drop it.
    let events: Vec<OpsEvent> = {
        let store = advisor.events.lock().await;
        let mut all = store.open_events();
        all.extend(store.history().iter().cloned());
        all
    };
    if events.is_empty() {
        return (Vec::new(), Vec::new());
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

    let (mut incidents, mut embs) =
        cluster_incidents_with_embeddings(&events, embeddings.as_deref(), DEFAULT_THRESHOLD);
    sort_incidents_with_embeddings(&mut incidents, &mut embs);
    (incidents, embs)
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
    /// Incident ids already notified (alert-storm dedup): one notification
    /// is sent per *incident*, not per event, and never re-sent on later
    /// passes. Persisted so a restart doesn't re-notify. Bounded.
    #[serde(default)]
    notified_ids: Vec<String>,
    /// Newest-last ring of representative embeddings of incidents seen on
    /// prior passes, used as the comparison corpus for novelty detection.
    /// Bounded to [`MAX_EMBED_HISTORY`]. Empty when no embedder is ever
    /// configured (the fallback path stores no embeddings).
    #[serde(default)]
    incident_embeddings: Vec<StoredEmbedding>,
}

/// A persisted incident embedding for novelty comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredEmbedding {
    /// The incident id this embedding came from.
    pub id: String,
    /// The representative (centroid) embedding.
    pub embedding: Vec<f32>,
    /// When it was recorded (ms epoch), for debugging/aging.
    pub ts_ms: i64,
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

    /// Has incident `id` already been notified on a prior pass?
    pub fn is_notified(&self, id: &str) -> bool {
        self.notified_ids.iter().any(|n| n == id)
    }

    /// Record that incident `id` has been notified (idempotent). The
    /// notified-id ring is bounded to [`MAX_HISTORY`] so a long-running
    /// process with churning incident ids cannot grow it unboundedly.
    pub fn mark_notified(&mut self, id: &str) {
        if !self.is_notified(id) {
            self.notified_ids.push(id.to_owned());
            if self.notified_ids.len() > MAX_HISTORY {
                let excess = self.notified_ids.len() - MAX_HISTORY;
                self.notified_ids.drain(0..excess);
            }
        }
    }

    /// The persisted incident embeddings (the novelty comparison corpus).
    pub fn incident_embeddings(&self) -> &[StoredEmbedding] {
        &self.incident_embeddings
    }

    /// Just the embedding vectors of the history, for [`score_novelty`].
    pub fn history_embeddings(&self) -> Vec<Vec<f32>> {
        self.incident_embeddings
            .iter()
            .map(|e| e.embedding.clone())
            .collect()
    }

    /// Append a representative embedding to the novelty corpus (bounded to
    /// [`MAX_EMBED_HISTORY`], dropping oldest). Empty embeddings (fallback
    /// path) are ignored — only real embeddings feed novelty detection.
    pub fn push_embedding(&mut self, id: &str, embedding: Vec<f32>, ts_ms: i64) {
        if embedding.is_empty() {
            return;
        }
        self.incident_embeddings.push(StoredEmbedding {
            id: id.to_owned(),
            embedding,
            ts_ms,
        });
        if self.incident_embeddings.len() > MAX_EMBED_HISTORY {
            let excess = self.incident_embeddings.len() - MAX_EMBED_HISTORY;
            self.incident_embeddings.drain(0..excess);
        }
    }

    /// Carry forward already-generated drafts from the previous `latest`
    /// onto the freshly-clustered `incidents`, matching by incident id.
    /// A stable incident keeps its draft so the LLM is not re-run for it.
    pub fn carry_drafts_into(&self, incidents: &mut [Incident]) {
        for inc in incidents.iter_mut() {
            if inc.draft.is_none()
                && let Some(prev) = self.latest.iter().find(|p| p.id == inc.id)
                && prev.draft.is_some()
            {
                inc.draft = prev.draft.clone();
            }
        }
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

/// Generates an LLM root-cause draft for an incident. Abstracted so the
/// correlation cycle can be driven with a fake drafter in tests (the
/// production drafter runs the `veda-ops` agent over the network).
pub trait IncidentDrafter {
    /// Draft a root-cause paragraph for `prompt`, or `None` on any
    /// failure / no-LLM (best-effort — never fails the cycle).
    fn draft(&self, prompt: &str) -> impl std::future::Future<Output = Option<String>> + Send;
}

/// Production drafter: runs the built-in `veda-ops` agent (one off-lock
/// turn, default tenant, no conversation). Any error — including no LLM
/// configured (`AgentTurnError::NoLlm`) → `None`.
pub struct OpsAgentDrafter<'a> {
    state: &'a AppState,
}

impl<'a> OpsAgentDrafter<'a> {
    pub fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl IncidentDrafter for OpsAgentDrafter<'_> {
    async fn draft(&self, prompt: &str) -> Option<String> {
        let tenant = crate::tenant::TenantId::default_tenant();
        match crate::rest::agents::run_agent_turn(
            self.state, &tenant, None, "veda-ops", &[], prompt, None,
        )
        .await
        {
            Ok((_agent, result, _citations, _grounding, _cache_hit, _guardrails)) => Some(result.response),
            Err(_) => None,
        }
    }
}

/// Build the incident-draft prompt: the representative + member count +
/// the bounded cross-signal RCA summary, asking for a root cause + one
/// runbook step. Pure + testable.
pub fn build_incident_draft_prompt(incident: &Incident, bundle: &RcaBundle) -> String {
    let mut prompt = format!(
        "A new ops incident formed in Veda by collapsing {count} correlated alert(s) into one. \
Give a concise ROOT-CAUSE paragraph and ONE concrete runbook step the operator should take. \
Be specific and brief.\n\n\
Incident:\n\
- representative: {rep}\n\
- alerts collapsed: {count}\n\
- rule_ids: {rules}\n\
- max_severity: {sev:?}",
        count = incident.count,
        rep = incident.representative,
        rules = incident.rule_ids.join(", "),
        sev = incident.max_severity,
    );
    prompt.push_str(&rca_summary(bundle));
    prompt
}

/// Best-effort: for each incident **without a draft yet**, resolve its
/// representative event id, fetch the cross-signal RCA bundle, prompt the
/// `drafter`, and set `incident.draft`. Already-drafted incidents (drafts
/// carried forward from the previous pass) are skipped so the LLM is not
/// re-run for a stable incident. Runs entirely OFF any advisor lock; a
/// failed draft simply leaves `draft = None` and never fails the cycle.
pub async fn draft_incidents<D: IncidentDrafter>(
    state: &AppState,
    incidents: &mut [Incident],
    drafter: &D,
) {
    for inc in incidents.iter_mut() {
        if inc.draft.is_some() {
            continue; // stable / already drafted — don't re-run the LLM.
        }
        let event_id = inc.representative_event_id().to_owned();
        let Some(bundle) = correlate_incident(state, &event_id).await else {
            continue; // representative event vanished — skip, leave None.
        };
        let prompt = build_incident_draft_prompt(inc, &bundle);
        if let Some(draft) = drafter.draft(&prompt).await {
            inc.draft = Some(draft);
        }
    }
}

/// Synthesize the one-shot [`OpsEvent`] used to push a per-incident
/// notification through the existing transition notifier. Carries the
/// "N alerts → 1 incident" framing + a one-line draft summary (if any).
/// Pure + testable.
pub fn incident_notification_event(incident: &Incident) -> OpsEvent {
    let title = if incident.is_novel {
        format!(
            "⚠ UNPRECEDENTED Incident: {} ({} alerts → 1 incident)",
            incident.representative, incident.count
        )
    } else {
        format!(
            "Incident: {} ({} alerts → 1 incident)",
            incident.representative, incident.count
        )
    };
    let mut detail = format!(
        "{} correlated alert(s) collapsed into one incident (rules: {}).",
        incident.count,
        incident.rule_ids.join(", "),
    );
    if incident.is_novel {
        let score = incident.novelty_score.unwrap_or(0.0);
        detail.push_str(&format!(
            "\n\n⚠ UNPRECEDENTED pattern — this incident is unlike anything in the incident history \
(novelty score {score:.2}; no similar prior incident). Investigate as a genuinely new failure mode."
        ));
    }
    let recommendation = match &incident.draft {
        Some(d) => {
            // One-line draft summary: first non-empty line, bounded.
            let line = d.lines().find(|l| !l.trim().is_empty()).unwrap_or(d).trim();
            let line: String = line.chars().take(280).collect();
            detail.push_str(&format!("\n\nDraft root-cause: {line}"));
            line
        }
        None => "Review the correlated incident in the Insights → Incidents view.".to_owned(),
    };
    // A genuinely unprecedented pattern is escalated to High regardless of
    // its constituent alerts' severities.
    let severity = if incident.is_novel {
        incident.max_severity.max(Severity::High)
    } else {
        incident.max_severity
    };
    OpsEvent {
        id: incident.id.clone(),
        rule_id: incident.rule_ids.first().cloned().unwrap_or_default(),
        scope: crate::advisor::event::Scope::Global,
        severity,
        status: crate::advisor::event::EventStatus::Open,
        title,
        detail,
        recommendation,
        runbook: None,
        remediation: crate::advisor::event::RemediationAction::None,
        opened_at_ms: incident.first_seen_ms,
        closed_at_ms: None,
        acknowledged: false,
        remediation_note: None,
    }
}

/// Recompute incidents from the current events, draft NEW incidents
/// (best-effort, via the production [`OpsAgentDrafter`]), notify on
/// newly-formed incidents (once each), then store + persist.
///
/// Invoked exclusively from the dedicated [`spawn_incident_correlator`]
/// task, **never** from `run_tick`, so neither the embedding nor the
/// draft/notify network calls run inside the advisor's locked tick.
/// Every network await runs OFF any advisor lock; the `incidents` mutex is
/// only re-locked *briefly* to read the carry-forward drafts/notified set
/// and to record + save.
pub async fn run_incident_correlation(state: &AppState) {
    let drafter = OpsAgentDrafter::new(state);
    run_incident_correlation_with(state, &drafter).await;
}

/// [`run_incident_correlation`] parameterized over the drafter, so tests
/// can inject a fake LLM. Mirrors the lock discipline of the recall /
/// faithfulness off-tick tasks: network awaits (embed, draft, notify) all
/// run with no advisor lock held.
pub async fn run_incident_correlation_with<D: IncidentDrafter>(state: &AppState, drafter: &D) {
    // 1. Cluster (embedding await off-lock inside compute_incidents). Reuse
    //    the clustering centroids as the per-incident representative
    //    embeddings — NO second embedding pass for novelty.
    let (mut incidents, incident_embs) = compute_incidents_with_embeddings(state).await;
    let advisor = state.advisor();

    // 2. Carry forward drafts, score novelty against the persisted
    //    embedding history, and identify newly-formed (not-yet-notified)
    //    incidents — all under a brief read of the store, no await held.
    //    `score_novelty` is pure (no I/O), so this stays lock-safe.
    let new_ids: Vec<String> = {
        let store = advisor.incidents.lock().await;
        store.carry_drafts_into(&mut incidents);

        // Novelty: reuse the clustering centroids (no re-embed). Only the
        // embedding path produces non-empty centroids; with no embedder
        // they are empty → annotate_novelty leaves is_novel=false /
        // novelty_score=None (no false novelty). Warm-up is applied inside.
        let history = store.history_embeddings();
        annotate_novelty(&mut incidents, &incident_embs, &history, NOVELTY_THRESHOLD);

        incidents
            .iter()
            .filter(|i| !store.is_notified(&i.id))
            .map(|i| i.id.clone())
            .collect()
    };

    // 3. Draft NEW (undrafted) incidents OFF any advisor lock.
    draft_incidents(state, &mut incidents, drafter).await;

    // 4. Notify each newly-formed incident exactly once, OFF any advisor
    //    lock. Reuses the transition notifier with a synthesized event
    //    (novel incidents get an "UNPRECEDENTED" framing + High severity).
    let channels = advisor.config.read().await.channels.clone();
    let notifier = Notifier::new(channels);
    for inc in incidents.iter().filter(|i| new_ids.contains(&i.id)) {
        let ev = incident_notification_event(inc);
        notifier.notify_transitions(std::slice::from_ref(&ev), &[]).await;
    }

    // 5. Record + mark notified + append NEW incidents' embeddings to the
    //    novelty corpus + persist, under a brief lock.
    let now_ms = crate::advisor::engine::now_ms();
    let mut store = advisor.incidents.lock().await;
    for (inc, emb) in incidents.iter().zip(incident_embs.iter()) {
        if new_ids.contains(&inc.id) {
            store.push_embedding(&inc.id, emb.clone(), now_ms);
        }
    }
    store.record(incidents);
    for id in &new_ids {
        store.mark_notified(id);
    }
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
            draft: None,
            is_novel: false,
            novelty_score: None,
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

    // ---- 3.1 incident drafts -------------------------------------------

    /// Fake drafter: returns a fixed draft and counts invocations so the
    /// "stable incident is not re-drafted" property is observable.
    struct FakeDrafter {
        reply: Option<String>,
        calls: std::sync::atomic::AtomicUsize,
    }
    impl FakeDrafter {
        fn new(reply: Option<&str>) -> Self {
            Self {
                reply: reply.map(str::to_owned),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl IncidentDrafter for FakeDrafter {
        async fn draft(&self, _prompt: &str) -> Option<String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.reply.clone()
        }
    }

    async fn seed_one_event(state: &AppState) {
        let mut store = state.advisor().events.lock().await;
        store.reconcile(
            vec![crate::advisor::rules::RuleHit {
                rule_id: "high_error_rate".into(),
                scope: Scope::Global,
                severity: Severity::High,
                title: "errors spiking".into(),
                detail: "5xx ratio over threshold".into(),
                recommendation: "inspect logs".into(),
                runbook: None,
                remediation: RemediationAction::None,
            }],
            1000,
        );
    }

    #[tokio::test]
    async fn injected_llm_sets_incident_draft() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_one_event(&state).await;
        let drafter = FakeDrafter::new(Some("Root cause: error spike. Step: inspect logs."));
        run_incident_correlation_with(&state, &drafter).await;

        let store = state.advisor().incidents.lock().await;
        assert_eq!(store.latest().len(), 1);
        assert_eq!(
            store.latest()[0].draft.as_deref(),
            Some("Root cause: error spike. Step: inspect logs.")
        );
        assert_eq!(drafter.call_count(), 1, "the one new incident is drafted once");
    }

    #[tokio::test]
    async fn no_llm_leaves_draft_none_and_cycle_completes() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_one_event(&state).await;
        // Drafter returns None (mirrors AgentTurnError::NoLlm).
        let drafter = FakeDrafter::new(None);
        run_incident_correlation_with(&state, &drafter).await;

        let store = state.advisor().incidents.lock().await;
        assert_eq!(store.latest().len(), 1, "cycle still completes");
        assert!(store.latest()[0].draft.is_none(), "draft stays None");
    }

    #[tokio::test]
    async fn stable_incident_is_not_redrafted() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_one_event(&state).await;
        let drafter = FakeDrafter::new(Some("draft v1"));

        // First pass drafts the incident.
        run_incident_correlation_with(&state, &drafter).await;
        assert_eq!(drafter.call_count(), 1);

        // Second pass: same incident (carry-forward draft) → NOT re-drafted.
        run_incident_correlation_with(&state, &drafter).await;
        assert_eq!(drafter.call_count(), 1, "stable incident not re-drafted");
        let store = state.advisor().incidents.lock().await;
        assert_eq!(store.latest()[0].draft.as_deref(), Some("draft v1"));
    }

    // ---- 3.2 per-incident notification ---------------------------------

    #[test]
    fn notification_event_frames_alert_storm_and_draft() {
        let mut inc = incident("inc_x", 5);
        inc.representative = "errors spiking".into();
        inc.rule_ids = vec!["high_error_rate".into(), "disk_pressure".into()];
        inc.draft = Some("Root cause: error spike.\nNext: inspect logs.".into());
        let ev = incident_notification_event(&inc);
        assert!(ev.title.contains("5 alerts → 1 incident"));
        assert!(ev.title.contains("errors spiking"));
        assert!(ev.detail.contains("Draft root-cause: Root cause: error spike."));
        // dedup_key is the incident id → one PagerDuty/webhook identity.
        assert_eq!(ev.id, "inc_x");
        assert_eq!(ev.severity, Severity::High);
    }

    #[tokio::test]
    async fn new_incident_notified_once_then_not_renotified() {
        use std::io::{Read, Write};
        use std::net::TcpListener as StdTcpListener;
        use std::sync::mpsc;

        // A capture webhook counting every POST it receives.
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut tmp = [0u8; 2048];
                let _ = stream.read(&mut tmp);
                let reply = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
                let _ = stream.write_all(reply.as_bytes());
                let _ = stream.flush();
                if tx.send(()).is_err() {
                    break;
                }
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        // Configure an enabled webhook (Low floor → all severities pass).
        {
            let mut cfg = state.advisor().config.write().await;
            cfg.channels.webhook = Some(crate::advisor::config::WebhookChannel {
                enabled: true,
                url: url.clone(),
                min_severity: Severity::Low,
            });
        }
        seed_one_event(&state).await;
        let drafter = FakeDrafter::new(Some("Root cause: x"));

        // First pass: one new incident → exactly one notification.
        run_incident_correlation_with(&state, &drafter).await;
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("exactly one incident notification");
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "only one notification on the first pass"
        );

        // Second pass: same incident → NOT re-notified (deduped).
        run_incident_correlation_with(&state, &drafter).await;
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(500)).is_err(),
            "stable incident must not be re-notified"
        );

        // The notified-id is persisted.
        let store = state.advisor().incidents.lock().await;
        assert!(store.is_notified(&store.latest()[0].id.clone()));
    }

    // ---- 4.1 novelty model + detector ----------------------------------

    #[test]
    fn score_novelty_flags_embedding_far_from_all_history() {
        // An embedding orthogonal to all history → max_sim 0 → score 1.0 →
        // novel.
        let history = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]];
        let current = vec![0.0, 0.0, 1.0];
        let (is_novel, score) = score_novelty(&current, &history, NOVELTY_THRESHOLD);
        assert!(is_novel, "orthogonal embedding must be novel");
        assert!((score - 1.0).abs() < 1e-6, "score should be ~1.0, got {score}");
    }

    #[test]
    fn score_novelty_does_not_flag_embedding_near_history() {
        // Near-identical to a history vector → max_sim ~1 → score ~0 → not
        // novel.
        let history = vec![vec![1.0, 0.0, 0.0]];
        let current = vec![0.999, 0.01, 0.0];
        let (is_novel, score) = score_novelty(&current, &history, NOVELTY_THRESHOLD);
        assert!(!is_novel, "near-duplicate embedding must not be novel");
        assert!(score < NOVELTY_THRESHOLD, "score {score} should be below threshold");
    }

    #[test]
    fn score_novelty_empty_history_scores_first_incident_maximally_novel() {
        // No history → max_sim 0 → score 1.0. (Warm-up — applied by the
        // caller, not here — is what prevents flagging on a fresh deploy.)
        let (is_novel, score) = score_novelty(&[1.0, 0.0], &[], NOVELTY_THRESHOLD);
        assert!(is_novel);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_novelty_empty_current_is_never_novel() {
        // No embedder (empty centroid) → never novel, score 0.
        let history = vec![vec![1.0, 0.0]];
        let (is_novel, score) = score_novelty(&[], &history, NOVELTY_THRESHOLD);
        assert!(!is_novel);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn store_push_embedding_ignores_empty_and_bounds_history() {
        let mut store = IncidentStore::new();
        store.push_embedding("a", vec![], 1); // empty → ignored
        assert!(store.incident_embeddings().is_empty());
        store.push_embedding("b", vec![1.0, 0.0], 2);
        assert_eq!(store.incident_embeddings().len(), 1);
        // Overfill past the cap → oldest dropped.
        for i in 0..(MAX_EMBED_HISTORY + 10) {
            store.push_embedding(&format!("x{i}"), vec![i as f32, 1.0], i as i64);
        }
        assert_eq!(store.incident_embeddings().len(), MAX_EMBED_HISTORY);
    }

    #[test]
    fn store_embeddings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(INCIDENTS_FILE);
        let mut store = IncidentStore::new();
        store.push_embedding("inc_a", vec![0.1, 0.2, 0.3], 99);
        store.save(&path);
        let loaded = IncidentStore::load(&path);
        assert_eq!(loaded.incident_embeddings().len(), 1);
        assert_eq!(loaded.incident_embeddings()[0].id, "inc_a");
        assert_eq!(loaded.history_embeddings(), vec![vec![0.1, 0.2, 0.3]]);
    }

    // ---- 4.2 novelty wiring --------------------------------------------

    #[test]
    fn cluster_with_embeddings_returns_aligned_centroids() {
        // The representative embedding of each incident is its cluster
        // centroid — reused for novelty (no re-embed).
        let events = vec![
            ev("a", Scope::Global, Severity::High, "thing a", 1),
            ev("b", Scope::Global, Severity::High, "thing b", 2),
        ];
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (incidents, centroids) = cluster_incidents_with_embeddings(&events, Some(&embs), 0.82);
        assert_eq!(incidents.len(), 2);
        assert_eq!(centroids.len(), 2);
        assert!(centroids.iter().all(|c| c.len() == 2), "centroids carry the embedding dim");
    }

    #[test]
    fn cluster_with_embeddings_fallback_yields_empty_centroids() {
        // No embeddings → fallback path → empty centroids → no novelty.
        let events = vec![ev("r", Scope::Global, Severity::High, "x", 1)];
        let (incidents, centroids) = cluster_incidents_with_embeddings(&events, None, 0.82);
        assert_eq!(incidents.len(), 1);
        assert_eq!(centroids.len(), 1);
        assert!(centroids[0].is_empty(), "fallback path stores no embedding");
    }

    #[tokio::test]
    async fn run_correlation_no_embedder_flags_no_novelty() {
        // No embedder configured → fallback clustering → is_novel stays
        // false and novelty_score stays None (never false novelty).
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_one_event(&state).await;
        let drafter = FakeDrafter::new(None);
        run_incident_correlation_with(&state, &drafter).await;
        let store = state.advisor().incidents.lock().await;
        assert_eq!(store.latest().len(), 1);
        assert!(!store.latest()[0].is_novel);
        assert!(store.latest()[0].novelty_score.is_none());
        // Fallback path persisted no embeddings.
        assert!(store.incident_embeddings().is_empty());
    }

    fn warm_history(n: usize) -> Vec<Vec<f32>> {
        // n history vectors all near [1,0,0].
        (0..n).map(|i| vec![1.0, 0.001 * i as f32, 0.0]).collect()
    }

    #[test]
    fn annotate_novelty_flags_dissimilar_and_not_similar_when_warmed_up() {
        let history = warm_history(NOVELTY_WARMUP); // warmed up
        let mut incidents = vec![incident("inc_far", 1), incident("inc_near", 1)];
        let embs = vec![
            vec![0.0, 0.0, 1.0],   // orthogonal to history → novel
            vec![1.0, 0.002, 0.0], // ~identical to history → not novel
        ];
        annotate_novelty(&mut incidents, &embs, &history, NOVELTY_THRESHOLD);
        assert!(incidents[0].is_novel, "dissimilar incident is novel");
        assert!(incidents[0].novelty_score.unwrap() >= NOVELTY_THRESHOLD);
        assert!(!incidents[1].is_novel, "similar incident is not novel");
        assert!(incidents[1].novelty_score.unwrap() < NOVELTY_THRESHOLD);
    }

    #[test]
    fn annotate_novelty_warmup_suppresses_until_k_history() {
        // Below warm-up: a dissimilar incident gets a SCORE but is NOT
        // flagged novel.
        let history = warm_history(NOVELTY_WARMUP - 1);
        let mut incidents = vec![incident("inc_far", 1)];
        let embs = vec![vec![0.0, 0.0, 1.0]];
        annotate_novelty(&mut incidents, &embs, &history, NOVELTY_THRESHOLD);
        assert!(!incidents[0].is_novel, "warm-up suppresses novelty flag");
        assert!(
            incidents[0].novelty_score.unwrap() >= NOVELTY_THRESHOLD,
            "score is still computed during warm-up"
        );
    }

    #[test]
    fn annotate_novelty_skips_empty_embeddings() {
        let history = warm_history(NOVELTY_WARMUP);
        let mut incidents = vec![incident("inc_fallback", 1)];
        let embs = vec![vec![]]; // fallback path — no embedding
        annotate_novelty(&mut incidents, &embs, &history, NOVELTY_THRESHOLD);
        assert!(!incidents[0].is_novel);
        assert!(incidents[0].novelty_score.is_none(), "no score without an embedding");
    }

    #[test]
    fn novel_incident_notification_is_escalated_and_framed() {
        let mut inc = incident("inc_novel", 2);
        inc.representative = "weird new failure".into();
        inc.max_severity = Severity::Low;
        inc.is_novel = true;
        inc.novelty_score = Some(0.93);
        let ev = incident_notification_event(&inc);
        assert!(ev.title.contains("UNPRECEDENTED"), "title flags unprecedented");
        assert!(ev.detail.contains("UNPRECEDENTED pattern"));
        assert!(ev.detail.contains("0.93"), "novelty score surfaced");
        assert_eq!(ev.severity, Severity::High, "novel incident escalated to High");
    }

    #[test]
    fn non_novel_incident_notification_unchanged() {
        let mut inc = incident("inc_plain", 3);
        inc.max_severity = Severity::Medium;
        let ev = incident_notification_event(&inc);
        assert!(!ev.title.contains("UNPRECEDENTED"));
        assert!(!ev.detail.contains("UNPRECEDENTED"));
        assert_eq!(ev.severity, Severity::Medium, "non-novel keeps its severity");
    }
}
