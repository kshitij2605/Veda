//! Ops-advisor event model + a persisted event store.
//!
//! A *rule* (see [`crate::advisor::rules`]) inspects an [`OpsSnapshot`]
//! and emits zero or more [`RuleHit`](crate::advisor::rules::RuleHit)s.
//! The [`EventStore`] turns those transient hits into durable
//! [`OpsEvent`]s with open/close lifecycle:
//!
//! * a hit whose event id is not yet open → a new `Open` event;
//! * a hit matching an already-open event → refreshes its detail /
//!   severity (no duplicate, no re-notify);
//! * an open event with no matching hit this round → `Closed`, moved
//!   to a bounded history ring.
//!
//! [`EventStore::reconcile`] returns the open/close transitions so
//! later tasks (notifications, auto-remediation) can act on the edges
//! without re-deriving them. Persistence is a single JSON file at
//! `{data_dir}/_ops_events.json`, mirroring the write-through shape of
//! [`crate::rest::saved_objects::SavedObjectStore`].

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::advisor::rules::RuleHit;

/// Maximum number of closed events retained in history.
pub const DEFAULT_HISTORY_CAP: usize = 500;

/// Event severity. Ordered so `High > Medium > Low`, letting callers
/// sort most-urgent-first with a plain reverse compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
}

/// Lifecycle state of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventStatus {
    Open,
    Closed,
}

/// What an event is scoped to. Provides a stable string [`Scope::key`]
/// used to build the event id (so the same condition over the same
/// target maps to a single event across rounds).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "target")]
pub enum Scope {
    Global,
    Collection(String),
    Shard { collection: String, shard: u32 },
    Node(String),
}

impl Scope {
    /// Stable, collision-resistant string key for this scope.
    pub fn key(&self) -> String {
        match self {
            Scope::Global => "global".to_owned(),
            Scope::Collection(c) => format!("collection:{c}"),
            Scope::Shard { collection, shard } => format!("shard:{collection}/{shard}"),
            Scope::Node(n) => format!("node:{n}"),
        }
    }
}

/// A remediation an event *could* trigger. This is only a data marker
/// here — nothing in this task executes it; the auto-remediation task
/// consumes it. Extend with new variants as remediations are added.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action", content = "params")]
pub enum RemediationAction {
    None,
    Compact { collection: String },
    Reindex { collection: String },
}

/// A durable advisor event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsEvent {
    /// Stable id: `"{rule_id}:{scope.key()}"`.
    pub id: String,
    pub rule_id: String,
    pub scope: Scope,
    pub severity: Severity,
    pub status: EventStatus,
    pub title: String,
    /// Root cause / what tripped the rule.
    pub detail: String,
    /// What the operator should do about it.
    pub recommendation: String,
    /// Optional command / endpoint hint (a runbook pointer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runbook: Option<String>,
    pub remediation: RemediationAction,
    pub opened_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_at_ms: Option<i64>,
    #[serde(default)]
    pub acknowledged: bool,
    /// Concise note recording the last auto-remediation outcome for this
    /// event (set by [`EventStore::annotate_remediation`]). Preserved
    /// across reconcile refreshes of an already-open event.
    #[serde(default)]
    pub remediation_note: Option<String>,
}

impl OpsEvent {
    /// Stable event id for a `(rule_id, scope)` pair.
    pub fn make_id(rule_id: &str, scope: &Scope) -> String {
        format!("{rule_id}:{}", scope.key())
    }

    /// Build a fresh `Open` event from a hit.
    fn from_hit(hit: RuleHit, now_ms: i64) -> Self {
        let id = Self::make_id(&hit.rule_id, &hit.scope);
        OpsEvent {
            id,
            rule_id: hit.rule_id,
            scope: hit.scope,
            severity: hit.severity,
            status: EventStatus::Open,
            title: hit.title,
            detail: hit.detail,
            recommendation: hit.recommendation,
            runbook: hit.runbook,
            remediation: hit.remediation,
            opened_at_ms: now_ms,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }
}

/// The result of a single [`EventStore::reconcile`] pass: the events
/// that newly opened and the ones that closed this round. Already-open
/// events that merely refreshed appear in neither list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileResult {
    pub opened: Vec<OpsEvent>,
    pub closed: Vec<OpsEvent>,
}

/// Stores open events keyed by id plus a bounded ring of closed
/// history, with JSON write-through to `{data_dir}/_ops_events.json`.
#[derive(Debug)]
pub struct EventStore {
    open: HashMap<String, OpsEvent>,
    history: VecDeque<OpsEvent>,
    history_cap: usize,
    path: PathBuf,
}

/// On-disk shape for the event store.
#[derive(Serialize, Deserialize, Default)]
struct PersistedEvents {
    open: Vec<OpsEvent>,
    history: Vec<OpsEvent>,
}

impl EventStore {
    /// New, empty store persisting under `data_dir`.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            open: HashMap::new(),
            history: VecDeque::new(),
            history_cap: DEFAULT_HISTORY_CAP,
            path: data_dir.join("_ops_events.json"),
        }
    }

    /// Turn this round's hits into open/close transitions.
    ///
    /// * a hit whose id is not open → opens an event (in `opened`);
    /// * a hit matching an open event → refreshes mutable fields
    ///   (detail / severity / title / recommendation / runbook /
    ///   remediation) in place, emitting no transition;
    /// * an open event with no matching hit → closed, stamped, moved
    ///   to history (in `closed`).
    ///
    /// Callers persist explicitly via [`EventStore::save`].
    pub fn reconcile(&mut self, hits: Vec<RuleHit>, now_ms: i64) -> ReconcileResult {
        let mut result = ReconcileResult::default();
        let mut seen: Vec<String> = Vec::with_capacity(hits.len());

        for hit in hits {
            let id = OpsEvent::make_id(&hit.rule_id, &hit.scope);
            seen.push(id.clone());
            match self.open.get_mut(&id) {
                Some(existing) => {
                    // Refresh in place: same event, possibly worse/better
                    // severity or updated root-cause text. No transition.
                    existing.severity = hit.severity;
                    existing.title = hit.title;
                    existing.detail = hit.detail;
                    existing.recommendation = hit.recommendation;
                    existing.runbook = hit.runbook;
                    existing.remediation = hit.remediation;
                    // NOTE: `remediation_note` and `acknowledged` are
                    // deliberately NOT reset here — a refresh of an
                    // already-open event must preserve the last
                    // auto-remediation outcome annotation.
                }
                None => {
                    let ev = OpsEvent::from_hit(hit, now_ms);
                    self.open.insert(id, ev.clone());
                    result.opened.push(ev);
                }
            }
        }

        // Close any open event not present in this round's hits.
        let stale: Vec<String> = self
            .open
            .keys()
            .filter(|id| !seen.contains(id))
            .cloned()
            .collect();
        for id in stale {
            if let Some(mut ev) = self.open.remove(&id) {
                ev.status = EventStatus::Closed;
                ev.closed_at_ms = Some(now_ms);
                self.push_history(ev.clone());
                result.closed.push(ev);
            }
        }

        result
    }

    fn push_history(&mut self, ev: OpsEvent) {
        if self.history.len() == self.history_cap {
            self.history.pop_front();
        }
        self.history.push_back(ev);
    }

    /// Open events, sorted by severity descending then oldest-first.
    pub fn open_events(&self) -> Vec<OpsEvent> {
        let mut out: Vec<OpsEvent> = self.open.values().cloned().collect();
        out.sort_by(|a, b| {
            b.severity
                .cmp(&a.severity)
                .then(a.opened_at_ms.cmp(&b.opened_at_ms))
        });
        out
    }

    /// The bounded ring of closed events, oldest-first.
    pub fn history(&self) -> &VecDeque<OpsEvent> {
        &self.history
    }

    /// Mark an open event acknowledged. Returns whether it existed.
    pub fn acknowledge(&mut self, id: &str) -> bool {
        match self.open.get_mut(id) {
            Some(ev) => {
                ev.acknowledged = true;
                true
            }
            None => false,
        }
    }

    /// Annotate an OPEN event with a concise auto-remediation outcome
    /// note (e.g. `"[auto] compacted 3 shards @1700000000000"`). The
    /// `ok` flag tags the note `[auto]` on success or `[auto-fail]` on
    /// failure. No-op (returns `false`) if no open event matches.
    pub fn annotate_remediation(
        &mut self,
        event_id: &str,
        note: &str,
        ok: bool,
        at_ms: i64,
    ) -> bool {
        match self.open.get_mut(event_id) {
            Some(ev) => {
                let tag = if ok { "[auto]" } else { "[auto-fail]" };
                ev.remediation_note = Some(format!("{tag} {note} @{at_ms}"));
                true
            }
            None => false,
        }
    }

    /// Persist open events + bounded history to JSON. Failure is
    /// logged, not surfaced — the in-memory copy is authoritative.
    pub fn save(&self) {
        let persisted = PersistedEvents {
            open: self.open.values().cloned().collect(),
            history: self.history.iter().cloned().collect(),
        };
        if let Some(parent) = self.path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match serde_json::to_vec_pretty(&persisted) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&self.path, bytes) {
                    tracing::warn!(error = %e, path = %self.path.display(), "ops_events: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "ops_events: serialize failed"),
        }
    }

    /// Load open events + history from disk into a new store. A missing
    /// or unparseable file yields an empty store (logged).
    pub fn load(data_dir: &Path) -> Self {
        let mut store = Self::new(data_dir);
        let bytes = match std::fs::read(&store.path) {
            Ok(b) => b,
            Err(_) => return store,
        };
        let persisted: PersistedEvents = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %store.path.display(),
                    "ops_events: parse failed, starting empty"
                );
                return store;
            }
        };
        for ev in persisted.open {
            store.open.insert(ev.id.clone(), ev);
        }
        for ev in persisted.history.into_iter().rev().take(store.history_cap).rev() {
            store.history.push_back(ev);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::rules::RuleHit;

    fn hit(rule_id: &str, scope: Scope, sev: Severity) -> RuleHit {
        RuleHit {
            rule_id: rule_id.to_owned(),
            scope,
            severity: sev,
            title: format!("{rule_id} tripped"),
            detail: "root cause".to_owned(),
            recommendation: "do the thing".to_owned(),
            runbook: Some("POST /v1/...".to_owned()),
            remediation: RemediationAction::None,
        }
    }

    #[test]
    fn reconcile_opens_then_dedups_then_closes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());

        // First pass with one hit: opens an event.
        let h = hit("compaction_backlog", Scope::Collection("docs".into()), Severity::Medium);
        let r1 = store.reconcile(vec![h.clone()], 1000);
        assert_eq!(r1.opened.len(), 1);
        assert_eq!(r1.closed.len(), 0);
        assert_eq!(r1.opened[0].id, "compaction_backlog:collection:docs");
        assert_eq!(store.open_events().len(), 1);

        // Second pass, same hit: NO re-open, NO close.
        let r2 = store.reconcile(vec![h.clone()], 2000);
        assert!(r2.opened.is_empty(), "duplicate hit must not re-open");
        assert!(r2.closed.is_empty());
        assert_eq!(store.open_events().len(), 1);
        // opened_at stays at the original open time.
        assert_eq!(store.open_events()[0].opened_at_ms, 1000);

        // Third pass with no hits: closes the event, moves to history.
        let r3 = store.reconcile(vec![], 3000);
        assert!(r3.opened.is_empty());
        assert_eq!(r3.closed.len(), 1);
        assert_eq!(r3.closed[0].status, EventStatus::Closed);
        assert_eq!(r3.closed[0].closed_at_ms, Some(3000));
        assert!(store.open_events().is_empty());
        assert_eq!(store.history().len(), 1);
    }

    #[test]
    fn refresh_updates_severity_in_place() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        let scope = Scope::Collection("docs".into());
        store.reconcile(vec![hit("r", scope.clone(), Severity::Low)], 1000);
        // Same id, higher severity → refreshed, not re-opened.
        let r = store.reconcile(vec![hit("r", scope, Severity::High)], 2000);
        assert!(r.opened.is_empty());
        assert_eq!(store.open_events()[0].severity, Severity::High);
    }

    #[test]
    fn open_events_sorted_by_severity_desc() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        store.reconcile(
            vec![
                hit("a", Scope::Node("n1".into()), Severity::Low),
                hit("b", Scope::Node("n2".into()), Severity::High),
                hit("c", Scope::Node("n3".into()), Severity::Medium),
            ],
            1000,
        );
        let sevs: Vec<Severity> = store.open_events().iter().map(|e| e.severity).collect();
        assert_eq!(sevs, vec![Severity::High, Severity::Medium, Severity::Low]);
    }

    #[test]
    fn acknowledge_marks_open_event() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        store.reconcile(vec![hit("r", Scope::Global, Severity::High)], 1000);
        let id = "r:global";
        assert!(store.acknowledge(id));
        assert!(store.open_events()[0].acknowledged);
        assert!(!store.acknowledge("nope:global"));
    }

    #[test]
    fn save_load_round_trips_open_and_history() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        // Open two, then close one (-> history).
        store.reconcile(
            vec![
                hit("open_rule", Scope::Collection("a".into()), Severity::High),
                hit("close_rule", Scope::Collection("b".into()), Severity::Low),
            ],
            1000,
        );
        store.reconcile(
            vec![hit("open_rule", Scope::Collection("a".into()), Severity::High)],
            2000,
        );
        assert_eq!(store.open_events().len(), 1);
        assert_eq!(store.history().len(), 1);
        store.save();

        let loaded = EventStore::load(tmp.path());
        assert_eq!(loaded.open_events().len(), 1);
        assert_eq!(loaded.open_events()[0].id, "open_rule:collection:a");
        assert_eq!(loaded.history().len(), 1);
        assert_eq!(loaded.history()[0].rule_id, "close_rule");
        assert_eq!(loaded.history()[0].status, EventStatus::Closed);
    }

    #[test]
    fn annotate_remediation_sets_note_and_reconcile_preserves_it() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        let scope = Scope::Collection("docs".into());
        store.reconcile(vec![hit("compaction_backlog", scope.clone(), Severity::Medium)], 1000);
        let id = "compaction_backlog:collection:docs";

        // Annotate the open event with a remediation outcome.
        assert!(store.annotate_remediation(id, "compacted 3 shards", true, 1500));
        let note = store.open_events()[0].remediation_note.clone().unwrap();
        assert!(note.contains("compacted 3 shards"));
        assert!(note.starts_with("[auto]"));
        assert!(note.contains("@1500"));

        // A subsequent reconcile refresh of the SAME open event must
        // PRESERVE the note (regression against clobbering).
        let r = store.reconcile(
            vec![hit("compaction_backlog", scope, Severity::High)],
            2000,
        );
        assert!(r.opened.is_empty(), "refresh, not re-open");
        let after = store.open_events()[0].clone();
        assert_eq!(after.severity, Severity::High, "severity still refreshed");
        assert_eq!(
            after.remediation_note,
            Some("[auto] compacted 3 shards @1500".to_owned()),
            "reconcile refresh must preserve remediation_note"
        );

        // Annotating a non-open event is a no-op.
        assert!(!store.annotate_remediation("nope:global", "x", true, 3000));
    }

    #[test]
    fn annotate_remediation_failure_tags_auto_fail() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = EventStore::new(tmp.path());
        store.reconcile(vec![hit("r", Scope::Global, Severity::High)], 1000);
        assert!(store.annotate_remediation("r:global", "boom", false, 2000));
        let note = store.open_events()[0].remediation_note.clone().unwrap();
        assert!(note.starts_with("[auto-fail]"), "got: {note}");
        assert!(note.contains("boom"));
    }

    #[test]
    fn severity_orders_high_above_low() {
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
    }

    #[test]
    fn scope_keys_are_stable() {
        assert_eq!(Scope::Global.key(), "global");
        assert_eq!(Scope::Collection("c".into()).key(), "collection:c");
        assert_eq!(
            Scope::Shard { collection: "c".into(), shard: 2 }.key(),
            "shard:c/2"
        );
        assert_eq!(Scope::Node("n".into()).key(), "node:n");
    }
}
