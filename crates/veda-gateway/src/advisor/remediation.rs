//! Opt-in auto-remediation for the ops advisor.
//!
//! The advisor engine reconciles a snapshot into open/close transitions
//! (see [`EventStore::reconcile`](crate::advisor::event::EventStore::reconcile)).
//! For each *newly opened* event this module decides — strictly opt-in —
//! whether to act on its [`RemediationAction`], dispatches the action,
//! and returns a [`RemediationOutcome`] the engine uses to annotate the
//! event (via [`EventStore::annotate_remediation`](crate::advisor::event::EventStore::annotate_remediation)).
//!
//! **Gating (all must hold to dispatch):**
//! * the event carries a non-[`RemediationAction::None`] action;
//! * the rule's config has `auto_remediate = true`;
//! * the rule is `enabled && !muted`;
//! * a per-scope [`RateLimiter`] permits another action this window.
//!
//! **No `dyn` async:** dispatch goes through a generic
//! `<D: RemediationDispatcher>` bound, so the trait can use a native
//! `async fn` (matching [`crate::advisor`]'s sibling `ChatModel`
//! pattern) without needing object safety.

use std::collections::HashMap;

use crate::advisor::config::AdvisorConfig;
use crate::advisor::event::{OpsEvent, RemediationAction};

/// Per-scope rate limiter: at most one remediation per scope key per
/// `min_interval_ms` window. Cheap last-fired timestamp map.
#[derive(Debug)]
pub struct RateLimiter {
    last: HashMap<String, i64>,
    min_interval_ms: i64,
}

/// Default minimum interval between remediations for one scope: 5 min.
pub const DEFAULT_MIN_INTERVAL_MS: i64 = 300_000;

impl RateLimiter {
    /// New limiter with the given minimum inter-action interval (ms).
    pub fn new(min_interval_ms: i64) -> Self {
        Self {
            last: HashMap::new(),
            min_interval_ms,
        }
    }

    /// `true` (and records `now_ms`) if `scope_key` has not fired within
    /// the last `min_interval_ms`; `false` (no record) if still inside
    /// the window.
    pub fn allow(&mut self, scope_key: &str, now_ms: i64) -> bool {
        match self.last.get(scope_key) {
            Some(&prev) if now_ms - prev < self.min_interval_ms => false,
            _ => {
                self.last.insert(scope_key.to_owned(), now_ms);
                true
            }
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_INTERVAL_MS)
    }
}

/// Executes a single [`RemediationAction`]. `Ok` carries a
/// human-readable outcome note; `Err` carries a failure note. Both are
/// recorded on the event by the engine.
#[allow(async_fn_in_trait)]
pub trait RemediationDispatcher: Send + Sync {
    async fn dispatch(&self, action: &RemediationAction, now_ms: i64) -> Result<String, String>;
}

/// Outcome of attempting to remediate one opened event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemediationOutcome {
    pub event_id: String,
    pub action: RemediationAction,
    pub ok: bool,
    pub note: String,
    pub at_ms: i64,
}

/// Run opt-in auto-remediation over this round's `opened` events.
///
/// For each opened event, in order, dispatch its action iff all gates
/// (see module docs) pass. Events that are skipped produce no outcome;
/// the returned vec contains exactly the dispatched attempts (success or
/// failure). The rate limiter is consulted (and stamped) only when every
/// other gate has already passed, so a gated-out event never consumes a
/// scope's budget.
pub async fn run_remediation<D: RemediationDispatcher>(
    opened: &[OpsEvent],
    cfg: &AdvisorConfig,
    dispatcher: &D,
    limiter: &mut RateLimiter,
    now_ms: i64,
) -> Vec<RemediationOutcome> {
    let mut outcomes = Vec::new();

    for ev in opened {
        // Gate 1: must have an actionable remediation.
        if ev.remediation == RemediationAction::None {
            continue;
        }
        // Gate 2/3: rule opt-in + active.
        let rc = cfg.for_rule(&ev.rule_id);
        if !rc.auto_remediate || !rc.enabled || rc.muted {
            continue;
        }
        // Gate 4: per-scope rate limit (stamped only past the gates).
        if !limiter.allow(&ev.scope.key(), now_ms) {
            continue;
        }

        let (ok, note) = match dispatcher.dispatch(&ev.remediation, now_ms).await {
            Ok(n) => (true, n),
            Err(n) => (false, n),
        };
        outcomes.push(RemediationOutcome {
            event_id: ev.id.clone(),
            action: ev.remediation.clone(),
            ok,
            note,
            at_ms: now_ms,
        });
    }

    outcomes
}

mod live;
pub use live::LiveDispatcher;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::config::RuleConfig;
    use crate::advisor::event::{EventStatus, OpsEvent, Scope, Severity};
    use std::sync::Mutex;

    /// Spy dispatcher: records every action it was asked to run and
    /// returns a canned Ok note. No real compaction.
    struct SpyDispatcher {
        seen: Mutex<Vec<RemediationAction>>,
    }
    impl SpyDispatcher {
        fn new() -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<RemediationAction> {
            self.seen.lock().unwrap().clone()
        }
    }
    impl RemediationDispatcher for SpyDispatcher {
        async fn dispatch(
            &self,
            action: &RemediationAction,
            _now_ms: i64,
        ) -> Result<String, String> {
            self.seen.lock().unwrap().push(action.clone());
            Ok("did the thing".to_owned())
        }
    }

    fn event(rule_id: &str, scope: Scope, action: RemediationAction) -> OpsEvent {
        OpsEvent {
            id: OpsEvent::make_id(rule_id, &scope),
            rule_id: rule_id.to_owned(),
            scope,
            severity: Severity::Medium,
            status: EventStatus::Open,
            title: "t".into(),
            detail: "d".into(),
            recommendation: "r".into(),
            runbook: None,
            remediation: action,
            opened_at_ms: 0,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }

    fn cfg_with(rule_id: &str, rc: RuleConfig) -> AdvisorConfig {
        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(rule_id.to_owned(), rc);
        cfg
    }

    fn compact(col: &str) -> RemediationAction {
        RemediationAction::Compact {
            collection: col.to_owned(),
        }
    }

    #[tokio::test]
    async fn auto_remediate_enabled_dispatches_once() {
        let spy = SpyDispatcher::new();
        let mut limiter = RateLimiter::new(300_000);
        let cfg = cfg_with(
            "compaction_backlog",
            RuleConfig {
                auto_remediate: true,
                ..Default::default()
            },
        );
        let opened = vec![event(
            "compaction_backlog",
            Scope::Collection("docs".into()),
            compact("docs"),
        )];

        let out = run_remediation(&opened, &cfg, &spy, &mut limiter, 1000).await;
        assert_eq!(out.len(), 1);
        assert!(out[0].ok);
        assert_eq!(out[0].event_id, "compaction_backlog:collection:docs");
        assert_eq!(out[0].action, compact("docs"));
        assert_eq!(out[0].at_ms, 1000);
        assert_eq!(spy.calls(), vec![compact("docs")]);
    }

    #[tokio::test]
    async fn auto_remediate_disabled_does_not_dispatch() {
        let spy = SpyDispatcher::new();
        let mut limiter = RateLimiter::new(300_000);
        // Default RuleConfig has auto_remediate = false.
        let cfg = AdvisorConfig::default();
        let opened = vec![event(
            "compaction_backlog",
            Scope::Collection("docs".into()),
            compact("docs"),
        )];

        let out = run_remediation(&opened, &cfg, &spy, &mut limiter, 1000).await;
        assert!(out.is_empty(), "auto_remediate=false must not dispatch");
        assert!(spy.calls().is_empty());
    }

    #[tokio::test]
    async fn muted_or_disabled_rule_does_not_dispatch() {
        let spy = SpyDispatcher::new();
        let mut limiter = RateLimiter::new(300_000);
        let opened = vec![event(
            "compaction_backlog",
            Scope::Collection("docs".into()),
            compact("docs"),
        )];

        // Muted (even with auto_remediate on).
        let cfg_muted = cfg_with(
            "compaction_backlog",
            RuleConfig {
                auto_remediate: true,
                muted: true,
                ..Default::default()
            },
        );
        assert!(
            run_remediation(&opened, &cfg_muted, &spy, &mut limiter, 1000)
                .await
                .is_empty()
        );

        // Disabled.
        let cfg_off = cfg_with(
            "compaction_backlog",
            RuleConfig {
                auto_remediate: true,
                enabled: false,
                ..Default::default()
            },
        );
        assert!(
            run_remediation(&opened, &cfg_off, &spy, &mut limiter, 2000)
                .await
                .is_empty()
        );

        assert!(spy.calls().is_empty());
    }

    #[tokio::test]
    async fn none_action_is_skipped() {
        let spy = SpyDispatcher::new();
        let mut limiter = RateLimiter::new(300_000);
        let cfg = cfg_with(
            "some_rule",
            RuleConfig {
                auto_remediate: true,
                ..Default::default()
            },
        );
        let opened = vec![event(
            "some_rule",
            Scope::Collection("docs".into()),
            RemediationAction::None,
        )];

        let out = run_remediation(&opened, &cfg, &spy, &mut limiter, 1000).await;
        assert!(out.is_empty());
        assert!(spy.calls().is_empty());
    }

    #[tokio::test]
    async fn rate_limit_blocks_second_then_allows_after_window() {
        let spy = SpyDispatcher::new();
        let mut limiter = RateLimiter::new(300_000);
        let cfg = cfg_with(
            "compaction_backlog",
            RuleConfig {
                auto_remediate: true,
                ..Default::default()
            },
        );
        // Two opened events, SAME scope, within the window.
        let opened = vec![
            event(
                "compaction_backlog",
                Scope::Collection("docs".into()),
                compact("docs"),
            ),
            event(
                "compaction_backlog",
                Scope::Collection("docs".into()),
                compact("docs"),
            ),
        ];

        let out1 = run_remediation(&opened, &cfg, &spy, &mut limiter, 1000).await;
        assert_eq!(out1.len(), 1, "only first dispatch within window");

        // Still within window → blocked.
        let out2 = run_remediation(&opened[..1], &cfg, &spy, &mut limiter, 1000 + 200_000).await;
        assert!(out2.is_empty(), "still rate-limited");

        // Past the window → allowed again.
        let out3 = run_remediation(&opened[..1], &cfg, &spy, &mut limiter, 1000 + 300_001).await;
        assert_eq!(out3.len(), 1, "dispatch again after window");

        assert_eq!(spy.calls().len(), 2);
    }

    #[test]
    fn rate_limiter_allow_semantics() {
        let mut l = RateLimiter::new(1000);
        assert!(l.allow("s", 0));
        assert!(!l.allow("s", 999), "within window");
        assert!(l.allow("s", 1000), "boundary is outside (>=)");
        assert!(l.allow("other", 1000), "independent scope");
    }
}
