//! Advisor configuration: per-rule knobs + defaults.
//!
//! This task only defines the structs, their `Default`s, and the
//! [`AdvisorConfig::for_rule`] accessor. Persistence and a REST surface
//! land in a later task; channel (notification) config is added then
//! too.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::advisor::event::Severity;

/// Per-rule configuration. Unset fields fall back to the rule's own
/// defaults at evaluation time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleConfig {
    /// Whether the rule runs at all.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Muted rules still run but their hits are suppressed (no events /
    /// notifications). Kept distinct from `enabled` so the UI can show
    /// "temporarily silenced" vs "off".
    #[serde(default)]
    pub muted: bool,
    /// Override the rule's natural severity for every hit it emits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity_override: Option<Severity>,
    /// Opaque per-rule threshold knobs (e.g. `{"sealed": 50}`). Each
    /// rule reads the keys it understands and falls back to its
    /// compiled-in default otherwise.
    #[serde(default)]
    pub thresholds: serde_json::Value,
    /// Opt-in to auto-remediation for this rule (consumed by the
    /// remediation task; inert here).
    #[serde(default)]
    pub auto_remediate: bool,
}

fn default_true() -> bool {
    true
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            muted: false,
            severity_override: None,
            thresholds: serde_json::Value::Null,
            auto_remediate: false,
        }
    }
}

/// Top-level advisor configuration. Channel (notification) config is
/// added in a later task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvisorConfig {
    /// Per-rule overrides keyed by `rule.id()`. Absent → defaults.
    #[serde(default)]
    pub rules: HashMap<String, RuleConfig>,
}

impl AdvisorConfig {
    /// The configuration for `rule_id`, or [`RuleConfig::default`] when
    /// none is set.
    pub fn for_rule(&self, rule_id: &str) -> RuleConfig {
        self.rules.get(rule_id).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_config_defaults() {
        let c = RuleConfig::default();
        assert!(c.enabled);
        assert!(!c.muted);
        assert!(c.severity_override.is_none());
        assert!(!c.auto_remediate);
    }

    #[test]
    fn for_rule_returns_default_when_unset() {
        let cfg = AdvisorConfig::default();
        let rc = cfg.for_rule("anything");
        assert!(rc.enabled);
        assert!(!rc.muted);
    }

    #[test]
    fn for_rule_returns_configured() {
        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(
            "high_search_latency".into(),
            RuleConfig {
                enabled: false,
                severity_override: Some(Severity::High),
                ..Default::default()
            },
        );
        let rc = cfg.for_rule("high_search_latency");
        assert!(!rc.enabled);
        assert_eq!(rc.severity_override, Some(Severity::High));
    }
}
