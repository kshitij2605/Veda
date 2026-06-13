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

/// Notification channel: a generic HTTP webhook. The notifier `POST`s
/// `{ status, event }` JSON to [`url`](WebhookChannel::url).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebhookChannel {
    #[serde(default)]
    pub enabled: bool,
    pub url: String,
    /// Only events with `severity >= min_severity` are delivered.
    #[serde(default = "min_low")]
    pub min_severity: Severity,
}

/// Slack incoming-webhook channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlackChannel {
    #[serde(default)]
    pub enabled: bool,
    pub webhook_url: String,
    #[serde(default = "min_low")]
    pub min_severity: Severity,
}

/// PagerDuty Events API v2 channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PagerDutyChannel {
    #[serde(default)]
    pub enabled: bool,
    pub routing_key: String,
    #[serde(default = "min_low")]
    pub min_severity: Severity,
}

/// Email channel over (async) SMTP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmailChannel {
    #[serde(default)]
    pub enabled: bool,
    pub smtp_host: String,
    pub smtp_port: u16,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    pub from: String,
    #[serde(default)]
    pub to: Vec<String>,
    #[serde(default = "min_low")]
    pub min_severity: Severity,
}

/// Default `min_severity` (deliver everything) for channels that omit it.
fn min_low() -> Severity {
    Severity::Low
}

/// All configured notification channels. Each is `Option` and, when
/// present, individually `enabled`-gated. Default = nothing configured
/// (silent), so existing persisted configs load unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChannelsConfig {
    #[serde(default)]
    pub webhook: Option<WebhookChannel>,
    #[serde(default)]
    pub slack: Option<SlackChannel>,
    #[serde(default)]
    pub pagerduty: Option<PagerDutyChannel>,
    #[serde(default)]
    pub email: Option<EmailChannel>,
}

/// Top-level advisor configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvisorConfig {
    /// Per-rule overrides keyed by `rule.id()`. Absent → defaults.
    #[serde(default)]
    pub rules: HashMap<String, RuleConfig>,
    /// Notification channels. Absent → all disabled (silent).
    #[serde(default)]
    pub channels: ChannelsConfig,
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

    #[test]
    fn channels_default_is_all_none() {
        let cfg = AdvisorConfig::default();
        assert!(cfg.channels.webhook.is_none());
        assert!(cfg.channels.slack.is_none());
        assert!(cfg.channels.pagerduty.is_none());
        assert!(cfg.channels.email.is_none());
    }

    #[test]
    fn legacy_config_without_channels_still_loads() {
        // A persisted config from before channels existed.
        let json = r#"{ "rules": {} }"#;
        let cfg: AdvisorConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.channels.webhook.is_none());
    }

    #[test]
    fn webhook_channel_defaults_min_severity_to_low() {
        let json = r#"{ "enabled": true, "url": "http://x" }"#;
        let ch: WebhookChannel = serde_json::from_str(json).unwrap();
        assert!(ch.enabled);
        assert_eq!(ch.min_severity, Severity::Low);
    }
}
