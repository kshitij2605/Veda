//! Ops-advisor notification fan-out.
//!
//! The advisor engine reconciles snapshots into open/close
//! *transitions* (see [`EventStore::reconcile`](crate::advisor::event::EventStore::reconcile))
//! and hands the `opened` / `closed` edges to [`Notifier::notify_transitions`].
//! Because reconcile already dedups, this layer is invoked only on
//! edges — it does no per-tick dedup of its own.
//!
//! Each configured + enabled channel receives every event meeting its
//! `min_severity` gate: an OPEN/trigger notification for `opened`
//! events, a CLOSE/resolve notification for `closed` ones.
//!
//! **Failure isolation:** a single channel or single event failing
//! (dead URL, SMTP refusal, …) is caught and logged via
//! [`tracing::warn`] — it never propagates and never blocks the other
//! channels/events. The engine cannot be taken down by a misconfigured
//! webhook.
//!
//! **Async-only HTTP:** all delivery uses the async [`reqwest::Client`]
//! and (for email) lettre's async [`AsyncSmtpTransport`]. No blocking
//! client is used anywhere — that would panic inside the runtime.

use lettre::message::header::ContentType;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use serde_json::json;

use crate::advisor::config::{
    ChannelsConfig, EmailChannel, PagerDutyChannel, SlackChannel, WebhookChannel,
};
use crate::advisor::event::{OpsEvent, Severity};

/// PagerDuty Events API v2 enqueue endpoint. A field on [`Notifier`]
/// overrides it in tests (real PagerDuty is unreachable from CI).
const PAGERDUTY_ENQUEUE_URL: &str = "https://events.pagerduty.com/v2/enqueue";

/// Severity gate: does `ev` meet the channel's `min` floor?
fn meets(min: Severity, ev: Severity) -> bool {
    ev >= min
}

/// Severity emoji marker used in Slack messages.
fn severity_emoji(sev: Severity) -> &'static str {
    match sev {
        Severity::High => "🔴",
        Severity::Medium => "🟠",
        Severity::Low => "🟡",
    }
}

/// PagerDuty payload severity string for an event severity.
fn pagerduty_severity(sev: Severity) -> &'static str {
    match sev {
        Severity::High => "critical",
        Severity::Medium => "warning",
        Severity::Low => "info",
    }
}

/// Fans advisor open/close transitions out to the configured channels.
pub struct Notifier {
    client: reqwest::Client,
    channels: ChannelsConfig,
    /// PagerDuty enqueue URL — overridable for tests.
    pagerduty_url: String,
}

impl Notifier {
    /// Build a notifier for the given channels.
    pub fn new(channels: ChannelsConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            channels,
            pagerduty_url: PAGERDUTY_ENQUEUE_URL.to_owned(),
        }
    }

    /// Deliver one round of transitions. Per-channel, per-event errors
    /// are caught + logged; this never errors out the caller. Returns
    /// the number of notifications successfully delivered.
    pub async fn notify_transitions(&self, opened: &[OpsEvent], closed: &[OpsEvent]) -> usize {
        let mut sent = 0usize;
        for (ev, is_open) in opened
            .iter()
            .map(|e| (e, true))
            .chain(closed.iter().map(|e| (e, false)))
        {
            sent += self.dispatch_event(ev, is_open).await;
        }
        sent
    }

    /// Send `ev` to every enabled channel whose `min_severity` it
    /// meets. Each send is isolated.
    async fn dispatch_event(&self, ev: &OpsEvent, is_open: bool) -> usize {
        let mut sent = 0usize;

        if let Some(cfg) = &self.channels.webhook {
            if cfg.enabled && meets(cfg.min_severity, ev.severity) {
                match self.send_webhook(cfg, ev, is_open).await {
                    Ok(()) => sent += 1,
                    Err(e) => tracing::warn!(channel = "webhook", event = %ev.id, error = %e, "notify: send failed"),
                }
            }
        }
        if let Some(cfg) = &self.channels.slack {
            if cfg.enabled && meets(cfg.min_severity, ev.severity) {
                match self.send_slack(cfg, ev, is_open).await {
                    Ok(()) => sent += 1,
                    Err(e) => tracing::warn!(channel = "slack", event = %ev.id, error = %e, "notify: send failed"),
                }
            }
        }
        if let Some(cfg) = &self.channels.pagerduty {
            if cfg.enabled && meets(cfg.min_severity, ev.severity) {
                match self.send_pagerduty(cfg, ev, is_open).await {
                    Ok(()) => sent += 1,
                    Err(e) => tracing::warn!(channel = "pagerduty", event = %ev.id, error = %e, "notify: send failed"),
                }
            }
        }
        if let Some(cfg) = &self.channels.email {
            if cfg.enabled && meets(cfg.min_severity, ev.severity) {
                match self.send_email(cfg, ev, is_open).await {
                    Ok(()) => sent += 1,
                    Err(e) => tracing::warn!(channel = "email", event = %ev.id, error = %e, "notify: send failed"),
                }
            }
        }
        sent
    }

    // -- channel senders ------------------------------------------------

    /// Generic webhook: `POST {url}` with `{ status, event }` JSON.
    async fn send_webhook(
        &self,
        cfg: &WebhookChannel,
        ev: &OpsEvent,
        is_open: bool,
    ) -> Result<(), reqwest::Error> {
        let body = webhook_body(ev, is_open);
        self.client
            .post(&cfg.url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Slack incoming webhook: `POST {webhook_url}` with a `text` field.
    async fn send_slack(
        &self,
        cfg: &SlackChannel,
        ev: &OpsEvent,
        is_open: bool,
    ) -> Result<(), reqwest::Error> {
        let body = slack_body(ev, is_open);
        self.client
            .post(&cfg.webhook_url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// PagerDuty Events API v2 trigger/resolve.
    async fn send_pagerduty(
        &self,
        cfg: &PagerDutyChannel,
        ev: &OpsEvent,
        is_open: bool,
    ) -> Result<(), reqwest::Error> {
        let body = pagerduty_body(&cfg.routing_key, ev, is_open);
        self.client
            .post(&self.pagerduty_url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Email via async SMTP — one message per `to` address.
    async fn send_email(
        &self,
        cfg: &EmailChannel,
        ev: &OpsEvent,
        is_open: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut transport =
            AsyncSmtpTransport::<Tokio1Executor>::relay(&cfg.smtp_host)?.port(cfg.smtp_port);
        if !cfg.username.is_empty() {
            transport =
                transport.credentials(Credentials::new(cfg.username.clone(), cfg.password.clone()));
        }
        let mailer = transport.build();
        for to in &cfg.to {
            let msg = build_email(cfg, ev, is_open, to)?;
            mailer.send(msg).await?;
        }
        Ok(())
    }
}

// -- pure body builders (unit-testable without sending) ----------------

/// Webhook JSON body: `{ status, event }`.
fn webhook_body(ev: &OpsEvent, is_open: bool) -> serde_json::Value {
    json!({
        "status": if is_open { "open" } else { "closed" },
        "event": ev,
    })
}

/// Slack incoming-webhook JSON body.
fn slack_body(ev: &OpsEvent, is_open: bool) -> serde_json::Value {
    let emoji = severity_emoji(ev.severity);
    let prefix = if is_open { String::new() } else { "RESOLVED: ".to_owned() };
    let text = format!(
        "{emoji} {prefix}*{title}*\nScope: `{scope}`\n{detail}\n_Recommendation:_ {rec}",
        title = ev.title,
        scope = ev.scope.key(),
        detail = ev.detail,
        rec = ev.recommendation,
    );
    json!({
        "text": text,
        "blocks": [
            { "type": "section", "text": { "type": "mrkdwn", "text": text } }
        ]
    })
}

/// PagerDuty Events API v2 body. `payload` is only required for trigger.
fn pagerduty_body(routing_key: &str, ev: &OpsEvent, is_open: bool) -> serde_json::Value {
    if is_open {
        json!({
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": ev.id,
            "payload": {
                "summary": ev.title,
                "severity": pagerduty_severity(ev.severity),
                "source": "veda",
            }
        })
    } else {
        json!({
            "routing_key": routing_key,
            "event_action": "resolve",
            "dedup_key": ev.id,
        })
    }
}

/// Build (but do not send) the email message for one recipient.
/// Subject: `"[{severity}] {title}"` (open) / `"[RESOLVED] {title}"`.
/// Body: detail + recommendation + optional runbook, plaintext.
pub fn build_email(
    cfg: &EmailChannel,
    ev: &OpsEvent,
    is_open: bool,
    to: &str,
) -> Result<Message, Box<dyn std::error::Error + Send + Sync>> {
    let subject = if is_open {
        let sev = match ev.severity {
            Severity::High => "High",
            Severity::Medium => "Medium",
            Severity::Low => "Low",
        };
        format!("[{sev}] {}", ev.title)
    } else {
        format!("[RESOLVED] {}", ev.title)
    };

    let mut body = format!("{}\n\nRecommendation: {}", ev.detail, ev.recommendation);
    if let Some(runbook) = &ev.runbook {
        body.push_str(&format!("\n\nRunbook: {runbook}"));
    }

    let msg = Message::builder()
        .from(cfg.from.parse()?)
        .to(to.parse()?)
        .subject(subject)
        .header(ContentType::TEXT_PLAIN)
        .body(body)?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::config::{
        EmailChannel, PagerDutyChannel, SlackChannel, WebhookChannel,
    };
    use crate::advisor::event::{EventStatus, RemediationAction, Scope};
    use std::io::{Read, Write};
    use std::net::TcpListener as StdTcpListener;
    use std::sync::mpsc;

    fn ev(severity: Severity, status: EventStatus) -> OpsEvent {
        OpsEvent {
            id: "compaction_backlog:collection:docs".to_owned(),
            rule_id: "compaction_backlog".to_owned(),
            scope: Scope::Collection("docs".to_owned()),
            severity,
            status,
            title: "Compaction backlog on docs".to_owned(),
            detail: "42 sealed segments".to_owned(),
            recommendation: "Trigger compaction".to_owned(),
            runbook: Some("POST /v1/collections/docs/compact".to_owned()),
            remediation: RemediationAction::None,
            opened_at_ms: 1000,
            closed_at_ms: None,
            acknowledged: false,
            remediation_note: None,
        }
    }

    /// A throwaway HTTP/1.1 mock handling exactly `n` POSTs. Each POST's
    /// `(path, body)` is sent over the channel; a `200 {}` is returned.
    /// Returns the `http://127.0.0.1:PORT` base URL.
    fn spawn_capture(n: usize) -> (String, mpsc::Receiver<(String, String)>) {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for _ in 0..n {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = Vec::new();
                let mut tmp = [0u8; 1024];
                loop {
                    let read = stream.read(&mut tmp).unwrap_or(0);
                    if read == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..read]);
                    let text = String::from_utf8_lossy(&buf);
                    if let Some(hdr_end) = text.find("\r\n\r\n") {
                        let headers = &text[..hdr_end];
                        let body_start = hdr_end + 4;
                        let content_len = headers
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if buf.len() - body_start >= content_len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&buf).to_string();
                let path = text
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("")
                    .to_owned();
                let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
                let _ = tx.send((path, body));
                let reply = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
                let _ = stream.write_all(reply.as_bytes());
                let _ = stream.flush();
            }
        });
        (url, rx)
    }

    #[test]
    fn meets_gates_by_severity() {
        assert!(meets(Severity::Low, Severity::Low));
        assert!(meets(Severity::Low, Severity::High));
        assert!(meets(Severity::High, Severity::High));
        assert!(!meets(Severity::High, Severity::Low));
        assert!(!meets(Severity::Medium, Severity::Low));
        assert!(meets(Severity::Medium, Severity::Medium));
    }

    #[tokio::test]
    async fn webhook_open_then_close_posts_status() {
        let (url, rx) = spawn_capture(2);
        let channels = ChannelsConfig {
            webhook: Some(WebhookChannel {
                enabled: true,
                url: url.clone(),
                min_severity: Severity::Low,
            }),
            ..Default::default()
        };
        let notifier = Notifier::new(channels);

        let opened = vec![ev(Severity::High, EventStatus::Open)];
        let closed = vec![ev(Severity::High, EventStatus::Closed)];
        let sent = notifier.notify_transitions(&opened, &closed).await;
        assert_eq!(sent, 2);

        let (_, body1) = rx.recv().unwrap();
        let v1: serde_json::Value = serde_json::from_str(&body1).unwrap();
        assert_eq!(v1["status"], "open");
        assert_eq!(v1["event"]["title"], "Compaction backlog on docs");

        let (_, body2) = rx.recv().unwrap();
        let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
        assert_eq!(v2["status"], "closed");
    }

    #[tokio::test]
    async fn min_severity_routing_drops_low_events() {
        // Channel floor = High. A Low event must NOT be POSTed; a High one must.
        let (url, rx) = spawn_capture(1);
        let channels = ChannelsConfig {
            webhook: Some(WebhookChannel {
                enabled: true,
                url: url.clone(),
                min_severity: Severity::High,
            }),
            ..Default::default()
        };
        let notifier = Notifier::new(channels);

        let low = vec![ev(Severity::Low, EventStatus::Open)];
        let sent_low = notifier.notify_transitions(&low, &[]).await;
        assert_eq!(sent_low, 0, "low event must be gated out");

        let high = vec![ev(Severity::High, EventStatus::Open)];
        let sent_high = notifier.notify_transitions(&high, &[]).await;
        assert_eq!(sent_high, 1, "high event must pass the gate");

        // Exactly one capture, and it's the High event.
        let (_, body) = rx.recv().unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "open");
        assert!(rx.try_recv().is_err(), "only the high event should be sent");
    }

    #[tokio::test]
    async fn slack_payload_has_title_and_severity_marker() {
        let (url, rx) = spawn_capture(1);
        let channels = ChannelsConfig {
            slack: Some(SlackChannel {
                enabled: true,
                webhook_url: url.clone(),
                min_severity: Severity::Low,
            }),
            ..Default::default()
        };
        let notifier = Notifier::new(channels);
        notifier
            .notify_transitions(&[ev(Severity::High, EventStatus::Open)], &[])
            .await;

        let (_, body) = rx.recv().unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let text = v["text"].as_str().unwrap();
        assert!(text.contains("Compaction backlog on docs"));
        assert!(text.contains("🔴"), "High should carry the red marker");
    }

    #[tokio::test]
    async fn slack_resolved_prefixed_on_close() {
        let (url, rx) = spawn_capture(1);
        let channels = ChannelsConfig {
            slack: Some(SlackChannel {
                enabled: true,
                webhook_url: url.clone(),
                min_severity: Severity::Low,
            }),
            ..Default::default()
        };
        let notifier = Notifier::new(channels);
        notifier
            .notify_transitions(&[], &[ev(Severity::Medium, EventStatus::Closed)])
            .await;

        let (_, body) = rx.recv().unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["text"].as_str().unwrap().contains("RESOLVED:"));
    }

    #[tokio::test]
    async fn pagerduty_trigger_and_resolve_with_dedup_key() {
        let (url, rx) = spawn_capture(2);
        let channels = ChannelsConfig {
            pagerduty: Some(PagerDutyChannel {
                enabled: true,
                routing_key: "rk-123".to_owned(),
                min_severity: Severity::Low,
            }),
            ..Default::default()
        };
        let mut notifier = Notifier::new(channels);
        notifier.pagerduty_url = format!("{url}/v2/enqueue");

        let opened = vec![ev(Severity::High, EventStatus::Open)];
        let closed = vec![ev(Severity::High, EventStatus::Closed)];
        notifier.notify_transitions(&opened, &closed).await;

        let (path1, body1) = rx.recv().unwrap();
        assert_eq!(path1, "/v2/enqueue");
        let v1: serde_json::Value = serde_json::from_str(&body1).unwrap();
        assert_eq!(v1["event_action"], "trigger");
        assert_eq!(v1["dedup_key"], "compaction_backlog:collection:docs");
        assert_eq!(v1["payload"]["severity"], "critical");
        assert_eq!(v1["routing_key"], "rk-123");

        let (_, body2) = rx.recv().unwrap();
        let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
        assert_eq!(v2["event_action"], "resolve");
        assert_eq!(v2["dedup_key"], "compaction_backlog:collection:docs");
    }

    #[test]
    fn pagerduty_severity_mapping() {
        assert_eq!(pagerduty_severity(Severity::High), "critical");
        assert_eq!(pagerduty_severity(Severity::Medium), "warning");
        assert_eq!(pagerduty_severity(Severity::Low), "info");
    }

    fn email_cfg() -> EmailChannel {
        EmailChannel {
            enabled: true,
            smtp_host: "smtp.example.com".to_owned(),
            smtp_port: 587,
            username: "u".to_owned(),
            password: "p".to_owned(),
            from: "alerts@veda.test".to_owned(),
            to: vec!["ops@veda.test".to_owned()],
            min_severity: Severity::Low,
        }
    }

    #[test]
    fn build_email_open_subject_and_body() {
        let cfg = email_cfg();
        let msg = build_email(&cfg, &ev(Severity::High, EventStatus::Open), true, "ops@veda.test")
            .unwrap();
        let raw = String::from_utf8(msg.formatted()).unwrap();
        assert!(raw.contains("Subject: [High] Compaction backlog on docs"));
        assert!(raw.contains("42 sealed segments"));
        assert!(raw.contains("Recommendation: Trigger compaction"));
        assert!(raw.contains("Runbook: POST /v1/collections/docs/compact"));
    }

    #[test]
    fn build_email_resolved_subject() {
        let cfg = email_cfg();
        let msg =
            build_email(&cfg, &ev(Severity::Low, EventStatus::Closed), false, "ops@veda.test")
                .unwrap();
        let raw = String::from_utf8(msg.formatted()).unwrap();
        assert!(raw.contains("Subject: [RESOLVED] Compaction backlog on docs"));
    }

    #[tokio::test]
    async fn failing_channel_does_not_block_others_or_panic() {
        // Webhook points at a dead port (nothing listening); slack at a
        // live capture server. The dead webhook must not stop slack.
        let (slack_url, rx) = spawn_capture(1);
        let channels = ChannelsConfig {
            webhook: Some(WebhookChannel {
                enabled: true,
                // 127.0.0.1:1 — reserved, refuses connections.
                url: "http://127.0.0.1:1/dead".to_owned(),
                min_severity: Severity::Low,
            }),
            slack: Some(SlackChannel {
                enabled: true,
                webhook_url: slack_url.clone(),
                min_severity: Severity::Low,
            }),
            ..Default::default()
        };
        let notifier = Notifier::new(channels);
        let sent = notifier
            .notify_transitions(&[ev(Severity::High, EventStatus::Open)], &[])
            .await;
        // Only slack succeeded; webhook failed but did not panic/propagate.
        assert_eq!(sent, 1);
        let (_, body) = rx.recv().unwrap();
        assert!(body.contains("Compaction backlog on docs"));
    }
}
