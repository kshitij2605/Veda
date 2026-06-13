# Veda Ops Advisor ("Insights") — AutoOps for Veda

**Date:** 2026-06-14
**Status:** Shipped on `main` (v0.2.5). Validated end-to-end against a live server.

This document describes the **Ops Advisor** — Veda's automated operations layer, built as
a Veda-native answer to Elastic's AutoOps: it watches the running system, **detects
issues from a rule catalog**, explains the **root cause**, gives a concrete
**recommendation**, **alerts** the operator, and can optionally **auto-remediate**.

---

## 1. Why this exists

Elastic's AutoOps samples cluster metrics and, from a catalog of 100+ rule-based events,
tells operators *what's wrong, why, and how to fix it* — with severity levels, root-cause
drill-downs, and Slack/PagerDuty/webhook alerts. A gap analysis found Veda had the raw
observability **foundation** (Prometheus metrics, health/readiness, OTel + Grafana,
Niriksha Observability/SystemHealth, `/coverage`, auto-compaction/GC/TTL) but **none of
the intelligence layer** — no rule catalog, no root-cause, no recommendations, no
alerting. A human had to interpret raw dashboards.

AutoOps's rule catalog is Elasticsearch-internals-specific (shard allocation, JVM heap,
disk watermarks, thread-pool rejections). Veda's internals differ (segments, MVCC,
compaction, shard logs, embeddings), so the Ops Advisor ships a **Veda-native** rule set.

---

## 2. Architecture

A background tick samples an operational snapshot, runs the rules + anomaly detectors,
reconciles an event store (open/auto-close), fires alerts on transitions, and optionally
auto-remediates. All in `crates/veda-gateway/src/advisor/`.

```
  spawn_advisor (tokio interval, VEDA_ADVISOR_INTERVAL_SECS, default 30s)
        │  each tick → run_tick(state)
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. snapshot.rs   collect(state) → OpsSnapshot  (+ history ring) │
  │ 2. rules/ + anomaly.rs   run_rules + detect_anomalies → hits    │
  │ 3. event.rs   EventStore.reconcile(hits) → {opened, closed}     │  (under lock; sync)
  │ 4. notify.rs   notify_transitions(opened, closed)               │  (lock-free; async)
  │ 5. remediation.rs   run_remediation(opened)  [opt-in]           │  (lock-free; async)
  │ 6. EventStore.annotate + save                                   │  (brief re-lock)
  └──────────────────────────────────────────────────────────────┘
        │
        ▼  served by rest/insights.rs  (admin-only)  +  Niriksha "Insights" page
```

Key modules:
- `snapshot.rs` — `OpsSnapshot` per tick from AppState (per-collection segment/record
  counts, compaction-in-progress, active collections), `/nodes`+raft, replica `safe_ts`,
  health probes (MinIO/NATS/embedding), query-log latency (p50/p95 + slow count), process
  RSS (`/proc/self/status`), and request/error counters parsed from the Prometheus
  exposition. A bounded `SnapshotHistory` ring buffer (cap 240) backs rate + anomaly math.
- `event.rs` — `OpsEvent`, `Severity{High,Medium,Low}`, `Scope{Global,Collection,Shard,
  Node}`, `RemediationAction{None,Compact,Reindex}`, and `EventStore` (open on first fire,
  dedup per `rule_id:scope`, refresh in place, **auto-close** when the condition clears,
  bounded history, persisted to `{data_dir}/_ops_events.json`). `reconcile(hits) ->
  {opened, closed}` is the transition signal alerting/remediation consume.
- `rules/` — a `Rule` trait + `catalog()` + `run_rules()` (skips disabled/muted, applies
  severity overrides).
- `anomaly.rs` — EWMA / rolling mean+stddev baselines; fire on N-σ deviation once warmed up.
- `config.rs` — `AdvisorConfig { rules, channels }`, per-rule `RuleConfig { enabled,
  muted, severity_override, thresholds, auto_remediate }`; persisted to
  `{data_dir}/_advisor_config.json`.
- `notify.rs` — async dispatch to the 4 channels on transitions.
- `remediation.rs` — opt-in safe actions, rate-limited.
- `engine.rs` — the tick + `spawn_advisor`.

**Availability note:** the tick does the reconcile phase synchronously under lock, then
releases the `events`/`history` locks **before** the notification/remediation network
awaits (re-acquiring `events` only briefly to annotate). The notifier's HTTP client has a
10s timeout. So a hung webhook can't freeze the Insights API or the tick loop.

---

## 3. The rule catalog (15 rules)

Threshold+correlation rules (thresholds tunable per rule via the API; defaults shown):

| Rule | Severity | Fires when | Recommendation / remediation |
|---|---|---|---|
| `compaction_backlog` | Medium | a collection's sealed segments > 20 (and not already compacting) | run compaction → `RemediationAction::Compact` |
| `seal_stalled` | Medium | growing segment > 100k records with 0 sealed | investigate seal threshold |
| `high_search_latency` | Medium | query p95 > 500 ms | check load / slow dependency / heavy queries |
| `slow_query_surge` | Low | slow-query count (> 1 s) over threshold in the window | inspect slow queries |
| `high_error_rate` | High | error/request rate > 5% (from counter deltas) | investigate 5xx source |
| `embedding_unhealthy` | Medium | embedding endpoint health == unavailable | check the embedder |
| `dependency_down` | High | MinIO or NATS health == unavailable | restore the dependency |
| `memory_pressure` | Medium | process RSS over threshold | `use_log_primary` / quantization / scale out |
| `replica_lag` | High | replica `safe_ts` physical lag > 60 s | check WAN shipping / replica health |
| `node_unhealthy` | High | a node reports `unhealthy`/`unavailable` | inspect the node |
| `no_compaction_progress` | Low | large corpus with zero sealed segments cluster-wide | check the seal/compaction pipeline |
| `anomaly_search_latency` | Med/High | p95 latency > baseline mean + Nσ | investigate the deviation |
| `anomaly_error_rate` | Med/High | error-rate > baseline + Nσ | investigate the deviation |
| `anomaly_insert_rate` | Med/High | insert-rate spike > baseline + Nσ | investigate the load spike |
| `anomaly_rss` | Med/High | RSS > baseline + Nσ | investigate memory growth |

Anomaly detectors warm up over `min_samples` (default 20) and fire above the configured
`sigma` (default 3σ → Medium, ≥5σ → High). Rules needing data the snapshot doesn't yet
carry (coverage orphan/monoculture, TTL backlog, applier lag, shard imbalance) are
documented follow-ups.

---

## 4. Events, alerting, remediation

**Event lifecycle.** A rule hit opens an `OpsEvent` (id = `rule_id:scope-key`, dedup), it
refreshes in place each tick while the condition holds, and **auto-closes** (moved to
bounded history) when the condition clears. Events carry severity, root-cause `detail`, a
`recommendation`, an optional `runbook` hint, the `RemediationAction`, timestamps, an
`acknowledged` flag, and a `remediation_note`.

**Alerting (on open/close transitions only).** Four channels, async, severity-routed
(per-channel `min_severity`), per-channel/per-event failure-isolated:
- **Generic webhook** — `POST {url}` with `{ status: "open"|"closed", event: {...} }`.
- **Slack** — incoming-webhook message with a severity emoji + title + recommendation
  (`RESOLVED:` prefix on close).
- **PagerDuty** — Events API v2: `trigger` on open / `resolve` on close, `dedup_key =
  event.id`, severity mapped High→critical / Medium→warning / Low→info.
- **Email** — async SMTP (`lettre`), subject `[severity] title`.
All HTTP is async with a 10s timeout (never `reqwest::blocking`). Channel secrets are
**write-only** — `GET /channels` returns only `{ configured, enabled, min_severity }`
(+ non-secret email fields).

**Opt-in auto-remediation.** Per rule, `auto_remediate` is **off by default**. When
enabled, an opened event can trigger its `RemediationAction` — `Compact` runs the real
compaction path (honoring the compaction lock); `Reindex` is best-effort — rate-limited to
≤1 per 5 minutes per scope, with the outcome annotated onto the event. Muted/disabled
rules never act.

---

## 5. REST API (admin-only)

| Method + path | Purpose |
|---|---|
| `GET /v1/admin/insights` | open events (`?severity=`, `?scope=` filters), severity-sorted |
| `GET /v1/admin/insights/history` | closed events (bounded) |
| `GET /v1/admin/insights/rules` | catalog: `{ id, default_severity?, anomaly, effective_config }` per rule |
| `PUT /v1/admin/insights/rules/{id}` | tune `enabled`/`muted`/`severity_override`/`thresholds`/`auto_remediate` (persisted) |
| `GET /v1/admin/insights/channels` | channel config, **secrets redacted** |
| `PUT /v1/admin/insights/channels` | set channel config (secrets write-only) |
| `POST /v1/admin/insights/run` | force one evaluation tick → `{ opened, closed }` |
| `POST /v1/admin/insights/{event_id}/ack` | acknowledge an open event |

### Example: configure a webhook, tune a rule, run

```bash
B=http://localhost:6340
# route all severities to a webhook
curl -X PUT $B/v1/admin/insights/channels -H 'Content-Type: application/json' \
  -d '{"webhook":{"enabled":true,"url":"https://hooks.example.com/veda","min_severity":"low"}}'
# raise the compaction-backlog threshold
curl -X PUT $B/v1/admin/insights/rules/compaction_backlog -H 'Content-Type: application/json' \
  -d '{"thresholds":{"sealed_segments":40},"auto_remediate":true}'
# evaluate now
curl -X POST $B/v1/admin/insights/run        # → {"opened":N,"closed":M}
curl $B/v1/admin/insights                     # → {"events":[ ... open events ... ]}
```

---

## 6. Niriksha "Insights" page

A dashboard page (`niriksha/src/pages/Insights.tsx`) with: an **Open events** list
(severity-colored badges, scope, relative time, inline ack) and a **History** view; an
**event detail drawer** (root cause + recommendation + runbook + remediation affordance);
a **Rules** tab (enable/mute/auto-remediate toggles, severity override, threshold editing);
a **Channels** tab (configure webhook/Slack/PagerDuty/email; secrets write-only); and a
**"Run check now"** button. A sidebar **open-event-count badge** (severity-colored) polls
in the background, plus a command-palette entry.

---

## 7. Configuration

- `VEDA_ADVISOR_INTERVAL_SECS` — advisor tick interval (default 30s).
- Rule thresholds / severities / mute / auto-remediate — via `PUT /insights/rules/{id}`,
  persisted to `{data_dir}/_advisor_config.json`.
- Alert channels — via `PUT /insights/channels`, persisted (secrets write-only).
- Events persist to `{data_dir}/_ops_events.json` (open + bounded history).

---

## 8. Validation (live)

Verified against a running server: configured a webhook → tripped `memory_pressure`
(threshold lowered) → `POST /run` returned `{opened:1}` → `GET /insights` showed the event
with severity + recommendation → the webhook received the event JSON → raised the
threshold → `POST /run` returned `{closed:1}` → the event auto-closed into history and the
webhook received the close notification (2 total). `GET /channels` returned the
**redacted** config (no secret leak); `/rules` listed all 15 rules including the 4 anomaly
detectors. Detection → root-cause → recommendation → alerting → auto-close all confirmed.

---

## 9. Known limitations & follow-ups

- **Rules deferred for missing snapshot data:** coverage orphan/monoculture, TTL backlog,
  log-applier lag, shard imbalance — add the cheap snapshot fields then the rules.
- **Capacity forecasting** (trend extrapolation) and **multi-deployment overview** are not
  built (current advice is point-in-time, like AutoOps's right-sizing).
- **More channels** (MS Teams, Opsgenie) are easy follow-ons to the channel framework.
- **Snapshot history** is in-memory (bounded ring); persisting the time-series to
  Mimir/Prometheus for long-range trends is a follow-up.
- **`Reindex` auto-remediation** is best-effort (logs); the full reindex pipeline wiring is
  deferred. `Compact` is fully wired.
- Cosmetic: the `memory_pressure` detail string rounds RSS to GiB.
