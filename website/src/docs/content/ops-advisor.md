# Ops Advisor

The **Ops Advisor** (surfaced as **Insights**) is Veda's automated operations layer. It watches the running system, detects issues from a rule catalog, explains the **root cause**, gives a concrete **recommendation**, **alerts** the operator, and can optionally **auto-remediate** — so you read decisions, not raw dashboards.

Where [Observability](/docs/observability) gives you the raw signals (metrics, traces, health), the Ops Advisor adds the intelligence layer on top: it turns those signals into actionable events with a *what's wrong, why, and how to fix it* for each.

## How it works

A background tick (every `VEDA_ADVISOR_INTERVAL_SECS`, default 30s) runs an evaluation loop:

1. **Snapshot** — collect an `OpsSnapshot`: per-collection segment/record counts and compaction state, node/raft status, replica `safe_ts` lag, dependency health (MinIO, NATS, embedder), query-log latency (p50/p95 + slow-query count), [node resource signals](/docs/resource-signals) (disk, CPU, load, file descriptors, memory), process RSS, and request/error counters. A bounded history ring buffer backs the rate and anomaly math.
2. **Evaluate** — run the rule catalog and the anomaly detectors against the snapshot to produce hits.
3. **Reconcile** — an event store opens new events, refreshes ongoing ones in place, and **auto-closes** events whose condition has cleared.
4. **Alert** — on open/close transitions only, dispatch to the configured channels.
5. **Remediate** — for opted-in rules, run the safe action (rate-limited).

The reconcile phase runs under a lock, which is released before any network calls, and the notifier's HTTP client has a 10s timeout — so a hung webhook can never freeze the Insights API or the tick loop.

## The rule catalog

Thirty-six rules ship today — 29 threshold/correlation rules and 7 anomaly detectors — including three log-content rules powered by [Log Insights](/docs/log-insights), seven [node resource signals](/docs/resource-signals), four [Service Reliability](/docs/service-reliability) rules (`slo_burn_rate_fast`, `slo_burn_rate_slow`, `slo_budget_exhausted`, `synthetic_failing`), two [index advisor](/docs/apm-observability#ann-index-advisor) rules (`index_recall_degraded`, `index_latency_high_for_recall`), three [capacity-forecast](/docs/rag-aiops#capacity-forecasting) rules (`disk_full_forecast`, `memory_growth_forecast`, `recall_decay_forecast`), a [change-point](/docs/aiops#change-point-detection) rule (`change_point_detected`), and two [LLM-ops](/docs/llm-observability#ttft-cost-budgets-quality-drift-feedback) rules (`llm_cost_budget`, `quality_drift`). Thresholds and severities are tunable per rule via the API; defaults are shown. [Semantic incident correlation](/docs/rag-aiops#semantic-incident-correlation) groups related events into clusters *beside* this catalog (it's a clustering surface, not a rule).

| Rule | Severity | Fires when | Recommendation |
|---|---|---|---|
| `compaction_backlog` | Medium | a collection's sealed segments > 20 (and not already compacting) | run compaction (auto-remediable) |
| `seal_stalled` | Medium | growing segment > 100k records with 0 sealed | investigate the seal threshold |
| `high_search_latency` | Medium | query p95 > 500 ms | check load / slow dependency / heavy queries |
| `slow_query_surge` | Low | slow-query count (> 1 s) over threshold in the window | inspect slow queries |
| `high_error_rate` | High | error/request rate > 5% | investigate the 5xx source |
| `embedding_unhealthy` | Medium | embedding endpoint unavailable | check the embedder |
| `dependency_down` | High | MinIO or NATS unavailable | restore the dependency |
| `memory_pressure` | Medium | process RSS over threshold | `use_log_primary` / quantization / scale out |
| `replica_lag` | High | replica `safe_ts` physical lag > 60 s | check WAN shipping / replica health |
| `node_unhealthy` | High | a node reports `unhealthy` / `unavailable` | inspect the node |
| `no_compaction_progress` | Low | large corpus with zero sealed segments cluster-wide | check the seal/compaction pipeline |
| `anomaly_search_latency` | Med/High | p95 latency > baseline mean + Nσ | investigate the deviation |
| `anomaly_error_rate` | Med/High | error-rate > baseline + Nσ | investigate the deviation |
| `anomaly_insert_rate` | Med/High | insert-rate spike > baseline + Nσ | investigate the load spike |
| `anomaly_rss` | Med/High | RSS > baseline + Nσ | investigate memory growth |
| `log_error_rate` | High | ERROR-log count over the window exceeds a threshold | investigate the error source |
| `anomaly_error_logs` | Med/High | error-log rate > baseline + Nσ | investigate the deviation |
| `new_error_category` | Medium | a new WARN/ERROR [semantic log category](/docs/log-insights#semantic-categorization) appears | inspect the new failure mode |
| `disk_space_low` | High | data-dir disk used > 90% | free space / run compaction / scale out |
| `high_cpu` | Medium | derived CPU% > 85% | check load / heavy queries |
| `high_load` | Medium | `load1 / num_cpus` > 2.0 | check CPU saturation |
| `fd_exhaustion` | High | `open_fds / fd_limit` > 0.8 | raise the fd limit / investigate leaks |
| `system_memory_low` | Medium | available system memory < 10% | scale out / reduce pressure |
| `anomaly_cpu` | Med/High | CPU% spikes above baseline | investigate the deviation |
| `anomaly_disk_used` | Med/High | disk-used trend rises above baseline | check the rapid fill |
| `slo_burn_rate_fast` | High | an SLO's fast-burn window burn rate exceeds its threshold | the budget is burning fast — page |
| `slo_burn_rate_slow` | Medium | an SLO's slow-burn window burn rate exceeds its threshold | sustained drift — open a ticket |
| `slo_budget_exhausted` | High | an SLO's error budget remaining ≤ 0 | freeze risky changes / investigate |
| `synthetic_failing` | High | a [synthetic probe](/docs/service-reliability#synthetic-self-checks) fails K times in a row (default 3) | the probed path is down |
| `index_recall_degraded` | High | a collection's proxy `recall` < `recall_min` (default 0.90) | raise `efSearch`/`efConstruction`, or reindex |
| `index_latency_high_for_recall` | Medium | `p95_ms` > 50 while `recall >= 0.95` | lower `efSearch`, or switch index type |
| `disk_full_forecast` | High | a least-squares trend projects the data-dir disk filling | free space / scale out before the ETA |
| `memory_growth_forecast` | Medium | the trend projects memory exhaustion | scale out / reduce pressure before the ETA |
| `recall_decay_forecast` | Medium | the trend projects recall dropping below target | retune / reindex before the ETA |
| `change_point_detected` | Medium | a metric series shifts to a new level or breaks onto a new trend and stays there | investigate the regime change |
| `llm_cost_budget` | Medium | LLM spend over a window crosses a USD budget (`veda_llm_cost_usd_total`) | check for a runaway agent / expensive model |
| `quality_drift` | Medium | a sustained drop in the faithfulness series (two-window statistic) | investigate the grounding regression |

The three [capacity-forecast](/docs/rag-aiops#capacity-forecasting) rules extrapolate the resource-snapshot history to a threshold-crossing ETA (warm-up + min-R² guards mean they never fire on flat/noisy data). `change_point_detected` is distinct from the anomaly detectors: anomalies fire on a single outlier point, while [change-point detection](/docs/aiops#change-point-detection) catches a *sustained* level shift or trend break (two-window least-squares, warm-up ≥20 points, min-magnitude 1.5σ). The two [LLM-ops](/docs/llm-observability) rules watch LLM spend (`llm_cost_budget`) and a sustained faithfulness drop (`quality_drift`, which reuses the same change-point statistic over the faithfulness series). The two [index advisor](/docs/apm-observability#ann-index-advisor) rules are per-collection, driven by an off-tick proxy recall probe. The four [Service Reliability](/docs/service-reliability) rules are scoped per-SLO/per-probe. The seven [node resource signals](/docs/resource-signals) (Linux `/proc` + `statvfs`) each skip cleanly when their signal is unavailable. Anomaly detectors use EWMA / rolling mean+stddev baselines. They warm up over `min_samples` (default 20) and fire above the configured `sigma` (default 3σ → Medium, ≥5σ → High).

## Events & lifecycle

A rule hit opens an `OpsEvent` (deduplicated by `rule_id:scope`). It refreshes in place each tick while the condition holds and **auto-closes** into a bounded history when the condition clears. Each event carries:

- **severity** (`High` / `Medium` / `Low`) and **scope** (`Global` / `Collection` / `Shard` / `Node`)
- a root-cause **`detail`**, a **`recommendation`**, and an optional **`runbook`** hint
- a **`RemediationAction`** (`None` / `Compact` / `Reindex`), timestamps, an `acknowledged` flag, and a `remediation_note`

Events persist to `{data_dir}/_ops_events.json` (open events plus bounded history) and survive restarts.

## Alerting channels

Alerts fire on open/close transitions only, asynchronously, severity-routed (each channel has a `min_severity`), and failure-isolated per channel and per event:

| Channel | Behavior |
|---|---|
| **Webhook** | `POST {url}` with `{ "status": "open" \| "closed", "event": {…} }` |
| **Slack** | Incoming-webhook message: severity emoji + title + recommendation (`RESOLVED:` on close) |
| **PagerDuty** | Events API v2 — `trigger` on open / `resolve` on close, `dedup_key = event.id`, severity mapped High→critical, Medium→warning, Low→info |
| **Email** | Async SMTP (`lettre`), subject `[severity] title` |

All HTTP is async with a 10s timeout. Channel secrets are **write-only** — `GET /channels` returns only `{ configured, enabled, min_severity }` (plus non-secret email fields), never the secret itself.

## Auto-remediation

Opt-in per rule (`auto_remediate`, **off by default**). When enabled, an opened event can trigger its `RemediationAction`:

- **`Compact`** runs the real compaction path (honoring the compaction lock) — fully wired.
- **`Reindex`** is best-effort today (logged; the full pipeline wiring is a follow-up).

Remediation is rate-limited to at most one action per five minutes per scope, and the outcome is annotated onto the event. Muted or disabled rules never act.

## REST API

All endpoints are admin-only.

| Method + path | Purpose |
|---|---|
| `GET /v1/admin/insights` | Open events (`?severity=`, `?scope=` filters), severity-sorted |
| `GET /v1/admin/insights/history` | Closed events (bounded) |
| `GET /v1/admin/insights/rules` | Catalog — `{ id, default_severity?, anomaly, effective_config }` per rule |
| `PUT /v1/admin/insights/rules/{id}` | Tune `enabled` / `muted` / `severity_override` / `thresholds` / `auto_remediate` |
| `GET /v1/admin/insights/channels` | Channel config (secrets redacted) |
| `PUT /v1/admin/insights/channels` | Set channel config (secrets write-only) |
| `POST /v1/admin/insights/run` | Force one evaluation tick → `{ opened, closed }` |
| `POST /v1/admin/insights/{event_id}/ack` | Acknowledge an open event |
| `POST /v1/admin/insights/{event_id}/triage` | [AI triage](/docs/service-reliability#ai-ops-triage) of an event — grounded explanation + tool trace (`404` unknown, `503` no LLM) |
| `GET /v1/admin/insights/{event_id}/rca` | [Cross-signal root-cause bundle](/docs/rag-aiops#cross-signal-rca) — logs + queries + traces + resources + incident |
| `GET /v1/admin/incidents` | [Semantic incident clusters](/docs/rag-aiops#semantic-incident-correlation) (dedup of correlated events) |
| `GET /v1/admin/forecast` | [Capacity forecasts](/docs/rag-aiops#capacity-forecasting) — disk / memory / recall ETAs |

### Example: configure a webhook, tune a rule, evaluate now

```bash
B=http://localhost:6340

# route all severities to a webhook
curl -X PUT $B/v1/admin/insights/channels \
  -H "Content-Type: application/json" \
  -d '{"webhook":{"enabled":true,"url":"https://hooks.example.com/veda","min_severity":"low"}}'

# raise the compaction-backlog threshold and let it self-heal
curl -X PUT $B/v1/admin/insights/rules/compaction_backlog \
  -H "Content-Type: application/json" \
  -d '{"thresholds":{"sealed_segments":40},"auto_remediate":true}'

# evaluate immediately
curl -X POST $B/v1/admin/insights/run   # -> {"opened":N,"closed":M}
curl $B/v1/admin/insights               # -> {"events":[ ... open events ... ]}
```

## In Niriksha

The dashboard's **Insights** page surfaces all of this without curl: an Open-events list with severity badges, scope, relative time, and inline acknowledge; a History view; an event-detail drawer (root cause + recommendation + runbook + remediation affordance); a **Rules** tab (enable/mute/auto-remediate toggles, severity overrides, threshold editing); a **Channels** tab; and a **"Run check now"** button. A severity-colored open-event-count badge in the sidebar polls in the background.

## Configuration

| Variable / location | Purpose |
|---|---|
| `VEDA_ADVISOR_INTERVAL_SECS` | Advisor tick interval (default 30s) |
| `PUT /insights/rules/{id}` → `{data_dir}/_advisor_config.json` | Rule thresholds, severities, mute, auto-remediate (persisted) |
| `PUT /insights/channels` → `{data_dir}/_advisor_config.json` | Alert channels (persisted; secrets write-only) |
| `{data_dir}/_ops_events.json` | Open events + bounded history (persisted) |

## Current limitations

- Some rules are deferred pending cheap snapshot fields: coverage orphan/monoculture, TTL backlog, log-applier lag, shard imbalance.
- Capacity forecasting (trend extrapolation) and a multi-deployment overview are not built — advice is point-in-time.
- Snapshot history is an in-memory bounded ring; persisting the time-series for long-range trends is a follow-up.
- `Reindex` auto-remediation is best-effort; `Compact` is fully wired.
