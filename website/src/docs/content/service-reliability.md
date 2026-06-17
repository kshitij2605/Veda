# Service Reliability

**Service Reliability** gives operators **service-level objectives with error budgets**, **synthetic self-checks** that exercise Veda's own critical paths, and **AI triage** that explains an incident in plain language — all on the service Veda actually owns.

It is deliberately *not* a full Digital Experience Monitoring product: there is no end-user application to instrument, so RUM, Core Web Vitals, mobile telemetry, synthetic *browser* journeys, and global probing locations are out of a vector database's lane. What's in lane is *"is Veda meeting its SLOs, are its own endpoints up, and what do I do when one isn't?"* This extends the existing [Ops Advisor](/docs/ops-advisor) engine — it is not a new product.

## Architecture

The SLO engine and synthetic probes run inside the existing background advisor tick; their outputs become SLIs and advisor rules that flow through the same [`EventStore`](/docs/ops-advisor#events-lifecycle) and [alert channels](/docs/ops-advisor#alerting-channels) as every other rule.

```
  advisor tick (VEDA_ADVISOR_INTERVAL_SECS)
    ├─ Phase 0  synthetic probes run in-process (lock-free): health / search / embed
    │              → ProbeResult ring → SyntheticSummary
    ├─ OpsSnapshot::collect()  (recent_latencies_ms, request/error counters, …)
    ├─ SLO engine: per-SLO good/total → persisted 60s-bucketed rolling counter (cap 30d)
    │              → SloStatus { sli_pct, error_budget_remaining_pct, burn_rate_fast/slow, … }
    ├─ rules: slo_burn_rate_fast/slow, slo_budget_exhausted, synthetic_failing
    │              → EventStore → channels (webhook/Slack/PagerDuty/email)
    └─ REST: /v1/admin/slos*, /v1/admin/synthetics*, /v1/admin/insights/{id}/triage
```

SLO and probe definitions persist to the data dir and reload on restart, like the rest of the advisor config. The AI triage reuses the [agent framework](/docs/agents-mcp) — it adds four read-only ops tools and one built-in agent, nothing new in the execution loop.

## SLOs & error budgets

An SLO targets a percentage over a window, with a fast and a slow burn window:

```json
{
  "id": "search-latency",
  "name": "Search under 200ms",
  "kind": { "kind": "latency", "threshold_ms": 200 },
  "target_pct": 99.0,
  "window_secs": 2592000,
  "fast_burn": { "window_secs": 3600,  "threshold": 14.4 },
  "slow_burn": { "window_secs": 86400, "threshold": 6.0 }
}
```

Each tick records **good** and **total** for every SLO into a persisted, 60-second-bucketed rolling counter (capped at 30 days). Over a window the engine computes:

```
sli_pct                    = good / total * 100
error_budget_remaining_pct = (1 - (bad/total) / (1 - target/100)) * 100   # floor-clamped at -1000
burn_rate                  = (bad/total) / (1 - target/100)               # per fast / slow window
```

`burn_rate` is the Google-SRE multi-window signal: **1.0 means you're spending the budget exactly on pace**, and `> 1.0` means too fast (e.g. `14.4` over the fast window is the classic "budget gone in ~2 days" alert). The status object exposes both windows:

```json
{
  "sli_pct": 99.2, "error_budget_remaining_pct": 41.0,
  "burn_rate_fast": 0.8, "burn_rate_slow": 1.1,
  "breaching_fast": false, "breaching_slow": false,
  "budget_exhausted": false, "compliant": true
}
```

`error_budget_remaining_pct` can go negative (over budget). `breaching_fast/slow` compare each window's burn rate to its configured threshold; `budget_exhausted` is the budget `<= 0`; `compliant` is `sli_pct >= target_pct`.

## SLI sourcing

Each tick the engine sources good/total per SLO from data the advisor already collects:

| Kind | Source | good / total |
|---|---|---|
| `latency` | snapshot `recent_latencies_ms` (query-log `took_ms`) | # under `threshold_ms` / # sampled |
| `availability` | `veda_requests_total` / `veda_errors_total` deltas | (requests − errors) / requests |
| `error_rate` | same counters | (requests − errors) / requests |
| `synthetic` | probe results since the last tick | ok runs / total runs |

Availability/error-rate use **deltas** between ticks, so the SLI tracks the per-window rate rather than the lifetime counter.

## SLO rules

Three rules join the [advisor catalog](/docs/ops-advisor#the-rule-catalog), each scoped per-SLO. They open events through the existing event store and alert channels, and are tunable via `/v1/admin/insights/rules`:

| Rule | Severity | Fires when |
|---|---|---|
| `slo_burn_rate_fast` | High | the SLO's fast-burn window burn rate exceeds its threshold — budget is being spent dangerously quickly right now |
| `slo_burn_rate_slow` | Medium | the slow-burn window burn rate exceeds its threshold — a sustained, lower-grade burn |
| `slo_budget_exhausted` | High | error budget remaining `<= 0` over the SLO window |

The fast/slow pairing is the standard Google-SRE multi-window/multi-burn-rate pattern: the fast window catches acute outages, the slow window catches chronic drift, and the event store's dedup/auto-close keeps each to one open event at a time.

## Synthetic self-checks

A synthetic probe exercises one of Veda's real critical paths, in-process at the start of each advisor tick:

| Probe kind | What it runs |
|---|---|
| `health` | the same dependency checks as `/readiness` |
| `search` | the internal query executor for a top-1 result on a collection |
| `embed` | the async embedding client on a text string |

```json
{ "id": "search-probe", "name": "Search path", "kind": { "kind": "search", "collection": "docs" }, "interval_secs": 60, "timeout_ms": 2000 }
```

Each run records `{ ts_ms, ok, latency_ms, error? }` into a bounded per-probe ring; the rolled-up summary (`consecutive_failures`, `last_ok`, `uptime_pct`, …) feeds both the `synthetic` SLO SLI and a rule:

- **`synthetic_failing`** (High, per-probe) — fires when `consecutive_failures >= K` (default **3**), so a single transient blip doesn't page.

This is the in-lane subset of synthetic monitoring: Veda probes **its own** endpoints from inside the process — no external load generator, headless browser, or multi-location probing.

## AI ops triage

Four read-only ops tools are added to the agent [tool catalog](/docs/agents-mcp#the-tool-catalog) — `get_insights`, `get_logs`, `get_resources`, `get_slo_status` — and a built-in **`veda-ops`** agent ships alongside `veda-agent` with an ops-focused prompt. It whitelists the 4 ops tools plus the read-only builtins, but **not** `upsert_record` — triage never mutates data.

`POST /v1/admin/insights/{event_id}/triage` (admin-only) looks up the event (`404` if missing), returns `503` when no LLM is configured, and otherwise runs `veda-ops` to produce a grounded explanation:

```json
{ "event_id": "…", "explanation": "…", "steps": [ /* full reasoning + tool trace */ ] }
```

The `steps[]` trace shows exactly which signals the agent consulted — so the operator sees how it reached its root cause and next steps.

## REST API

All endpoints are admin-only.

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/admin/slos` | List SLO defs / create-or-overwrite (POST needs an explicit `id`) |
| `GET/PUT/DELETE` | `/v1/admin/slos/{id}` | Get / update / delete one SLO (`404` if missing) |
| `GET` | `/v1/admin/slos/{id}/status` | Current SLI / error-budget / burn-rate status |
| `GET/POST` | `/v1/admin/synthetics` | List probe defs / create-or-overwrite |
| `GET/PUT/DELETE` | `/v1/admin/synthetics/{id}` | Get / update / delete one probe |
| `GET` | `/v1/admin/synthetics/{id}/results` | Recent per-probe results + summary |
| `POST` | `/v1/admin/insights/{event_id}/triage` | AI triage of an event (`404` unknown, `503` no LLM) |

## In Niriksha

The dashboard's **SLOs** page (in the Observe nav) shows one card per SLO with the SLI %, an error-budget gauge, and fast/slow burn cells (breach-colored), plus a create/edit form. The **System Health** page gains a **Synthetics** panel (per-probe uptime %, last result, consecutive failures), and the Insights event drawer gains a **"Triage with AI"** button that renders the explanation and step trace inline (with a graceful notice when no LLM is configured).

## Configuration

| Variable | Purpose |
|---|---|
| `VEDA_ADVISOR_INTERVAL_SECS` | Advisor tick (default 30s) — how often SLIs are recorded, probes run, and the SLO/synthetic rules evaluate |
| `VEDA_LLM_ENDPOINT` | The OpenAI-compatible LLM used by `veda-ops`; unset → triage returns `503` (everything else still works) |
| `PUT /v1/admin/insights/rules/{id}` | Tune the SLO/synthetic rules (mute `slo_burn_rate_slow`, change `synthetic_failing`'s `K`, …) |

SLO and probe definitions are managed through their REST surfaces and persist in the data dir — no new external dependency.

## Live validation

Verified against a running server on a Linux host:

- **SLO breach end-to-end** — a latency SLO with `threshold_ms = 0` (every query counts as bad), 30 searches, and a forced tick returned `sli_pct 0.0`, `error_budget_remaining_pct -1000`, `burn_rate ≈ 100×`, with `breaching_fast` / `breaching_slow` / `budget_exhausted` all true.
- **Events + alerting** — the tick opened 3 events (`slo_burn_rate_fast` High, `slo_budget_exhausted` High, `slo_burn_rate_slow` Medium) and **each fired the configured webhook exactly once** (transition-only).
- **Synthetic probe** — a Health probe ran inside the tick (ok, ~255 ms, uptime 100%).
- **AI triage** — `POST …/triage` returned a grounded explanation; `veda-ops` autonomously called all four ops tools (`get_slo_status`, `get_insights`, `get_logs`, `get_resources`) and produced a root cause + next steps.

## Out of scope (deliberately)

- **RUM, Core Web Vitals, mobile telemetry, synthetic browser journeys, and global probing locations** — these monitor an end-user application, which a vector database does not have. Veda monitors its own service.
- **SLI granularity** is bounded by the tick cadence (`VEDA_ADVISOR_INTERVAL_SECS`) — the finest resolution is one tick; sub-tick spikes are averaged into the bucket.
- The latency SLI reads the snapshot's `recent_latencies_ms` ring (capped at 2000 recent samples), so at very high QPS the per-tick latency window is bounded by that cap.
