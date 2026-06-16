# Veda Service Reliability — SLOs, Synthetic Self-Checks & AI Ops Triage

**Date:** 2026-06-16
**Status:** Shipped on `feat/slo-dem`. Validated end-to-end against a live server.

Service Reliability is Veda's in-lane answer to a Digital Experience Monitoring (DEM)
platform: give operators **service-level objectives with error budgets**, **synthetic
self-checks** that exercise Veda's own critical paths, and **AI triage** that explains an
incident in plain language — all on the service Veda actually owns. It deliberately does NOT
try to be Elastic's full DEM product: there is no end-user application to instrument, so
**RUM, Core Web Vitals, mobile telemetry, synthetic browser journeys, and global probing
locations are out of a vector DB's lane**. What's in lane is *"is Veda meeting its SLOs, are
its own endpoints up, and what do I do when one isn't?"* This extends the existing Ops Advisor
engine (snapshot → rules → events → channels → Niriksha); it is not a new product.

---

## 1. Architecture

This reuses the Ops Advisor pipeline end-to-end. The SLO engine and synthetic probes run
inside the existing background advisor tick; their outputs become SLIs and advisor rules that
flow through the same `EventStore` and alert channels as every other rule.

```
  advisor tick (VEDA_ADVISOR_INTERVAL_SECS)
    │
    ├─ Phase 0  synthetic probes run IN-PROCESS (lock-free): health / search / embed
    │              → ProbeResult ring (_synthetic_results.json) → SyntheticSummary
    │
    ├─ OpsSnapshot::collect()  (recent_latencies_ms, request/error counters, …)
    │
    ├─ SLO engine: per-SLO good/total recorded into a persisted 60s-bucketed
    │              rolling counter (_slo_state.json, cap 44640 = 30d)
    │              → SloStatus { sli_pct, error_budget_remaining_pct, burn_rate_fast/slow, … }
    │
    ├─ rules: slo_burn_rate_fast/slow, slo_budget_exhausted, synthetic_failing
    │              → EventStore (open/dedup/auto-close) → channels (webhook/Slack/PD/email)
    │
    └─ REST: /v1/admin/slos*, /v1/admin/synthetics*, /v1/admin/insights/{id}/triage
                 → Niriksha SLOs page + Synthetics panel + "Triage with AI"
```

- The SLO engine lives in `crates/veda-gateway/src/advisor/slo.rs`; synthetic probes in
  `crates/veda-gateway/src/advisor/synthetic.rs`.
- Both persist to the data dir (`_slo_state.json`, `_synthetics.json`, `_synthetic_results.json`)
  and reload on restart, like the rest of the advisor config.
- The AI triage reuses the agent framework (`AgentExecutor`, `ToolCatalog`) — it adds 4
  read-only ops tools and one built-in agent, nothing new in the execution loop.

---

## 2. SLO engine & error-budget math

An SLO is defined by `SloDef`:

```
SloDef { id, name, kind, target_pct, window_secs, fast_burn{window_secs,threshold}, slow_burn{...} }
```

`SloKind` is internally tagged on `kind` (snake_case):

- `{ kind: "latency", threshold_ms }` — fraction of queries faster than `threshold_ms`.
- `{ kind: "availability" }` — fraction of requests that did not error.
- `{ kind: "error_rate" }` — same counters, framed as the error fraction.
- `{ kind: "synthetic", probe_id }` — fraction of synthetic-probe runs that succeeded.

Each advisor tick records **good** and **total** for every SLO into a **persisted, bucketed
rolling counter** (`_slo_state.json`, 60-second buckets, cap **44640 = 30 days**). Over a given
window the engine sums good/total and computes (with `target = target_pct`):

```
allowed_bad                  = 1 - target/100
sli_pct                      = good/total * 100
error_budget_remaining_pct   = (1 - (bad/total) / allowed_bad) * 100      # floor-clamped at -1000
burn_rate                    = (bad/total) / allowed_bad                  # over the fast / slow window
```

`error_budget_remaining_pct` can go **negative** (over budget) and is clamped at a floor of
**-1000** to keep the gauge bounded. `burn_rate` is the Google-SRE multi-window signal: **1.0
means you are spending the budget exactly on pace**, and `> 1.0` means too fast (e.g. `burn_rate
= 14.4` over the fast window is the classic "budget gone in ~2 days" alert). The status object
exposes both windows:

```
SloStatus { sli_pct, error_budget_remaining_pct,
            burn_rate_fast, burn_rate_slow,
            breaching_fast, breaching_slow,
            budget_exhausted, compliant }
```

`breaching_fast/slow` compare each window's burn rate to that window's configured `threshold`;
`budget_exhausted` is `error_budget_remaining_pct <= 0`; `compliant` is `sli_pct >= target_pct`.

---

## 3. SLI sourcing per kind

Each tick, the engine sources good/total per SLO from data the advisor already collects:

| Kind | Source | good / total |
|---|---|---|
| Latency | snapshot `recent_latencies_ms` (query-log `took_ms`) | # under `threshold_ms` / # sampled |
| Availability | `veda_requests_total` / `veda_errors_total` counter deltas | (requests − errors) / requests |
| Error rate | same counters | (requests − errors) / requests |
| Synthetic | probe results since the last tick (`SyntheticSummary`) | ok runs / total runs |

The Availability/ErrorRate counters are **deltas** between the previous and current tick, so the
SLI tracks the per-window rate rather than the lifetime counter. Latency uses the snapshot's
in-memory `recent_latencies_ms` ring (the same query-log `took_ms` series the latency rules use).

---

## 4. SLO rules

Three rules join the advisor catalog, each scoped per-SLO (`Scope::Slo`). They open events
through the existing `EventStore` (open / dedup / auto-close) and the existing alert channels
(webhook / Slack / PagerDuty / email), and are tunable per-rule via `/v1/admin/insights/rules`:

- **`slo_burn_rate_fast`** (High) — the SLO's `fast_burn` window burn rate exceeds its threshold.
  The fast page: budget is being spent dangerously quickly right now.
- **`slo_burn_rate_slow`** (Medium) — the `slow_burn` window burn rate exceeds its threshold.
  The slow ticket: a sustained, lower-grade burn.
- **`slo_budget_exhausted`** (High) — `error_budget_remaining_pct <= 0` over the SLO window.

The two-window pairing is the standard Google-SRE multi-window/multi-burn-rate pattern: the fast
window catches acute outages, the slow window catches chronic drift, and the dedup/auto-close in
the event store keeps each to one open event at a time.

---

## 5. Synthetic self-checks

A synthetic probe is defined by `SyntheticProbe`:

```
SyntheticProbe { id, name, kind, interval_secs, timeout_ms }
```

`ProbeKind` is internally tagged: `{ kind: "health" }`, `{ kind: "search", collection, query? }`,
`{ kind: "embed", text }`. Probes run **in-process at the start of each advisor tick** (Phase 0,
lock-free w.r.t. events/history), exercising Veda's real critical paths:

- **Health** → runs the same dependency checks as `/readiness`.
- **Search** → calls the internal query executor for a top-1 result on `collection` (using
  `query` if given).
- **Embed** → calls the async embedding client on `text`.

Each run yields `ProbeResult { ts_ms, ok, latency_ms, error? }` into a bounded per-probe ring;
defs persist to `_synthetics.json` and results to `_synthetic_results.json`. The rolled-up
`SyntheticSummary { consecutive_failures, last_ok, last_latency_ms, uptime_pct, last_ts_ms }`
feeds two consumers: the **`Synthetic` SLO SLI** (above) and a new rule:

- **`synthetic_failing`** (High, per-probe `Scope::Synthetic`) — fires when
  `consecutive_failures >= K` (default **K = 3**), so a single transient blip doesn't page.

This is the in-lane subset of synthetic monitoring: Veda probes **its own** endpoints from
inside the process. There is no external load generator, no headless browser, and no
multi-location probing — those belong to an end-user-facing app, not a vector DB.

---

## 6. AI ops triage

Four **read-only** ops tools are added to the agent `ToolCatalog`:

- `get_insights` — open/closed advisor events.
- `get_logs` — recent captured logs (level/text filter).
- `get_resources` — the latest node-resource snapshot.
- `get_slo_status` — per-SLO SLI / error-budget / burn-rate status.

A built-in, in-memory **`veda-ops`** agent ships alongside the existing `veda-agent`: it carries
an ops-focused system prompt and **whitelists the 4 ops tools plus the read-only builtins — but
NOT `upsert_record`** (triage never mutates data).

`POST /v1/admin/insights/{event_id}/triage` (admin-only) looks up the event (404 if missing),
returns **503 when no LLM is configured** (`VEDA_LLM_ENDPOINT` unset), and otherwise builds a
prompt from the event and runs `veda-ops` via `AgentExecutor`, returning:

```json
{ "event_id": "...", "explanation": "...", "steps": [ ... ] }
```

`steps[]` is the full reasoning/tool trace (internally-tagged `AgentStep`:
`reasoning` / `tool_call` / `tool_result`), so the operator sees exactly which signals the agent
consulted to reach its root cause and next steps.

---

## 7. REST API

All endpoints are admin-only (`bearerAuth`).

| Endpoint | Method | Description |
|---|---|---|
| `/v1/admin/slos` | GET / POST | List SLO defs / create-or-overwrite (POST requires an explicit `id`) |
| `/v1/admin/slos/{id}` | GET / PUT / DELETE | Get / update / delete one SLO (404 if missing) |
| `/v1/admin/slos/{id}/status` | GET | Current SLI / error-budget / burn-rate status |
| `/v1/admin/synthetics` | GET / POST | List probe defs / create-or-overwrite (POST requires an explicit `id`) |
| `/v1/admin/synthetics/{id}` | GET / PUT / DELETE | Get / update / delete one probe (404 if missing) |
| `/v1/admin/synthetics/{id}/results` | GET | Recent per-probe results + summary |
| `/v1/admin/insights/{event_id}/triage` | POST | AI triage of an event (404 unknown, 503 no LLM) |

---

## 8. Niriksha

- **SLOs page** (`/slos`, in the Observe nav + command palette) — one card per SLO showing the
  SLI %, an error-budget gauge, fast/slow burn cells (breach-colored), and a create/edit form.
- **Synthetics panel** on the **System Health** page — per-probe uptime %, last result, and
  consecutive failures.
- **"Triage with AI"** button on the Insights event drawer — renders the explanation and the
  step trace inline, with a graceful 503 notice when no LLM is configured.

---

## 9. Configuration

- `VEDA_ADVISOR_INTERVAL_SECS` — advisor tick interval (default 30s); drives how often SLIs are
  recorded, probes run, and the SLO/synthetic rules are evaluated.
- `VEDA_LLM_ENDPOINT` — the OpenAI-compatible LLM used by `veda-ops`. Unset → triage returns
  **503** (everything else still works).
- Per-rule tuning via `PUT /v1/admin/insights/rules/{id}` — e.g. mute `slo_burn_rate_slow`,
  change the severity of `synthetic_failing`, or adjust its `K` threshold.
- SLO and probe definitions are managed through their REST surfaces and persist in the data dir;
  no new external dependency.

---

## 10. Validation (live)

Verified against a running server on the Linux host:
- **SLO breach end-to-end:** created a latency SLO with `threshold_ms = 0` (so every query counts
  as bad), ran **30 searches**, and forced a tick (`POST /v1/admin/insights/run`). Status came
  back `sli_pct 0.0`, `total 30`, `error_budget_remaining_pct -1000`, `burn_rate_fast/slow ≈ 100×`,
  with `breaching_fast`, `breaching_slow`, and `budget_exhausted` all `true`.
- **Events + alerting:** the tick opened **3 events** — `slo_burn_rate_fast` (High),
  `slo_budget_exhausted` (High), and `slo_burn_rate_slow` (Medium) — and **each fired the
  configured webhook exactly once** (transition-only, no per-tick spam).
- **Synthetic probe in-process:** a Health probe ran inside the tick (ok, ~255 ms, uptime 100%).
- **AI triage grounded:** `POST /v1/admin/insights/{id}/triage` returned a grounded explanation —
  the `veda-ops` agent **autonomously called all 4 ops tools** (`get_slo_status`, `get_insights`,
  `get_logs`, `get_resources`) and produced a root cause + next steps. (503 is the documented
  no-LLM fallback.)

---

## 11. Limitations & follow-ups

- **SLI granularity is bounded by the tick cadence.** SLIs are recorded once per advisor tick
  (`VEDA_ADVISOR_INTERVAL_SECS`), so the finest resolution is one tick; sub-tick spikes are
  averaged into the bucket.
- **Save-each-tick.** The rolling counter is persisted every tick (`_slo_state.json`); at very
  high tick rates / many SLOs this is extra write amplification, acceptable at the default cadence.
- **Latency SLI window.** The latency SLI reads the snapshot's `recent_latencies_ms`, which is
  capped at **2000** recent samples — at very high QPS the per-tick latency window is bounded by
  that cap rather than every request.
- **Out of scope, deliberately:** RUM, Core Web Vitals, mobile telemetry, synthetic *browser*
  journeys, and global probing locations — these monitor an end-user application, which a vector
  DB does not have. Veda monitors its own service.
