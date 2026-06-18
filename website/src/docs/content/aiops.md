# AIOps

Veda's AIOps layer is built **inside the database**, on the primitives Veda already owns — the embedder, the retriever, the chunker, the agent loop, the query log, the trace pipeline, and the resource snapshot. Every operations feature reuses one of those primitives and degrades gracefully (a template fallback, an embedding-overlap fallback, a novelty skip, or a clear `503`) when no LLM or embedder is configured.

This page is the single overview. The deep-dive pages — [Ops Advisor](/docs/ops-advisor), [Log Insights](/docs/log-insights), [Node Resource Signals](/docs/resource-signals), [Service Reliability](/docs/service-reliability), [APM & Performance](/docs/apm-observability), and [RAG & AIOps Intelligence](/docs/rag-aiops) — carry the per-area detail.

## The competitive thesis

Elastic's AIOps is **statistical and template-based over text**: log-rate analysis bucketizes by message *templates*, anomaly detection runs population/time-series statistics, and the "explain a spike" feature counts term frequencies. **Veda's AIOps is semantic and embedding-based over every signal it owns** — logs, traces, queries, resources, retrieval quality, and incidents — in *one* engine. Because Veda owns the embedder, the log categorizer, the incident correlator, the cross-signal RCA bundle, *and* the agent, it answers questions a statistical engine can only approximate:

- spike attribution names the **meaning-cluster** driving a spike ("*this kind of failure* rose"), not just which template's count went up — and can say a failure mode **is new**;
- incidents **cluster by similarity** and flag **novelty** ("this pattern is unprecedented") by cosine distance to a history of past incidents, not by string match;
- root-cause drafts are **grounded in Veda's own RAG** — the triage agent reasons over a cross-signal bundle (logs + queries + traces + resources + incident) joined locally, because Veda holds every one of those signals.

A text-search engine that doesn't own the embeddings or the agent can't faithfully score grounding; a metrics store that doesn't own an embedder can't semantically cluster incidents. Veda can, because it *is* the whole pipeline.

## The Ops Advisor engine

The [Ops Advisor](/docs/ops-advisor) is the spine. A background tick samples an operational snapshot, runs the rules + anomaly + change-point detectors, reconciles an event store, fires alerts on transitions, and optionally auto-remediates:

```
  spawn_advisor (tokio interval, VEDA_ADVISOR_INTERVAL_SECS, default 30s)
    Phase 0  synthetic probes (lock-free) + log categorization     (off-lock; may embed)
    1. collect → OpsSnapshot  (+ history ring, cap 240)
    2. rules + anomaly + change-point → hits                       (pure / in-tick)
    3. EventStore.reconcile(hits) → { opened, closed }             (under lock; sync)
    4. notify_transitions(...)                                     (lock-free; async)
    5. run_remediation(opened)  [opt-in]                           (lock-free; async)
    6. SLO engine records good/total; annotate + save              (brief re-lock)
```

**The availability invariant:** the reconcile phase runs synchronously under the `events`/`history` locks, then those locks are **released before** any notification / remediation network await. Anything that embeds or calls the network (log categorization, spike attribution, incident correlation, novelty, drafts) runs **off-lock**, and the notifier's HTTP client has a 10s timeout — so a hung webhook, a slow embedder, or a slow LLM can never freeze the Insights API or the tick loop.

### Cadence & the off-tick runners

The three most expensive operations run on **separate, low-cadence tasks off the tick** so they never stall it:

| Runner | Env interval | Default | Does |
|---|---|---|---|
| Advisor tick | `VEDA_ADVISOR_INTERVAL_SECS` | 30s | snapshot → rules/anomaly/change-point → events → notify → remediation → SLO record |
| Recall probe | `VEDA_RECALL_PROBE_INTERVAL_SECS` | 600s | brute-force `recall@k` proxy (min 50 records) |
| Incident correlator | `VEDA_INCIDENT_INTERVAL_SECS` | 300s | semantic incident clustering + novelty + proactive RCA drafts |

## Detection

**34 rules** span operational health, node resources, log content, SLO/reliability, the index advisor, capacity forecasting, and change-point — see the full [rule catalog](/docs/ops-advisor#the-rule-catalog). Detection comes in four flavours:

- **Threshold/correlation rules** — e.g. `disk_space_low`, `high_error_rate`, `replica_lag`. Each resource/SLO/index rule skips cleanly when its signal is absent (no false alerts).
- **Anomaly detection** — 7 EWMA / rolling mean+stddev detectors (`anomaly_search_latency`, `anomaly_error_rate`, `anomaly_rss`, …) firing above a configured sigma (default 3σ → Medium, ≥5σ → High) once warmed up.
- **Change-point detection** *(new)* — see below.
- **Forecasting** — least-squares ETAs with an R² confidence guard (`disk_full_forecast`, `memory_growth_forecast`, `recall_decay_forecast`); see [capacity forecasting](/docs/rag-aiops#capacity-forecasting).

### Change-point detection

Single-point anomaly detectors fire on one outlier and reset — they miss a **sustained regime change** (the disk-used baseline steps up after a config change; RSS climbs onto a new trend). Change-point detection answers a different question: *did the series move to a new level, or break onto a new trend, and stay there?*

It runs a **two-window least-squares statistic** over the snapshot series (`disk_used_pct`, `rss_bytes`, query `p95`, error-log count, per-collection `recall`), detecting both a **LEVEL** shift (the mean steps up/down) and a **TREND** break (the slope changes). Three guards prevent crying wolf: a **well-fit gate** (the two-window model must fit materially better than a single-window fit), a **warm-up** (≥20 points), and a **min-magnitude** guard (1.5σ). It is **pure and in-tick** — no network, no LLM — so it costs nothing on the tick path. The `change_point_detected` (Medium) rule opens an event through the normal pipeline, and `GET /v1/admin/change-points` returns the raw detections (metric, scope, kind, before/after, magnitude in σ).

## Semantic, vector-native AIOps

These are the features a statistical engine cannot replicate. Each calls `embed_text` → the text model (CPU-only out of the box — see [Embedding & CPU/GPU](#embedding-model-cpugpu)) and degrades gracefully without an embedder.

- **Semantic log categorization** — embeds each distinct recent WARN/ERROR message and clusters by cosine into `LogCategory` groups, so differently-worded messages about the same failure land together (not just identical templates). Fallback: normalized-template grouping. See [Log Insights](/docs/log-insights#semantic-categorization).
- **Semantic spike attribution** *(new)* — see below.
- **Semantic incident correlation** — embeds each event's title+detail and clusters related events into one `Incident`, collapsing a storm into a single page. See [incident correlation](/docs/rag-aiops#semantic-incident-correlation).
- **Incident novelty detection** *(new)* — see below.

### Semantic spike root-cause attribution

When the error/warn-log rate spikes, the operator's first question is "what is driving it?". A statistical "log rate analysis" bucketizes by template and reports which templates grew — but a template is a syntactic artifact: two lines that *mean* the same thing but differ in wording land in different buckets, and a count can't tell you a genuinely new *kind* of failure appeared.

Veda answers it **semantically** by reusing the embedding log categorizer. Given a spike `window` and a `baseline` window, it embeds and clusters the logs in each, then attributes the `total_delta` to the meaning-clusters driving it:

```json
{
  "window": "…", "baseline": "…", "total_delta": 128,
  "drivers": [
    { "category": "…", "representative": "connection refused to shard-3",
      "spike_count": 96, "baseline_count": 2, "growth": 48.0,
      "share_of_growth": 0.75, "is_new": false }
  ],
  "method": "embedding"
}
```

Each driver names a semantic **category**, its `spike_count` vs `baseline_count`, its `growth` ratio, its `share_of_growth` (what fraction of the delta it accounts for), and `is_new` (no presence in the baseline — a brand-new failure mode). Embedding runs off-lock; without an embedder it falls back to the normalized-template path and reports `method: "template"`. `GET /v1/admin/log-rate-analysis?window_secs=&baseline_secs=&min_level=`.

### Incident novelty detection

Not every incident is equally interesting — a flapping rule you've seen a hundred times deserves a different response from a failure pattern that has never occurred before. "Novel" is a semantic judgment, and Veda already embeds every incident for correlation, so the answer is one cosine away.

The incident store keeps a persisted **embedding history** of past incidents; a new incident's `novelty_score = 1 − max_cosine_to_history` (1.0 = nothing in history looks like it; 0.0 = a near-duplicate), and `is_novel` is flagged above the threshold. Two guards prevent false novelty: a **warm-up** (≥5 historical incidents — below that nothing is flagged, so a fresh deployment doesn't call everything new) and an **embedder-free skip** (no embedder → `novelty_score` null, not fabricated). A novel incident **escalates its notification**: tagged "⚠ UNPRECEDENTED pattern" with severity raised to **High**, so a genuinely new failure mode jumps the queue. `is_novel` / `novelty_score` ride on `GET /v1/admin/incidents`.

## Root-cause & agentic

- **Cross-signal RCA bundle** — `GET /v1/admin/insights/{event_id}/rca` joins five signals for one event into a single bundle: correlated logs, slow/errored queries in the window, traces matched by `trace_id`, the nearest resource snapshot, and the related incident. The cross-signal correlation an APM tool charges for, joined locally because Veda holds every signal. See [cross-signal RCA](/docs/rag-aiops#cross-signal-rca).
- **The `veda-ops` triage agent** — a built-in agent with read-only ops tools (`get_insights`, `get_logs`, `get_resources`, `get_slo_status`, `correlate_incident`) and **not** `upsert_record` — triage never mutates data. `POST /v1/admin/insights/{event_id}/triage` returns a grounded explanation + the full tool trace. See [AI triage](/docs/service-reliability#ai-ops-triage).
- **Proactive incident RCA drafts** *(new)* — on a newly-correlated incident (off-tick), the `veda-ops` agent auto-runs over the incident's RCA bundle and drafts a root-cause paragraph + runbook step into `Incident.draft`, so the first-pass RCA is already there when an operator opens it. Best-effort and **503-safe** (`null` without an LLM), and **new-incident-only** — one draft per incident, not one per tick.
- **Alert-storm noise reduction** *(new)* — notifications fire per **incident**, not per event: a storm of twelve correlated events that folds into one incident pages you **once** ("N alerts → 1 incident"), deduped via a persisted set of notified incident-ids. `GET /v1/admin/incidents` carries `noise_reduction { event_count, incident_count, suppressed }`, where `suppressed = event_count − incident_count` — the pages you *didn't* get.

## SLOs, alerting & remediation

The [Service Reliability](/docs/service-reliability) layer adds error budgets with Google-SRE multi-window burn-rate alerting over six SLI kinds — `latency`, `availability`, `error_rate`, `synthetic`, and the RAG-native `recall` and `faithfulness`. Alerting fires on event open/close transitions (and per-incident, above) across four severity-routed channels — webhook, Slack, PagerDuty, email — all async with a 10s timeout. Opt-in [auto-remediation](/docs/ops-advisor#auto-remediation) (`Compact` / `Reindex`) is off by default and rate-limited to ≤1 action per 5 minutes per scope.

## REST surface

All `/v1/admin/*` are admin-only. The new v0.3.1 endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/change-points` | Change-point detections — sustained level shifts / trend breaks (warm-up ≥20 pts, min-magnitude 1.5σ) |
| `GET` | `/v1/admin/log-rate-analysis` | Semantic spike attribution — the meaning-clusters driving a log-rate spike (`?window_secs=&baseline_secs=&min_level=`) |
| `GET` | `/v1/admin/incidents` | Semantic incident clusters — now with `draft`, `is_novel` / `novelty_score`, and a `noise_reduction` summary |

The complete AIOps endpoint table (insights, rules, channels, triage, RCA, resources, SLOs, synthetics, correlations, forecast, index-advisor, depmap, pprof, logs) lives in the [REST API reference](/docs/rest-api).

## Embedding model & CPU/GPU

**The semantic AIOps edge runs CPU-only out of the box. No GPU is required.**

- The **text embedder** (default `BAAI/bge-m3`, 1024-d) is served by **HuggingFace TEI on CPU** under Compose `--profile embedding`. Every semantic AIOps feature — log categorization, spike attribution, incident correlation, novelty, faithfulness — calls `embed_text` → the text model, so it runs CPU-only.
- The **multimodal embedder** (Qwen3-VL-Embedding, vLLM) needs a GPU and is used **only** for image / audio / video. No AIOps feature touches it.
- **Latency is a non-issue:** the semantic AIOps work runs off-tick at low cadence (categorization in the advisor's off-lock Phase 0; incident correlation every 300s; recall probe every 600s), so CPU embedding speed doesn't matter there.

The embedder is fully pluggable via `EMBEDDING_ENDPOINT` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` (a curated preset catalog ships bge-m3, qwen3-embedding, and voyage-3 models). The chat **LLM** used for triage and proactive drafts is the separate `VEDA_LLM_ENDPOINT`; those paths are `503`-safe without it, and the rest of the pipeline is unaffected.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_ADVISOR_INTERVAL_SECS` | `30` | Advisor tick cadence (snapshot / rules / anomaly / change-point / SLO record) |
| `VEDA_INCIDENT_INTERVAL_SECS` | `300` | Off-tick incident correlation + novelty + proactive drafts |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Off-tick proxy recall probe (min 50 records); sources the `recall` SLO SLI |
| `VEDA_LOG_CAPTURE_LEVEL` | `WARN` | In-process log-capture threshold |
| `EMBEDDING_*` | `BAAI/bge-m3` / `1024` | Text embedder (CPU TEI) — drives all semantic AIOps |
| `VEDA_LLM_ENDPOINT` | *(unset)* | Chat LLM for triage / drafts (`503`-safe when unset) |

Rule thresholds, alert channels, SLO/probe defs, the event store, and the SLO counter all persist to the data dir and reload on restart.

## Deep-dive guides

| Guide | Covers |
|---|---|
| [Ops Advisor](/docs/ops-advisor) | The engine: snapshot → rules → events → channels → opt-in remediation; the full 34-rule catalog |
| [Log Insights](/docs/log-insights) | In-process log capture + trace correlation, semantic log categorization, logs-as-a-collection |
| [Node Resource Signals](/docs/resource-signals) | Disk / CPU / load / fd / memory from `/proc` + `statvfs`, the resource rules |
| [Service Reliability](/docs/service-reliability) | SLOs + error budgets + burn-rate, synthetic self-checks, the `veda-ops` triage agent |
| [APM & Performance](/docs/apm-observability) | Tracing, profiling, GenAI-semconv + cost, slow-query correlation, the index advisor |
| [RAG & AIOps Intelligence](/docs/rag-aiops) | Citations + faithfulness, quality SLOs, eval harness, agent memory, streaming, incident correlation, forecasting, the RCA bundle |

## Vs Elastic AIOps

| Capability | Elastic AIOps | Veda |
|---|---|---|
| Anomaly detection | statistical | EWMA/σ over 7 series |
| Log categorization | template-based | **semantic** — groups by *meaning*, not template |
| Forecasting | ✓ | least-squares ETA + R² confidence guard |
| Correlation / RCA | ✓ | **cross-signal bundle** joined locally (logs+queries+traces+resources+incident) |
| AI assistant | ✓ | `veda-ops` triage agent, grounded in Veda's own RAG |
| Change-point detection | ✓ | two-window level/trend shifts |
| Spike attribution | statistical (term frequency) | **semantic** — names the *meaning-cluster*, flags `is_new` |
| Incident dedup | ✓ | **semantic clustering** — collapses differently-worded events |
| Incident novelty | — | **embedding novelty** (`1 − max_cosine_to_history`) — flags unprecedented patterns |
