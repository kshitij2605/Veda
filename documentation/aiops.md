# Veda AIOps — Operations Intelligence Reference

**Status:** Living reference. Consolidates the AIOps/operations-intelligence story shipped
across v0.2.5 → v0.3.1 (2026-06-14 → 2026-06-18). This is the single overview; the dated
deep-dive guides (indexed in §11) carry the per-cycle detail and live validation.

Veda's AIOps layer is built **inside the database**, on the primitives Veda already owns —
the embedder, the retriever, the chunker, the agent loop, the query log, the trace pipeline,
and the resource snapshot. Every operations feature reuses one of these primitives and
degrades gracefully (a template fallback, an embedding-overlap fallback, a novelty skip, or
a clear **503**) when no LLM or embedder is configured.

## The competitive thesis

Elastic's AIOps is **statistical and template-based over text**: log-rate analysis
bucketizes by message *templates*, anomaly detection runs population/time-series statistics,
and the "explain a spike" feature counts term frequencies. **Veda's AIOps is
semantic/embedding-based over every signal it owns** — logs, traces, queries, resources,
retrieval quality, and incidents — in *one* engine. Because Veda owns the embedder, the log
categorizer, the incident correlator, the cross-signal RCA bundle, *and* the agent, it
answers questions a statistical engine can only approximate:

- spike attribution names the **meaning-cluster** driving a spike ("*this kind of failure*
  rose"), not just which template's count went up — and can say a failure mode **is new**;
- incidents **cluster by similarity** and flag **novelty** ("this pattern is unprecedented")
  by cosine distance to a history of past incidents, not by string match;
- root-cause drafts are **grounded in Veda's own RAG** — the triage agent reasons over a
  cross-signal bundle (logs + queries + traces + resources + incident) joined locally,
  because Veda holds every one of those signals.

A text-search engine that doesn't own the embeddings or the agent can't faithfully score
grounding; a metrics store that doesn't own an embedder can't semantically cluster
incidents. Veda can, because it *is* the whole pipeline.

---

## 1. The Ops Advisor engine

The Ops Advisor is the spine of Veda's AIOps. A background tick samples an operational
snapshot, runs the rules + anomaly + change-point detectors, reconciles an event store, fires
alerts on transitions, and optionally auto-remediates. All in
`crates/veda-gateway/src/advisor/`.

### The tick pipeline

```
  spawn_advisor (tokio interval, VEDA_ADVISOR_INTERVAL_SECS, default 30s)
        │  each tick → run_tick(state)
        ▼
  Phase 0  synthetic probes (lock-free) + log categorization        (off-lock; may embed/await)
  1. snapshot.rs   collect(state) → OpsSnapshot  (+ history ring, cap 240)
  2. rules/ + anomaly.rs + changepoint.rs   → hits                  (pure / in-tick)
  3. event.rs   EventStore.reconcile(hits) → { opened, closed }     (under lock; sync)
  4. notify.rs   notify_transitions(...)                            (lock-free; async)
  5. remediation.rs   run_remediation(opened)  [opt-in]             (lock-free; async)
  6. SLO engine records good/total; EventStore.annotate + save      (brief re-lock)
```

### Lock discipline (the availability invariant)

The reconcile phase runs **synchronously under the `events`/`history` locks**, then those
locks are **released before** any notification / remediation network await (the store is
re-locked only briefly to annotate). The notifier's HTTP client has a 10s timeout. So a hung
webhook, a slow embedder, or a slow LLM can never freeze the Insights API or the tick loop.
Anything that embeds or calls the network (log categorization, spike attribution, incident
correlation, novelty, drafts) runs **off-lock**.

### Modules

- `snapshot.rs` — `OpsSnapshot` per tick: per-collection segment/record counts,
  compaction-in-progress, nodes + raft, replica `safe_ts`, health probes (MinIO/NATS/
  embedding), query-log latency (p50/p95 + slow count), process RSS, request/error counters,
  node resources (`/proc` + `statvfs`), log error/warn counts, change-points, and the index
  snapshot. A bounded `SnapshotHistory` ring (cap 240) backs the rate/anomaly/forecast math.
- `event.rs` — `OpsEvent`, `Severity{High,Medium,Low}`, `Scope{Global,Collection,Shard,Node,
  Slo,Synthetic,Index}`, `RemediationAction{None,Compact,Reindex}`, and `EventStore`
  (open-on-first-fire, dedup per `rule_id:scope`, refresh-in-place, **auto-close** on clear,
  bounded history, persisted to `{data_dir}/_ops_events.json`). `reconcile(hits)` returns the
  `{opened, closed}` transition signal alerting/remediation consume.
- `rules/` — a `Rule` trait + `catalog()` + `run_rules()` (skips disabled/muted, applies
  severity overrides).
- `anomaly.rs` — EWMA / rolling mean+stddev baselines; fire on N-σ once warmed up.
- `changepoint.rs` — sustained level/trend regime shifts (see §3c).
- `slo.rs` / `synthetic.rs` — the SLO engine and in-process probes (see §6).
- `config.rs` — `AdvisorConfig { rules, channels }`, persisted to `_advisor_config.json`.
- `notify.rs` / `remediation.rs` / `engine.rs` — channels, opt-in actions, the tick.

### Cadence + the off-tick runners

The main advisor tick runs every `VEDA_ADVISOR_INTERVAL_SECS` (default **30s**). Three of the
most expensive operations run on **separate, low-cadence tasks off the tick** so they never
stall it:

| Runner | Env interval | Default | Does |
|---|---|---|---|
| Advisor tick | `VEDA_ADVISOR_INTERVAL_SECS` | 30s | snapshot → rules/anomaly/change-point → events → notify → remediation → SLO record |
| Recall probe | `VEDA_RECALL_PROBE_INTERVAL_SECS` | 600s | brute-force `recall@k` proxy (min 50 records) |
| Incident correlator | `VEDA_INCIDENT_INTERVAL_SECS` | 300s | semantic incident clustering + novelty + proactive RCA drafts |

---

## 2. Detection

### (a) Rule catalog (~34 rules)

All rules are tunable per-rule via `PUT /v1/admin/insights/rules/{id}`
(enabled / muted / severity_override / thresholds / auto_remediate), persisted to
`_advisor_config.json`. Each resource/SLO/index rule **skips cleanly** when its required
signal is absent — no false alerts.

**Operational health**

| Rule | Severity | Fires when |
|---|---|---|
| `compaction_backlog` | Medium | a collection's sealed segments > 20 (and not already compacting) → `Compact` |
| `seal_stalled` | Medium | growing segment > 100k records with 0 sealed |
| `high_search_latency` | Medium | query p95 > 500 ms |
| `slow_query_surge` | Low | slow-query (>1s) count over threshold in the window |
| `high_error_rate` | High | error/request rate > 5% (counter deltas) |
| `embedding_unhealthy` | Medium | embedding endpoint health unavailable |
| `dependency_down` | High | MinIO or NATS health unavailable |
| `memory_pressure` | Medium | process RSS over threshold |
| `replica_lag` | High | replica `safe_ts` physical lag > 60 s |
| `node_unhealthy` | High | a node reports unhealthy/unavailable |
| `no_compaction_progress` | Low | large corpus, zero sealed segments cluster-wide |

**Node resources** (`/proc` + `statvfs`, Linux; null off-Linux):
`disk_space_low` (High, data-dir used % > 90 — warns *before* writes/seals/compaction fail),
`high_cpu` (Medium, >85%), `high_load` (Medium, load1/core > 2.0),
`fd_exhaustion` (High, open/limit > 0.8), `system_memory_low` (Medium, available/total < 0.1).

**Log content:**
`log_error_rate` (ERROR-log count over a 5-min window exceeds threshold),
`new_error_category` (a newly-appeared WARN/ERROR semantic category).

**SLO / reliability:**
`slo_burn_rate_fast` (High), `slo_burn_rate_slow` (Medium), `slo_budget_exhausted` (High),
`synthetic_failing` (High, after K consecutive failures, default K=3).

**Index advisor:**
`index_recall_degraded` (High, `recall < 0.90`),
`index_latency_high_for_recall` (Medium, `p95_ms > 50` while `recall >= 0.95`).

**Forecasting:**
`disk_full_forecast` (High), `memory_growth_forecast` (Medium), `recall_decay_forecast` (Medium).

**Change-point:** `change_point_detected` (Medium).

### (b) Anomaly detection (7 EWMA/σ detectors)

EWMA / rolling mean+stddev baselines that fire above a configured sigma (default 3σ → Medium,
≥5σ → High) once warmed up (default `min_samples` 20):
`anomaly_search_latency`, `anomaly_error_rate`, `anomaly_insert_rate`, `anomaly_rss`,
`anomaly_cpu`, `anomaly_disk_used`, `anomaly_error_logs`.

### (c) Change-point detection

Single-point anomaly detectors miss a **sustained regime change** (the disk-used baseline
steps up after a config change; RSS climbs on a new trend). `changepoint.rs` runs a
**two-window least-squares statistic** over the snapshot series (`disk_used_pct`, `rss_bytes`,
query `p95`, error-log count, per-collection `recall`), detecting both a **LEVEL** shift and a
**TREND** break. Three guards: a well-fit gate (the two-window model must fit materially better
than a single-window fit), a warm-up (≥20 points), and a min-magnitude guard (1.5σ). It is
**pure and in-tick** — no network, no LLM — so it costs nothing on the tick path.

### (d) Forecasting

A **least-squares trend** over the snapshot history, extrapolated to each rule's threshold,
produces a threshold-crossing **ETA**: when the disk fills, when memory exhausts, when recall
decays below target. Two guards keep it honest — **warm-up** (won't forecast without enough
history) and **min-R²** (won't forecast off a flat/noisy series) — so it never fires on
flat/noisy data. Surfaced via the three forecast rules and `GET /v1/admin/forecast`
(slope / R² / threshold / ETA per metric).

---

## 3. Semantic / vector-native AIOps (the differentiators)

These are the features a statistical engine cannot replicate. **Every one calls `embed_text`
→ the text model → runs CPU-only out of the box** (see §9), and every one degrades gracefully
without an embedder.

- **Semantic log categorization** — `LogCategorizer.categorize()` embeds each *distinct*
  recent WARN/ERROR message (cache-aware, off-lock) and greedily clusters by cosine similarity
  into `LogCategory { id, representative, count, max_severity, examples, first/last_seen }`. It
  groups *semantically similar* errors (differently-worded messages about the same failure),
  not just identical templates. Fallback without an embedder: **normalized-template** grouping
  (numbers/ids/quoted strings collapsed to placeholders). `GET /v1/admin/logs/categories`.
- **Semantic spike root-cause attribution** — `spike_attribution.rs` answers "what is driving
  the log-rate spike?" semantically. Given a spike `window` and a `baseline`, it embeds and
  clusters both, then attributes `total_delta` to the **meaning-clusters** driving it:
  `drivers: [{ category, representative, spike_count, baseline_count, growth,
  share_of_growth, is_new }]`. `is_new` flags a brand-new failure mode. Embedding runs
  off-lock; without an embedder it falls back to the normalized-template path and reports
  `method="template"` (vs `method="embedding"`). `GET /v1/admin/log-rate-analysis`.
- **Semantic incident correlation & dedup** — each ops-event's title+detail is embedded and
  clustered by cosine into an `Incident { representative, member_event_ids, rule_ids, count,
  max_severity, first_seen, last_seen }`, collapsing a storm of related events into one
  incident. Fallback without an embedder: a **rule_id + scope** grouping (dedup still works).
  Runs on the off-tick incident correlator. `GET /v1/admin/incidents`.
- **Embedding-based incident novelty detection** — the `IncidentStore` keeps a persisted
  **embedding history** of past incidents; a new incident's `novelty_score =
  1 − max_cosine_to_history` (1.0 = nothing looks like it). `is_novel` is flagged above
  threshold. Guards: warm-up (≥5 historical incidents — below that, nothing is flagged) and an
  **embedder-free skip** (no embedder → `novelty_score` null, not fabricated). A novel incident
  escalates its notification ("⚠ UNPRECEDENTED pattern", severity → High).

---

## 4. Root-cause & agentic

- **Cross-signal RCA bundle** — `GET /v1/admin/insights/{event_id}/rca` **joins** five signals
  for one event into a single bundle: the correlated **logs** (log buffer), the slow/errored
  **queries** in the window (query log), the **traces** matched by `trace_id` + the event time
  window, the nearest **resource snapshot**, and the related **incident**. This is the
  cross-signal correlation an APM tool charges for — but Veda holds every signal, so the join
  is local.
- **The `veda-ops` triage agent** — a built-in, in-memory agent (alongside `veda-agent`) with
  an ops-focused system prompt. It whitelists **read-only** ops tools and **not** `upsert_record`
  (triage never mutates data): `get_insights`, `get_logs`, `get_resources`, `get_slo_status`,
  and `correlate_incident` (the RCA bundle). `POST /v1/admin/insights/{event_id}/triage`
  (404 unknown event, **503** no LLM) runs the agent and returns
  `{ explanation, steps[], metrics, citations, grounding }` — `steps[]` is the full
  reasoning/tool-call/tool-result trace, so the operator sees which signals it consulted.
- **Proactive incident RCA drafts** — on a **newly-correlated** incident (off-tick), the
  `veda-ops` agent auto-runs over the incident's RCA bundle and drafts a root-cause paragraph +
  runbook step into `Incident.draft`. Best-effort and **503-safe** (`None` without an LLM),
  **new-incident-only** (one draft per incident, not per tick).
- **Alert-storm noise reduction** — notifications fire per **incident**, not per event: a storm
  of twelve correlated events that folds into one incident pages you **once** ("N alerts → 1
  incident"), deduped via a persisted set of notified incident-ids. `GET /v1/admin/incidents`
  carries `noise_reduction { event_count, incident_count, suppressed }` where
  `suppressed = event_count − incident_count` (the pages you *didn't* get).

---

## 5. SLOs & error budgets

An `SloDef { id, name, kind, target_pct, window_secs, fast_burn{window_secs,threshold},
slow_burn{...} }`. `SloKind` (internally tagged on `kind`, snake_case):

- `latency { threshold_ms }` — fraction of queries faster than `threshold_ms`.
- `availability` — fraction of requests that did not error.
- `error_rate` — same counters, framed as the error fraction.
- `synthetic { probe_id }` — fraction of synthetic-probe runs that succeeded.
- `recall { collection, k? }` — sourced from the ANN **recall probe** (proxy; a drift guard).
- `faithfulness { agent? }` — sourced from a persisted faithfulness sample of converse turns.

Each advisor tick records **good/total** per SLO into a **persisted, 60s-bucketed rolling
counter** (`_slo_state.json`, cap **44640 = 30 days**). Over a window:

```
allowed_bad                = 1 - target/100
sli_pct                    = good/total * 100
error_budget_remaining_pct = (1 - (bad/total)/allowed_bad) * 100      # negative = over budget; floor -1000
burn_rate                  = (bad/total)/allowed_bad                  # 1.0 = on pace; >1.0 = too fast
```

`SloStatus` exposes both windows: `{ sli_pct, error_budget_remaining_pct, burn_rate_fast,
burn_rate_slow, breaching_fast, breaching_slow, budget_exhausted, compliant }`. The two-window
fast/slow pairing is the standard Google-SRE multi-window/multi-burn-rate pattern (fast catches
acute outages, slow catches chronic drift). The recall + faithfulness quality SLOs reuse the
**same** burn-rate / error-budget rules — Veda owns both the recall probe and the faithfulness
scorer, so the SLO layer just reads SLIs that already exist.

---

## 6. Alerting & remediation

**Alerting fires on event open/close transitions only** (and, for incidents, per-incident — see
§4), so there's no per-tick spam. Four channels, async, severity-routed (per-channel
`min_severity`), per-channel/per-event failure-isolated:

- **Generic webhook** — `POST {url}` with `{ status: "open"|"closed", event: {...} }`.
- **Slack** — incoming-webhook message (severity emoji + title + recommendation; `RESOLVED:` on close).
- **PagerDuty** — Events API v2: `trigger` on open / `resolve` on close, `dedup_key = event.id`,
  severity mapped High→critical / Medium→warning / Low→info.
- **Email** — async SMTP (`lettre`), subject `[severity] title`.

All HTTP is async with a 10s timeout (never `reqwest::blocking`). Channel secrets are
**write-only** — `GET /channels` returns only `{ configured, enabled, min_severity }`.

**Opt-in auto-remediation.** Per rule, `auto_remediate` is **off by default**. When enabled, an
opened event can trigger its `RemediationAction` — `Compact` runs the real compaction path
(honoring the compaction lock); `Reindex` is best-effort — **rate-limited to ≤1 per 5 minutes
per scope**, with the outcome annotated onto the event. Muted/disabled rules never act.

**In-process synthetic self-checks.** Probes (`SyntheticProbe { id, name, kind, interval_secs,
timeout_ms }`) run at the start of each tick (Phase 0, lock-free), exercising Veda's real
critical paths: `health` (the readiness dependency checks), `search` (the internal query
executor, top-1), and `embed` (the async embedding client). The rolled-up `SyntheticSummary`
feeds the `synthetic` SLO SLI and the `synthetic_failing` rule.

---

## 7. REST surface — AIOps endpoints

All `/v1/admin/*` are admin-only (`bearerAuth`); the agent routes are tenant/RLS-scoped.

| Method + path | Description |
|---|---|
| `GET /v1/admin/insights` | Open events (`?severity=`, `?scope=`), severity-sorted |
| `GET /v1/admin/insights/history` | Closed events (bounded) |
| `GET /v1/admin/insights/rules` | Rule catalog + effective config per rule |
| `PUT /v1/admin/insights/rules/{id}` | Tune enabled/muted/severity_override/thresholds/auto_remediate |
| `GET /PUT /v1/admin/insights/channels` | Channel config (GET redacts secrets; PUT write-only) |
| `POST /v1/admin/insights/run` | Force one evaluation tick → `{ opened, closed }` |
| `POST /v1/admin/insights/{id}/ack` | Acknowledge an open event |
| `POST /v1/admin/insights/{id}/triage` | AI triage via `veda-ops` (404 unknown, **503** no LLM) |
| `GET /v1/admin/insights/{id}/rca` | Cross-signal RCA bundle (logs+queries+traces+resources+incident) |
| `GET /v1/admin/resources` | Latest node-resource snapshot (disk/cpu/load/fds/memory/rss) |
| `GET /POST /v1/admin/slos` | List / create-or-overwrite SLO defs |
| `GET /PUT /DELETE /v1/admin/slos/{id}` | Get / update / delete one SLO |
| `GET /v1/admin/slos/{id}/status` | Current SLI / error-budget / burn-rate status |
| `GET /POST /v1/admin/synthetics` | List / create-or-overwrite probe defs |
| `GET /PUT /DELETE /v1/admin/synthetics/{id}` | Get / update / delete one probe |
| `GET /v1/admin/synthetics/{id}/results` | Recent per-probe results + summary |
| `GET /v1/admin/correlations` | Slow-query/error correlation, bucketed + ranked worst-first |
| `GET /v1/admin/change-points` | Change-point detections (sustained level/trend shifts) |
| `GET /v1/admin/log-rate-analysis` | Semantic spike attribution (`?window_secs=&baseline_secs=&min_level=`) |
| `GET /v1/admin/incidents` | Semantic incident clusters (+ `draft`, `is_novel`/`novelty_score`, `noise_reduction`) |
| `GET /v1/admin/forecast` | Capacity forecasts (disk-full / memory / recall-decay ETAs) |
| `GET /v1/admin/index-advisor` | Per-collection index_type/params/p95/recall + recommendations + open events |
| `GET /v1/admin/depmap` | Declarative service dependency graph (9 nodes / 8 edges) |
| `GET /v1/admin/pprof` | On-demand CPU profile (SVG / pprof; `profiling` feature build; 409 if running) |
| `GET /v1/admin/logs`, `/logs/categories`, `/slow-queries` | Log tail / semantic categories / slow queries |

---

## 8. Embedding model & CPU/GPU (important)

**The semantic AIOps edge runs CPU-only out of the box. No GPU is required.**

- **Text embedder (default `BAAI/bge-m3`, 1024-d)** is served by **HuggingFace TEI on CPU** —
  `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` in `docker-compose.yml` under
  `--profile embedding` (`MODEL_ID` default `BAAI/bge-m3`). BGE-M3 uniquely ships dense +
  sparse + multi-vector from one forward pass.
- **Multimodal embedder (Qwen3-VL-Embedding-2B/8B, vLLM)** needs a **GPU** and is used **only**
  for image / audio / video. The compose default for the multimodal service is
  `Qwen/Qwen3-VL-Embedding-8B` on an NVIDIA device.
- **All semantic AIOps features** — log categorization, spike attribution, incident
  correlation, novelty, faithfulness (embedding-overlap fallback) — call **`embed_text` → the
  text model**, so they run **CPU-only** with the default `--profile embedding` stack. None of
  them touch the multimodal/GPU path.

**Latency is a non-issue for AIOps.** CPU embedding is slower than GPU, but the semantic AIOps
work runs **off-tick at low cadence** (categorization in the advisor's off-lock Phase 0;
incident correlation every `VEDA_INCIDENT_INTERVAL_SECS`=300s; recall probe every 600s), so
CPU speed doesn't matter there. (Per-request *search*/*converse* embedding on CPU adds tens of
ms — that's the only place CPU embedding is in a hot path, and it's still fine.)

**Fully pluggable.** The embedder is an HTTP endpoint configured by `EMBEDDING_ENDPOINT` /
`EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` (default 1024). A curated `text_presets()` catalog in
`veda-core::embedding` ships bge-m3 (default), qwen3-embedding-8b/4b/0.6b, voyage-3-large, and
voyage-3.5 — each carrying the model's dimension/modality/size envelope; the deployment supplies
only endpoint + key. (`multimodal_presets()` ships qwen3-vl-embedding-2b/8b.)

**Graceful degradation without an embedder:** log categorization + spike attribution fall back
to **normalized-template** grouping (`method="template"`), faithfulness falls back to the
**LLM-judge** path, and incident correlation falls back to **rule_id + scope** grouping;
novelty is **skipped** (`novelty_score` null). Semantic quality needs an embedder — which is
CPU-fine.

**The LLM is separate.** The chat model used for AI triage and proactive incident drafts is the
OpenAI-compatible endpoint `VEDA_LLM_ENDPOINT` / `VEDA_LLM_API_KEY` / `VEDA_LLM_MODEL` (a hosted
API or a local model). The draft/triage/eval/streaming paths are **503-safe** without it; the
rest of the AIOps pipeline is unaffected.

---

## 9. Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_ADVISOR_INTERVAL_SECS` | `30` | Advisor tick cadence (snapshot/rules/anomaly/change-point/SLO record) |
| `VEDA_INCIDENT_INTERVAL_SECS` | `300` | Off-tick incident correlation + novelty + proactive drafts |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Off-tick proxy recall probe (min 50 records); sources the `recall` SLO SLI |
| `VEDA_LOG_CAPTURE_LEVEL` | `WARN` | In-process log-capture threshold (ERROR/WARN/INFO/DEBUG/TRACE) |
| `EMBEDDING_ENDPOINT` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | — / — / `BAAI/bge-m3` / `1024` | Text embedder (CPU TEI); drives all semantic AIOps |
| `VEDA_LLM_ENDPOINT` / `VEDA_LLM_API_KEY` / `VEDA_LLM_MODEL` | *(unset)* | OpenAI-compatible chat LLM for triage / drafts / eval / streaming (503-safe when unset) |
| `VEDA_LLM_PRICES` | *(built-in default)* | JSON per-1k-token price table for the cost metric |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | Gates all OTLP span/metric export; when set, log/query records carry `trace_id`/`span_id` |
| `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG` | `parentbased_traceidratio` / `1.0` | Head sampler strategy + argument |

Rule thresholds/severities/mute/auto-remediate, alert channels, SLO/probe defs, the event
store, the SLO counter, and recall results all persist to the data dir (`_advisor_config.json`,
`_ops_events.json`, `_slo_state.json`, `_synthetics.json`, `_synthetic_results.json`,
`_recall_results.json`).

---

## 10. Prometheus metrics (at `/v1/admin/metrics`, always available)

| Metric | Type | Notes |
|---|---|---|
| `veda_index_recall` / `veda_index_p95_ms` | gauge (per `collection`) | recall-probe outputs |
| `veda_llm_tokens_total` | counter (`type`, `model`) | token accounting |
| `veda_agent_run_duration_seconds` | histogram (`model`) | agent run latency |
| `veda_llm_cost_usd_total` | counter (`model`) | accumulated **micro-USD** (divide by 1e6) |
| `veda_agent_faithfulness` | gauge | latest faithfulness score (0–1) |
| `veda_agent_citations` | counter/gauge | citation accounting per run |

---

## 11. Deep-dive guides

| Guide | Version | Covers |
|---|---|---|
| [2026-06-14-autoops-ops-advisor.md](2026-06-14-autoops-ops-advisor.md) | v0.2.5 | The Ops Advisor engine: snapshot → rules → events → channels → opt-in remediation; the original 15-rule catalog + 4 anomaly detectors |
| [2026-06-15-log-insights.md](2026-06-15-log-insights.md) | v0.2.6 | In-process log capture + trace correlation, **semantic log categorization**, log-content rules, logs-as-a-collection ingest |
| [2026-06-16-node-resource-signals.md](2026-06-16-node-resource-signals.md) | v0.2.7 | Disk/CPU/load/fd/system-memory signals from `/proc` + `statvfs`, the resource rules, `/v1/admin/resources` |
| [2026-06-16-service-reliability-slo.md](2026-06-16-service-reliability-slo.md) | v0.2.8 | SLOs + error budgets + multi-window burn rate, synthetic self-checks, the `veda-ops` AI triage agent |
| [2026-06-17-apm-observability.md](2026-06-17-apm-observability.md) | v0.2.9 | Tracing/sampler/W3C propagation, CPU profiling, GenAI-semconv spans + cost/token, slow-query correlation, dependency map, the ANN index advisor |
| [2026-06-18-rag-aiops-intelligence.md](2026-06-18-rag-aiops-intelligence.md) | v0.3.0 | Citations + faithfulness, quality SLOs (recall/faithfulness), eval harness + flywheel, agent memory, streaming converse, **semantic incident correlation**, forecasting, the **cross-signal RCA bundle** |
| [2026-06-18-agentic-aiops.md](2026-06-18-agentic-aiops.md) | v0.3.1 | **Change-point detection**, **semantic spike attribution**, **proactive incident drafts + noise reduction**, **incident novelty detection** |

---

## 12. Vs Elastic AIOps

Veda's vector-native approach is the edge wherever the signal is semantic.

| Capability | Elastic AIOps | Veda |
|---|---|---|
| Anomaly detection | ✅ statistical | ✅ EWMA/σ over 7 series |
| Log categorization | ✅ template-based | ✅ **semantic (embedding cosine clusters)** — edge: groups by *meaning*, not template |
| Forecasting | ✅ | ✅ least-squares ETA + R² confidence guard |
| Correlation / RCA | ✅ | ✅ **cross-signal bundle** joined locally (logs+queries+traces+resources+incident) |
| AI assistant | ✅ | ✅ `veda-ops` triage agent, grounded in Veda's own RAG |
| Change-point detection | ✅ | ✅ two-window level/trend shifts |
| Spike attribution | ✅ statistical (term frequency / template) | ✅ **semantic** — edge: names the *meaning-cluster*, flags `is_new` |
| Incident dedup | ✅ | ✅ **semantic clustering** — edge: collapses differently-worded events |
| Incident novelty | ✗ | ✅ **embedding novelty** (`1 − max_cosine_to_history`) — edge: flags unprecedented patterns |
