# Veda APM & Performance Observability — Tracing, Profiling, GenAI/RAG Traces, Correlation, Cost & the Index Advisor

**Date:** 2026-06-17
**Status:** Shipped on `feat/apm-observability` (also on `main`), v0.2.9. Validated end-to-end against a live server.

APM & Performance Observability is Veda's in-lane answer to an Application Performance Monitoring
platform. The framing matters: **Veda is a monitored service, not an APM-agent vendor.** Elastic
APM (and Datadog/Dynatrace/New Relic) ship *agents* that instrument your application code in a dozen
languages, plus mobile SDKs, serverless layers, and tail-based sampling. Veda has exactly one
service to instrument — itself — so the in-lane subset is *"emit clean, standards-compliant traces
and metrics for the work Veda does, propagate context so we stitch into your trace, and turn the
performance data Veda already has into actionable signals."*

**Deliberately out of lane** (these monitor *your* application, which a vector DB does not own):
- **APM agents for application languages** (Java/Python/Node/Go/Ruby/.NET auto-instrumentation).
- **Mobile / RUM** end-user telemetry.
- **Serverless** layers / Lambda extensions.
- **Tail-based sampling** (a collector-tier concern; Veda ships head-based, parent-respecting
  sampling and lets your OTel Collector do the rest).

Everything here is OTel-first and **export-gated**: spans and the new metrics export only when
`OTEL_EXPORTER_OTLP_ENDPOINT` is configured. The no-OTLP path is byte-for-byte unaffected. This
extends the existing observability and Ops Advisor surfaces; it is not a new product.

---

## 1. Distributed tracing — sampler, W3C propagation, dependency spans

Three gaps closed, all in the existing tracing pipeline.

**Configurable sampler.** The OTel tracer reads `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`,
defaulting to `parentbased_traceidratio` @ `1.0` — i.e. **no behavior change** unless you set it.
Set `OTEL_TRACES_SAMPLER_ARG=0.1` to sample 10% of head traces; `parentbased_*` means a sampling
decision made upstream (your app) is respected, so a trace is sampled consistently end-to-end.

**Outbound W3C context injection.** Veda now injects W3C `traceparent`/`tracestate` into its
outbound calls:
- the LLM, embedding, and rerank `reqwest` clients (HTTP headers), and
- remote-shard gRPC calls (tonic request metadata).

So a trace that arrives from your application continues *through* Veda and *out* to its
dependencies, instead of dead-ending at the gateway.

**Missing dependency spans.** The hops that previously had no span now emit one: remote-shard gRPC,
rerank, local-shard search, vector-index search, and NATS. Combined with propagation, the service
trace is now complete from gateway → query → shard → index, and across the embedding/LLM/rerank
calls.

**Python SDK.** `veda-client` gained optional W3C propagation behind the `veda-client[otel]` extra.
With OpenTelemetry installed it injects context on every request; without it, the code path is a
no-op (no hard dependency).

---

## 2. On-demand CPU profiling

`GET /v1/admin/pprof` (admin-only) samples the running server on demand:

| Param | Default | Meaning |
|---|---|---|
| `seconds` | 5 | profile duration |
| `frequency` | 100 | sampling rate (Hz) |
| `format` | `flamegraph` | `flamegraph` (SVG) or `pprof` (protobuf) |

It returns an `image/svg+xml` flamegraph or an `application/octet-stream` pprof protobuf you can
open in `go tool pprof` / Pyroscope / Speedscope. Constraints:

- **409** if a profile is already running (single-flight — the `pprof` sampler is process-global).
- **Empty when idle** — with no CPU work during the window there are no samples, so an idle server
  yields an empty profile. This is by design; profile *under load*.
- **Behind a cargo feature.** Profiling pulls in the `pprof` crate (and its native unwinder), so it
  is gated behind a `profiling` feature and **excluded from default builds**. Build with
  `cargo build -p veda-server --features profiling`; without the feature the endpoint is absent.

---

## 3. Agent / LLM / GenAI tracing + RunMetrics + cost/token

Veda's RAG/agent loop is the most "APM-shaped" thing it does, so it now emits **OTel GenAI
semantic-convention** spans and rich per-run metrics.

### The gen_ai.* span tree

```
invoke_agent                       (root — one per /converse or triage run)
  ├─ chat {model}                  (one per LLM call)
  │     gen_ai.system              (e.g. "openai" — inferred from the model name)
  │     gen_ai.request.model
  │     gen_ai.usage.input_tokens
  │     gen_ai.usage.output_tokens
  │     gen_ai.response.finish_reasons
  ├─ execute_tool {id}             (one per tool call)
  │     gen_ai.tool.name
  │     gen_ai.tool.call.id
  │     (error status on failure)
  └─ chat {model} …                (next step)
```

The attribute keys are centralized in `veda-core::genai_semconv` so they stay consistent. Because
these follow the OTel GenAI semantic conventions, the traces drop **natively** into Datadog,
Grafana (Tempo), Phoenix, and Langfuse without custom mapping. Spans export only when OTLP is
configured; the no-OTLP path is unaffected.

### RunMetrics

Every agent run returns a `RunMetrics`, surfaced as `metrics` on both `ConverseResponse` and the
triage response:

```json
{ "total_ms": 685, "llm_ms": 677, "tool_ms": 2, "steps": 2,
  "prompt_tokens": 4369, "completion_tokens": 36, "total_tokens": 4405 }
```

Each `AgentStep::ToolResult` also carries a per-step `duration_ms`, so a slow tool is visible in the
step trace, not just in the aggregate.

### Cost / token metrics

Prometheus metrics (always available at `/v1/admin/metrics`):

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `veda_llm_tokens_total` | counter | `type` (input/output), `model` | token accounting |
| `veda_agent_run_duration_seconds` | histogram | `model` | end-to-end run latency |
| `veda_llm_cost_usd_total` | counter | `model` | **accumulated in micro-USD** |

> **Micro-USD note.** Prometheus counters are integers and this one is a `u64`, so cost is
> accumulated in **micro-USD**. Divide the scraped value by **1e6** to get dollars. (A floating
> sub-cent counter would lose precision; micro-USD keeps it exact.)

**Price table.** Prices come from `VEDA_LLM_PRICES`, a JSON map of per-1k-token prices:

```json
{ "gpt-4o": { "input": 2.5, "output": 10.0 } }
```

If the env var is unset, a small built-in default table applies. **An unpriced model still emits
token metrics — it just skips the cost metric.** Cost accounting never fails a request or a run; an
unknown model simply has no `veda_llm_cost_usd_total` series.

---

## 4. Slow-query / error correlation

`GET /v1/admin/correlations` (admin-only) answers *"which kind of query is slow or failing?"* — the
APM "transaction breakdown" applied to vector search. It groups the recent query-log window by:

`collection` · `query_type` · `has_filter` · `reranked` · `index_type` · `result_count`-bucket ·
`top_k`-bucket

and returns, per bucket: `count`, `p50_ms`, `p95_ms`, `slow_count`, and `error_rate`, **ranked
worst-first**. So instead of one global p95, you see that (say) *filtered + reranked HNSW queries on
collection X* are the slow/erroring cohort.

Two enabling changes in the query log:
- **Enrichment.** `QueryLogEntry` gained `collection, query_type, has_filter, reranked,
  result_count, status, error_kind, index_type, shard_ids` (serde back-compatible — old logs still
  parse).
- **Errors are now logged.** The query log was **success-only**, so failures left no record. It now
  writes one entry per request — including failures, with `status` / `error_kind` — **independent of
  trace sampling** (the log is the durable record; tracing is the sampled detail). Without this,
  `error_rate` in `/correlations` would always read zero.

A Niriksha **Correlations** panel renders the ranked buckets.

---

## 5. Retrieval-quality traces + dependency map

### Retrieval-quality span attributes

The search span now always carries database/retrieval context, following the OTel DB semantic
conventions where they apply:

- `db.system = veda`, `db.collection`
- `retrieval.query_type`, `retrieval.top_k`, `retrieval.result_count`, `retrieval.reranked`,
  `retrieval.has_filter`, `retrieval.index_type`

When a request sets `"explain": true`, the `ExplainTiming` per-phase breakdown is attached under an
`explain` group: `retrieval.filter_ms`, `retrieval.ann_search_ms`, `retrieval.scoring_ms`,
`retrieval.fusion_ms`, `retrieval.total_ms`. So a trace viewer shows not just *that* a search was
slow but *which phase* (filter vs ANN vs scoring vs fusion).

### Dependency map

`GET /v1/admin/depmap` returns a declarative service graph — **9 nodes** (gateway, query, shards,
index, embedding, llm, rerank, blob, nats) and **8 directed edges**. It is static topology (not
discovered from live traffic), giving a stable architecture view; the Niriksha **Dependencies**
panel renders it with a RAG-trace legend so the GenAI span tree and the topology line up.

---

## 6. ANN index advisor

The advisor closes the loop from *observing* recall/latency to *recommending* an index change.

### Proxy recall probe

`advisor/recall_probe.rs` samples query vectors, runs the live ANN search for top-k, and compares
against an **exact brute-force** scan over a `corpus_cap`=50k subset to compute **`recall@k`**. It
is flagged **`is_proxy=true`**: this is recall against the index's own corpus subset, *not* against
a labeled ground-truth set. It is a drift signal, not a benchmark number.

The probe runs on a **separate, low-cadence task off the advisor tick** — it is the most expensive
thing the advisor does (a brute-force scan), so it must not stall the 30s tick. Cadence is
`VEDA_RECALL_PROBE_INTERVAL_SECS` (default **600**), and it requires **at least 50 records** before
it runs. Results persist to `_recall_results.json` (`RecallResults` store) with an `IndexSnapshot`
in the advisor snapshot. Gauges: `veda_index_recall{collection}`, `veda_index_p95_ms{collection}`.

### Rules

Two rules join the advisor catalog (per-collection `Scope::Index`), flowing through the existing
event store + alert channels and tunable via `/v1/admin/insights/rules`:

- **`index_recall_degraded`** (High) — `recall < recall_min` (default **0.90**). Recommendation is
  directional: raise `efSearch`/`efConstruction`, or reindex.
- **`index_latency_high_for_recall`** (Medium) — `p95_ms > 50` while `recall >= recall_ok`
  (**0.95**). Recall is fine but you're paying too much for it: lower `efSearch`, or switch index
  type.

### REST

`GET /v1/admin/index-advisor` returns, per collection: `index_type`, `params`, `p95_ms`, `recall`,
`is_proxy`, `recommendations[]`, and `open_events[]`. The Niriksha **IndexAdvisor** panel surfaces
this with an **opt-in reindex** affordance.

> **Directional, not automatic.** Reindex doesn't yet *change* ANN params, so the recommendations
> are advisory hints ("raise efSearch") rather than one-click retunes — see Limitations.

---

## 7. REST API

All endpoints are admin-only (`bearerAuth`).

| Endpoint | Method | Description |
|---|---|---|
| `/v1/admin/correlations` | GET | Slow-query/error correlation, bucketed and ranked worst-first |
| `/v1/admin/pprof` | GET | On-demand CPU profile (SVG / pprof proto); `profiling` feature build; 409 if running |
| `/v1/admin/depmap` | GET | Declarative service dependency graph (9 nodes / 8 edges) |
| `/v1/admin/index-advisor` | GET | Per-collection index_type/params/p95/recall + recommendations + open events |
| `/v1/agents/{id}/converse` | POST | Now returns `metrics` (RunMetrics) alongside the answer + step trace |
| `/v1/admin/insights/{event_id}/triage` | POST | Triage response now also returns `metrics` |

## Metrics (Prometheus, at `/v1/admin/metrics`)

| Metric | Type | Labels |
|---|---|---|
| `veda_llm_tokens_total` | counter | `type`, `model` |
| `veda_agent_run_duration_seconds` | histogram | `model` |
| `veda_llm_cost_usd_total` | counter (micro-USD) | `model` |
| `veda_index_recall` | gauge | `collection` |
| `veda_index_p95_ms` | gauge | `collection` |

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | Gates all OTLP export (spans + the new metrics) |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Head sampler strategy |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Sampler argument (e.g. ratio) |
| `VEDA_LLM_PRICES` | *(built-in default)* | JSON per-1k-token price table for cost metrics |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Recall-probe cadence (off the advisor tick; min 50 records) |

Profiling is enabled at **build** time: `cargo build -p veda-server --features profiling`.

---

## 8. Validation (live)

Verified against a running server on the Linux host:

- **Correlation:** `/correlations` captured **13 queries including 1 error** (error_rate ≈ 0.077,
  p95 4 ms) and ranked the buckets per dimension worst-first.
- **Profiling:** `/pprof` returned a **2.1 MB flamegraph SVG** when sampled under load; an idle
  server returned an empty profile (no samples), as designed.
- **Agent metrics + cost/token:** `/converse` returned a grounded answer with
  `metrics { total_ms 685, llm_ms 677, tool_ms 2, steps 2, total_tokens 4405 }`, and Prometheus
  showed `veda_llm_tokens_total{type=input}=4369` / `{type=output}=36` for the live Qwen model.
  `veda_llm_cost_usd_total` was **absent** for that model — it is unpriced, so tokens were emitted
  and cost was skipped, exactly as designed.
- **Dependency map:** `/depmap` returned **9 nodes / 8 edges**.
- **Index advisor:** the recall probe computed a proxy **recall of 1.0**; after `recall_min` was
  raised above the observed recall, `index_recall_degraded` fired once with a directional
  recommendation, and `/index-advisor` surfaced it.

---

## 9. Limitations & follow-ups

- **Proxy recall ≠ labeled recall@k.** The probe measures ANN top-k against an exact brute-force
  scan over a sampled subset (`is_proxy=true`), which is a drift/regression signal, not a
  ground-truth benchmark against a labeled relevance set.
- **Recommendations are directional.** Reindex doesn't yet change ANN parameters, so the advisor's
  "raise efSearch" / "switch index type" outputs are advisory hints, not an automated retune.
- **`gen_ai.system` is inferred** from the model name rather than a configured provider identity, so
  exotic/self-hosted model names may be mis-attributed.
- **GenAI semconv is experimental.** The OTel GenAI semantic conventions are still evolving; the
  attribute names may shift in a future OTel release and we will track them.
- **pprof needs load and a feature build.** The profiler is empty on an idle server (no samples) and
  is absent unless built with `--features profiling`.
- **Out of scope, deliberately:** APM agents for application languages, mobile/RUM, serverless
  layers, and tail-based sampling — these instrument an end-user application or live in the
  collector tier, not in a vector DB. Veda emits standards-compliant telemetry for the service it
  owns and propagates context so it stitches into yours.
