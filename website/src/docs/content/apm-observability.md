# APM & Performance Observability

Veda emits clean, standards-compliant traces and metrics for the work it does, propagates trace context so it stitches into *your* distributed trace, and turns the performance data it already has into actionable signals — distributed tracing, on-demand profiling, GenAI/RAG traces with cost accounting, slow-query correlation, a dependency map, and an ANN index advisor.

The framing matters: **Veda is a monitored service, not an APM-agent vendor.** It has exactly one service to instrument — itself. So application-language APM agents (Java/Python/Node/Go/…), mobile/RUM telemetry, serverless layers, and tail-based sampling are deliberately out of lane — those monitor *your* application or live in the collector tier. Veda emits the telemetry for the service it owns and propagates context so it lines up with yours.

Everything here is OTel-first and **export-gated**: spans and the new metrics export only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. The no-OTLP path is byte-for-byte unaffected. This extends the existing [Observability](/docs/observability) and [Ops Advisor](/docs/ops-advisor) surfaces.

## Distributed tracing

- **Configurable head sampler.** The tracer reads `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`, defaulting to `parentbased_traceidratio` @ `1.0` (no change unless you set it). Set the arg to `0.1` to sample 10% of head traces; `parentbased_*` respects an upstream decision so a trace is sampled consistently end-to-end.
- **Outbound W3C propagation.** Veda injects `traceparent` / `tracestate` into its outbound LLM, embedding, and rerank HTTP calls and its remote-shard gRPC metadata — so a trace from your app continues *through* Veda and *out* to its dependencies instead of dead-ending at the gateway.
- **Complete dependency spans.** The hops that previously had no span now emit one: remote-shard gRPC, rerank, local-shard search, vector-index search, and NATS. The service trace is now complete from gateway → query → shard → index, and across the embedding/LLM/rerank calls.
- **Python SDK.** `veda-client` gained optional W3C propagation behind the `veda-client[otel]` extra — a no-op when OpenTelemetry isn't installed (no hard dependency).

## On-demand CPU profiling

`GET /v1/admin/pprof` samples the running server on demand and returns a flamegraph (SVG) or a `pprof` protobuf for `go tool pprof` / Pyroscope / Speedscope.

| Param | Default | Meaning |
|---|---|---|
| `seconds` | 5 | profile duration |
| `frequency` | 100 | sampling rate (Hz) |
| `format` | `flamegraph` | `flamegraph` (SVG) or `pprof` (protobuf) |

- **409** if a profile is already running (the sampler is process-global, single-flight).
- **Empty when idle** — no CPU work in the window means no samples. Profile *under load*.
- **Behind a build feature** — the profiler pulls in a native unwinder, so it's gated: `cargo build -p veda-server --features profiling`. Without it, the endpoint is absent.

## GenAI / RAG tracing, run metrics & cost

Veda's RAG/agent loop is the most APM-shaped thing it does, so it emits **OpenTelemetry GenAI semantic-convention** spans. Because they follow the convention, the traces drop **natively** into Datadog, Grafana (Tempo), Phoenix, and Langfuse with no custom mapping.

```
invoke_agent                       (root — one per /converse or triage run)
  ├─ chat {model}                  gen_ai.system, gen_ai.request.model,
  │                                gen_ai.usage.input_tokens / output_tokens,
  │                                gen_ai.response.finish_reasons
  ├─ execute_tool {id}             gen_ai.tool.name, gen_ai.tool.call.id
  └─ chat {model} …
```

### Run metrics

Every agent run returns a `RunMetrics`, surfaced as `metrics` on both the [`/converse`](/docs/agents-mcp) and the [triage](/docs/service-reliability#ai-ops-triage) responses:

```json
{ "total_ms": 685, "llm_ms": 677, "tool_ms": 2, "steps": 2,
  "prompt_tokens": 4369, "completion_tokens": 36, "total_tokens": 4405 }
```

Each tool step also carries a per-step `duration_ms`, so a slow tool is visible in the step trace, not just the aggregate.

### Cost & token metrics

Always available at `/v1/admin/metrics`:

| Metric | Type | Labels |
|---|---|---|
| `veda_llm_tokens_total` | counter | `type` (input/output), `model` |
| `veda_agent_run_duration_seconds` | histogram | `model` |
| `veda_llm_cost_usd_total` | counter (**micro-USD**) | `model` |

Cost is accumulated in **micro-USD** (the counter is an integer `u64`) — divide the scraped value by `1e6` for dollars. Prices come from `VEDA_LLM_PRICES`, a JSON map of per-1k-token prices (`{ "gpt-4o": { "input": 2.5, "output": 10.0 } }`); a small built-in table applies if it's unset. An **unpriced model still emits token metrics** — it just skips the cost series. Cost accounting never fails a request.

## Slow-query / error correlation

`GET /v1/admin/correlations` is the APM "transaction breakdown" applied to vector search: it answers *"which kind of query is slow or failing?"* It groups the recent query-log window by `collection`, `query_type`, `has_filter`, `reranked`, `index_type`, and `result_count` / `top_k` buckets, and returns per bucket `count`, `p50_ms`, `p95_ms`, `slow_count`, and `error_rate` — **ranked worst-first**. So instead of one global p95, you see that (say) *filtered + reranked HNSW queries on collection X* are the slow cohort.

This is backed by query-log enrichment (the log now records `collection`, `query_type`, `has_filter`, `reranked`, `result_count`, `status`, `error_kind`, `index_type`, `shard_ids`) and by the log now writing **one entry per request including failures** — independent of trace sampling, so `error_rate` is real.

## Retrieval-quality traces & dependency map

The search span always carries retrieval context (`db.system=veda`, `db.collection`, `retrieval.query_type` / `top_k` / `result_count` / `reranked` / `has_filter` / `index_type`). When a request sets `"explain": true`, the per-phase breakdown is attached too (`retrieval.filter_ms`, `ann_search_ms`, `scoring_ms`, `fusion_ms`, `total_ms`) — so a trace shows not just *that* a search was slow but *which phase*.

`GET /v1/admin/depmap` returns a declarative service graph — **9 nodes** (gateway, query, shards, index, embedding, llm, rerank, blob, nats) and **8 directed edges** — a stable architecture view (static topology, not discovered from traffic). The Niriksha **Dependencies** panel renders it with a RAG-trace legend.

## ANN index advisor

The index advisor closes the loop from *observing* recall and latency to *recommending* a change.

A **proxy recall probe** samples query vectors, runs the live ANN search for top-k, and compares against an exact brute-force scan over a 50k-record subset to compute `recall@k`. It is flagged **`is_proxy=true`** — recall against the index's own corpus subset, a drift signal, *not* a labeled benchmark. It runs on a **separate low-cadence task off the advisor tick** (`VEDA_RECALL_PROBE_INTERVAL_SECS`, default 600s; needs ≥50 records) because a brute-force scan must not stall the 30s tick. Gauges: `veda_index_recall{collection}`, `veda_index_p95_ms{collection}`.

Two rules join the [advisor catalog](/docs/ops-advisor#the-rule-catalog) (per-collection), flowing through the existing event store and alert channels:

| Rule | Severity | Fires when | Recommendation |
|---|---|---|---|
| `index_recall_degraded` | High | `recall < recall_min` (default 0.90) | raise `efSearch`/`efConstruction`, or reindex |
| `index_latency_high_for_recall` | Medium | `p95_ms > 50` while `recall >= 0.95` | recall is fine but costly — lower `efSearch`, or switch index type |

`GET /v1/admin/index-advisor` returns, per collection: `index_type`, `params`, `p95_ms`, `recall`, `is_proxy`, `recommendations[]`, and `open_events[]`. The Niriksha **IndexAdvisor** panel surfaces it with an opt-in reindex affordance. Recommendations are **directional hints** ("raise efSearch") today, not one-click retunes.

## REST API

All endpoints are admin-only.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/correlations` | Slow-query / error correlation, bucketed and ranked worst-first |
| `GET` | `/v1/admin/pprof` | On-demand CPU profile (SVG / pprof proto); `profiling` build; `409` if running |
| `GET` | `/v1/admin/depmap` | Declarative service dependency graph (9 nodes / 8 edges) |
| `GET` | `/v1/admin/index-advisor` | Per-collection index type / params / p95 / recall + recommendations + open events |

`POST /v1/agents/{id}/converse` and `POST /v1/admin/insights/{event_id}/triage` now also return `metrics` (`RunMetrics`) alongside their answer + step trace.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | Gates all OTLP export (spans + the new metrics) |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Head sampler strategy |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Sampler argument (e.g. ratio) |
| `VEDA_LLM_PRICES` | *(built-in default)* | JSON per-1k-token price table for cost metrics |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Recall-probe cadence (off the advisor tick; min 50 records) |

Profiling is enabled at build time: `cargo build -p veda-server --features profiling`.

## Current limitations

- **Proxy recall ≠ labeled recall@k** — the probe measures ANN top-k against a brute-force scan over a sampled subset (`is_proxy=true`); it's a drift/regression signal, not a ground-truth benchmark.
- **Recommendations are directional** — reindex doesn't yet change ANN parameters, so "raise efSearch" / "switch index type" are advisory hints.
- **`gen_ai.system` is inferred** from the model name, so exotic/self-hosted names may be mis-attributed.
- **GenAI semconv is experimental** — the OTel attribute names may shift in a future release; we track them.
- **Profiling needs load and a feature build** — empty on an idle server, and absent unless built with `--features profiling`.
- **Out of scope, deliberately:** APM agents for application languages, mobile/RUM, serverless layers, and tail-based sampling.
