# Observability & Telemetry Research for Vector Databases

**Date:** 2026-04-09

---

## 1. Competitor Analysis

### 1.1 Qdrant

**Metrics:** Prometheus/OpenMetrics via `/metrics` endpoint (port 6333) on every node. Qdrant Cloud adds `/sys_metrics` for cluster-wide system metrics (CPU, memory, disk). Hardware utilization can be included in API responses via `hardware_reporting` config. Metrics prefix is configurable.

**Tracing:** No built-in distributed tracing. Client-side OTel instrumentation available via third-party OpenLLMetry. OTel-compatible data export announced for Qdrant Cloud (March 2025).

**Logging:** Standard structured logging. `/metrics` endpoint excluded from request logs to avoid overhead.

**Dashboards:** Pre-built Grafana dashboard provided. Integrations with Datadog, Grafana Cloud.

**Alerting:** Via Grafana, Prometheus Alertmanager, or Datadog -- not built-in.

**Shortcomings:** Self-hosted users lack system-level metrics (`/sys_metrics` is Cloud-only). Per-node scraping required even behind load balancers. No server-side distributed tracing -- relies entirely on client-side instrumentation.

### 1.2 Weaviate

**Metrics:** Prometheus endpoint at `<host>:2112/metrics`. Covers request latencies, import speed, vector/object storage time, memory usage. v1.34 added 30+ new metrics (LSM operations, WAL recovery, memtable flushing, async replication). Multi-tenant metrics can be grouped via `PROMETHEUS_MONITORING_GROUP`.

**Tracing:** No built-in tracing. Client-side OTel via OpenLLMetry's `opentelemetry-instrumentation-weaviate` package.

**Logging:** Standard logging; no documented structured JSON format.

**Dashboards:** Example Grafana dashboards in their GitHub repo. Integrations with Datadog, Dynatrace, New Relic, Grafana Cloud.

**Alerting:** Via external tools (Prometheus, Datadog).

**Shortcomings:** Metrics do NOT follow semver -- breaking metric changes can land in minor releases. Users have filed issues requesting async indexing queue metrics (GitHub #4819). The fact that 30+ metrics were added in a single v1.34 release indicates significant prior gaps. No native tracing.

### 1.3 Milvus / Zilliz

**Metrics:** Prometheus at `<component>:9091/metrics` for each component (proxy, datanode, indexnode, querynode). Helm chart supports `ServiceMonitor` for auto-discovery. Covers query latency, import speed, memory, cache hit ratios, proxy queue wait times, JSON filter performance.

**Tracing:** Partial. Recent releases improved trace observability for scalar expressions and added gRPC latency metrics. No full OpenTelemetry integration yet -- the 2025 roadmap mentions exploring OTel+Jaeger for correlating DB perf with upstream services.

**Logging:** Grafana Loki integration for log aggregation. Promtail collects and labels logs. Structured log streams indexed by labels (not full-text).

**Dashboards:** Pre-built Grafana dashboards. Integrations with Datadog, New Relic, Dynatrace, IBM Instana.

**Alerting:** Via Grafana alerts or Prometheus Alertmanager.

**Shortcomings:** Most complex setup -- many components to monitor independently. Label cardinality explosions reported with dynamic client IDs in `milvus_query_type`. CVE-2026-26190 (CVSS 9.8) revealed unauthenticated access on the metrics port -- a serious security gap. No built-in distributed tracing across components. Consistency mode affects metric accuracy.

### 1.4 Pinecone

**Metrics:** Console metrics dashboard built-in. Prometheus endpoints for Standard/Enterprise plans (with HTTP service discovery across regions). 25+ metrics via Datadog integration covering request count, latency, read/write units, index fullness, vector counts.

**Tracing:** No server-side tracing (managed service). Java SDK added client-side observability. OTel client wrapping via third-party tools.

**Logging:** No logs exposed to users. Metrics-only observability model.

**Dashboards:** Built-in console metrics tab. Pre-built dashboards in Grafana Cloud, Datadog, New Relic, Splunk.

**Alerting:** Via Datadog monitors, Grafana alerts, or New Relic.

**Shortcomings:** No log access at all. No tracing. Prometheus only on paid plans. Limited visibility into internal operations. Vendor lock-in makes it hard to build custom observability.

### 1.5 ChromaDB

**Metrics:** No built-in Prometheus endpoint. No native metrics.

**Tracing:** Native OpenTelemetry export via `CHROMA_OPEN_TELEMETRY__ENDPOINT` config. Sends traces to any OTel-compatible backend (Honeycomb, Jaeger, Grafana Tempo). This is Chroma's primary observability mechanism.

**Logging:** Standard Python logging. No structured format documented.

**Dashboards:** None built-in. Third-party via OpenLLMetry/OpenLIT integrations.

**Shortcomings:** Observability is minimal. No Prometheus metrics. OTel support sends traces but lacks metrics breadth. Designed for single-node; observability tooling reflects that scope.

### 1.6 LanceDB

**Metrics:** No Prometheus endpoint. Added `explain_plan` and `analyze_plan` for merge-insert operations with cumulative CPU time per operator. New FTS tuning knobs and metrics.

**Tracing:** No built-in tracing. Client-side via Dynatrace/OpenLLMetry (OTel-based) and HoneyHive decorator-based tracing for RAG pipelines.

**Dashboards:** None built-in. Dynatrace provides cost/latency/error dashboards.

**Shortcomings:** Youngest product -- observability is thinnest. Query plan analysis is useful but no standard metrics endpoint. Relies entirely on third-party integrations and client-side instrumentation.

### Competitor Summary Table

| Capability | Qdrant | Weaviate | Milvus | Pinecone | Chroma | Lance |
|---|---|---|---|---|---|---|
| Prometheus `/metrics` | Yes | Yes | Yes (per component) | Paid plans | No | No |
| OpenTelemetry | Cloud-only, client-side | Client-side only | Roadmap | No | **Yes (native export)** | No |
| Grafana dashboards | Pre-built | Pre-built | Pre-built | Built-in console | No | No |
| Server-side tracing | No | No | No | No | Partial | No |
| Query explain/profile | No | No | No | No | No | `explain_plan` |

### Common User Complaints Across Competitors

- No server-side distributed tracing (everyone relies on client-side instrumentation)
- No index build progress visibility
- No query profiling/explain plans
- Metrics breaking between versions (Weaviate)
- Unauthenticated metrics endpoints (Milvus CVE-2026-26190)
- No async operation visibility (compaction, replication queues)

---

## 2. Open-Source Observability Ecosystem (Rust)

### 2.1 OpenTelemetry (OTEL) Rust SDK

**Maturity:** Actively maintained. Metrics and Logs SDKs are **Stable** (as of 0.30 release). Traces remain **Beta**. Minimum supported Rust version: 1.75.

**Core crates:**

| Crate | Purpose |
|---|---|
| `opentelemetry` | API crate (context, baggage, propagators, metrics, tracing, logging bridge) |
| `opentelemetry_sdk` | SDK implementation (133M+ all-time downloads) |
| `opentelemetry-otlp` | OTLP exporter for logs, metrics, and traces |
| `tracing-opentelemetry` | Bridge from Rust's `tracing` spans to OTel spans |
| `opentelemetry-appender-tracing` | Bridge from `tracing` events to OTel log records |

**Limitations:** Version churn -- `tracing-opentelemetry` 0.26 aligns with `opentelemetry` 0.25, breaking changes still occur. Trace signal not yet stable. Log export requires a separate appender crate (not via `tracing-opentelemetry`).

**OTLP backends:** OTel Collector, Grafana Tempo, Jaeger v2, SigNoz, Uptrace, Datadog, New Relic, Google Cloud -- virtually every modern backend accepts OTLP natively.

### 2.2 Prometheus Metrics

**Two Rust ecosystems:**

| Approach | Crate | Notes |
|---|---|---|
| Prometheus-specific | `prometheus` (by TiKV) | Battle-tested, used by TiKV itself. Direct `Registry` control. |
| Generic facade | `metrics` + `metrics-exporter-prometheus` | Ergonomic facade (mirrors `tracing` for logs). Backend-agnostic. |
| Bridge | `metrics-prometheus` | Combines both ecosystems in one binary. |
| Official client | `prometheus/client_rust` | Official Prometheus Rust client, OpenMetrics-compatible. |

**Pull vs push:** Prometheus uses a pull model (scrapes `/metrics` every 15-30s). This is standard for databases.

**Key metric types for a database:**
- Histograms: query/write latency, p99s
- Gauges: active connections, memory usage, segment counts, shard states
- Counters: total queries, inserts, errors, bytes read/written

### 2.3 Grafana LGTM Stack

The **Grafana + Prometheus + Loki + Tempo** stack is the de facto open-source observability stack:

| Component | Role |
|---|---|
| Prometheus | Metrics storage + PromQL queries |
| Loki | Log aggregation (indexes labels only, 10x cheaper than Elasticsearch) |
| Tempo | Distributed trace backend, accepts OTLP natively |
| Grafana | Unified visualization with cross-signal correlation |
| Grafana Alloy | Unified telemetry collector/pipeline (replaces Promtail) |

These integrate tightly: same label sets across metrics, logs, and traces enable drill-down from a dashboard panel to specific log lines to individual trace spans. Pre-built dashboards exist for PostgreSQL, MySQL, MongoDB, Redis on Grafana's marketplace -- patterns (query latency histograms, connection gauges, throughput counters) transfer directly.

### 2.4 Tracing in Rust

The `tracing` crate is the de facto standard for Rust instrumentation:

| Crate | Role |
|---|---|
| `tracing` | Core: spans, events, structured fields |
| `tracing-subscriber` | Composable layer system (fmt, OTel, env-filter) |
| `tracing-opentelemetry` | Bridges `tracing` spans to OTel spans |
| `opentelemetry-appender-tracing` | Bridges `tracing` events to OTel log records |
| `tracing-loki` | Ships logs directly to Loki (pre-1.0, rough edges) |

**Best practices for a database:**
1. Use `#[instrument]` on public API methods with structured fields (`collection.name`, `query.vector_dim`, `error.code`)
2. Layer ordering: place env-filter before expensive export layers
3. Always use batch span processors in production
4. Use environment-based filtering (`RUST_LOG`) to control verbosity per module

### 2.5 Alternatives

- **Jaeger v2:** Built on OTel Collector. Solid for trace visualization but traces-only. Accepts OTLP natively.
- **SigNoz:** Open-source, OTel-native, ClickHouse backend. Strong alternative to Grafana stack as single-binary solution.
- **Datadog/New Relic:** Support OTLP ingestion. Expensive at scale. Rust works via OTLP export.
- **ClickHouse for observability storage:** Used by SigNoz and Uptrace. 10-20x compression vs Elasticsearch.

### Recommended Stack for Veda

| Signal | Instrument With | Export Via | Store In |
|---|---|---|---|
| Metrics | `metrics` + `metrics-exporter-prometheus` | Prometheus scrape | Prometheus |
| Traces | `tracing` + `tracing-opentelemetry` | OTLP | Grafana Tempo |
| Logs | `tracing` + `opentelemetry-appender-tracing` | OTLP | Grafana Loki |
| Visualization | -- | -- | Grafana dashboards |

---

## 3. Veda's Current State

### Audit Summary

| Component | Status | Detail |
|---|---|---|
| Metrics | Defined, never recorded | Custom `Metrics` struct with Prometheus export at `/v1/admin/metrics`, but handlers never call `record_query()`/`record_insert()` -- always zero |
| Logging | Infrastructure only | `tracing` + `tracing-subscriber` with JSON output, but only 3 log statements total in main.rs |
| Distributed Tracing | Not wired | `tracing` crate present, but no `#[instrument]`, no OpenTelemetry, no spans |
| Health Check | Hardcoded | `/v1/admin/health` always returns `"healthy"` without checking MinIO/NATS/embedding |
| Query Profiling | Minimal | Search returns `took_ms` but no per-component breakdown |
| Monitoring Stack | None | No Prometheus, Grafana, or collector in Docker Compose |

### Existing Infrastructure

**Metrics (`crates/veda-core/src/metrics.rs`):**
- Custom `Metrics` struct using atomic counters
- Prometheus text format exporter via `export_prometheus()`
- Metrics defined: `veda_query_total`, `veda_query_latency_microseconds_total`, `veda_insert_total`, `veda_insert_latency_microseconds_total`, `veda_errors_total`
- Endpoint: `GET /v1/admin/metrics` (behind auth)
- **Problem:** No handler ever calls the recording methods

**Logging (`crates/veda-server/src/main.rs`):**
- `tracing` v0.1 + `tracing-subscriber` v0.3 with JSON feature
- `RUST_LOG` env var supported (default: "info")
- Only 3 log statements in entire codebase

**Health (`crates/veda-gateway/src/rest/admin.rs`):**
- `GET /v1/admin/health` returns hardcoded `{"status": "healthy", "version": "0.1.0"}`

**Query timing:**
- `search()` handler measures `Instant::now()` and returns `took_ms` in response
- No timing for inserts, embedding calls, blob operations, or index builds

---

## 4. Opportunities & Recommendations

### Differentiation Opportunity

**No vector database has proper server-side distributed tracing.** This is the single biggest gap in the market. Veda can be the first to ship native OpenTelemetry traces from query planning through index search to embedding generation -- giving operators trace-level visibility into every request.

### Implementation Roadmap

#### Phase 1 -- Foundation (high impact, low effort)

1. **Wire up existing metrics** -- The `Metrics` struct and `/v1/admin/metrics` endpoint already exist. Add `record_query()`/`record_insert()`/`record_error()` calls in handlers. Add histograms for latency (not just counters).

2. **Real health checks** -- Make `/v1/admin/health` actually probe MinIO, NATS, and embedding endpoints. Add `/v1/admin/readiness` for Kubernetes.

3. **HTTP request logging middleware** -- `tower-http` trace feature is already in `Cargo.toml` but not wired. One line of middleware gives structured request/response logging.

#### Phase 2 -- Prometheus & Grafana (table stakes)

4. **Upgrade to `metrics` crate** -- Replace hand-rolled `Metrics` with `metrics` facade + `metrics-exporter-prometheus`. Proper histograms (p50/p95/p99), labeled metrics, industry-standard exposition.

5. **Add Prometheus + Grafana to Docker Compose** -- Optional `monitoring` profile. Ship pre-built Grafana dashboard covering: query latency, insert throughput, error rate, segment counts, memory, active connections.

6. **Database-specific metrics** -- Index build progress, segment seal/compact times, embedding call latency, blob store operation latency, shard distribution.

#### Phase 3 -- OpenTelemetry Tracing (differentiation)

7. **`#[instrument]` on all handlers and hot paths** -- search, insert, embed-insert, process-document, embedding client calls, blob store operations. Structured span fields: `collection.name`, `query.top_k`, `modality`, `model.name`.

8. **OTLP export** -- Add `tracing-opentelemetry` + `opentelemetry-otlp`. Users set `OTEL_EXPORTER_OTLP_ENDPOINT` and traces flow to Tempo/Jaeger/Datadog/whatever they use.

9. **Query explain** -- Return optional trace breakdown in search responses: filter eval time, ANN search time, fusion time, total. No competitor does this well.

#### Phase 4 -- Full Stack (polish)

10. **Grafana Tempo + Loki** in Docker Compose `monitoring` profile for logs+traces alongside metrics.

11. **Authenticated metrics endpoint** -- Already behind `/v1/admin/` which supports auth. Avoid Milvus CVE mistake.

12. **Async operation visibility** -- Metrics/traces for compaction, index rebuilds, segment seals. Invisible in every competitor and a common complaint.

---

## Sources

- [Qdrant Monitoring & Telemetry](https://qdrant.tech/documentation/operations/monitoring/)
- [Qdrant Observability](https://qdrant.tech/documentation/observability/)
- [Weaviate Monitoring Docs](https://docs.weaviate.io/deploy/configuration/monitoring)
- [Weaviate v1.34 Release](https://weaviate.io/blog/weaviate-1-34-release)
- [Milvus Monitor Overview](https://milvus.io/docs/monitor_overview.md)
- [Zilliz Blog - Monitoring with Grafana & Loki](https://zilliz.com/blog/monitoring-in-milvus-with-grafana-and-loki)
- [Pinecone Monitor Performance](https://docs.pinecone.io/guides/production/monitoring)
- [Datadog Pinecone Integration](https://www.datadoghq.com/blog/monitor-pinecone-vector-databases-datadog/)
- [Chroma Observability Docs](https://docs.trychroma.com/guides/deploy/observability)
- [LanceDB Dynatrace Integration](https://www.dynatrace.com/hub/detail/lancedb/)
- [OpenTelemetry Rust docs](https://opentelemetry.io/docs/languages/rust/)
- [opentelemetry-rust GitHub](https://github.com/open-telemetry/opentelemetry-rust)
- [TiKV rust-prometheus GitHub](https://github.com/tikv/rust-prometheus)
- [Grafana LGTM Stack](https://dasroot.net/posts/2026/04/observability-stack-prometheus-grafana-loki/)
- [SigNoz OpenTelemetry Rust guide](https://signoz.io/blog/opentelemetry-rust/)
