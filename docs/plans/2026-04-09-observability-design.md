# Veda Observability Design

**Date:** 2026-04-09
**Status:** Approved

## Problem

1. The custom `Metrics` struct defines 5 atomic counters but no handler ever calls the recording methods — metrics are always zero.
2. Health endpoint returns hardcoded `"healthy"` without probing dependencies (MinIO, NATS, embedding services).
3. `tracing` crate is present but no `#[instrument]` annotations or spans exist — no distributed tracing.
4. `tower-http` has the `trace` feature enabled but `TraceLayer` is not wired into the router — no request logging.
5. No monitoring stack in Docker Compose — no Prometheus, Grafana, or trace collection.
6. No query profiling beyond a single `took_ms` in search responses.
7. Background operations (compaction, index rebuilds, segment seals) are invisible.

## Design

### 1. Architecture

OTel-First approach: OpenTelemetry as the single instrumentation layer for all three signals. Bundled Grafana LGTM + Mimir stack under an optional `monitoring` Docker Compose profile.

```
  Veda Server
  ┌─────────────────────────────────────┐
  │  tracing crate (#[instrument])      │──── spans ────┐
  │  metrics crate (counters/histos)    │──── /metrics ──┼──┐
  │  tracing-subscriber (JSON stdout)   │──── logs ──────┘  │
  └─────────────────────────────────────┘                   │
         │ OTLP (traces + logs)          │ Prometheus scrape │
         ▼                               ▼                  │
  ┌──────────────┐              ┌──────────────┐            │
  │  OTel        │              │  Prometheus   │◄───────────┘
  │  Collector   │              │  :9090        │
  │  :4317       │              └──────┬───────┘
  └──────┬───────┘                     │ remote-write
     ┌───┴───┐                  ┌──────▼───────┐
     ▼       ▼                  │  Mimir       │──── MinIO (veda-mimir)
┌────────┐ ┌────────┐          │  :9009       │
│ Tempo  │ │ Loki   │          └──────┬───────┘
│ :3200  │ │ :3100  │                 │
└────┬───┘ └────┬───┘─────────────────┘
     └─────┬────┘
           ▼
     ┌──────────┐
     │ Grafana  │
     │ :3000    │
     └──────────┘
```

**Data flow:**
- **Metrics:** Veda exposes `/v1/admin/metrics` (Prometheus format). Prometheus scrapes every 15s, remote-writes to Mimir for long-term storage.
- **Traces:** `tracing` spans bridged to OTel via `tracing-opentelemetry`, exported as OTLP to OTel Collector, forwarded to Tempo.
- **Logs:** `tracing` events bridged to OTel log records via `opentelemetry-appender-tracing`, exported as OTLP to OTel Collector, forwarded to Loki. Also continues to stdout (JSON).
- **Visualization:** Grafana connects to Mimir (metrics), Tempo (traces), Loki (logs). Pre-built dashboard ships with Veda.

**Vendor-neutral:** Users who prefer Datadog/Jaeger/etc. set `OTEL_EXPORTER_OTLP_ENDPOINT` to their backend and skip the `monitoring` profile.

### 2. Metrics

**Replace** the hand-rolled `Metrics` struct with `metrics` facade + `metrics-exporter-prometheus`.

**New workspace dependencies:**
- `metrics = "0.24"`
- `metrics-exporter-prometheus = "0.16"`

**Metrics exposed:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `veda_requests_total` | Counter | `method`, `endpoint`, `status` | HTTP request count |
| `veda_request_duration_seconds` | Histogram | `method`, `endpoint` | HTTP request latency |
| `veda_search_duration_seconds` | Histogram | `collection` | Search latency (end-to-end) |
| `veda_insert_total` | Counter | `collection` | Records inserted |
| `veda_insert_duration_seconds` | Histogram | `collection` | Insert latency |
| `veda_embedding_duration_seconds` | Histogram | `model`, `modality` | Embedding call latency |
| `veda_blob_operation_duration_seconds` | Histogram | `operation` (put/get/delete) | Blob store latency |
| `veda_segments_total` | Gauge | `collection`, `state` (growing/sealed) | Segment count |
| `veda_index_build_duration_seconds` | Histogram | `collection`, `index_type` | Index rebuild time |
| `veda_compaction_duration_seconds` | Histogram | `collection` | Compaction time |
| `veda_errors_total` | Counter | `type` | Error count by category |
| `veda_active_collections` | Gauge | — | Number of loaded collections |
| `veda_records_total` | Gauge | `collection` | Record count per collection |

**Implementation:**
- Record metrics in handlers using `metrics::counter!()`, `metrics::histogram!()`, `metrics::gauge!()` macros.
- Tower middleware layer for `veda_requests_total` and `veda_request_duration_seconds` — catches all HTTP traffic automatically.
- Initialize Prometheus exporter in `start_server()`, wire to existing `/v1/admin/metrics` endpoint.
- Delete old `Metrics` struct and its tests.

### 3. Health Checks

**Replace** hardcoded health endpoint with real dependency probing.

**`GET /v1/admin/health`** response:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "minio": "healthy",
    "nats": "healthy",
    "embedding": "healthy",
    "multimodal_embedding": "unavailable"
  }
}
```

**Status logic:**
- `"healthy"` — all critical deps (MinIO, NATS) up; all optional deps up.
- `"degraded"` — all critical deps up; one or more optional deps down.
- `"unhealthy"` — any critical dep down.

**Probes (2-second timeout each):**
- **MinIO:** HTTP GET to `{VEDA_BLOB_ENDPOINT}/minio/health/live`
- **NATS:** TCP connect to `{NATS_URL}` (port 4222)
- **Embedding:** HTTP GET to `{EMBEDDING_ENDPOINT}/health`
- **Multimodal embedding:** HTTP GET to `{MULTIMODAL_EMBEDDING_ENDPOINT}/health`

Missing/unconfigured services reported as `"unconfigured"` (not `"unavailable"`).

**`GET /v1/admin/readiness`** — same checks, returns HTTP 503 if `"unhealthy"`. For Kubernetes readiness gates.

### 4. Distributed Tracing (OpenTelemetry)

**New workspace dependencies:**
- `opentelemetry = "0.27"`
- `opentelemetry_sdk = "0.27"`
- `opentelemetry-otlp = "0.27"`
- `tracing-opentelemetry = "0.28"`
- `opentelemetry-appender-tracing = "0.27"`

**Tracing subscriber setup** (in `veda-server/src/main.rs`):

Replace current `tracing_subscriber::fmt().json().init()` with layered subscriber:
1. **fmt layer** — JSON to stdout (keeps current behavior)
2. **env-filter layer** — `RUST_LOG` control (keeps current behavior)
3. **OTel trace layer** — `tracing-opentelemetry` bridges spans to OTel traces, exports via OTLP
4. **OTel log layer** — `opentelemetry-appender-tracing` bridges events to OTel log records, exports via OTLP

OTel layers are **conditional** — only initialized if `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Without it, Veda behaves exactly as today. Zero overhead when disabled.

**Instrumented paths:**

| Handler/Function | Span Name | Key Fields |
|---|---|---|
| `search()` | `search` | `collection`, `top_k`, `has_filter` |
| `insert()` | `insert` | `collection`, `record_count` |
| `embed_insert()` | `embed_insert` | `collection`, `record_count`, `model` |
| `process_document()` | `process_document` | `collection`, `file_name`, `chunking_strategy` |
| `EmbeddingClient::embed_text()` | `embed_text` | `model`, `input_len` |
| `EmbeddingClient::embed_bytes()` | `embed_bytes` | `model`, `modality`, `input_size` |
| `ObjectBlobStore::put/get/delete` | `blob_{op}` | `key`, `size` |
| `SegmentManager::insert()` | `segment_insert` | `collection`, `shard_id` |
| `SearchExecutor::execute()` | `execute_search` | `collection`, `index_type` |
| `seal_growing()` | `seal_segment` | `collection`, `shard_id` |
| `rebuild_executors()` | `rebuild_executors` | `collection` |

**Trace context propagation:** Extract `traceparent` header from incoming HTTP requests (W3C Trace Context).

**Configuration:**

| Env Var | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — (disabled) | OTLP endpoint. Set to enable tracing+log export |
| `OTEL_SERVICE_NAME` | `veda` | Service name in traces |
| `OTEL_TRACES_SAMPLER` | `parentbased_always_on` | Sampling strategy |

### 5. Query Explain

Opt-in latency breakdown in search responses when `"explain": true` is passed.

**Request:**
```json
{
  "query": {"vector": {"field": "embedding", "values": [0.1, ...]}},
  "limit": 10,
  "filter": "category = 'electronics'",
  "explain": true
}
```

**Response:**
```json
{
  "results": [...],
  "took_ms": 3,
  "explain": {
    "filter_ms": 0.4,
    "ann_search_ms": 1.8,
    "scoring_ms": 0.2,
    "fusion_ms": 0.1,
    "total_ms": 2.5
  }
}
```

When `"explain"` is omitted or `false`, the `explain` field is absent. Timing code gated behind the flag — no overhead in normal operation.

**Implementation:** Pass `explain` flag through to `SearchExecutor`. Each phase wraps work in `Instant::now()` / `elapsed()` only when explain is true. Timings collected in `ExplainTiming` struct.

### 6. HTTP Request Logging Middleware

Wire `tower-http::TraceLayer` into the router (dependency and feature flag already exist).

Logs every HTTP request with: method, URI, status code, duration, request ID (generated UUID, set as `x-request-id` response header), trace ID (from OTel context when enabled).

`/v1/admin/metrics` and `/v1/admin/health` excluded from request logging.

### 7. Async Operation Visibility

Background operations get their own metrics and root trace spans (not attached to any request).

**Metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `veda_compaction_in_progress` | Gauge | `collection` | 1 if compacting, 0 otherwise |
| `veda_compaction_duration_seconds` | Histogram | `collection` | Time per compaction run |
| `veda_seal_duration_seconds` | Histogram | `collection`, `shard_id` | Time to seal a growing segment |
| `veda_index_rebuild_duration_seconds` | Histogram | `collection`, `index_type` | Time to rebuild query executors |
| `veda_index_rebuild_in_progress` | Gauge | `collection` | 1 if rebuilding, 0 otherwise |

Traces appear as independent root spans in Tempo, searchable by `operation.type`.

### 8. Docker Compose Monitoring Stack

All under `profiles: ["monitoring"]`.

**Services:**

| Service | Image | Port | Role |
|---|---|---|---|
| `otel-collector` | `otel/opentelemetry-collector-contrib` | 4317 | Receives OTLP traces+logs from Veda, routes to Tempo+Loki |
| `prometheus` | `prom/prometheus` | 9090 | Scrapes Veda metrics, remote-writes to Mimir |
| `mimir` | `grafana/mimir` | 9009 | Long-term metrics storage, uses MinIO bucket `veda-mimir` |
| `tempo` | `grafana/tempo` | 3200 | Trace storage, receives OTLP from collector |
| `loki` | `grafana/loki` | 3100 | Log storage, receives OTLP from collector |
| `grafana` | `grafana/grafana` | 3000 | Dashboards, pre-configured datasources |

**Configuration files** (`config/monitoring/`):
- `otel-collector.yaml` — receivers (OTLP), exporters (Tempo, Loki), pipeline config
- `prometheus.yml` — scrape config for Veda, remote-write to Mimir
- `mimir.yml` — single-binary mode, S3 backend pointing at MinIO `veda-mimir` bucket
- `tempo.yml` — local storage, OTLP receiver
- `loki.yml` — local storage, OTLP receiver
- `grafana/provisioning/datasources.yaml` — auto-provision Mimir, Tempo, Loki
- `grafana/provisioning/dashboards/veda.json` — pre-built dashboard

**Pre-built Grafana dashboard panels:**
- Request rate and error rate
- Request latency p50/p95/p99
- Search latency by collection
- Insert throughput by collection
- Embedding call latency by model/modality
- Blob store operation latency
- Active collections and record counts
- Segment counts (growing vs sealed)
- Health status overview
- Background operations (compaction/seal/rebuild activity and duration)

**Veda env vars** (set in docker-compose when monitoring profile active):
- `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`

**Storage:** Mimir uses MinIO (existing, new `veda-mimir` bucket). Tempo and Loki use local Docker volumes.

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `metrics`, `metrics-exporter-prometheus`, `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`, `opentelemetry-appender-tracing` |
| `crates/veda-core/Cargo.toml` | Add `metrics` |
| `crates/veda-core/src/metrics.rs` | Delete old `Metrics` struct, re-export `metrics` crate macros |
| `crates/veda-core/tests/metrics_test.rs` | Rewrite for new metrics |
| `crates/veda-gateway/Cargo.toml` | Add `metrics`, `opentelemetry` (for trace context extraction) |
| `crates/veda-gateway/src/rest/mod.rs` | Add TraceLayer + metrics middleware to router |
| `crates/veda-gateway/src/rest/admin.rs` | Rewrite health, add readiness, update metrics endpoint |
| `crates/veda-gateway/src/rest/search.rs` | Add `#[instrument]`, explain support, record metrics |
| `crates/veda-gateway/src/rest/records.rs` | Add `#[instrument]`, record metrics |
| `crates/veda-gateway/src/rest/embed_insert.rs` | Add `#[instrument]`, record metrics |
| `crates/veda-gateway/src/rest/document.rs` | Add `#[instrument]`, record metrics |
| `crates/veda-core/src/embedding_client.rs` | Add `#[instrument]`, record embedding latency |
| `crates/veda-storage/src/blob_store.rs` | Add `#[instrument]`, record blob op latency |
| `crates/veda-server/Cargo.toml` | Add `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`, `opentelemetry-appender-tracing` |
| `crates/veda-server/src/main.rs` | Layered tracing subscriber with conditional OTel |
| `docker-compose.yml` | Add 6 monitoring services under `monitoring` profile |
| `config/monitoring/otel-collector.yaml` | New: OTel Collector config |
| `config/monitoring/prometheus.yml` | New: Prometheus scrape + remote-write config |
| `config/monitoring/mimir.yml` | New: Mimir single-binary config |
| `config/monitoring/tempo.yml` | New: Tempo config |
| `config/monitoring/loki.yml` | New: Loki config |
| `config/monitoring/grafana/provisioning/datasources.yaml` | New: Grafana datasource provisioning |
| `config/monitoring/grafana/provisioning/dashboards/veda.json` | New: Pre-built Grafana dashboard |

## What's NOT Changing

- REST API contract (all existing endpoints unchanged)
- Auth system (JWT, RBAC)
- Index, storage, query, coordinator crates (only adding `#[instrument]` annotations)
- Existing Docker services (MinIO, NATS, embedding, OCR)
- Default behavior without `monitoring` profile or `OTEL_EXPORTER_OTLP_ENDPOINT`
