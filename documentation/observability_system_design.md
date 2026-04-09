# Observability System Design

## 1. Overview

Veda's observability system provides full-stack monitoring across three telemetry signals: **metrics**, **traces**, and **logs**. The system follows an OpenTelemetry-first approach, using the `metrics` crate facade for Prometheus-compatible metrics and the `tracing` crate with optional OTel bridging for distributed traces and structured logs.

**Key design principles:**
- **Zero overhead when disabled** — OTel trace and log export only activates when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Without it, Veda behaves identically to a non-instrumented build.
- **Vendor-neutral** — OTLP export works with any compatible backend (Grafana stack, Datadog, Jaeger, etc.). The bundled monitoring stack is optional.
- **Opt-in depth** — Query explain timing is only computed when `"explain": true` is passed. Health probes run per-request but are lightweight.

---

## 2. Architecture

```
  Veda Server (:6340)
  ┌─────────────────────────────────────────┐
  │  metrics crate (counters/histograms)    │──── /v1/admin/metrics ──┐
  │  tracing crate (#[instrument] spans)    │──── OTLP spans ────────┤
  │  tracing-subscriber (JSON stdout)       │──── OTLP logs ─────────┘
  └─────────────────────────────────────────┘         │
                                                      │
           Prometheus scrape (every 15s)        OTLP gRPC (:4317)
                    │                                 │
                    ▼                                 ▼
            ┌──────────────┐                 ┌──────────────────┐
            │  Prometheus   │                 │  OTel Collector   │
            │  :9090        │                 │  :4317 / :4318   │
            └──────┬───────┘                 └────────┬─────────┘
                   │ remote-write                 ┌───┴───┐
            ┌──────▼───────┐                     ▼       ▼
            │  Mimir        │               ┌────────┐ ┌────────┐
            │  :9009        │               │ Tempo  │ │ Loki   │
            │  (MinIO S3)   │               │ :3200  │ │ :3100  │
            └──────┬───────┘               └────┬───┘ └────┬───┘
                   └──────────────┬──────────────┘          │
                                  ▼                         │
                            ┌──────────┐                    │
                            │ Grafana  │◄───────────────────┘
                            │ :3000    │
                            └──────────┘
```

**Data flow:**
- **Metrics:** Veda exposes `/v1/admin/metrics` in Prometheus text format. Prometheus scrapes every 15s, remote-writes to Mimir for long-term storage backed by MinIO.
- **Traces:** `tracing` spans bridged to OTel via `tracing-opentelemetry`, exported as OTLP gRPC to OTel Collector, forwarded to Tempo.
- **Logs:** `tracing` events bridged to OTel log records via `opentelemetry-appender-tracing`, exported as OTLP to OTel Collector, forwarded to Loki. JSON output to stdout continues regardless.
- **Visualization:** Grafana connects to Mimir (metrics), Tempo (traces), Loki (logs). A pre-built dashboard ships with Veda.

---

## 3. Metrics

### 3.1 Instrumentation Layer

Veda uses the `metrics` crate (v0.24) as a recording facade with `metrics-exporter-prometheus` (v0.16) as the backend. The global recorder is installed once at server startup in `veda-gateway/src/server.rs`:

```rust
let prometheus_handle = PrometheusBuilder::new()
    .install_recorder()
    .expect("failed to install Prometheus recorder");
```

The `PrometheusHandle` is stored in `AppState` and used by the `/v1/admin/metrics` endpoint to render the Prometheus text exposition format.

### 3.2 Metric Constants

All metric names are defined as `pub const` in `veda-core/src/metrics.rs` (16 constants). Every name uses the `veda_` prefix and follows Prometheus naming conventions (snake_case, units in suffix).

| Constant | Metric Name | Type | Description |
|---|---|---|---|
| `REQUESTS_TOTAL` | `veda_requests_total` | Counter | HTTP request count |
| `REQUEST_DURATION` | `veda_request_duration_seconds` | Histogram | HTTP request latency |
| `SEARCH_DURATION` | `veda_search_duration_seconds` | Histogram | End-to-end search latency |
| `INSERT_TOTAL` | `veda_insert_total` | Counter | Records inserted |
| `INSERT_DURATION` | `veda_insert_duration_seconds` | Histogram | Insert operation latency |
| `EMBEDDING_DURATION` | `veda_embedding_duration_seconds` | Histogram | Embedding HTTP call latency |
| `BLOB_OP_DURATION` | `veda_blob_operation_duration_seconds` | Histogram | Blob store operation latency |
| `SEGMENTS_TOTAL` | `veda_segments_total` | Gauge | Segment count |
| `INDEX_BUILD_DURATION` | `veda_index_build_duration_seconds` | Histogram | Index rebuild time |
| `COMPACTION_DURATION` | `veda_compaction_duration_seconds` | Histogram | Compaction time |
| `ERRORS_TOTAL` | `veda_errors_total` | Counter | Error count by category |
| `ACTIVE_COLLECTIONS` | `veda_active_collections` | Gauge | Number of loaded collections |
| `RECORDS_TOTAL` | `veda_records_total` | Gauge | Record count per collection |
| `COMPACTION_IN_PROGRESS` | `veda_compaction_in_progress` | Gauge | 1 if compacting, 0 otherwise |
| `INDEX_REBUILD_IN_PROGRESS` | `veda_index_rebuild_in_progress` | Gauge | 1 if rebuilding, 0 otherwise |
| `SEAL_DURATION` | `veda_seal_duration_seconds` | Histogram | Segment seal time |

### 3.3 Recording Locations

Metrics are recorded at two levels: a global HTTP middleware layer for request-level metrics, and inline in handlers/services for domain-specific metrics.

#### HTTP Request Metrics (middleware)

**File:** `veda-gateway/src/rest/metrics_middleware.rs`

The `request_metrics()` middleware wraps every HTTP request and records:

| Metric | Labels | Notes |
|---|---|---|
| `veda_requests_total` | `method`, `endpoint`, `status` | Incremented by 1 per request |
| `veda_request_duration_seconds` | `method`, `endpoint` | Elapsed wall-clock time in seconds |

**Exclusions:** The middleware skips `/v1/admin/health`, `/v1/admin/metrics`, and `/v1/admin/readiness` to avoid self-referential noise in dashboards.

#### Search Handler

**File:** `veda-gateway/src/rest/search.rs`

| Metric | Labels | When |
|---|---|---|
| `veda_search_duration_seconds` | `collection` | After every search (single or multi-vector) |

#### Insert / Embed-Insert Handlers

**Files:** `veda-gateway/src/rest/records.rs`, `veda-gateway/src/rest/embed_insert.rs`

Both handlers record identical metrics after the write path completes:

| Metric | Labels | When |
|---|---|---|
| `veda_seal_duration_seconds` | `collection` | After sealing growing segments |
| `veda_index_rebuild_in_progress` | `collection` | Set to 1.0 before rebuild, 0.0 after |
| `veda_index_build_duration_seconds` | `collection`, `index_type` | After rebuilding query executors |
| `veda_insert_duration_seconds` | `collection` | Total insert wall-clock time |
| `veda_insert_total` | `collection` | Incremented by number of records inserted |

#### Embedding Client

**File:** `veda-core/src/embedding_client.rs`

| Metric | Labels | When |
|---|---|---|
| `veda_embedding_duration_seconds` | `model`, `modality` | After each HTTP embedding call (text or bytes) |

Recorded in both `embed_text_http()` (modality: `"text"`) and `embed_bytes_http()` (modality: runtime-detected).

#### Blob Store

**File:** `veda-storage/src/blob_store.rs`

| Metric | Labels | When |
|---|---|---|
| `veda_blob_operation_duration_seconds` | `operation` | After each put/get/delete on `ObjectBlobStore` |

The `operation` label is one of `"put"`, `"get"`, or `"delete"`.

### 3.4 Prometheus Endpoint

**Endpoint:** `GET /v1/admin/metrics`
**Authentication:** Behind auth middleware when auth is enabled
**Content-Type:** `text/plain; version=0.0.4; charset=utf-8`

The handler calls `prometheus_handle.render()` to produce the full Prometheus text exposition. If no recorder is installed (e.g., in tests), returns `# metrics recorder not installed`.

### 3.5 Metrics Validation Test

**File:** `veda-core/tests/metrics_test.rs`

A single test (`test_metrics_names_are_valid_prometheus_format`) validates all 16 metric names conform to Prometheus naming rules:
- Not empty
- Starts with a letter or underscore
- Contains only alphanumeric characters and underscores

---

## 4. Distributed Tracing

### 4.1 Tracing Subscriber Setup

**File:** `veda-server/src/main.rs`

The tracing subscriber is a layered stack built at startup. The stack composition depends on whether `OTEL_EXPORTER_OTLP_ENDPOINT` is set:

**Without OTel (default):**
```
registry
  └── EnvFilter (RUST_LOG, default: "info")
  └── fmt layer (JSON to stdout)
```

**With OTel:**
```
registry
  └── EnvFilter (RUST_LOG, default: "info")
  └── fmt layer (JSON to stdout)
  └── OTel trace layer (tracing-opentelemetry)
  └── OTel log layer (opentelemetry-appender-tracing)
```

### 4.2 OTel Export Configuration

| Component | Crate | Protocol | Destination |
|---|---|---|---|
| Trace exporter | `opentelemetry-otlp` (SpanExporter) | gRPC (Tonic) | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| Log exporter | `opentelemetry-otlp` (LogExporter) | gRPC (Tonic) | `OTEL_EXPORTER_OTLP_ENDPOINT` |

Both exporters use batch processing (`with_batch_exporter`) with the Tokio runtime for async flush. The `service.name` OTel resource attribute is set from `OTEL_SERVICE_NAME` (default: `"veda"`).

### 4.3 Instrumented Functions

All `#[instrument]` annotations use `skip` to avoid logging large request/response bodies and `fields` to capture structured trace attributes:

| Function | Span Name | Fields | File |
|---|---|---|---|
| `search()` | `search` | `collection`, `top_k`, `has_filter` | `veda-gateway/src/rest/search.rs:97` |
| `insert()` | `insert` | `collection`, `record_count` | `veda-gateway/src/rest/records.rs:52` |
| `embed_insert()` | `embed_insert` | `collection`, `record_count` | `veda-gateway/src/rest/embed_insert.rs:49` |
| `process_document()` | `process_document` | `collection` | `veda-gateway/src/rest/document.rs:64` |
| `process_documents_batch()` | `process_documents_batch` | `collection` | `veda-gateway/src/rest/document.rs:176` |
| `embed_text()` | `embed_text` | `model`, `input_len` | `veda-core/src/embedding_client.rs:24` |
| `embed_bytes()` | `embed_bytes` | `model`, `modality`, `input_size` | `veda-core/src/embedding_client.rs:36` |
| `ObjectBlobStore::put()` | `blob_put` | `key`, `size` | `veda-storage/src/blob_store.rs:211` |
| `ObjectBlobStore::get()` | `blob_get` | `key` | `veda-storage/src/blob_store.rs:229` |
| `ObjectBlobStore::delete()` | `blob_delete` | `key` | `veda-storage/src/blob_store.rs:245` |

### 4.4 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset — disabled)* | OTLP endpoint. Set to enable trace+log export |
| `OTEL_SERVICE_NAME` | `veda` | Service name in traces |
| `RUST_LOG` | `info` | Log level filter (applies to both stdout and OTel) |

---

## 5. Health Checks

### 5.1 Endpoints

| Method | Path | Auth | HTTP Status |
|---|---|---|---|
| GET | `/v1/admin/health` | Public | Always 200 |
| GET | `/v1/admin/readiness` | Public | 200 if healthy/degraded, 503 if unhealthy |

Both endpoints return the same `HealthResponse` structure:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "minio": "healthy",
    "nats": "healthy",
    "embedding": "healthy",
    "multimodal_embedding": "unconfigured"
  }
}
```

### 5.2 Dependency Probes

**File:** `veda-gateway/src/rest/admin.rs` — `run_health_checks()`

Each probe has a 2-second timeout. A shared `reqwest::Client` (with `no_proxy()`) is constructed per invocation.

| Dependency | Type | Probe Method | Env Var | Status Values |
|---|---|---|---|---|
| MinIO | Critical | HTTP GET `{endpoint}/minio/health/live` | `VEDA_BLOB_ENDPOINT` | healthy, unavailable, unconfigured |
| NATS | Critical | TCP connect (strips `nats://` prefix) | `VEDA_MQ_ENDPOINT` | healthy, unavailable, unconfigured |
| Embedding | Optional | HTTP GET `{endpoint}/health` | `EMBEDDING_ENDPOINT` | healthy, unavailable, unconfigured |
| Multimodal Embedding | Optional | HTTP GET `{endpoint}/health` | `MULTIMODAL_EMBEDDING_ENDPOINT` | healthy, unavailable, unconfigured |

**"unconfigured"** is returned when the corresponding env var is empty or unset. This distinguishes "not configured" from "configured but down".

### 5.3 Status Computation

**Function:** `compute_status()`

Three-tier status logic:

| Status | Condition |
|---|---|
| `unhealthy` | Any critical dependency (MinIO or NATS) is `"unavailable"` |
| `degraded` | All critical deps healthy, but any optional dep is `"unavailable"` |
| `healthy` | All configured dependencies are healthy (or unconfigured) |

The readiness endpoint (`/v1/admin/readiness`) returns HTTP 503 when status is `"unhealthy"`, suitable for Kubernetes readiness gates.

---

## 6. HTTP Request Logging Middleware

### 6.1 Middleware Stack

**File:** `veda-gateway/src/rest/mod.rs` — `build_router()`

The router applies middleware layers in the following order (outermost to innermost):

```
Request → DefaultBodyLimit (256MB)
        → TraceLayer (tower-http)
        → request_metrics (custom)
        → auth_middleware (protected routes only)
        → Handler
```

Axum layers are applied bottom-to-top in code, so the code reads:

```rust
public
    .merge(protected)
    .layer(from_fn(metrics_middleware::request_metrics))  // Layer 3: custom metrics
    .layer(TraceLayer::new_for_http())                     // Layer 2: HTTP tracing
    .layer(DefaultBodyLimit::max(256 * 1024 * 1024))       // Layer 1: body limit
    .with_state(app_state)
```

### 6.2 TraceLayer

`tower_http::trace::TraceLayer::new_for_http()` generates a span per HTTP request with method, URI, and version. Combined with the JSON fmt layer, this produces structured request/response logs to stdout.

### 6.3 Request Metrics Middleware

**File:** `veda-gateway/src/rest/metrics_middleware.rs`

The `request_metrics()` function is an axum `from_fn` middleware that:

1. Extracts method and path from the request
2. Checks if the path is in the skip list (`/v1/admin/health`, `/v1/admin/metrics`, `/v1/admin/readiness`)
3. If skipped, passes through without recording
4. Otherwise, starts a timer, calls the next handler, then records:
   - `veda_requests_total` (counter) with labels: `method`, `endpoint`, `status`
   - `veda_request_duration_seconds` (histogram) with labels: `method`, `endpoint`

---

## 7. Query Explain

### 7.1 Request Format

The search endpoint accepts an optional `"explain": true` field:

```json
{
  "query": {"vector": {"field": "embedding", "values": [0.1, ...]}},
  "limit": 10,
  "explain": true
}
```

When `explain` is omitted or `false`, the `explain` field is absent from the response (via `#[serde(skip_serializing_if = "Option::is_none")]`).

### 7.2 ExplainTiming Structure

**File:** `veda-gateway/src/rest/search.rs`

```rust
pub struct ExplainTiming {
    pub filter_ms: f64,
    pub ann_search_ms: f64,
    pub scoring_ms: f64,
    pub fusion_ms: f64,
    pub total_ms: f64,
}
```

### 7.3 Timing Collection

Timing is collected by wrapping the search execution in `Instant::now()` / `elapsed()` only when `explain` is true:

**Single-vector search:** The entire `execute_single_vector()` call is timed and reported as `ann_search_ms`. Other phases (`filter_ms`, `scoring_ms`, `fusion_ms`) are reported as `0.0` because the current SearchExecutor does not expose per-phase timing internally.

**Multi-vector search:** The entire `execute_multi_vector()` call (which includes per-field search + fusion) is timed and reported as `fusion_ms`. Other phases are `0.0`.

**Response example:**
```json
{
  "results": [...],
  "count": 10,
  "took_ms": 3,
  "explain": {
    "filter_ms": 0.0,
    "ann_search_ms": 1.8,
    "scoring_ms": 0.0,
    "fusion_ms": 0.0,
    "total_ms": 1.8
  }
}
```

---

## 8. Docker Compose Monitoring Stack

All monitoring services are defined under `profiles: ["monitoring"]` in `docker-compose.yml`. They start only when explicitly activated with `docker compose --profile monitoring up`.

### 8.1 Services

| Service | Image | Port | Role | Dependencies |
|---|---|---|---|---|
| `otel-collector` | `otel/opentelemetry-collector-contrib` | 4317 (gRPC), 4318 (HTTP) | Receives OTLP traces+logs from Veda, routes to Tempo+Loki | tempo, loki |
| `prometheus` | `prom/prometheus` | 9090 | Scrapes Veda metrics, remote-writes to Mimir | mimir |
| `mimir` | `grafana/mimir` | 9009 | Long-term metrics storage (backed by MinIO bucket `veda-mimir`) | minio (healthy) |
| `tempo` | `grafana/tempo` | 3200 | Distributed trace storage, receives OTLP from collector | — |
| `loki` | `grafana/loki` | 3100 | Log aggregation, receives OTLP from collector | — |
| `grafana` | `grafana/grafana` | 3000 | Unified visualization, pre-provisioned datasources + dashboard | prometheus, tempo, loki |

### 8.2 Docker Volumes

The monitoring stack adds 5 named volumes: `prometheus-data`, `mimir-data`, `tempo-data`, `loki-data`, `grafana-data`.

### 8.3 Veda Environment Variables (monitoring mode)

When running with the monitoring profile, set these in `.env`:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=veda
```

---

## 9. Monitoring Stack Configuration

All configuration files live under `config/monitoring/`.

### 9.1 OTel Collector

**File:** `config/monitoring/otel-collector.yaml`

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  otlphttp/tempo:
    endpoint: http://tempo:4318
  otlphttp/loki:
    endpoint: http://loki:3100/otlp

processors:
  batch:
    timeout: 5s
    send_batch_size: 1024

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/tempo]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/loki]
```

**Pipeline routing:**
- Traces: OTLP receiver → batch processor → Tempo (OTLP/HTTP)
- Logs: OTLP receiver → batch processor → Loki (OTLP endpoint at `/otlp`)

The batch processor aggregates up to 1024 spans/log records or flushes every 5 seconds.

### 9.2 Prometheus

**File:** `config/monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

remote_write:
  - url: http://mimir:9009/api/v1/push

scrape_configs:
  - job_name: "veda"
    metrics_path: "/v1/admin/metrics"
    static_configs:
      - targets: ["veda:6340"]
```

Prometheus scrapes Veda's `/v1/admin/metrics` endpoint every 15 seconds and remote-writes all data to Mimir for long-term storage.

### 9.3 Mimir

**File:** `config/monitoring/mimir.yml`

Mimir runs in single-binary mode with S3 (MinIO) as the blocks storage backend:

```yaml
multitenancy_enabled: false

blocks_storage:
  backend: s3
  s3:
    endpoint: minio:9000
    bucket_name: veda-mimir
    access_key_id: veda
    secret_access_key: vedapassword
    insecure: true
  tsdb:
    dir: /data/tsdb

server:
  http_listen_port: 9009
```

Mimir reuses the existing MinIO instance (shared with Veda's blob storage) but writes to a separate `veda-mimir` bucket. All ring-based components (compactor, ingester, store gateway) use `memberlist` with `replication_factor: 1` for single-node deployment.

**Note:** The MinIO credentials are hardcoded in this config file. If `MINIO_PASSWORD` is changed in `.env`, this file must be updated to match.

### 9.4 Tempo

**File:** `config/monitoring/tempo.yml`

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:4318

storage:
  trace:
    backend: local
    local:
      path: /var/tempo/traces
    wal:
      path: /var/tempo/wal
```

Tempo receives traces via OTLP/HTTP on port 4318 (from the OTel Collector) and stores them locally on the `tempo-data` Docker volume.

### 9.5 Loki

**File:** `config/monitoring/loki.yml`

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

limits_config:
  allow_structured_metadata: true

schema_config:
  configs:
    - from: "2024-01-01"
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
```

Loki uses local filesystem storage with TSDB schema v13. Structured metadata is enabled to preserve JSON log fields from `tracing` events.

### 9.6 Grafana Datasource Provisioning

**File:** `config/monitoring/grafana/provisioning/datasources.yaml`

Three datasources are auto-provisioned at startup:

| Name | UID | Type | URL | Default |
|---|---|---|---|---|
| Mimir | `mimir` | prometheus | `http://mimir:9009/prometheus` | Yes |
| Tempo | `tempo` | tempo | `http://tempo:3200` | No |
| Loki | `loki` | loki | `http://loki:3100` | No |

The explicit `uid` fields ensure dashboard panels reference datasources reliably.

Grafana is configured with:
- Admin password: `${GRAFANA_PASSWORD:-vedagrafana}`
- Anonymous access: enabled with Viewer role (for easy onboarding)

---

## 10. Grafana Dashboard

**File:** `config/monitoring/grafana/provisioning/dashboards/veda.json`

**Dashboard:** "Veda Overview" (UID: `veda-overview`)
**Refresh:** 10 seconds
**Default time range:** Last 1 hour

The dashboard is organized into 4 rows with 12 panels:

### Row 1: Overview

| Panel | Type | Query | Labels |
|---|---|---|---|
| Request Rate | Time series | `rate(veda_requests_total[5m])` | `method`, `endpoint` |
| Error Rate | Time series | `rate(veda_errors_total[5m])` | `type` |
| Active Collections | Stat | `veda_active_collections` | — |

### Row 2: Latency

| Panel | Type | Query | Unit |
|---|---|---|---|
| Request Latency p99 | Time series | `histogram_quantile(0.99, rate(veda_request_duration_seconds_bucket[5m]))` | seconds |
| Search Latency p99 | Time series | `histogram_quantile(0.99, rate(veda_search_duration_seconds_bucket[5m]))` | seconds |
| Insert Latency p99 | Time series | `histogram_quantile(0.99, rate(veda_insert_duration_seconds_bucket[5m]))` | seconds |

### Row 3: Throughput

| Panel | Type | Query | Unit |
|---|---|---|---|
| Insert Rate | Time series | `rate(veda_insert_total[5m])` | ops |
| Embedding Latency | Time series | `histogram_quantile(0.99, rate(veda_embedding_duration_seconds_bucket[5m]))` | seconds |
| Blob Op Latency | Time series | `histogram_quantile(0.99, rate(veda_blob_operation_duration_seconds_bucket[5m]))` | seconds |

### Row 4: Background Operations

| Panel | Type | Query | Notes |
|---|---|---|---|
| Index Rebuild In Progress | Stat | `veda_index_rebuild_in_progress` | Yellow threshold at ≥ 1 |
| Seal Duration p99 | Time series | `histogram_quantile(0.99, rate(veda_seal_duration_seconds_bucket[5m]))` | seconds |
| Index Build Duration | Time series | `histogram_quantile(0.99, rate(veda_index_build_duration_seconds_bucket[5m]))` | seconds |

All panels use the Mimir datasource (UID: `mimir`) with palette-classic coloring and list-style legends.

---

## 11. AppState Integration

### 11.1 PrometheusHandle in State

**File:** `veda-gateway/src/rest/state.rs`

The `AppStateInner` struct carries an `Option<PrometheusHandle>`:

```rust
pub struct AppStateInner {
    // ... other fields ...
    pub prometheus_handle: Option<PrometheusHandle>,
}
```

### 11.2 Constructors

| Constructor | Auth | Prometheus | Use Case |
|---|---|---|---|
| `AppState::new()` | Disabled | None | Tests |
| `AppState::with_auth()` | Enabled | None | Tests with auth |
| `AppState::with_auth_and_users()` | Enabled (custom users) | None | Tests with custom users |
| `AppState::with_prometheus()` | Disabled | Yes | Development server |
| `AppState::with_auth_and_prometheus()` | Enabled | Yes | Production server |

The `with_prometheus()` and `with_auth_and_prometheus()` constructors use `Arc::get_mut` to set the handle on the inner state before any clones exist.

### 11.3 Server Startup

**File:** `veda-gateway/src/server.rs`

```rust
pub async fn start_server(config: ServerConfig) -> std::io::Result<()> {
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let app_state = AppState::with_prometheus(&config.data_dir, prometheus_handle);
    app_state.load_embedding_from_env().await;
    app_state.load_blob_store_from_env().await;
    let router = build_router(app_state);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
```

---

## 12. Crate Dependencies

### 12.1 Workspace Dependencies (root Cargo.toml)

```toml
metrics = "0.24"
metrics-exporter-prometheus = "0.16"
opentelemetry = "0.27"
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.27", features = ["tonic", "metrics", "logs", "trace"] }
tracing-opentelemetry = "0.28"
opentelemetry-appender-tracing = "0.27"
```

### 12.2 Per-Crate Dependencies

| Crate | Dependencies Added |
|---|---|
| `veda-core` | `metrics`, `tracing` |
| `veda-storage` | `metrics`, `tracing` |
| `veda-gateway` | `metrics`, `metrics-exporter-prometheus`, `opentelemetry`, `tracing-opentelemetry` |
| `veda-query` | `tracing` |
| `veda-server` | `metrics`, `metrics-exporter-prometheus`, `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`, `opentelemetry-appender-tracing` |

---

## 13. Resolved Limitations

All 16 metric constants defined in `veda-core::metrics` are now recorded at their corresponding call sites:
- `SEGMENTS_TOTAL` is updated after seal and compaction in insert handlers.
- `COMPACTION_DURATION` and `COMPACTION_IN_PROGRESS` are recorded when `SegmentManager::compact_if_needed()` runs after segment sealing.
- Compaction is triggered automatically after seal in insert and embed-insert handlers using `CompactionConfig::default()`.

Query explain now returns per-phase timing (`filter_ms`, `ann_search_ms`, `scoring_ms`) from `SearchTimings` produced by `SearchExecutor::search()`, rather than timing the outer boundary call.
