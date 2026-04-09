# Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full-stack observability for Veda — metrics (Prometheus), distributed tracing (OpenTelemetry), centralized logging, health checks, query explain, and a bundled Grafana monitoring stack.

**Architecture:** Instrument Veda with the `metrics` crate (Prometheus exporter) and `tracing` + `tracing-opentelemetry` (OTLP export). Bundle Prometheus, Mimir, Tempo, Loki, OTel Collector, and Grafana under a Docker Compose `monitoring` profile. OTel layers are conditional — only active when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

**Tech Stack:** `metrics` 0.24, `metrics-exporter-prometheus` 0.16, `opentelemetry` 0.27, `opentelemetry_sdk` 0.27, `opentelemetry-otlp` 0.27, `tracing-opentelemetry` 0.28, `opentelemetry-appender-tracing` 0.27, `tower-http` TraceLayer, Grafana LGTM + Mimir stack.

**Design doc:** `docs/plans/2026-04-09-observability-design.md`

---

### Task 1: Add metrics crate dependencies

**Files:**
- Modify: `Cargo.toml` (workspace root, lines 18-26)
- Modify: `crates/veda-core/Cargo.toml` (lines 7-13)
- Modify: `crates/veda-gateway/Cargo.toml` (lines 7-24)
- Modify: `crates/veda-server/Cargo.toml` (lines 7-19)

**Step 1: Add workspace dependencies**

In root `Cargo.toml`, add after the `bytes = "1"` line (line 26):

```toml
metrics = "0.24"
metrics-exporter-prometheus = "0.16"
opentelemetry = "0.27"
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.27", features = ["tonic", "metrics", "logs", "trace"] }
tracing-opentelemetry = "0.28"
opentelemetry-appender-tracing = "0.27"
```

**Step 2: Add `metrics` to veda-core**

In `crates/veda-core/Cargo.toml`, add under `[dependencies]`:

```toml
metrics = { workspace = true }
```

**Step 3: Add `metrics` and `opentelemetry` to veda-gateway**

In `crates/veda-gateway/Cargo.toml`, add under `[dependencies]`:

```toml
metrics = { workspace = true }
opentelemetry = { workspace = true }
```

**Step 4: Add OTel crates to veda-server**

In `crates/veda-server/Cargo.toml`, add under `[dependencies]`:

```toml
metrics = { workspace = true }
metrics-exporter-prometheus = { workspace = true }
opentelemetry = { workspace = true }
opentelemetry_sdk = { workspace = true }
opentelemetry-otlp = { workspace = true }
tracing-opentelemetry = { workspace = true }
opentelemetry-appender-tracing = { workspace = true }
```

**Step 5: Verify it compiles**

Run: `cargo check`
Expected: Compiles without errors.

**Step 6: Commit**

```bash
git add Cargo.toml crates/veda-core/Cargo.toml crates/veda-gateway/Cargo.toml crates/veda-server/Cargo.toml
git commit -m "chore: add metrics and opentelemetry dependencies"
```

---

### Task 2: Replace hand-rolled Metrics with `metrics` crate

**Files:**
- Modify: `crates/veda-core/src/metrics.rs`
- Modify: `crates/veda-core/tests/metrics_test.rs`

**Step 1: Rewrite metrics.rs**

Replace the entire contents of `crates/veda-core/src/metrics.rs` with:

```rust
//! Re-exports from the `metrics` crate and Veda-specific metric helpers.
//!
//! All metric recording uses the `metrics` crate macros directly in handlers.
//! This module defines metric name constants to avoid typos.

// Metric name constants
pub const REQUESTS_TOTAL: &str = "veda_requests_total";
pub const REQUEST_DURATION: &str = "veda_request_duration_seconds";
pub const SEARCH_DURATION: &str = "veda_search_duration_seconds";
pub const INSERT_TOTAL: &str = "veda_insert_total";
pub const INSERT_DURATION: &str = "veda_insert_duration_seconds";
pub const EMBEDDING_DURATION: &str = "veda_embedding_duration_seconds";
pub const BLOB_OP_DURATION: &str = "veda_blob_operation_duration_seconds";
pub const SEGMENTS_TOTAL: &str = "veda_segments_total";
pub const INDEX_BUILD_DURATION: &str = "veda_index_build_duration_seconds";
pub const COMPACTION_DURATION: &str = "veda_compaction_duration_seconds";
pub const ERRORS_TOTAL: &str = "veda_errors_total";
pub const ACTIVE_COLLECTIONS: &str = "veda_active_collections";
pub const RECORDS_TOTAL: &str = "veda_records_total";
pub const COMPACTION_IN_PROGRESS: &str = "veda_compaction_in_progress";
pub const INDEX_REBUILD_IN_PROGRESS: &str = "veda_index_rebuild_in_progress";
pub const SEAL_DURATION: &str = "veda_seal_duration_seconds";
```

**Step 2: Rewrite metrics_test.rs**

Replace `crates/veda-core/tests/metrics_test.rs` with:

```rust
use veda_core::metrics;

#[test]
fn test_metric_names_are_valid_prometheus_names() {
    let names = [
        metrics::REQUESTS_TOTAL,
        metrics::REQUEST_DURATION,
        metrics::SEARCH_DURATION,
        metrics::INSERT_TOTAL,
        metrics::INSERT_DURATION,
        metrics::EMBEDDING_DURATION,
        metrics::BLOB_OP_DURATION,
        metrics::SEGMENTS_TOTAL,
        metrics::INDEX_BUILD_DURATION,
        metrics::COMPACTION_DURATION,
        metrics::ERRORS_TOTAL,
        metrics::ACTIVE_COLLECTIONS,
        metrics::RECORDS_TOTAL,
        metrics::COMPACTION_IN_PROGRESS,
        metrics::INDEX_REBUILD_IN_PROGRESS,
        metrics::SEAL_DURATION,
    ];

    for name in &names {
        // Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*
        assert!(!name.is_empty(), "metric name must not be empty");
        assert!(
            name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'),
            "metric '{name}' must start with letter or underscore"
        );
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "metric '{name}' contains invalid characters"
        );
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p veda-core -- metrics`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/veda-core/src/metrics.rs crates/veda-core/tests/metrics_test.rs
git commit -m "refactor: replace hand-rolled metrics with metrics crate constants"
```

---

### Task 3: Update metrics endpoint to use Prometheus exporter

**Files:**
- Modify: `crates/veda-gateway/src/rest/admin.rs` (lines 1-6, 90-97)
- Modify: `crates/veda-gateway/src/rest/state.rs` (lines 27-48)
- Modify: `crates/veda-gateway/src/server.rs`

**Step 1: Add PrometheusHandle to AppStateInner**

In `crates/veda-gateway/src/rest/state.rs`, add import at the top:

```rust
use metrics_exporter_prometheus::PrometheusHandle;
```

Add field to `AppStateInner` (after `blob_store`):

```rust
/// Prometheus metrics handle for rendering /metrics endpoint.
pub prometheus_handle: Option<PrometheusHandle>,
```

In all three constructors (`new`, `with_auth`, `with_auth_and_users`), add:

```rust
prometheus_handle: None,
```

Add a method to `AppState`:

```rust
/// Set the Prometheus handle (called from server startup).
pub async fn set_prometheus_handle(&self, handle: PrometheusHandle) {
    // We need interior mutability here. Since this is called once at startup,
    // we'll store it via an Option set during construction.
    // Actually, since AppStateInner is behind Arc, we need a different approach.
}
```

Actually — since `AppStateInner` is constructed once and wrapped in `Arc`, the cleaner approach is to pass the handle at construction time. Modify `ServerConfig` in `crates/veda-gateway/src/server.rs`:

```rust
use metrics_exporter_prometheus::PrometheusHandle;

pub struct ServerConfig {
    pub listen_addr: String,
    pub data_dir: PathBuf,
    pub prometheus_handle: Option<PrometheusHandle>,
}
```

Thread it into `AppState::new()` — add a `prometheus_handle` parameter to `AppState::new()` and pass it through:

In `crates/veda-gateway/src/rest/state.rs`, change `AppState::new` signature:

```rust
pub fn new(data_dir: &Path, prometheus_handle: Option<metrics_exporter_prometheus::PrometheusHandle>) -> Self {
```

And set `prometheus_handle` in `AppStateInner`.

Do the same for `with_auth` and `with_auth_and_users`.

**Step 2: Update server.rs to install Prometheus recorder and pass handle**

Replace `crates/veda-gateway/src/server.rs` with:

```rust
use std::path::PathBuf;

use metrics_exporter_prometheus::PrometheusBuilder;

use crate::rest::{build_router, state::AppState};

/// Configuration for the Veda API server.
pub struct ServerConfig {
    pub listen_addr: String,
    pub data_dir: PathBuf,
}

/// Start the HTTP server and block until shutdown.
pub async fn start_server(config: ServerConfig) -> std::io::Result<()> {
    // Install Prometheus metrics recorder (global).
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let app_state = AppState::new(&config.data_dir, Some(prometheus_handle));
    app_state.load_embedding_from_env().await;
    app_state.load_blob_store_from_env().await;
    let router = build_router(app_state);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    tracing::info!("Veda listening on {}", config.listen_addr);
    axum::serve(listener, router).await?;
    Ok(())
}
```

**Step 3: Update admin.rs metrics endpoint**

Replace the `metrics()` handler in `crates/veda-gateway/src/rest/admin.rs`:

```rust
pub async fn metrics(
    State(state): State<AppState>,
) -> (StatusCode, [(axum::http::header::HeaderName, &'static str); 1], String) {
    let body = match &state.inner.prometheus_handle {
        Some(handle) => handle.render(),
        None => String::from("# metrics not configured\n"),
    };
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}
```

Remove the `use veda_core::metrics::Metrics;` import from admin.rs (line 5).

**Step 4: Fix test constructors**

Search for all calls to `AppState::new(` across the codebase and add the `None` prometheus handle parameter. Same for `with_auth(` and `with_auth_and_users(`.

**Step 5: Verify it compiles and tests pass**

Run: `cargo test`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/veda-gateway/src/rest/admin.rs crates/veda-gateway/src/rest/state.rs crates/veda-gateway/src/server.rs
git commit -m "feat: wire Prometheus exporter to /v1/admin/metrics endpoint"
```

---

### Task 4: Record metrics in search handler

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs`

**Step 1: Add metrics recording to search handler**

At the top of `search.rs`, add:

```rust
use veda_core::metrics as m;
```

After the `took_ms` calculation (line 130), before the `Ok(...)` return, add:

```rust
metrics::histogram!(m::SEARCH_DURATION, "collection" => name.clone())
    .record(start.elapsed().as_secs_f64());
```

In the error branches (when search fails), add:

```rust
metrics::counter!(m::ERRORS_TOTAL, "type" => "search").increment(1);
```

**Step 2: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 3: Commit**

```bash
git add crates/veda-gateway/src/rest/search.rs
git commit -m "feat: record search latency metrics"
```

---

### Task 5: Record metrics in insert and embed-insert handlers

**Files:**
- Modify: `crates/veda-gateway/src/rest/records.rs`
- Modify: `crates/veda-gateway/src/rest/embed_insert.rs`

**Step 1: Add metrics to insert handler**

In `records.rs`, add `use veda_core::metrics as m;` and add timing:

```rust
use std::time::Instant;
```

At start of `insert()` handler, add `let start = Instant::now();`.

After successful insert, before the `Ok(...)` return:

```rust
metrics::histogram!(m::INSERT_DURATION, "collection" => name.clone())
    .record(start.elapsed().as_secs_f64());
metrics::counter!(m::INSERT_TOTAL, "collection" => name.clone())
    .increment(inserted as u64);
```

**Step 2: Add metrics to embed_insert handler**

Same pattern in `embed_insert.rs`:
- Add timing with `Instant::now()` at start
- Record `INSERT_DURATION` and `INSERT_TOTAL` after success
- Record `ERRORS_TOTAL` on embedding failures

**Step 3: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 4: Commit**

```bash
git add crates/veda-gateway/src/rest/records.rs crates/veda-gateway/src/rest/embed_insert.rs
git commit -m "feat: record insert and embed-insert metrics"
```

---

### Task 6: Record metrics in embedding client and blob store

**Files:**
- Modify: `crates/veda-core/src/embedding_client.rs`
- Modify: `crates/veda-storage/src/blob_store.rs`
- Modify: `crates/veda-storage/Cargo.toml`

**Step 1: Add `metrics` dependency to veda-storage**

In `crates/veda-storage/Cargo.toml`, add:

```toml
metrics = { workspace = true }
```

**Step 2: Record embedding latency**

In `embedding_client.rs`, add `use crate::metrics as m;` and `use std::time::Instant;`.

In `embed_text_http()`, wrap the HTTP call with timing:

```rust
let start = Instant::now();
// ... existing HTTP call ...
metrics::histogram!(m::EMBEDDING_DURATION, "model" => self.config.name.clone(), "modality" => "text")
    .record(start.elapsed().as_secs_f64());
```

Same for `embed_bytes_http()` with `"modality" => modality.to_string()`.

**Step 3: Record blob store operation latency**

In `blob_store.rs`, add `use veda_core::metrics as m;` and `use std::time::Instant;`.

In `ObjectBlobStore`'s `put`, `get`, `delete` methods, wrap each with:

```rust
let start = Instant::now();
// ... existing operation ...
metrics::histogram!(m::BLOB_OP_DURATION, "operation" => "put")
    .record(start.elapsed().as_secs_f64());
```

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: Compiles.

**Step 5: Commit**

```bash
git add crates/veda-core/src/embedding_client.rs crates/veda-storage/src/blob_store.rs crates/veda-storage/Cargo.toml
git commit -m "feat: record embedding and blob store operation metrics"
```

---

### Task 7: Add HTTP request metrics middleware

**Files:**
- Create: `crates/veda-gateway/src/rest/metrics_middleware.rs`
- Modify: `crates/veda-gateway/src/rest/mod.rs`

**Step 1: Create metrics middleware**

Create `crates/veda-gateway/src/rest/metrics_middleware.rs`:

```rust
use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use veda_core::metrics as m;

/// Middleware that records HTTP request count and duration for all endpoints.
pub async fn request_metrics(req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    // Skip metrics for health and metrics endpoints to avoid noise.
    if path == "/v1/admin/health" || path == "/v1/admin/metrics" || path == "/v1/admin/readiness" {
        return next.run(req).await;
    }

    let start = Instant::now();
    let response = next.run(req).await;
    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    metrics::counter!(m::REQUESTS_TOTAL, "method" => method.clone(), "endpoint" => path.clone(), "status" => status)
        .increment(1);
    metrics::histogram!(m::REQUEST_DURATION, "method" => method, "endpoint" => path)
        .record(duration);

    response
}
```

**Step 2: Wire middleware into router**

In `crates/veda-gateway/src/rest/mod.rs`, add:

```rust
pub mod metrics_middleware;
```

Add the metrics middleware as an outer layer (after merging public + protected, before `DefaultBodyLimit`):

```rust
use crate::rest::metrics_middleware::request_metrics;

pub fn build_router(app_state: state::AppState) -> Router {
    let public = Router::new()
        .route("/v1/admin/health", get(admin::health))
        .route("/v1/admin/readiness", get(admin::readiness))
        .route("/v1/auth/login", post(admin::login));

    let protected = Router::new()
        // ... existing routes ...
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth::auth_middleware,
        ));

    public
        .merge(protected)
        .layer(axum::middleware::from_fn(request_metrics))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(app_state)
}
```

**Step 3: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 4: Commit**

```bash
git add crates/veda-gateway/src/rest/metrics_middleware.rs crates/veda-gateway/src/rest/mod.rs
git commit -m "feat: add HTTP request metrics middleware"
```

---

### Task 8: Implement real health checks and readiness endpoint

**Files:**
- Modify: `crates/veda-gateway/src/rest/admin.rs`
- Modify: `crates/veda-gateway/src/rest/mod.rs` (add readiness route)

**Step 1: Rewrite health handler**

Replace the `HealthResponse` struct and `health()` handler in `admin.rs` with:

```rust
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub checks: HashMap<String, String>,
}

pub async fn health(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let checks = run_health_checks(&state).await;
    let status = compute_status(&checks);
    Json(HealthResponse {
        status,
        version: "0.1.0".into(),
        checks,
    })
}

pub async fn readiness(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, (StatusCode, Json<HealthResponse>)> {
    let checks = run_health_checks(&state).await;
    let status = compute_status(&checks);
    let resp = HealthResponse {
        status: status.clone(),
        version: "0.1.0".into(),
        checks,
    };
    if status == "unhealthy" {
        Err((StatusCode::SERVICE_UNAVAILABLE, Json(resp)))
    } else {
        Ok(Json(resp))
    }
}

fn compute_status(checks: &HashMap<String, String>) -> String {
    let any_critical_down = checks.get("minio").is_some_and(|s| s == "unavailable")
        || checks.get("nats").is_some_and(|s| s == "unavailable");
    if any_critical_down {
        "unhealthy".into()
    } else if checks.values().any(|s| s == "unavailable") {
        "degraded".into()
    } else {
        "healthy".into()
    }
}

async fn run_health_checks(state: &AppState) -> HashMap<String, String> {
    let mut checks = HashMap::new();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .no_proxy()
        .build()
        .unwrap_or_default();

    // MinIO
    let blob_endpoint = std::env::var("VEDA_BLOB_ENDPOINT").unwrap_or_default();
    if blob_endpoint.is_empty() {
        checks.insert("minio".into(), "unconfigured".into());
    } else {
        let url = format!("{}/minio/health/live", blob_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("minio".into(), status.into());
    }

    // NATS
    let nats_url = std::env::var("VEDA_MQ_ENDPOINT").unwrap_or_default();
    if nats_url.is_empty() {
        checks.insert("nats".into(), "unconfigured".into());
    } else {
        let addr = nats_url.trim_start_matches("nats://");
        let status = match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::net::TcpStream::connect(addr),
        ).await {
            Ok(Ok(_)) => "healthy",
            _ => "unavailable",
        };
        checks.insert("nats".into(), status.into());
    }

    // Embedding
    let embed_endpoint = std::env::var("EMBEDDING_ENDPOINT").unwrap_or_default();
    if embed_endpoint.is_empty() {
        checks.insert("embedding".into(), "unconfigured".into());
    } else {
        let url = format!("{}/health", embed_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("embedding".into(), status.into());
    }

    // Multimodal embedding
    let mm_endpoint = std::env::var("MULTIMODAL_EMBEDDING_ENDPOINT").unwrap_or_default();
    if mm_endpoint.is_empty() {
        checks.insert("multimodal_embedding".into(), "unconfigured".into());
    } else {
        let url = format!("{}/health", mm_endpoint.trim_end_matches('/'));
        let status = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => "healthy",
            _ => "unavailable",
        };
        checks.insert("multimodal_embedding".into(), status.into());
    }

    checks
}
```

**Step 2: Add readiness route**

In `crates/veda-gateway/src/rest/mod.rs`, add the readiness route to public routes:

```rust
.route("/v1/admin/readiness", get(admin::readiness))
```

**Step 3: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 4: Commit**

```bash
git add crates/veda-gateway/src/rest/admin.rs crates/veda-gateway/src/rest/mod.rs
git commit -m "feat: real health checks with dependency probing and readiness endpoint"
```

---

### Task 9: Add HTTP request logging via tower-http TraceLayer

**Files:**
- Modify: `crates/veda-gateway/src/rest/mod.rs`

**Step 1: Add TraceLayer to router**

In `crates/veda-gateway/src/rest/mod.rs`, add import:

```rust
use tower_http::trace::TraceLayer;
```

Add `TraceLayer::new_for_http()` as a layer on the merged router (outermost layer, after metrics middleware):

```rust
public
    .merge(protected)
    .layer(axum::middleware::from_fn(request_metrics))
    .layer(TraceLayer::new_for_http())
    .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
    .with_state(app_state)
```

**Step 2: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 3: Commit**

```bash
git add crates/veda-gateway/src/rest/mod.rs
git commit -m "feat: add tower-http TraceLayer for HTTP request logging"
```

---

### Task 10: Add `#[instrument]` annotations to handlers

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs`
- Modify: `crates/veda-gateway/src/rest/records.rs`
- Modify: `crates/veda-gateway/src/rest/embed_insert.rs`
- Modify: `crates/veda-gateway/src/rest/document.rs`

**Step 1: Instrument search handler**

In `search.rs`, add `use tracing::instrument;` and annotate `search()`:

```rust
#[instrument(name = "search", skip(state, req), fields(collection = %name, top_k = req.limit, has_filter = req.filter.is_some()))]
pub async fn search(
```

**Step 2: Instrument insert handler**

In `records.rs`, add `use tracing::instrument;` and annotate `insert()`:

```rust
#[instrument(name = "insert", skip(state, req), fields(collection = %name, record_count = req.records.len()))]
pub async fn insert(
```

**Step 3: Instrument embed_insert handler**

In `embed_insert.rs`, add `use tracing::instrument;` and annotate:

```rust
#[instrument(name = "embed_insert", skip(state, req), fields(collection = %name, record_count = req.records.len()))]
pub async fn embed_insert(
```

**Step 4: Instrument process_document handler**

In `document.rs`, add `use tracing::instrument;` and annotate `process_document()`:

```rust
#[instrument(name = "process_document", skip(state, multipart), fields(collection = %name))]
pub async fn process_document(
```

**Step 5: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 6: Commit**

```bash
git add crates/veda-gateway/src/rest/search.rs crates/veda-gateway/src/rest/records.rs crates/veda-gateway/src/rest/embed_insert.rs crates/veda-gateway/src/rest/document.rs
git commit -m "feat: add tracing instrument annotations to all handlers"
```

---

### Task 11: Add `#[instrument]` to embedding client and blob store

**Files:**
- Modify: `crates/veda-core/Cargo.toml` (add `tracing` dependency)
- Modify: `crates/veda-core/src/embedding_client.rs`
- Modify: `crates/veda-storage/Cargo.toml` (add `tracing` dependency)
- Modify: `crates/veda-storage/src/blob_store.rs`

**Step 1: Add tracing to veda-core and veda-storage**

In `crates/veda-core/Cargo.toml`, add:

```toml
tracing = { workspace = true }
```

In `crates/veda-storage/Cargo.toml`, add:

```toml
tracing = { workspace = true }
```

**Step 2: Instrument embedding client**

In `embedding_client.rs`, add `use tracing::instrument;`.

Annotate `embed_text()`:

```rust
#[instrument(name = "embed_text", skip(self, text), fields(model = %self.config.name, input_len = text.len()))]
pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
```

Annotate `embed_bytes()`:

```rust
#[instrument(name = "embed_bytes", skip(self, data), fields(model = %self.config.name, modality = %modality, input_size = data.len()))]
pub fn embed_bytes(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
```

**Step 3: Instrument blob store**

In `blob_store.rs`, add `use tracing::instrument;`.

Annotate `ObjectBlobStore` methods:

```rust
#[instrument(name = "blob_put", skip(self, data), fields(key = %key, size = data.len()))]
fn put(&self, key: &str, data: &[u8]) -> Result<()> {

#[instrument(name = "blob_get", skip(self), fields(key = %key))]
fn get(&self, key: &str) -> Result<Vec<u8>> {

#[instrument(name = "blob_delete", skip(self), fields(key = %key))]
fn delete(&self, key: &str) -> Result<()> {
```

**Step 4: Verify it compiles and tests pass**

Run: `cargo test`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/veda-core/Cargo.toml crates/veda-core/src/embedding_client.rs crates/veda-storage/Cargo.toml crates/veda-storage/src/blob_store.rs
git commit -m "feat: add tracing instrumentation to embedding client and blob store"
```

---

### Task 12: Set up conditional OpenTelemetry tracing subscriber

**Files:**
- Modify: `crates/veda-server/src/main.rs`

**Step 1: Replace tracing setup with layered subscriber**

Replace lines 19-26 of `main.rs` (the tracing initialization block) with:

```rust
// Initialize tracing with optional OpenTelemetry export.
let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
    .unwrap_or_else(|_| "info".into());

let fmt_layer = tracing_subscriber::fmt::layer().json();

let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok().filter(|s| !s.is_empty());

if let Some(endpoint) = otel_endpoint {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let service_name = std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "veda".into());

    // Trace exporter
    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("failed to create OTLP trace exporter");

    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(trace_exporter)
        .with_resource(opentelemetry_sdk::Resource::builder()
            .with_service_name(service_name.clone())
            .build())
        .build();

    let tracer = tracer_provider.tracer("veda");
    let otel_trace_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Log exporter
    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("failed to create OTLP log exporter");

    let log_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_batch_exporter(log_exporter)
        .with_resource(opentelemetry_sdk::Resource::builder()
            .with_service_name(service_name)
            .build())
        .build();

    let otel_log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&log_provider);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_trace_layer)
        .with(otel_log_layer)
        .init();

    tracing::info!("OpenTelemetry export enabled to {}", endpoint);
} else {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p veda-server`
Expected: Compiles.

**Step 3: Verify it runs without OTEL_EXPORTER_OTLP_ENDPOINT**

Run: `cargo run --bin veda-server -- /tmp/veda-test-otel 0.0.0.0:0 &` then `kill %1`
Expected: Starts normally, no OTel-related errors in output.

**Step 4: Commit**

```bash
git add crates/veda-server/src/main.rs
git commit -m "feat: conditional OpenTelemetry tracing and log export via OTLP"
```

---

### Task 13: Add query explain support to search

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs`

**Step 1: Add explain types and request field**

Add to `SearchRequest`:

```rust
#[serde(default)]
pub explain: bool,
```

Add explain types:

```rust
#[derive(Debug, Serialize)]
pub struct ExplainTiming {
    pub filter_ms: f64,
    pub ann_search_ms: f64,
    pub scoring_ms: f64,
    pub fusion_ms: f64,
    pub total_ms: f64,
}
```

Add to `SearchResponse`:

```rust
#[serde(skip_serializing_if = "Option::is_none")]
pub explain: Option<ExplainTiming>,
```

**Step 2: Add timing in search handler**

Modify the `search()` handler to conditionally measure per-phase timing. Wrap each phase (filter parsing, ANN search, fusion) with `Instant::now()` / `elapsed()` when `req.explain` is true.

For single vector search:

```rust
let results = if let Some(ref vec_query) = req.query.vector {
    let metric = resolve_metric(&schema, &vec_query.field);
    if req.explain {
        let t0 = Instant::now();
        // Filter would be parsed inside executor — capture outer timing
        let search_start = Instant::now();
        let hits = execute_single_vector(executor, vec_query, &req, metric)?;
        let ann_ms = search_start.elapsed().as_secs_f64() * 1000.0;
        explain_timing = Some(ExplainTiming {
            filter_ms: 0.0, // filter is inside executor, measured as part of ann_search
            ann_search_ms: ann_ms,
            scoring_ms: 0.0,
            fusion_ms: 0.0,
            total_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
        hits
    } else {
        execute_single_vector(executor, vec_query, &req, metric)?
    }
}
```

**Step 3: Return explain in response**

```rust
Ok(Json(SearchResponse {
    results,
    count,
    took_ms,
    explain: explain_timing,
}))
```

**Step 4: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/search.rs
git commit -m "feat: add opt-in query explain with per-phase timing breakdown"
```

---

### Task 14: Add async operation metrics (seal, rebuild, compaction)

**Files:**
- Modify: `crates/veda-gateway/src/rest/records.rs` (seal_growing and rebuild_executors calls)
- Modify: `crates/veda-gateway/src/rest/embed_insert.rs` (same)

**Step 1: Add timing around seal_growing**

In `records.rs`, where `sm.seal_growing()` is called, wrap with metrics:

```rust
let seal_start = std::time::Instant::now();
sm.seal_growing().map_err(|e| { /* ... */ })?;
metrics::histogram!(m::SEAL_DURATION, "collection" => name.clone(), "shard_id" => shard_id.to_string())
    .record(seal_start.elapsed().as_secs_f64());
```

**Step 2: Add timing around rebuild_executors**

Where `rebuild_executors()` is called:

```rust
metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(1.0);
let rebuild_start = std::time::Instant::now();
rebuild_executors(&state, &name).await?;
metrics::histogram!(m::INDEX_BUILD_DURATION, "collection" => name.clone(), "index_type" => "rebuild")
    .record(rebuild_start.elapsed().as_secs_f64());
metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(0.0);
```

**Step 3: Do the same in embed_insert.rs**

Apply the same seal and rebuild metrics.

**Step 4: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/records.rs crates/veda-gateway/src/rest/embed_insert.rs
git commit -m "feat: add async operation metrics for seal and index rebuild"
```

---

### Task 15: Add monitoring stack Docker Compose services

**Files:**
- Modify: `docker-compose.yml`
- Create: `config/monitoring/otel-collector.yaml`
- Create: `config/monitoring/prometheus.yml`
- Create: `config/monitoring/mimir.yml`
- Create: `config/monitoring/tempo.yml`
- Create: `config/monitoring/loki.yml`

**Step 1: Create OTel Collector config**

Create `config/monitoring/otel-collector.yaml`:

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

**Step 2: Create Prometheus config**

Create `config/monitoring/prometheus.yml`:

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

**Step 3: Create Mimir config**

Create `config/monitoring/mimir.yml`:

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

compactor:
  data_dir: /data/compactor
  sharding_ring:
    kvstore:
      store: memberlist

ingester:
  ring:
    kvstore:
      store: memberlist
    replication_factor: 1

store_gateway:
  sharding_ring:
    kvstore:
      store: memberlist

server:
  http_listen_port: 9009
```

**Step 4: Create Tempo config**

Create `config/monitoring/tempo.yml`:

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

**Step 5: Create Loki config**

Create `config/monitoring/loki.yml`:

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

**Step 6: Add monitoring services to docker-compose.yml**

Add before the `volumes:` section in `docker-compose.yml`:

```yaml
  # --- Monitoring Stack (optional, enabled with --profile monitoring) ---
  otel-collector:
    profiles: ["monitoring"]
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"
      - "4318:4318"
    volumes:
      - ./config/monitoring/otel-collector.yaml:/etc/otelcol-contrib/config.yaml:ro
    depends_on:
      - tempo
      - loki

  prometheus:
    profiles: ["monitoring"]
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    depends_on:
      - mimir

  mimir:
    profiles: ["monitoring"]
    image: grafana/mimir:latest
    ports:
      - "9009:9009"
    volumes:
      - ./config/monitoring/mimir.yml:/etc/mimir/config.yaml:ro
      - mimir-data:/data
    command: ["-config.file=/etc/mimir/config.yaml"]
    depends_on:
      minio:
        condition: service_healthy

  tempo:
    profiles: ["monitoring"]
    image: grafana/tempo:latest
    ports:
      - "3200:3200"
    volumes:
      - ./config/monitoring/tempo.yml:/etc/tempo/config.yaml:ro
      - tempo-data:/var/tempo
    command: ["-config.file=/etc/tempo/config.yaml"]

  loki:
    profiles: ["monitoring"]
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./config/monitoring/loki.yml:/etc/loki/config.yaml:ro
      - loki-data:/loki
    command: ["-config.file=/etc/loki/config.yaml"]

  grafana:
    profiles: ["monitoring"]
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-vedagrafana}
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
      - tempo
      - loki
```

Also add `OTEL_EXPORTER_OTLP_ENDPOINT` to the veda service environment (conditionally relevant when monitoring profile is active):

```yaml
OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT:-}
OTEL_SERVICE_NAME: ${OTEL_SERVICE_NAME:-veda}
```

Add new volumes:

```yaml
  prometheus-data:
  mimir-data:
  tempo-data:
  loki-data:
  grafana-data:
```

**Step 7: Commit**

```bash
git add docker-compose.yml config/monitoring/
git commit -m "feat: add monitoring stack Docker Compose services (Prometheus, Mimir, Tempo, Loki, OTel Collector, Grafana)"
```

---

### Task 16: Add Grafana provisioning and pre-built dashboard

**Files:**
- Create: `config/monitoring/grafana/provisioning/datasources.yaml`
- Create: `config/monitoring/grafana/provisioning/dashboards/dashboards.yaml`
- Create: `config/monitoring/grafana/provisioning/dashboards/veda.json`

**Step 1: Create datasources provisioning**

Create `config/monitoring/grafana/provisioning/datasources.yaml`:

```yaml
apiVersion: 1

datasources:
  - name: Mimir
    type: prometheus
    access: proxy
    url: http://mimir:9009/prometheus
    isDefault: true
    editable: false

  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    editable: false
    jsonData:
      tracesToMetrics:
        datasourceUid: mimir
      serviceMap:
        datasourceUid: mimir
      nodeGraph:
        enabled: true
      tracesToLogs:
        datasourceUid: loki

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
    jsonData:
      derivedFields:
        - datasourceUid: tempo
          matcherRegex: "\"trace_id\":\"(\\w+)\""
          name: TraceID
          url: "$${__value.raw}"
```

**Step 2: Create dashboard provisioning config**

Create `config/monitoring/grafana/provisioning/dashboards/dashboards.yaml`:

```yaml
apiVersion: 1

providers:
  - name: Veda
    orgId: 1
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: false
```

**Step 3: Create pre-built Veda dashboard**

Create `config/monitoring/grafana/provisioning/dashboards/veda.json` — a Grafana dashboard JSON with panels for:

- Request rate (counter `veda_requests_total`)
- Request latency p50/p95/p99 (histogram `veda_request_duration_seconds`)
- Search latency by collection (histogram `veda_search_duration_seconds`)
- Insert throughput (counter `veda_insert_total`)
- Embedding latency by model (histogram `veda_embedding_duration_seconds`)
- Blob operation latency (histogram `veda_blob_operation_duration_seconds`)
- Active collections (gauge `veda_active_collections`)
- Error rate (counter `veda_errors_total`)
- Background operations: seal/rebuild duration and in-progress gauges

This is a standard Grafana dashboard JSON file. Use PromQL queries targeting the metric names defined in `veda_core::metrics`.

**Step 4: Commit**

```bash
git add config/monitoring/grafana/
git commit -m "feat: add Grafana provisioning with datasources and pre-built Veda dashboard"
```

---

### Task 17: Update README and docs

**Files:**
- Modify: `README.md`
- Modify: `documentation/system_architecture.md`

**Step 1: Add monitoring section to README**

Add a new `## Monitoring` section to `README.md` after the `## Benchmarks` section, documenting:

- How to start the monitoring stack (`docker compose --profile monitoring up`)
- Grafana URL (http://localhost:3000)
- Available dashboards
- How to configure OTLP export for external backends
- Quick reference of env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`)

Update the profiles table to include the `monitoring` profile.

**Step 2: Update system_architecture.md**

Update section 3.9 (EmbeddingClient) to mention `#[instrument]` annotations.
Update the Docker Compose section to include monitoring services.
Add a new section covering the observability architecture.

**Step 3: Commit**

```bash
git add README.md documentation/system_architecture.md
git commit -m "docs: add monitoring stack documentation and update architecture docs"
```

---

### Task 18: End-to-end verification

**Step 1: Verify cargo build succeeds**

Run: `cargo build`
Expected: Successful build.

**Step 2: Verify all tests pass**

Run: `cargo test`
Expected: All tests pass.

**Step 3: Verify Docker Compose monitoring stack starts**

Run: `docker compose --profile monitoring up -d`
Expected: All monitoring services start and become healthy.

Verify:
- Grafana accessible at http://localhost:3000
- Prometheus accessible at http://localhost:9090
- Veda metrics visible in Prometheus (query `veda_requests_total`)
- Grafana dashboard "Veda" is auto-provisioned and shows panels

**Step 4: Verify OTLP tracing works**

Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317` on the veda service, restart it, make a search request, and verify traces appear in Grafana Tempo.

**Step 5: Verify health endpoint**

```bash
curl http://localhost:6340/v1/admin/health
```

Expected: JSON with real dependency status checks.

**Step 6: Verify query explain**

```bash
curl -X POST http://localhost:6340/v1/collections/test/search \
  -H "Content-Type: application/json" \
  -d '{"query": {"vector": {"field": "embedding", "values": [0.1, 0.2]}}, "limit": 5, "explain": true}'
```

Expected: Response includes `explain` field with timing breakdown.

**Step 7: Commit any fixes**

```bash
git commit -m "fix: address issues found in e2e verification"
```
