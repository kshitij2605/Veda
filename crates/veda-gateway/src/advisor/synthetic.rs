//! Synthetic self-check probes for the ops advisor.
//!
//! Periodic *in-process* round-trips against Veda's own paths
//! (health / search / embed), recording per-probe uptime + latency. The
//! results feed two consumers:
//!
//! * the [`SloKind::Synthetic`](crate::advisor::slo::SloKind::Synthetic)
//!   SLI — the engine counts `(good, total)` probe results per tick into
//!   the rolling SLO counter, exactly like the availability/latency SLIs;
//! * the `synthetic_failing` rule — fires when a probe accrues
//!   `consecutive_failures >= K`.
//!
//! ## Model
//!
//! A [`SyntheticProbe`] declares *what* to probe ([`ProbeKind`]) and
//! *how often* (`interval_secs`, with a per-probe `timeout_ms`).
//! [`run_probe`] performs the in-process call and returns a
//! [`ProbeResult`] (ok / latency / error). [`SyntheticRegistry`] persists
//! the probe definitions (mirrors
//! [`SloRegistry`](crate::advisor::slo::SloRegistry)); [`SyntheticResults`]
//! is a per-probe bounded ring of results with a derived
//! [`SyntheticSummary`].
//!
//! ## Cadence
//!
//! Probes are driven by the advisor tick (see
//! [`crate::advisor::engine::run_due_probes`]); a probe whose
//! `interval_secs` has elapsed since its last result runs on the next
//! tick. Effective cadence is therefore bounded *below* by the advisor
//! tick interval (`VEDA_ADVISOR_INTERVAL_SECS`, default 30s) — a probe
//! configured with a sub-tick interval simply runs every tick. This is
//! intentional for v1: the advisor loop is the single in-process timer.
//!
//! ## No self-HTTP
//!
//! [`run_probe`] never opens a socket back to the gateway and never runs
//! the auth/RBAC stack — it calls the underlying functions directly
//! (`run_health_checks`, the `DistributedQueryExecutor`, the embedding
//! registry), so a probe is cheap and cannot deadlock on its own server.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::rest::state::AppState;

/// Per-probe cap on retained results. ~200 results per probe bounds the
/// ring's memory + clone cost; the newest are always retained.
pub const MAX_RESULTS_PER_PROBE: usize = 200;

/// What a synthetic probe exercises. Each variant maps to one in-process
/// path (no self-HTTP, no auth): see [`run_probe`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ProbeKind {
    /// Dependency readiness — runs the same health-check the
    /// `/v1/admin/readiness` endpoint uses; `ok` = not "unhealthy".
    Health,
    /// In-process search against `collection`. A zero query vector
    /// sized to the collection's first vector field, top-1; `ok` = the
    /// executor returns without error. `query` is currently advisory.
    Search {
        collection: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
    },
    /// Embed `text` via the configured text embedder; `ok` = a non-empty
    /// vector is returned. Degrades (ok=false) when no embedder exists.
    Embed { text: String },
}

/// A user-defined synthetic probe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticProbe {
    /// Stable id. May be omitted on create (`POST /v1/admin/synthetics`), in
    /// which case the gateway assigns `synthetic-<uuid>`.
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub kind: ProbeKind,
    /// Minimum seconds between runs (bounded below by the tick interval).
    pub interval_secs: u64,
    /// Per-probe timeout for the in-process call.
    pub timeout_ms: u64,
}

/// The outcome of one probe run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeResult {
    pub ts_ms: u64,
    pub ok: bool,
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A derived, point-in-time rollup over a probe's result ring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticSummary {
    pub probe_id: String,
    pub name: String,
    /// Length of the trailing run of `ok == false` results (0 if the
    /// most recent result is ok, or there are no results).
    pub consecutive_failures: u32,
    /// `ok` of the most recent result; `None` if no results yet.
    pub last_ok: Option<bool>,
    pub last_latency_ms: Option<f64>,
    /// `ok / total` over the retained ring, as a percentage; `None` when
    /// the ring is empty.
    pub uptime_pct: Option<f64>,
    pub last_ts_ms: Option<u64>,
}

/// A thin, persisted registry of [`SyntheticProbe`]s. Mirrors
/// [`SloRegistry`](crate::advisor::slo::SloRegistry): a single JSON file
/// at `{data_dir}/_synthetics.json`, write-through on mutation by the
/// caller, missing/unparseable → empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyntheticRegistry {
    probes: Vec<SyntheticProbe>,
}

impl SyntheticRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self { probes: Vec::new() }
    }

    /// All probes, in insertion order.
    pub fn list(&self) -> &[SyntheticProbe] {
        &self.probes
    }

    /// Find a probe by id.
    pub fn get(&self, id: &str) -> Option<&SyntheticProbe> {
        self.probes.iter().find(|p| p.id == id)
    }

    /// Insert a new probe or replace an existing one with the same id.
    pub fn upsert(&mut self, probe: SyntheticProbe) {
        match self.probes.iter_mut().find(|p| p.id == probe.id) {
            Some(existing) => *existing = probe,
            None => self.probes.push(probe),
        }
    }

    /// Remove the probe with `id`. Returns whether it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.probes.len();
        self.probes.retain(|p| p.id != id);
        self.probes.len() != before
    }

    /// Persist to `path` as pretty JSON. Failure is logged, not surfaced.
    pub fn save(&self, path: &Path) {
        write_json(path, self, "synthetics");
    }

    /// Load from `path`. A missing or unparseable file yields an empty
    /// registry (logged on parse error).
    pub fn load(path: &Path) -> Self {
        load_json(path, "synthetics").unwrap_or_default()
    }
}

/// Per-probe bounded result rings, persisted to
/// `{data_dir}/_synthetic_results.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyntheticResults {
    /// probe_id → newest-last ring of results.
    rings: HashMap<String, Vec<ProbeResult>>,
}

impl SyntheticResults {
    /// Empty results.
    pub fn new() -> Self {
        Self {
            rings: HashMap::new(),
        }
    }

    /// Append a result for `probe_id`, pruning the oldest beyond
    /// [`MAX_RESULTS_PER_PROBE`].
    pub fn record(&mut self, probe_id: &str, result: ProbeResult) {
        let ring = self.rings.entry(probe_id.to_owned()).or_default();
        ring.push(result);
        if ring.len() > MAX_RESULTS_PER_PROBE {
            let excess = ring.len() - MAX_RESULTS_PER_PROBE;
            ring.drain(0..excess);
        }
    }

    /// All retained results for `probe_id`, oldest-first.
    pub fn recent(&self, probe_id: &str) -> Vec<ProbeResult> {
        self.rings.get(probe_id).cloned().unwrap_or_default()
    }

    /// Results for `probe_id` with `ts_ms > since_ms` (strictly after).
    pub fn since(&self, probe_id: &str, since_ms: u64) -> Vec<ProbeResult> {
        self.rings
            .get(probe_id)
            .map(|v| v.iter().filter(|r| r.ts_ms > since_ms).cloned().collect())
            .unwrap_or_default()
    }

    /// Derive the [`SyntheticSummary`] for `probe_id`. `name` is carried
    /// through for display (the results store does not know probe names).
    pub fn summary(&self, probe_id: &str, name: &str) -> SyntheticSummary {
        let ring = self.rings.get(probe_id);
        let mut consecutive_failures = 0u32;
        let mut last_ok = None;
        let mut last_latency_ms = None;
        let mut last_ts_ms = None;
        let mut uptime_pct = None;

        if let Some(ring) = ring.filter(|r| !r.is_empty()) {
            let total = ring.len() as f64;
            let oks = ring.iter().filter(|r| r.ok).count() as f64;
            uptime_pct = Some(oks / total * 100.0);

            let last = ring.last().unwrap();
            last_ok = Some(last.ok);
            last_latency_ms = Some(last.latency_ms);
            last_ts_ms = Some(last.ts_ms);

            // Trailing run of failures (resets on the first ok from the end).
            for r in ring.iter().rev() {
                if r.ok {
                    break;
                }
                consecutive_failures += 1;
            }
        }

        SyntheticSummary {
            probe_id: probe_id.to_owned(),
            name: name.to_owned(),
            consecutive_failures,
            last_ok,
            last_latency_ms,
            uptime_pct,
            last_ts_ms,
        }
    }

    /// Persist to `path` as pretty JSON. Failure is logged, not surfaced.
    pub fn save(&self, path: &Path) {
        write_json(path, self, "synthetic_results");
    }

    /// Load from `path`. Missing/unparseable → empty (logged on parse error).
    pub fn load(path: &Path) -> Self {
        load_json(path, "synthetic_results").unwrap_or_default()
    }
}

/// Run one probe in-process, timing the call and applying `timeout_ms`.
///
/// Never panics, never opens a socket back to the gateway, never runs the
/// auth/RBAC stack. On timeout or any error the result is `ok = false`
/// with an `error` string; the latency is always the wall-clock elapsed
/// for the attempt.
pub async fn run_probe(state: &AppState, probe: &SyntheticProbe) -> ProbeResult {
    let started = Instant::now();
    let now_ms = crate::advisor::engine::now_ms().max(0) as u64;
    let budget = std::time::Duration::from_millis(probe.timeout_ms.max(1));

    let call = async {
        match &probe.kind {
            ProbeKind::Health => probe_health().await,
            ProbeKind::Search { collection, .. } => probe_search(state, collection).await,
            ProbeKind::Embed { text } => probe_embed(state, text).await,
        }
    };

    let outcome = match tokio::time::timeout(budget, call).await {
        Ok(inner) => inner,
        Err(_) => Err(format!("timed out after {} ms", probe.timeout_ms)),
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

    match outcome {
        Ok(()) => ProbeResult {
            ts_ms: now_ms,
            ok: true,
            latency_ms,
            error: None,
        },
        Err(e) => ProbeResult {
            ts_ms: now_ms,
            ok: false,
            latency_ms,
            error: Some(e),
        },
    }
}

/// Health probe: the readiness/health dependency check. `ok` unless the
/// rollup status is "unhealthy" (a critical dependency is down).
async fn probe_health() -> Result<(), String> {
    let checks = crate::rest::admin::run_health_checks().await;
    let unhealthy = checks.get("minio").is_some_and(|s| s == "unavailable")
        || checks.get("nats").is_some_and(|s| s == "unavailable");
    if unhealthy {
        Err(format!("dependency unhealthy: {checks:?}"))
    } else {
        Ok(())
    }
}

/// Search probe: a top-1 zero-vector search against the collection's
/// first dense vector field, through the in-process query executor.
async fn probe_search(state: &AppState, collection: &str) -> Result<(), String> {
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas
            .get(collection)
            .cloned()
            .ok_or_else(|| format!("collection '{collection}' not found"))?
    };
    let field = schema
        .vector_fields()
        .into_iter()
        .next()
        .ok_or_else(|| format!("collection '{collection}' has no vector field"))?;
    let (field_name, dim, metric) = match field.field_type() {
        veda_core::types::FieldType::Vector {
            dimension, metric, ..
        } => (field.name().to_owned(), *dimension as usize, *metric),
        _ => return Err("first vector field is not dense".into()),
    };

    let executors = state.inner.query_executors.read().await;
    let executor = executors
        .get(collection)
        .ok_or_else(|| "no query executor for collection".to_owned())?;

    let request = veda_query::search::VectorSearchRequest {
        field: field_name,
        query_vector: vec![0.0f32; dim],
        k: 1,
        metric,
        filter: None,
        filter_expr: None,
        ef_search: None,
        truncate_to: None,
        read_ts: None,
    };
    executor
        .search(&request)
        .map(|_| ())
        .map_err(|e| format!("search failed: {e}"))
}

/// Embed probe: embed `text` via the first text-modality model in the
/// registry. Degrades (Err) when no text embedder is configured.
async fn probe_embed(state: &AppState, text: &str) -> Result<(), String> {
    use veda_core::embedding::Modality;
    use veda_core::embedding_client::EmbeddingClient;

    let model_config = {
        let registry = state.inner.embedding_registry.read().await;
        let candidates = registry.find_by_modality(Modality::Text);
        candidates
            .first()
            .map(|c| (*c).clone())
            .ok_or_else(|| "no text embedder configured".to_owned())?
    };
    let client = EmbeddingClient::new(model_config);
    let v = client
        .embed_text(text)
        .await
        .map_err(|e| format!("embed_text failed: {e}"))?;
    if v.is_empty() {
        Err("embedder returned an empty vector".into())
    } else {
        Ok(())
    }
}

/// Write `value` to `path` as pretty JSON; failure is logged, not surfaced.
fn write_json<T: Serialize>(path: &Path, value: &T, label: &str) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_vec_pretty(value) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(path, bytes) {
                tracing::warn!(error = %e, path = %path.display(), "{label}: write failed");
            }
        }
        Err(e) => tracing::warn!(error = %e, "{label}: serialize failed"),
    }
}

/// Load `T` from `path`; missing → `None`, parse error → `None` (logged).
fn load_json<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "{label}: parse failed, starting empty");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(ts_ms: u64, ok: bool) -> ProbeResult {
        ProbeResult {
            ts_ms,
            ok,
            latency_ms: 1.0,
            error: if ok { None } else { Some("boom".into()) },
        }
    }

    fn health_probe(id: &str) -> SyntheticProbe {
        SyntheticProbe {
            id: id.to_owned(),
            name: format!("{id} health"),
            kind: ProbeKind::Health,
            interval_secs: 60,
            timeout_ms: 2000,
        }
    }

    #[test]
    fn ring_is_bounded_to_cap() {
        let mut r = SyntheticResults::new();
        let n = MAX_RESULTS_PER_PROBE as u64 + 50;
        for i in 0..n {
            r.record("p", result(i, true));
        }
        let recent = r.recent("p");
        assert_eq!(recent.len(), MAX_RESULTS_PER_PROBE);
        // Newest retained, oldest dropped.
        assert_eq!(recent.last().unwrap().ts_ms, n - 1);
        assert_eq!(recent.first().unwrap().ts_ms, 50);
    }

    #[test]
    fn summary_consecutive_failures_counts_trailing_run() {
        let mut r = SyntheticResults::new();
        // ok, ok, fail, fail, fail (trailing) → 3 consecutive failures.
        r.record("p", result(1, true));
        r.record("p", result(2, true));
        r.record("p", result(3, false));
        r.record("p", result(4, false));
        r.record("p", result(5, false));
        let s = r.summary("p", "P");
        assert_eq!(s.consecutive_failures, 3);
        assert_eq!(s.last_ok, Some(false));
        assert_eq!(s.last_ts_ms, Some(5));
        // 2 ok / 5 total = 40% uptime.
        assert!((s.uptime_pct.unwrap() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn summary_resets_consecutive_on_ok() {
        let mut r = SyntheticResults::new();
        r.record("p", result(1, false));
        r.record("p", result(2, false));
        r.record("p", result(3, true)); // trailing ok → reset to 0
        let s = r.summary("p", "P");
        assert_eq!(s.consecutive_failures, 0);
        assert_eq!(s.last_ok, Some(true));
        assert!((s.uptime_pct.unwrap() - (1.0 / 3.0 * 100.0)).abs() < 1e-9);
    }

    #[test]
    fn summary_empty_ring_is_all_none() {
        let r = SyntheticResults::new();
        let s = r.summary("missing", "M");
        assert_eq!(s.consecutive_failures, 0);
        assert_eq!(s.last_ok, None);
        assert_eq!(s.last_latency_ms, None);
        assert_eq!(s.uptime_pct, None);
        assert_eq!(s.last_ts_ms, None);
    }

    #[test]
    fn since_filters_strictly_after_ts() {
        let mut r = SyntheticResults::new();
        for i in 1..=5 {
            r.record("p", result(i * 100, true));
        }
        // ts_ms > 250 → 300, 400, 500.
        let after = r.since("p", 250);
        assert_eq!(after.len(), 3);
        assert!(after.iter().all(|x| x.ts_ms > 250));
        // Everything when since is 0.
        assert_eq!(r.since("p", 0).len(), 5);
        // Nothing for an unknown probe.
        assert!(r.since("nope", 0).is_empty());
    }

    #[test]
    fn registry_upsert_get_remove_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("_synthetics.json");

        let mut reg = SyntheticRegistry::new();
        assert!(reg.list().is_empty());
        reg.upsert(health_probe("a"));
        reg.upsert(health_probe("b"));
        assert_eq!(reg.list().len(), 2);

        // Upsert same id replaces, not appends.
        let mut a2 = health_probe("a");
        a2.interval_secs = 5;
        reg.upsert(a2);
        assert_eq!(reg.list().len(), 2);
        assert_eq!(reg.get("a").unwrap().interval_secs, 5);

        reg.save(&path);
        let loaded = SyntheticRegistry::load(&path);
        assert_eq!(loaded.list().len(), 2);
        assert_eq!(loaded.get("a").unwrap().interval_secs, 5);

        let mut reg = loaded;
        assert!(reg.remove("a"));
        assert!(!reg.remove("nope"));
        assert!(reg.get("a").is_none());
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn registry_load_missing_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.json");
        assert!(SyntheticRegistry::load(&path).list().is_empty());
    }

    #[test]
    fn results_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("_synthetic_results.json");
        let mut r = SyntheticResults::new();
        r.record("a", result(1, true));
        r.record("a", result(2, false));
        r.record("b", result(3, true));
        r.save(&path);

        let loaded = SyntheticResults::load(&path);
        assert_eq!(loaded.recent("a").len(), 2);
        assert_eq!(loaded.recent("b").len(), 1);
        assert_eq!(loaded.summary("a", "A").consecutive_failures, 1);
    }
}
