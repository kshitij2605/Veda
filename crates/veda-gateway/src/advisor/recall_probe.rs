//! Proxy recall probe (ANN-vs-exact) + recall-vs-latency tracker.
//!
//! This is the **measurement layer** for the index/Ops advisor: it
//! estimates, per collection, how well the configured ANN index recovers
//! the *true* nearest neighbours, and pairs that with the observed query
//! p95. The recommendation rules / REST / UI that consume it are Phase 6b
//! and live elsewhere.
//!
//! ## What it measures (and why it is a *proxy*)
//!
//! For a small sample of query vectors drawn from the collection's own
//! stored vectors, [`probe_recall`] computes:
//!
//! * the **ANN top-k** via the existing in-process search path
//!   ([`veda_query::distributed::DistributedQueryExecutor::search`]) — the
//!   exact code a live `/search` runs; and
//! * the **exact top-k** by brute force over the corpus, reusing the
//!   shared distance fn ([`veda_index::distance::compute_distance`]) — no
//!   reimplemented metric.
//!
//! `recall@k = mean over queries of |ANN ∩ exact| / k`.
//!
//! It is a **proxy** (`is_proxy = true`) for two reasons, both bounded to
//! keep cost predictable on a background task:
//!
//! 1. The brute-force exact scan is capped at `corpus_cap` vectors. When
//!    the corpus is larger than the cap, the "exact" set is exact only
//!    over the sampled prefix, so the recall figure is approximate.
//! 2. Query vectors are sampled from the stored corpus, not real traffic;
//!    they are representative but not the live query distribution.
//!
//! ## Cost / safety
//!
//! Brute force is O(corpus_cap · dim) per query, ×`sample_queries`. With
//! the small defaults ([`DEFAULT_SAMPLE_QUERIES`], [`DEFAULT_CORPUS_CAP`])
//! a probe is a few million flops — cheap, but **not** free, which is why
//! it MUST run on the dedicated low-cadence [`run_recall_probes`] task and
//! NEVER inside the advisor tick or while holding an advisor lock. See the
//! module-level lock discipline on [`run_recall_probes`].

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::rest::state::AppState;

/// Default number of query vectors sampled per probe. Small on purpose —
/// the brute-force exact scan is per-query, so this bounds cost.
pub const DEFAULT_SAMPLE_QUERIES: usize = 20;
/// Default top-k for the recall comparison.
pub const DEFAULT_K: usize = 10;
/// Default cap on vectors scanned for the exact brute-force top-k. Above
/// this the exact set is computed over a sampled prefix and recall is
/// approximate (still flagged `is_proxy`).
pub const DEFAULT_CORPUS_CAP: usize = 50_000;
/// A collection with fewer than this many vectors is skipped (`None`):
/// recall is meaningless when k approaches the corpus size.
pub const MIN_RECORDS: usize = 50;
/// Per-collection minimum seconds between probes (env override below).
pub const DEFAULT_PROBE_INTERVAL_SECS: u64 = 600;
/// Per-collection cap on retained recall samples (bounded history).
pub const MAX_SAMPLES_PER_COLLECTION: usize = 100;

/// The index params the tracker reports for a collection. Only
/// `index_type` is persisted on the schema today; the HNSW graph knobs
/// (`m` / `ef_construction` / `ef_search`) are not surfaced on
/// `CollectionSchema`, so they are `None` until that plumbing exists.
/// Phase 6b's rules read `index_type` (+ the recall/p95 pairing) and can
/// tighten this struct later without a shape change here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexParams {
    /// HNSW max-connections (`m`); `None` when not surfaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<usize>,
    /// HNSW `ef_construction`; `None` when not surfaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ef_construction: Option<usize>,
    /// HNSW `ef_search`; `None` when not surfaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
}

/// One proxy-recall measurement for a collection at a point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallSample {
    /// User-facing collection name (tenant prefix stripped).
    pub collection: String,
    /// Lowercase index-type string of the probed vector field
    /// (e.g. "hnsw" / "flat" / "ivf_pq").
    pub index_type: String,
    /// Mean recall@k over the sampled queries, in [0.0, 1.0].
    pub recall_at_k: f64,
    pub k: usize,
    /// How many query vectors actually contributed to the mean.
    pub sampled_queries: usize,
    /// Observed query p95 (ms) for this collection at probe time, when
    /// available from the query log; `None` if no recent queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_ms: Option<f64>,
    pub params: IndexParams,
    pub ts_ms: u64,
    /// Always `true`: this measurement is an approximation (sampled
    /// queries + capped brute force). See the module docs.
    pub is_proxy: bool,
}

/// Mean recall@k of an ANN result set against the exact set, over a set
/// of (ann, exact) id lists. Pure + deterministic so it can be unit
/// tested without any index. `k` is the requested top-k; recall for one
/// query is `|ann ∩ exact| / k` (using the requested `k` as the
/// denominator, not `exact.len()`, so a short exact set — corpus smaller
/// than k — does not inflate recall). Queries with an empty exact set are
/// skipped. Returns `None` when no query contributed.
pub fn mean_recall_at_k(
    per_query: &[(Vec<uuid::Uuid>, Vec<uuid::Uuid>)],
    k: usize,
) -> Option<f64> {
    if k == 0 {
        return None;
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (ann, exact) in per_query {
        if exact.is_empty() {
            continue;
        }
        let exact_set: std::collections::HashSet<&uuid::Uuid> = exact.iter().collect();
        let hits = ann.iter().filter(|id| exact_set.contains(id)).count();
        // Denominator is min(k, |exact|) so a corpus smaller than k can
        // still reach 1.0 (you cannot recall more than exists).
        let denom = k.min(exact.len()) as f64;
        sum += hits as f64 / denom;
        n += 1;
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

/// Probe one collection's proxy recall@k. Returns `None` (a clean skip)
/// when the collection is missing, has no dense vector field, or holds
/// fewer than [`MIN_RECORDS`] vectors.
///
/// Reuses the live in-process ANN path (the same
/// `DistributedQueryExecutor` a real `/search` uses) for the ANN top-k
/// and the shared [`veda_index::distance::compute_distance`] for the
/// exact brute force; it never reimplements a distance metric.
///
/// **Cost:** O(`corpus_cap` · dim · `sample_queries`). Bounded by the
/// caps; still must run off the advisor tick (see [`run_recall_probes`]).
pub async fn probe_recall(
    state: &AppState,
    collection: &str,
    sample_queries: usize,
    k: usize,
    corpus_cap: usize,
) -> Option<RecallSample> {
    use veda_core::record::FieldValue;
    use veda_core::types::FieldType;

    let k = k.max(1);

    // --- resolve the first dense vector field from the schema ---------
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(collection).cloned()?
    };
    let field = schema.vector_fields().into_iter().next()?;
    let (field_name, dim, metric) = match field.field_type() {
        FieldType::Vector {
            dimension, metric, ..
        } => (field.name().to_owned(), *dimension as usize, *metric),
        _ => return None,
    };
    let index_type = index_type_str(field.index_type()).to_owned();

    // --- gather corpus vectors + ids (capped) under a brief seg lock --
    // We collect (record_id, vector) pairs across all shards/segments up
    // to `corpus_cap`. This clones at most `corpus_cap` vectors; the lock
    // is released as soon as the gather completes (before any search).
    let mut corpus: Vec<(uuid::Uuid, Vec<f32>)> = Vec::new();
    {
        let segs = state.inner.segment_managers.read().await;
        let shard_map = segs.get(collection)?;
        'gather: for sm in shard_map.values() {
            let gs = sm.growing_segment();
            for doc_id in 0..gs.count() {
                if gs.is_deleted(doc_id) {
                    continue;
                }
                if let Some(rec) = gs.get_record(doc_id) {
                    if let Some(FieldValue::Vector(v)) = rec.get(&field_name) {
                        if v.as_slice().len() == dim {
                            corpus.push((rec.id(), v.as_slice().to_vec()));
                            if corpus.len() >= corpus_cap {
                                break 'gather;
                            }
                        }
                    }
                }
            }
            for handle in sm.sealed_segments() {
                for doc_id in 0..handle.metadata.row_count as u32 {
                    if handle.reader.delete_bitmap().is_deleted(doc_id) {
                        continue;
                    }
                    if let Ok(rec) = handle.reader.get_record(doc_id) {
                        if let Some(FieldValue::Vector(v)) = rec.get(&field_name) {
                            if v.as_slice().len() == dim {
                                corpus.push((rec.id(), v.as_slice().to_vec()));
                                if corpus.len() >= corpus_cap {
                                    break 'gather;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if corpus.len() < MIN_RECORDS {
        return None; // too few records → recall not meaningful.
    }

    // --- pick query vectors: evenly-spaced sample over the corpus -----
    let n = corpus.len();
    let want = sample_queries.max(1).min(n);
    let stride = (n / want).max(1);
    let query_idxs: Vec<usize> = (0..n).step_by(stride).take(want).collect();

    // --- per-query ANN (in-process) vs exact (brute force) ------------
    // The `query_executors` RwLock is a READ lock and the search runs
    // fully in-process (no network) — same pattern as the synthetic
    // search probe. It is NOT an advisor lock, so this never blocks the
    // advisor tick. Released at the end of this scope.
    let mut per_query: Vec<(Vec<uuid::Uuid>, Vec<uuid::Uuid>)> =
        Vec::with_capacity(query_idxs.len());
    {
        let executors = state.inner.query_executors.read().await;
        let executor = executors.get(collection)?;
        for &qi in &query_idxs {
            let query_vec = corpus[qi].1.clone();

            // Exact top-k via brute force over the capped corpus.
            let exact_ids = exact_top_k(&corpus, &query_vec, k, metric);

            // ANN top-k via the live in-process search path.
            let request = veda_query::search::VectorSearchRequest {
                field: field_name.clone(),
                query_vector: query_vec,
                k,
                metric,
                filter: None,
                filter_expr: None,
                truncate_to: None,
                read_ts: None,
            };
            let ann_ids = match executor.search(&request) {
                Ok((resp, _timings)) => {
                    resp.results.into_iter().map(|r| r.id).collect::<Vec<_>>()
                }
                Err(_) => Vec::new(),
            };

            per_query.push((ann_ids, exact_ids));
        }
    }

    let recall_at_k = mean_recall_at_k(&per_query, k)?;
    let sampled_queries = per_query.iter().filter(|(_, e)| !e.is_empty()).count();

    Some(RecallSample {
        collection: collection.to_owned(),
        index_type,
        recall_at_k,
        k,
        sampled_queries,
        p95_ms: None,
        params: IndexParams::default(),
        ts_ms: crate::advisor::engine::now_ms().max(0) as u64,
        is_proxy: true,
    })
}

/// Exact top-k record ids for `query` over `corpus`, ranked by ascending
/// distance via the shared metric. Ties broken by id for determinism.
fn exact_top_k(
    corpus: &[(uuid::Uuid, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: veda_core::types::DistanceMetric,
) -> Vec<uuid::Uuid> {
    let mut scored: Vec<(f32, uuid::Uuid)> = corpus
        .iter()
        .map(|(id, v)| (veda_index::distance::compute_distance(query, v, metric), *id))
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// Lowercase string form of a field's index type (local copy of the
/// gateway's search-path mapper; kept here so the probe has no cross-
/// module private dep).
fn index_type_str(it: veda_core::types::IndexType) -> &'static str {
    use veda_core::types::IndexType as I;
    match it {
        I::Hnsw => "hnsw",
        I::DiskAnn => "diskann",
        I::IvfPq => "ivf_pq",
        I::Flat => "flat",
        I::RaBitQ => "rabitq",
        I::RaBitQRerank => "rabitq_rerank",
        I::HnswRaBitQ => "hnsw_rabitq",
        I::Cagra => "cagra",
        I::RaBitQExtRerank => "rabitq_ext_rerank",
        I::Sq4 => "sq4",
        I::Sq8 => "sq8",
        I::SuffixArray => "suffix_array",
        I::Inverted => "inverted",
        I::Range => "range",
        I::Bitmap => "bitmap",
        I::Hash => "hash",
        I::None => "none",
    }
}

/// Per-collection latest [`RecallSample`] + a small bounded history,
/// persisted to `{data_dir}/_recall_results.json`. Mirrors
/// [`crate::advisor::synthetic::SyntheticResults`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallResults {
    /// collection → newest-last ring of recall samples.
    rings: HashMap<String, Vec<RecallSample>>,
}

impl RecallResults {
    /// Empty results.
    pub fn new() -> Self {
        Self {
            rings: HashMap::new(),
        }
    }

    /// Append a sample for its collection, pruning the oldest beyond
    /// [`MAX_SAMPLES_PER_COLLECTION`].
    pub fn record(&mut self, sample: RecallSample) {
        let ring = self.rings.entry(sample.collection.clone()).or_default();
        ring.push(sample);
        if ring.len() > MAX_SAMPLES_PER_COLLECTION {
            let excess = ring.len() - MAX_SAMPLES_PER_COLLECTION;
            ring.drain(0..excess);
        }
    }

    /// Most-recent sample for `collection`, if any.
    pub fn latest(&self, collection: &str) -> Option<&RecallSample> {
        self.rings.get(collection).and_then(|r| r.last())
    }

    /// Most-recent sample for `collection`, cloned out — a cheap owned
    /// accessor for the advisor tick, which clones under a brief lock and
    /// then drops the guard before computing the SLI.
    pub fn latest_recall(&self, collection: &str) -> Option<RecallSample> {
        self.latest(collection).cloned()
    }

    /// All retained samples for `collection`, oldest-first.
    pub fn history(&self, collection: &str) -> Vec<RecallSample> {
        self.rings.get(collection).cloned().unwrap_or_default()
    }

    /// The latest sample for every tracked collection.
    pub fn latest_all(&self) -> Vec<RecallSample> {
        self.rings
            .values()
            .filter_map(|r| r.last().cloned())
            .collect()
    }

    /// Timestamp of the most-recent sample for `collection`, or 0.
    pub fn last_ts_ms(&self, collection: &str) -> u64 {
        self.rings
            .get(collection)
            .and_then(|r| r.last())
            .map(|s| s.ts_ms)
            .unwrap_or(0)
    }

    /// Persist to `path` as pretty JSON. Failure is logged, not surfaced.
    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    tracing::warn!(error = %e, path = %path.display(), "recall_results: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "recall_results: serialize failed"),
        }
    }

    /// Load from `path`. Missing/unparseable → empty (logged on parse error).
    pub fn load(path: &Path) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Self::new(),
        };
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "recall_results: parse failed, starting empty");
                Self::new()
            }
        }
    }
}

/// Recall-probe results file name under `data_dir`.
pub const RECALL_RESULTS_FILE: &str = "_recall_results.json";

/// The per-collection minimum seconds between probes (env override,
/// default [`DEFAULT_PROBE_INTERVAL_SECS`]).
pub fn probe_interval_secs() -> u64 {
    std::env::var("VEDA_RECALL_PROBE_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_PROBE_INTERVAL_SECS)
}

/// Probe every collection whose per-collection rate limit has elapsed,
/// recording one [`RecallSample`] each and persisting the results.
///
/// **Lock discipline (the #1 correctness constraint of this task):** the
/// real index + brute-force searches run with **no advisor lock held**.
/// We take the `recall_results` lock only *briefly* — once up front to
/// read each collection's last-probe timestamp (rate limit), and once at
/// the end to record + save. The probe itself ([`probe_recall`]) holds
/// only `AppState` read locks, never an advisor lock. This function is
/// invoked exclusively from the dedicated [`spawn_recall_prober`] task,
/// **never** from `run_tick`, so a long brute-force scan can never stall
/// the advisor's rules / anomaly / notify pipeline.
pub async fn run_recall_probes(state: &AppState) {
    let advisor = state.advisor();
    let now_ms = crate::advisor::engine::now_ms().max(0) as u64;
    let interval_ms = probe_interval_secs().saturating_mul(1_000);

    // Which collections exist (scoped keys) → user-facing names.
    let scoped_names: Vec<String> = {
        let segs = state.inner.segment_managers.read().await;
        segs.keys().cloned().collect()
    };
    if scoped_names.is_empty() {
        return;
    }

    // Decide which are due under a brief recall_results lock, then DROP it.
    let due: Vec<String> = {
        let results = advisor.recall_results.lock().await;
        scoped_names
            .into_iter()
            .filter(|scoped| {
                let display = crate::tenant::display_name(scoped);
                let last = results.last_ts_ms(display);
                last == 0 || now_ms.saturating_sub(last) >= interval_ms
            })
            .collect()
    };
    if due.is_empty() {
        return;
    }

    // Run the probes with NO advisor lock held. Each probe samples its
    // own queries and does the ANN-vs-exact comparison.
    let mut samples: Vec<RecallSample> = Vec::new();
    for scoped in &due {
        let display = crate::tenant::display_name(scoped).to_owned();
        if let Some(mut sample) = probe_recall(
            state,
            scoped,
            DEFAULT_SAMPLE_QUERIES,
            DEFAULT_K,
            DEFAULT_CORPUS_CAP,
        )
        .await
        {
            // Pair with the observed p95 from the query log (cheap read).
            sample.collection = display.clone();
            sample.p95_ms = collection_p95_ms(state, scoped).await;
            emit_recall_metrics(&display, &sample);
            samples.push(sample);
        }
    }
    if samples.is_empty() {
        return;
    }

    // Re-lock briefly to record + persist.
    let mut results = advisor.recall_results.lock().await;
    for s in samples {
        results.record(s);
    }
    results.save(&state.inner.data_dir.join(RECALL_RESULTS_FILE));
}

/// Observed query p95 (ms) for one scoped collection over the recent
/// query-log window. `None` when there are no recent queries.
async fn collection_p95_ms(state: &AppState, scoped: &str) -> Option<f64> {
    let data_dir = state.inner.data_dir.clone();
    let scoped = scoped.to_owned();
    let mut took: Vec<u64> = tokio::task::spawn_blocking(move || {
        crate::rest::query_log::read_window_entries(&data_dir, &scoped, 300)
            .into_iter()
            .map(|e| e.took_ms)
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    if took.is_empty() {
        return None;
    }
    took.sort_unstable();
    let rank = (0.95 * (took.len() as f64 - 1.0)).round() as usize;
    Some(took[rank.min(took.len() - 1)] as f64)
}

/// Emit the per-collection recall + p95 gauges. Inline `metrics::gauge!`
/// (no registration), labelled by collection.
fn emit_recall_metrics(collection: &str, sample: &RecallSample) {
    metrics::gauge!("veda_index_recall", "collection" => collection.to_owned())
        .set(sample.recall_at_k);
    if let Some(p95) = sample.p95_ms {
        metrics::gauge!("veda_index_p95_ms", "collection" => collection.to_owned()).set(p95);
    }
}

/// Spawn the background recall-probe loop. Ticks every
/// [`probe_interval_secs`] (default 600s) — **much slower than the
/// advisor tick (30s)** — and is a *separate* tokio task from
/// [`crate::advisor::spawn_advisor`], so the expensive brute-force scans
/// never run inside `run_tick`. A failing/slow [`run_recall_probes`]
/// cannot crash the loop (panics are isolated to the task).
pub fn spawn_recall_prober(state: AppState) -> tokio::task::JoinHandle<()> {
    let secs = probe_interval_secs();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(secs));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            run_recall_probes(&state).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenant;
    use uuid::Uuid;
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::{DistanceMetric, IndexType};
    use veda_core::vector::DenseVector;

    // ---- pure recall math ------------------------------------------------

    #[test]
    fn mean_recall_perfect_when_sets_match() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let pq = vec![(vec![a, b, c], vec![a, b, c])];
        assert_eq!(mean_recall_at_k(&pq, 3), Some(1.0));
    }

    #[test]
    fn mean_recall_under_one_when_ann_misses() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();
        // ANN got a,b but missed c (returned d instead) → 2/3 recall.
        let pq = vec![(vec![a, b, d], vec![a, b, c])];
        let r = mean_recall_at_k(&pq, 3).unwrap();
        assert!((r - (2.0 / 3.0)).abs() < 1e-9, "recall was {r}");
    }

    #[test]
    fn mean_recall_averages_across_queries_and_skips_empty_exact() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        // q1: perfect (1.0); q2: empty exact → skipped; q3: 0.0.
        let pq = vec![
            (vec![a, b], vec![a, b]),
            (vec![a], vec![]),
            (vec![a], vec![b]),
        ];
        // mean over the two non-empty: (1.0 + 0.0) / 2 = 0.5.
        assert_eq!(mean_recall_at_k(&pq, 2), Some(0.5));
    }

    #[test]
    fn mean_recall_none_when_no_contributors() {
        let a = Uuid::new_v4();
        assert_eq!(mean_recall_at_k(&[(vec![a], vec![])], 2), None);
        assert_eq!(mean_recall_at_k(&[], 2), None);
        assert_eq!(mean_recall_at_k(&[(vec![a], vec![a])], 0), None);
    }

    // ---- RecallResults store --------------------------------------------

    fn sample(collection: &str, recall: f64, ts: u64) -> RecallSample {
        RecallSample {
            collection: collection.into(),
            index_type: "hnsw".into(),
            recall_at_k: recall,
            k: 10,
            sampled_queries: 20,
            p95_ms: Some(12.0),
            params: IndexParams::default(),
            ts_ms: ts,
            is_proxy: true,
        }
    }

    #[test]
    fn results_record_latest_and_history() {
        let mut r = RecallResults::new();
        r.record(sample("docs", 0.9, 100));
        r.record(sample("docs", 0.95, 200));
        r.record(sample("imgs", 0.8, 150));
        assert_eq!(r.latest("docs").unwrap().recall_at_k, 0.95);
        assert_eq!(r.latest("docs").unwrap().ts_ms, 200);
        assert_eq!(r.history("docs").len(), 2);
        assert_eq!(r.last_ts_ms("docs"), 200);
        assert_eq!(r.last_ts_ms("missing"), 0);
        assert_eq!(r.latest_all().len(), 2);
    }

    #[test]
    fn results_history_is_bounded() {
        let mut r = RecallResults::new();
        let n = MAX_SAMPLES_PER_COLLECTION as u64 + 25;
        for i in 0..n {
            r.record(sample("docs", 0.5, i));
        }
        let hist = r.history("docs");
        assert_eq!(hist.len(), MAX_SAMPLES_PER_COLLECTION);
        // newest kept, oldest dropped.
        assert_eq!(hist.last().unwrap().ts_ms, n - 1);
        assert_eq!(hist.first().unwrap().ts_ms, 25);
    }

    #[test]
    fn results_save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(RECALL_RESULTS_FILE);
        let mut r = RecallResults::new();
        r.record(sample("docs", 0.9, 1));
        r.record(sample("docs", 0.92, 2));
        r.save(&path);

        let loaded = RecallResults::load(&path);
        assert_eq!(loaded.history("docs").len(), 2);
        assert_eq!(loaded.latest("docs").unwrap().recall_at_k, 0.92);
        assert!(loaded.latest("docs").unwrap().is_proxy);
    }

    #[test]
    fn results_load_missing_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.json");
        assert!(RecallResults::load(&path).latest_all().is_empty());
    }

    // ---- probe_recall end-to-end on a real collection -------------------

    fn vec_schema(scoped: &str, index_type: IndexType) -> CollectionSchema {
        CollectionSchema::builder(scoped)
            .add_field(
                FieldDefinition::vector("v", 8, DistanceMetric::Cosine)
                    .with_index_type(index_type),
            )
            .shard_count(1)
            .build()
            .unwrap()
    }

    fn record_with_vec(values: Vec<f32>) -> Record {
        let mut r = Record::new();
        r.set("v", FieldValue::Vector(DenseVector::new(values)));
        r
    }

    /// Seed a Flat-index collection with `n` distinct 8-dim vectors.
    async fn seed_flat_collection(state: &AppState, user_name: &str, n: usize) -> String {
        let scoped = format!("{}__{}", tenant::DEFAULT_TENANT, user_name);
        let schema = vec_schema(&scoped, IndexType::Flat);
        let collection_dir = state.inner.data_dir.join(&scoped);
        std::fs::create_dir_all(&collection_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(state, &schema, &collection_dir)
            .await
            .unwrap();
        {
            let mut managers = state.inner.segment_managers.write().await;
            let shard_map = managers.get_mut(&scoped).unwrap();
            let sm = shard_map.get_mut(&0).unwrap();
            for i in 0..n {
                // Distinct, well-separated vectors: a moving spike per record.
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0 + (i as f32) * 0.01;
                v[(i + 1) % 8] = 0.5;
                sm.insert(record_with_vec(v)).unwrap();
            }
            // Seal the growing segment to disk so the query executor's
            // searcher (which loads SEALED segments from disk) can see the
            // data — this mirrors the production insert→seal→rebuild path.
            sm.seal_growing().unwrap();
        }
        // Rebuild the executor so its LocalShardSearcher loads the sealed
        // segment + builds the (Flat) index. Same call the insert handler
        // makes after data changes.
        crate::rest::records::rebuild_executors(state, &scoped)
            .await
            .unwrap();
        scoped
    }

    #[tokio::test]
    async fn probe_recall_flat_index_is_near_perfect() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let scoped = seed_flat_collection(&state, "docs", 200).await;

        let sample = probe_recall(&state, &scoped, 15, 10, 50_000)
            .await
            .expect("flat collection should yield a recall sample");

        assert_eq!(sample.index_type, "flat");
        assert!(sample.is_proxy);
        assert_eq!(sample.k, 10);
        assert!(sample.sampled_queries > 0);
        // Flat == exact search over the same corpus → recall must be 1.0.
        assert!(
            (sample.recall_at_k - 1.0).abs() < 1e-9,
            "flat recall should be 1.0, got {}",
            sample.recall_at_k
        );
    }

    #[tokio::test]
    async fn probe_recall_too_few_records_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        // Fewer than MIN_RECORDS → skip cleanly.
        let scoped = seed_flat_collection(&state, "tiny", MIN_RECORDS - 1).await;
        assert!(probe_recall(&state, &scoped, 10, 10, 50_000).await.is_none());
    }

    #[tokio::test]
    async fn probe_recall_missing_collection_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        assert!(probe_recall(&state, "nope", 10, 10, 50_000).await.is_none());
    }

    #[tokio::test]
    async fn run_recall_probes_records_and_rate_limits() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_flat_collection(&state, "docs", 120).await;

        run_recall_probes(&state).await;
        {
            let results = state.advisor().recall_results.lock().await;
            let latest = results.latest("docs").expect("docs recall recorded");
            assert!((latest.recall_at_k - 1.0).abs() < 1e-9);
            assert_eq!(results.history("docs").len(), 1);
        }

        // Second immediate run is rate-limited (default 600s) → no new sample.
        run_recall_probes(&state).await;
        {
            let results = state.advisor().recall_results.lock().await;
            assert_eq!(
                results.history("docs").len(),
                1,
                "second run within the interval must be rate-limited"
            );
        }
    }
}
