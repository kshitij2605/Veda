//! Semantic spike root-cause attribution.
//!
//! When the WARN/ERROR log rate spikes, this module attributes the spike
//! to the semantic **categories** driving it — Veda's vector-native answer
//! to Elastic's statistical "log rate analysis". Rather than bucketing by
//! a fixed field, it clusters the recent warnings/errors into semantic
//! categories (reusing the [`crate::logs::categorize`] machinery) and
//! compares each category's count in the **spike window** against the
//! **baseline window** to find which categories grew.
//!
//! Two layers:
//!
//! * A **pure** [`attribute`] over pre-categorized per-window counts — for
//!   each category `growth = max(0, spike_count - baseline_count)`,
//!   `share_of_growth = growth / total_growth`, and `is_new` when the
//!   category was absent in the baseline. Drivers are ranked by growth
//!   desc. Fully unit-testable with constructed counts; no I/O.
//! * A thin async [`analyze_log_rate`] gatherer that pulls WARN+ERROR
//!   records for the two windows from the [`LogBuffer`], categorizes their
//!   **union** (embedding path off-lock & cache-aware when a text model is
//!   configured, normalized-template fallback otherwise → `method =
//!   "template"`), counts each category per window, and calls [`attribute`].
//!   No lock is ever held across an embedding `.await`.

use std::collections::HashMap;

use serde::Serialize;
use veda_core::embedding::Modality;

use crate::logs::buffer::LogRecord;
use crate::logs::categorize::{
    cluster_embedded, cluster_normalized, normalize_message,
};
use crate::rest::state::AppState;

/// Maximum number of distinct messages embedded / considered per call.
const MAX_MESSAGES: usize = 400;
/// Maximum number of drivers returned.
const MAX_DRIVERS: usize = 25;
/// Cosine threshold for the embedding clustering path (matches
/// `categorize::DEFAULT_THRESHOLD`).
const DEFAULT_THRESHOLD: f32 = 0.82;

/// One semantic category's contribution to the spike.
#[derive(Debug, Clone, Serialize)]
pub struct Driver {
    /// Stable category id (shared with `categorize::LogCategory::id`).
    pub category: String,
    /// Representative (normalized) message for the category.
    pub representative: String,
    /// Count of this category in the spike window.
    pub spike_count: u64,
    /// Count of this category in the baseline window.
    pub baseline_count: u64,
    /// `spike_count - baseline_count`, floored at 0.
    pub growth: u64,
    /// Fraction of the total positive growth attributed to this category,
    /// in [0, 1]. 0.0 when there is no positive growth.
    pub share_of_growth: f64,
    /// True when the category was absent in the baseline window.
    pub is_new: bool,
}

/// Attribution of a log-rate spike to the semantic categories driving it.
#[derive(Debug, Clone, Serialize)]
pub struct SpikeAttribution {
    /// Spike window length in seconds.
    pub window_secs: u64,
    /// Baseline window length in seconds.
    pub baseline_secs: u64,
    /// Total WARN+ERROR count in the spike window.
    pub spike_count: u64,
    /// Total WARN+ERROR count in the baseline window.
    pub baseline_count: u64,
    /// `spike_count - baseline_count` (may be negative → no spike).
    pub total_delta: i64,
    /// Categories driving the spike, ranked by growth desc. Only
    /// categories with positive growth are included.
    pub drivers: Vec<Driver>,
    /// Categorization method used: `"embedding"` or `"template"`.
    pub method: String,
}

impl SpikeAttribution {
    /// An empty/zero attribution (no logs in either window).
    fn empty(window_secs: u64, baseline_secs: u64, method: &str) -> Self {
        Self {
            window_secs,
            baseline_secs,
            spike_count: 0,
            baseline_count: 0,
            total_delta: 0,
            drivers: Vec::new(),
            method: method.to_owned(),
        }
    }
}

/// A category's count in one window, paired with its representative.
/// Input to the pure [`attribute`].
#[derive(Debug, Clone)]
pub struct CategoryCount {
    pub category: String,
    pub representative: String,
    pub count: u64,
}

/// Pure attribution core. Given the per-category counts for the spike
/// window and the baseline window, compute the per-category growth, the
/// share of total growth, and the `is_new` flag, then rank drivers by
/// growth desc. Only categories with positive growth become drivers.
///
/// `total_delta` is the difference of the *window totals* (may be
/// negative — a flatter or quieter spike window has no significant
/// drivers). Shares are over the sum of positive per-category growths, so
/// they sum to ~1.0 whenever any category grew.
pub fn attribute(
    spike: &[CategoryCount],
    baseline: &[CategoryCount],
    window_secs: u64,
    baseline_secs: u64,
    method: &str,
) -> SpikeAttribution {
    let baseline_by_id: HashMap<&str, u64> = baseline
        .iter()
        .map(|c| (c.category.as_str(), c.count))
        .collect();

    let spike_total: u64 = spike.iter().map(|c| c.count).sum();
    let baseline_total: u64 = baseline.iter().map(|c| c.count).sum();

    // Per-category growth = max(0, spike - baseline).
    let mut drivers: Vec<Driver> = spike
        .iter()
        .map(|c| {
            let baseline_count = baseline_by_id.get(c.category.as_str()).copied().unwrap_or(0);
            let growth = c.count.saturating_sub(baseline_count);
            Driver {
                category: c.category.clone(),
                representative: c.representative.clone(),
                spike_count: c.count,
                baseline_count,
                growth,
                share_of_growth: 0.0,
                is_new: baseline_count == 0,
            }
        })
        .filter(|d| d.growth > 0)
        .collect();

    let total_growth: u64 = drivers.iter().map(|d| d.growth).sum();
    if total_growth > 0 {
        for d in drivers.iter_mut() {
            d.share_of_growth = d.growth as f64 / total_growth as f64;
        }
    }

    // Rank by growth desc, then spike_count desc, then id for determinism.
    drivers.sort_by(|a, b| {
        b.growth
            .cmp(&a.growth)
            .then(b.spike_count.cmp(&a.spike_count))
            .then(a.category.cmp(&b.category))
    });
    drivers.truncate(MAX_DRIVERS);

    SpikeAttribution {
        window_secs,
        baseline_secs,
        spike_count: spike_total,
        baseline_count: baseline_total,
        total_delta: spike_total as i64 - baseline_total as i64,
        drivers,
        method: method.to_owned(),
    }
}

/// Fold a list of `(category_id, representative, ts_ms)` assignments into
/// per-category counts (representative = the first seen for the id).
fn count_by_category(items: &[(String, String, i64)]) -> Vec<CategoryCount> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, CategoryCount> = HashMap::new();
    for (id, rep, _ts) in items {
        match map.get_mut(id) {
            Some(cc) => cc.count += 1,
            None => {
                order.push(id.clone());
                map.insert(
                    id.clone(),
                    CategoryCount {
                        category: id.clone(),
                        representative: rep.clone(),
                        count: 1,
                    },
                );
            }
        }
    }
    order.into_iter().filter_map(|k| map.remove(&k)).collect()
}

/// Pull WARN+ERROR records for the spike + baseline windows, categorize
/// their union, count per window, and attribute the spike.
///
/// * Spike window: `[now - window_secs, now]`.
/// * Baseline window: `[now - window_secs - baseline_secs, now -
///   window_secs]` — the equal-or-longer span immediately before the
///   spike window, so a category's baseline rate is its rate just prior.
///
/// `level_min` (e.g. `"WARN"` or `"ERROR"`) filters the records pulled
/// from the buffer. Categorization reuses the `categorize` clustering:
/// the embedding path (cache-aware, embedded OFF any lock) when a text
/// model is configured, else the normalized-template fallback
/// (`method = "template"`). Returns a stable empty result when there are
/// no logs in either window.
///
/// **Lock discipline:** the embedding-registry read lock is dropped
/// before any embed `.await`; no lock is held across a network call.
pub async fn analyze_log_rate(
    state: &AppState,
    window_secs: u64,
    baseline_secs: u64,
    level_min: &str,
) -> SpikeAttribution {
    let window_secs = window_secs.max(1);
    let baseline_secs = baseline_secs.max(1);
    let now_ms = now_ms();

    let spike_lo = now_ms - (window_secs as i64) * 1000;
    let baseline_hi = spike_lo;
    let baseline_lo = baseline_hi - (baseline_secs as i64) * 1000;

    // Pull WARN+ERROR records covering both windows (newest-first), then
    // keep only those within [baseline_lo, now].
    let records: Vec<LogRecord> = state
        .log_buffer()
        .recent(MAX_MESSAGES, Some(level_min), None)
        .into_iter()
        .filter(|r| r.ts_ms >= baseline_lo && r.ts_ms <= now_ms)
        .collect();

    if records.is_empty() {
        return SpikeAttribution::empty(window_secs, baseline_secs, "template");
    }

    // Resolve a text embedding model config (clone it so we don't hold the
    // registry borrow across the awaits). None → template fallback.
    let text_model = {
        let registry = state.embedding_registry().read().await;
        registry
            .find_by_modality(Modality::Text)
            .first()
            .map(|c| (*c).clone())
    };

    // Categorize the union. Each entry is (category_id, representative,
    // ts_ms) for one record.
    let (assignments, method) = match text_model {
        Some(model) => match categorize_embedded(&records, model).await {
            Some(a) => (a, "embedding"),
            None => (categorize_template(&records), "template"),
        },
        None => (categorize_template(&records), "template"),
    };

    // Split by window and count per category.
    let mut spike_items: Vec<(String, String, i64)> = Vec::new();
    let mut baseline_items: Vec<(String, String, i64)> = Vec::new();
    for a in assignments {
        if a.2 >= spike_lo {
            spike_items.push(a);
        } else {
            baseline_items.push(a);
        }
    }

    attribute(
        &count_by_category(&spike_items),
        &count_by_category(&baseline_items),
        window_secs,
        baseline_secs,
        method,
    )
}

/// Embedding categorization of the union: embed each distinct message
/// (no shared cache — this is a one-shot analysis) off-lock, cluster by
/// cosine via `categorize::cluster_embedded`, then map every record to
/// its category id by re-running the same greedy assignment. Returns
/// `None` on the first embedding error so the caller falls back.
async fn categorize_embedded(
    records: &[LogRecord],
    model: veda_core::embedding::EmbeddingModelConfig,
) -> Option<Vec<(String, String, i64)>> {
    use veda_core::embedding_client::EmbeddingClient;
    let client = EmbeddingClient::new(model);

    // Embed distinct messages only (NO lock held across these awaits).
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    for r in records {
        if cache.contains_key(&r.message) {
            continue;
        }
        match client.embed_text(&r.message).await {
            Ok(emb) => {
                cache.insert(r.message.clone(), emb);
            }
            Err(_) => return None,
        }
    }

    // Build (msg, emb, ts, level) items in buffer order, cluster them, and
    // map each record to a category id by its representative template.
    let items: Vec<(String, Vec<f32>, i64, String)> = records
        .iter()
        .map(|r| {
            (
                r.message.clone(),
                cache.get(&r.message).cloned().unwrap_or_default(),
                r.ts_ms,
                r.level.clone(),
            )
        })
        .collect();

    let cats = cluster_embedded(&items, DEFAULT_THRESHOLD);

    // Assign each record to its nearest category centroid. We re-derive
    // the assignment greedily against the category representatives'
    // embeddings via cosine; simpler & deterministic: re-cluster yields
    // categories whose `examples`/`representative` we can match. Instead,
    // run the same greedy pass to label every record.
    Some(label_records_embedded(&items, &cats))
}

/// Label each record with a category id by re-running the greedy cosine
/// assignment used to build `cats`. Mirrors `cluster_embedded`'s pass so
/// labels are consistent with the clustering, then maps centroid index →
/// category id.
fn label_records_embedded(
    items: &[(String, Vec<f32>, i64, String)],
    cats: &[crate::logs::categorize::LogCategory],
) -> Vec<(String, String, i64)> {
    use crate::logs::categorize::cosine;

    // Rebuild centroids by replaying the greedy assignment (same order &
    // threshold as cluster_embedded), tracking each centroid's category
    // id from the representative template.
    struct C {
        centroid: Vec<f32>,
        count: f32,
        id: String,
        rep: String,
    }
    let mut accs: Vec<C> = Vec::new();
    let mut out: Vec<(String, String, i64)> = Vec::with_capacity(items.len());

    for (_msg, emb, ts, _level) in items {
        let mut best: Option<usize> = None;
        let mut best_sim = DEFAULT_THRESHOLD;
        for (i, acc) in accs.iter().enumerate() {
            let sim = cosine(&acc.centroid, emb);
            if sim >= best_sim {
                best_sim = sim;
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                let acc = &mut accs[i];
                let n = acc.count;
                for k in 0..acc.centroid.len() {
                    acc.centroid[k] = (acc.centroid[k] * n + emb[k]) / (n + 1.0);
                }
                acc.count += 1.0;
                out.push((acc.id.clone(), acc.rep.clone(), *ts));
            }
            None => {
                let rep = normalize_message(_msg);
                // The id derived here matches cluster_embedded's
                // category_id(rep); look it up in cats for safety, falling
                // back to the locally-derived rep.
                let id = cats
                    .iter()
                    .find(|c| c.representative == rep)
                    .map(|c| c.id.clone())
                    .unwrap_or_else(|| category_id(&rep));
                accs.push(C {
                    centroid: emb.clone(),
                    count: 1.0,
                    id: id.clone(),
                    rep: rep.clone(),
                });
                out.push((id, rep, *ts));
            }
        }
    }
    out
}

/// Derive a category id from a representative template — must match
/// `categorize::category_id`.
fn category_id(template: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    template.hash(&mut h);
    format!("cat_{:016x}", h.finish())
}

/// Template categorization of the union: group by normalized template via
/// `categorize::cluster_normalized`, then label each record with the id of
/// the category whose representative equals its normalized template.
fn categorize_template(records: &[LogRecord]) -> Vec<(String, String, i64)> {
    let norm_items: Vec<(String, i64, String)> = records
        .iter()
        .map(|r| (r.message.clone(), r.ts_ms, r.level.clone()))
        .collect();
    let cats = cluster_normalized(&norm_items);
    // Map normalized-template → (id, representative).
    let by_rep: HashMap<String, (String, String)> = cats
        .iter()
        .map(|c| (c.representative.clone(), (c.id.clone(), c.representative.clone())))
        .collect();
    records
        .iter()
        .map(|r| {
            let rep = normalize_message(&r.message);
            match by_rep.get(&rep) {
                Some((id, representative)) => (id.clone(), representative.clone(), r.ts_ms),
                None => (category_id(&rep), rep, r.ts_ms),
            }
        })
        .collect()
}

/// Current wall-clock time in Unix milliseconds.
fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(id: &str, rep: &str, count: u64) -> CategoryCount {
        CategoryCount {
            category: id.to_string(),
            representative: rep.to_string(),
            count,
        }
    }

    #[test]
    fn attribute_dominant_category_ranks_first_with_correct_share() {
        // Spike: catA grew 2→20 (+18), catB grew 5→8 (+3); catC flat 4→4.
        let spike = vec![cc("a", "db timeout", 20), cc("b", "slow query", 8), cc("c", "retry", 4)];
        let baseline = vec![cc("a", "db timeout", 2), cc("b", "slow query", 5), cc("c", "retry", 4)];
        let attr = attribute(&spike, &baseline, 300, 900, "embedding");

        assert_eq!(attr.spike_count, 32);
        assert_eq!(attr.baseline_count, 11);
        assert_eq!(attr.total_delta, 21);
        // catC has zero growth → not a driver.
        assert_eq!(attr.drivers.len(), 2);
        // catA dominates.
        assert_eq!(attr.drivers[0].category, "a");
        assert_eq!(attr.drivers[0].growth, 18);
        assert!(!attr.drivers[0].is_new);
        // shares: 18/21 and 3/21.
        assert!((attr.drivers[0].share_of_growth - 18.0 / 21.0).abs() < 1e-9);
        assert!((attr.drivers[1].share_of_growth - 3.0 / 21.0).abs() < 1e-9);
        // shares sum ~1.
        let sum: f64 = attr.drivers.iter().map(|d| d.share_of_growth).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert_eq!(attr.method, "embedding");
    }

    #[test]
    fn attribute_is_new_flag_set_for_absent_baseline() {
        let spike = vec![cc("new", "panic in shard", 10)];
        let baseline = vec![cc("old", "warn cache", 3)];
        let attr = attribute(&spike, &baseline, 300, 300, "template");
        assert_eq!(attr.drivers.len(), 1);
        assert_eq!(attr.drivers[0].category, "new");
        assert!(attr.drivers[0].is_new, "category absent in baseline is new");
        assert_eq!(attr.drivers[0].baseline_count, 0);
        assert!((attr.drivers[0].share_of_growth - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attribute_flat_windows_have_no_drivers() {
        let spike = vec![cc("a", "x", 5), cc("b", "y", 3)];
        let baseline = vec![cc("a", "x", 5), cc("b", "y", 3)];
        let attr = attribute(&spike, &baseline, 300, 300, "template");
        assert_eq!(attr.total_delta, 0);
        assert!(attr.drivers.is_empty(), "no positive growth → no drivers");
    }

    #[test]
    fn attribute_quieter_spike_window_has_no_drivers() {
        // Spike window is quieter than baseline → negative delta, no growth.
        let spike = vec![cc("a", "x", 1)];
        let baseline = vec![cc("a", "x", 10)];
        let attr = attribute(&spike, &baseline, 300, 300, "template");
        assert_eq!(attr.total_delta, -9);
        assert!(attr.drivers.is_empty());
    }

    #[test]
    fn count_by_category_folds_and_keeps_first_representative() {
        let items = vec![
            ("a".to_string(), "rep a".to_string(), 10i64),
            ("a".to_string(), "rep a2".to_string(), 20i64),
            ("b".to_string(), "rep b".to_string(), 30i64),
        ];
        let counts = count_by_category(&items);
        assert_eq!(counts.len(), 2);
        let a = counts.iter().find(|c| c.category == "a").unwrap();
        assert_eq!(a.count, 2);
        assert_eq!(a.representative, "rep a", "first representative wins");
    }

    #[test]
    fn categorize_template_groups_by_normalized_template() {
        let records = vec![
            rec("WARN", "request 1 failed", 10),
            rec("ERROR", "request 2 failed", 20),
            rec("WARN", "disk full", 30),
        ];
        let assignments = categorize_template(&records);
        assert_eq!(assignments.len(), 3, "one assignment per record");
        // The two "request N failed" share an id; "disk full" differs.
        let ids: Vec<&str> = assignments.iter().map(|a| a.0.as_str()).collect();
        assert_eq!(ids[0], ids[1], "same normalized template → same id");
        assert_ne!(ids[0], ids[2]);
        // Representative carries the <num> template.
        assert!(assignments[0].1.contains("<num>"));
    }

    fn rec(level: &str, message: &str, ts: i64) -> LogRecord {
        LogRecord {
            ts_ms: ts,
            level: level.to_string(),
            target: "test".to_string(),
            message: message.to_string(),
            fields: serde_json::Map::new(),
            trace_id: None,
            span_id: None,
        }
    }

    #[tokio::test]
    async fn analyze_empty_buffer_is_stable_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let attr = analyze_log_rate(&state, 300, 900, "WARN").await;
        assert_eq!(attr.spike_count, 0);
        assert_eq!(attr.baseline_count, 0);
        assert_eq!(attr.total_delta, 0);
        assert!(attr.drivers.is_empty());
        assert_eq!(attr.method, "template");
        assert_eq!(attr.window_secs, 300);
        assert_eq!(attr.baseline_secs, 900);
    }

    #[tokio::test]
    async fn analyze_template_path_attributes_window_growth() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let now = now_ms();
        let buf = state.log_buffer();
        // Baseline window (older than window_secs=300 → > 300s ago): a few
        // "db timeout" + one "slow query".
        for _ in 0..2 {
            buf.push(rec("ERROR", "db timeout on shard 7", now - 600_000));
        }
        buf.push(rec("WARN", "slow query 12 ms", now - 600_000));
        // Spike window (within last 300s): many "db timeout".
        for _ in 0..10 {
            buf.push(rec("ERROR", "db timeout on shard 9", now - 1_000));
        }
        buf.push(rec("WARN", "slow query 34 ms", now - 1_000));

        let attr = analyze_log_rate(&state, 300, 900, "WARN").await;
        assert_eq!(attr.method, "template", "no embedder configured");
        assert!(attr.total_delta > 0, "spike window has more logs");
        // The "db timeout" category dominates the growth.
        assert!(!attr.drivers.is_empty());
        let top = &attr.drivers[0];
        assert!(top.representative.contains("db timeout"), "got {}", top.representative);
        assert!(top.growth >= 8, "db timeout grew ~10-2=8");
        assert!(top.share_of_growth > 0.5);
    }
}
