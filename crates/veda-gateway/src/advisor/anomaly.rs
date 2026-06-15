//! Lightweight statistical baseline / anomaly detection over the
//! recent [`SnapshotHistory`].
//!
//! No ML crate, no external calls: for each tracked signal we build a
//! short numeric series from the history, compute the mean and
//! population standard deviation over the **baseline window** (all
//! samples except the most recent), and fire a [`RuleHit`] when the
//! **current** (most-recent) value sits in the upper tail —
//! `current > mean + sigma * stddev`. These signals are all
//! "bad when high", so we only watch the upper tail.
//!
//! The hits are `Scope::Global` with `RemediationAction::None`; the
//! advisor engine (a later task) feeds them into the same
//! `EventStore::reconcile` as the threshold-rule hits, so anomalies get
//! the same open/close lifecycle and notifications.
//!
//! Guards (keep it boring and robust):
//! * warm-up — fewer than `min_samples` points → never fire;
//! * `stddev > 0` over the baseline → otherwise we skip (a flat series
//!   has no meaningful spread and would divide by zero);
//! * the signal's rule must be `enabled && !muted` in [`AdvisorConfig`].

use crate::advisor::config::AdvisorConfig;
use crate::advisor::event::{RemediationAction, Scope, Severity};
use crate::advisor::rules::RuleHit;
use crate::advisor::snapshot::SnapshotHistory;

/// Default upper-tail threshold in standard deviations.
const DEFAULT_SIGMA: f64 = 3.0;
/// Default warm-up: minimum series length before we trust a baseline.
const DEFAULT_MIN_SAMPLES: usize = 20;
/// Deviation (in sigmas) at or above which a hit is escalated to High.
const HIGH_SIGMA: f64 = 5.0;

/// Rule ids for the anomaly detectors, exposed so the rules-listing
/// REST surface (a later task) can show them in the catalog alongside
/// the threshold rules.
const ANOMALY_RULE_IDS: &[&str] = &[
    "anomaly_search_latency",
    "anomaly_error_rate",
    "anomaly_insert_rate",
    "anomaly_rss",
    "anomaly_error_logs",
];

/// The anomaly-detector rule ids (stable, unique).
pub fn anomaly_rule_ids() -> &'static [&'static str] {
    ANOMALY_RULE_IDS
}

/// Population mean and standard deviation of `values`.
///
/// Returns `(0.0, 0.0)` for an empty slice. Uses the population
/// (divide-by-N) standard deviation, matching "spread of the baseline
/// I have observed" rather than a sample estimate.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}

/// Detect anomalies across every tracked signal.
///
/// For each signal we build its recent series (oldest-first), split off
/// the most-recent value as `current`, compute mean+stddev over the
/// preceding baseline, and emit a `RuleHit` when `current` exceeds
/// `mean + sigma*stddev`. See the module docs for the guards.
pub fn detect_anomalies(history: &SnapshotHistory, cfg: &AdvisorConfig) -> Vec<RuleHit> {
    let mut hits = Vec::new();

    // p95 search latency — present on every snapshot.
    if let Some(hit) = detect_signal(
        cfg,
        "anomaly_search_latency",
        "search p95 latency",
        "ms",
        history.series(|s| s.query.p95_ms),
    ) {
        hits.push(hit);
    }

    // Error rate (errors/sec) — derived from the counter via pairwise
    // deltas; points where a rate can't be computed are skipped.
    if let Some(hit) = detect_signal(
        cfg,
        "anomaly_error_rate",
        "error rate",
        "err/s",
        rate_series(history, "veda_errors_total"),
    ) {
        hits.push(hit);
    }

    // Insert rate (records/sec). v1 watches the UPPER tail only — a
    // spike in writes; a *drop* (which can also matter) is intentionally
    // not flagged here to keep the detector single-tailed and simple.
    if let Some(hit) = detect_signal(
        cfg,
        "anomaly_insert_rate",
        "insert rate",
        "rec/s",
        rate_series(history, "veda_records_total"),
    ) {
        hits.push(hit);
    }

    // Resident set size — skip snapshots where it is unknown (None).
    if let Some(hit) = detect_signal(
        cfg,
        "anomaly_rss",
        "process RSS",
        "MiB",
        rss_series(history),
    ) {
        hits.push(hit);
    }

    // ERROR-log volume in the recent window — present on every snapshot.
    if let Some(hit) = detect_signal(
        cfg,
        "anomaly_error_logs",
        "error-log volume",
        "logs/5m",
        history.series(|s| s.error_log_count as f64),
    ) {
        hits.push(hit);
    }

    hits
}

/// Build a per-snapshot rate (per-second delta) series for `metric`
/// across the history, oldest-first. A rate needs two consecutive
/// snapshots, so the series is one shorter than the snapshot count;
/// points where the gap is non-positive or the counter went backwards
/// (a reset) are dropped.
fn rate_series(history: &SnapshotHistory, metric: &str) -> Vec<f64> {
    // `series` only yields f64, so encode a missing counter as NaN and
    // skip any interval that touches a NaN endpoint.
    let times = history.series(|s| s.taken_at_ms as f64);
    let counts =
        history.series(|s| s.counters.get(metric).copied().unwrap_or(f64::NAN));
    let mut out = Vec::new();
    for i in 1..times.len() {
        let (prev, now) = (counts[i - 1], counts[i]);
        if prev.is_nan() || now.is_nan() {
            continue;
        }
        let dt_s = (times[i] - times[i - 1]) / 1000.0;
        if dt_s <= 0.0 {
            continue;
        }
        let delta = now - prev;
        if delta < 0.0 {
            continue; // counter reset — can't interpret a partial rate.
        }
        out.push(delta / dt_s);
    }
    out
}

/// RSS-in-MiB series, skipping snapshots where RSS is unknown.
fn rss_series(history: &SnapshotHistory) -> Vec<f64> {
    history
        .series(|s| s.rss_bytes.map(|b| b as f64 / 1_048_576.0).unwrap_or(f64::NAN))
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect()
}

/// Core detector: given a finished `series` (oldest-first) for one
/// signal, apply the warm-up / config / stddev guards and, if the
/// current value is an upper-tail outlier, return a `RuleHit`.
fn detect_signal(
    cfg: &AdvisorConfig,
    rule_id: &str,
    label: &str,
    unit: &str,
    series: Vec<f64>,
) -> Option<RuleHit> {
    let rc = cfg.for_rule(rule_id);
    if !rc.enabled || rc.muted {
        return None;
    }
    let sigma = rc
        .thresholds
        .get("sigma")
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_SIGMA);
    let min_samples = rc
        .thresholds
        .get("min_samples")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MIN_SAMPLES);

    if series.len() < min_samples {
        return None; // warm-up: not enough data to trust a baseline.
    }

    let (current, baseline) = series.split_last()?;
    let (mean, stddev) = mean_std(baseline);
    if stddev <= 0.0 {
        return None; // flat baseline → no meaningful spread.
    }

    let deviation = (current - mean) / stddev;
    if deviation < sigma {
        return None; // within normal range (or below the baseline).
    }

    // Severity by deviation magnitude; the configured sigma is the
    // floor (Medium), 5σ and above escalates to High.
    let severity = if deviation >= HIGH_SIGMA {
        Severity::High
    } else {
        Severity::Medium
    };

    let n = baseline.len();
    Some(RuleHit {
        rule_id: rule_id.into(),
        scope: Scope::Global,
        severity,
        title: format!("Anomalous {label}"),
        detail: format!(
            "current {label} = {current:.2} {unit}, baseline mean {mean:.2} ± {stddev:.2} {unit} over {n} samples → {deviation:.1}σ above normal."
        ),
        recommendation: format!(
            "Investigate the recent {label} spike: check recent load, a slow/failing dependency, or a heavy query/ingest burst."
        ),
        runbook: Some("GET /v1/admin/metrics; use \"explain\": true on search".into()),
        remediation: RemediationAction::None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::config::RuleConfig;
    use crate::advisor::snapshot::{
        GlobalSnapshot, HealthSnapshot, OpsSnapshot, QuerySnapshot, SnapshotHistory,
        DEFAULT_HISTORY_CAP,
    };
    use std::collections::HashMap;

    /// Build a snapshot carrying a chosen p95 latency, RSS, and an
    /// arbitrary set of counters at a given timestamp.
    fn snap(taken_at_ms: i64, p95_ms: f64, rss_bytes: Option<u64>, counters: &[(&str, f64)]) -> OpsSnapshot {
        let mut c = HashMap::new();
        for (k, v) in counters {
            c.insert((*k).to_owned(), *v);
        }
        OpsSnapshot {
            taken_at_ms,
            global: GlobalSnapshot {
                active_collections: 0,
                total_records: 0,
            },
            collections: Vec::new(),
            nodes: Vec::new(),
            replica_safe_ts: None,
            health: HealthSnapshot {
                minio: "healthy".into(),
                nats: "healthy".into(),
                embedding: "healthy".into(),
                multimodal_embedding: "healthy".into(),
            },
            query: QuerySnapshot {
                window_secs: 300,
                count: 0,
                p50_ms: 0.0,
                p95_ms,
                slow_count: 0,
            },
            rss_bytes,
            disk_total_bytes: None,
            disk_free_bytes: None,
            disk_used_pct: None,
            cpu_total_jiffies: None,
            cpu_busy_jiffies: None,
            num_cpus: 1,
            load1: None,
            load5: None,
            load15: None,
            open_fds: None,
            fd_limit: None,
            mem_total_bytes: None,
            mem_available_bytes: None,
            counters: c,
            error_log_count: 0,
            warn_log_count: 0,
        }
    }

    /// History of `n` snapshots whose `error_log_count` is `errs(i)`.
    fn error_log_history(n: usize, errs: impl Fn(usize) -> u64) -> SnapshotHistory {
        let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        for i in 0..n {
            let mut s = snap(1000 + i as i64 * 1000, 0.0, None, &[]);
            s.error_log_count = errs(i);
            h.push(s);
        }
        h
    }

    /// History of `n` snapshots whose p95 latency is `latency(i)`.
    fn latency_history(n: usize, latency: impl Fn(usize) -> f64) -> SnapshotHistory {
        let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        for i in 0..n {
            h.push(snap(1000 + i as i64 * 1000, latency(i), None, &[]));
        }
        h
    }

    fn find<'a>(hits: &'a [RuleHit], id: &str) -> Option<&'a RuleHit> {
        hits.iter().find(|h| h.rule_id == id)
    }

    #[test]
    fn mean_std_known_vector() {
        // [2,4,4,4,5,5,7,9]: mean 5, population stddev 2.
        let (m, s) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((m - 5.0).abs() < 1e-9, "mean {m}");
        assert!((s - 2.0).abs() < 1e-9, "stddev {s}");
        // empty → (0,0), no panic.
        assert_eq!(mean_std(&[]), (0.0, 0.0));
    }

    #[test]
    fn steady_series_no_anomaly() {
        // 30 samples wobbling tightly around 100; current also ~100.
        let h = latency_history(30, |i| 100.0 + (i % 3) as f64 * 0.5 - 0.5);
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        assert!(find(&hits, "anomaly_search_latency").is_none(), "hits: {hits:?}");
    }

    #[test]
    fn current_spike_fires_medium() {
        // Baseline alternates 75/125 → mean 100, population stddev 25.
        // Current 200 → (200-100)/25 = 4σ: above the 3σ floor, below 5σ.
        let h = latency_history(30, |i| {
            if i == 29 {
                200.0
            } else if i % 2 == 0 {
                75.0
            } else {
                125.0
            }
        });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        let hit = find(&hits, "anomaly_search_latency").expect("latency anomaly fired");
        assert_eq!(hit.scope, Scope::Global);
        assert_eq!(hit.severity, Severity::Medium);
        assert!(!hit.detail.is_empty());
        assert!(!hit.recommendation.is_empty());
        assert_eq!(hit.remediation, RemediationAction::None);
    }

    #[test]
    fn extreme_spike_fires_high() {
        // Tight baseline ~100 (small spread), then an enormous jump → ≥5σ.
        let h = latency_history(30, |i| if i == 29 { 5000.0 } else { 100.0 + (i % 2) as f64 });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        let hit = find(&hits, "anomaly_search_latency").expect("latency anomaly fired");
        assert_eq!(hit.severity, Severity::High);
    }

    #[test]
    fn warmup_suppresses_fire() {
        // Only 10 samples (< default 20) even with a blatant spike → no fire.
        let h = latency_history(10, |i| if i == 9 { 5000.0 } else { 100.0 + (i % 2) as f64 });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        assert!(find(&hits, "anomaly_search_latency").is_none());
    }

    #[test]
    fn flat_series_no_fire_no_panic() {
        // Perfectly flat baseline at 100, current jumps — stddev 0 → skip.
        let h = latency_history(30, |i| if i == 29 { 999.0 } else { 100.0 });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        assert!(find(&hits, "anomaly_search_latency").is_none());
    }

    #[test]
    fn disabled_or_muted_suppressed() {
        let h = latency_history(30, |i| if i == 29 { 200.0 } else { 100.0 + (i % 3) as f64 * 0.4 });

        let mut disabled = AdvisorConfig::default();
        disabled.rules.insert(
            "anomaly_search_latency".into(),
            RuleConfig { enabled: false, ..Default::default() },
        );
        assert!(find(&detect_anomalies(&h, &disabled), "anomaly_search_latency").is_none());

        let mut muted = AdvisorConfig::default();
        muted.rules.insert(
            "anomaly_search_latency".into(),
            RuleConfig { muted: true, ..Default::default() },
        );
        assert!(find(&detect_anomalies(&h, &muted), "anomaly_search_latency").is_none());
    }

    #[test]
    fn error_rate_spike_fires() {
        // Build a monotonically-increasing veda_errors_total counter:
        // a steady ~10 err/s baseline, then a final-interval spike.
        let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        let mut total = 0.0;
        // 31 snapshots → 30 rate points (>= min_samples 20).
        for i in 0..31 {
            // steady 10/interval for the baseline, 1000 on the last step.
            if i > 0 {
                total += if i == 30 { 1000.0 } else { 10.0 + (i % 2) as f64 };
            }
            h.push(snap(1000 + i as i64 * 1000, 0.0, None, &[("veda_errors_total", total)]));
        }
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        let hit = find(&hits, "anomaly_error_rate").expect("error-rate anomaly fired");
        assert_eq!(hit.scope, Scope::Global);
        assert!(!hit.detail.is_empty());
    }

    #[test]
    fn rss_spike_fires() {
        let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        for i in 0..30 {
            let bytes = if i == 29 {
                4 * 1024 * 1024 * 1024 // 4 GiB current
            } else {
                (1024 + (i % 3) as u64) * 1024 * 1024 // ~1 GiB baseline
            };
            h.push(snap(1000 + i as i64 * 1000, 0.0, Some(bytes), &[]));
        }
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        assert!(find(&hits, "anomaly_rss").is_some(), "hits: {hits:?}");
    }

    #[test]
    fn error_logs_spike_fires() {
        // Steady ~4 errors/window baseline, then a spike to 100.
        let h = error_log_history(30, |i| if i == 29 { 100 } else { 4 + (i % 2) as u64 });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        let hit = find(&hits, "anomaly_error_logs").expect("error-log anomaly fired");
        assert_eq!(hit.scope, Scope::Global);
        assert!(!hit.detail.is_empty());
    }

    #[test]
    fn error_logs_warmup_suppresses() {
        // Only 10 samples (< default 20) even with a spike → no fire.
        let h = error_log_history(10, |i| if i == 9 { 100 } else { 4 + (i % 2) as u64 });
        let hits = detect_anomalies(&h, &AdvisorConfig::default());
        assert!(find(&hits, "anomaly_error_logs").is_none());
    }

    #[test]
    fn rule_ids_exposed_and_unique() {
        let ids = anomaly_rule_ids();
        assert!(ids.contains(&"anomaly_search_latency"));
        assert!(ids.contains(&"anomaly_error_rate"));
        assert!(ids.contains(&"anomaly_insert_rate"));
        assert!(ids.contains(&"anomaly_rss"));
        assert!(ids.contains(&"anomaly_error_logs"));
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "duplicate anomaly rule id");
    }
}
