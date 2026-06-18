//! Change-point detection — sustained level/trend shifts in the
//! snapshot-history metric series.
//!
//! Pure math over the in-memory [`SnapshotHistory`] (no network, safe to
//! run inside the advisor tick). Where [`crate::advisor::anomaly`] catches a
//! single outlier (EWMA/σ) and [`crate::advisor::forecast`] projects a
//! linear ETA, this catches a **regime shift**: the series settles to a new
//! level (a step), or its trend breaks (flat → rising). That is the classic
//! "what changed?" AIOps primitive — it points at a recent
//! deploy / config / load change.
//!
//! Statistic: we scan every interior split index `k`, splitting the window
//! into a left and a right segment. For each split we compute
//! * a **level** statistic — `|mean_right - mean_left|`, and
//! * a **trend** statistic — `|slope_right - slope_left|` (least-squares
//!   slope of each segment, in value-per-second).
//!
//! We keep the split that maximises the level statistic, and the split that
//! maximises the trend statistic. Both are normalized: level by the series
//! stddev (so it is unit-free), trend by `level_norm` derived from the same
//! stddev over the mean time-step (so the two are comparable). The dominant
//! normalized statistic decides [`ChangeKind`]; the larger normalized value
//! becomes `magnitude` and feeds `confidence`.
//!
//! The guards mirror [`crate::advisor::forecast`]: a **warm-up**
//! ([`DEFAULT_MIN_POINTS`], default 20) so a short series yields `None`, and
//! a **min-magnitude** floor ([`DEFAULT_MIN_MAGNITUDE`]) so a flat or merely
//! noisy-but-stationary series never fires. Each segment must also retain a
//! minimum length so a one-point edge split can't masquerade as a shift.

use serde::{Deserialize, Serialize};

use crate::advisor::anomaly::mean_std;
use crate::advisor::snapshot::SnapshotHistory;

/// Default warm-up: minimum series length before we look for a change.
pub const DEFAULT_MIN_POINTS: usize = 20;
/// Default minimum normalized magnitude for a change point to fire. A level
/// shift must move the mean by ~1.5σ of the series; below this a noisy but
/// stationary series is treated as no change.
pub const DEFAULT_MIN_MAGNITUDE: f64 = 1.5;
/// Minimum points each side of a split must retain (so an edge split on one
/// or two points can't masquerade as a regime shift).
const MIN_SEGMENT: usize = 5;

/// The kind of regime shift detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// A step / level shift — the mean moved to a new plateau.
    Level,
    /// A trend break — the slope changed markedly (e.g. flat → rising).
    Trend,
}

/// A detected change point in one metric series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Metric key, e.g. `disk_used_pct`, `rss_bytes`, `query_p95_ms`,
    /// `error_log_count`, `recall`.
    pub metric: String,
    /// Optional scope target (e.g. a collection name for recall).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub kind: ChangeKind,
    /// Epoch-ms of the first sample on the *after* side of the split.
    pub at_ms: u64,
    /// Representative value before the change (segment mean for Level, the
    /// segment's left-edge fitted value for Trend's "before" is just the
    /// before-segment mean — kept simple).
    pub before: f64,
    /// Representative value after the change.
    pub after: f64,
    /// Normalized magnitude of the shift (level: |Δmean|/σ; trend:
    /// |Δslope| normalized to the same σ scale). Always finite, >= the
    /// `min_magnitude` guard when present.
    pub magnitude: f64,
    /// Confidence in [0, 1] — a squashed function of `magnitude`.
    pub confidence: f64,
}

/// A least-squares line over `(x, y)` points plus its residual
/// sum-of-squares (SSE) — how well the line explains the points.
#[derive(Clone, Copy)]
struct SegFit {
    slope: f64,
    sse: f64,
}

/// Fit a line to `points` and return slope/intercept/SSE. A degenerate
/// segment (< 2 points or no x-spread) gets slope 0 and the line at the
/// mean, with SSE = total variance.
fn fit(points: &[(f64, f64)]) -> SegFit {
    let n = points.len();
    let nf = n as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / nf.max(1.0);
    let my = mean_y(points);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in points {
        let dx = x - mean_x;
        sxx += dx * dx;
        sxy += dx * (y - my);
    }
    let slope = if n < 2 || sxx <= 0.0 { 0.0 } else { sxy / sxx };
    let intercept = my - slope * mean_x;
    let sse = points
        .iter()
        .map(|&(x, y)| {
            let r = y - (slope * x + intercept);
            r * r
        })
        .sum::<f64>();
    SegFit { slope, sse }
}

/// Mean of the `value` column of `points`.
fn mean_y(points: &[(f64, f64)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64
}

/// Detect the single most-significant change point in a `(ts_secs, value)`
/// series (oldest-first).
///
/// Returns `None` when:
/// * fewer than `min_points` samples (warm-up), or the series is too short
///   to host two `MIN_SEGMENT`-length segments,
/// * the series has zero variance (perfectly flat),
/// * the best normalized statistic is below `min_magnitude` (a flat or
///   noisy-but-stationary series).
///
/// Otherwise the best split is classified [`ChangeKind::Level`] when the
/// normalized level shift dominates, else [`ChangeKind::Trend`].
pub fn detect_change_point(
    points: &[(f64, f64)],
    min_points: usize,
    min_magnitude: f64,
) -> Option<ChangePoint> {
    let n = points.len();
    if n < min_points || n < 2 * MIN_SEGMENT {
        return None; // warm-up / too short.
    }

    let values: Vec<f64> = points.iter().map(|p| p.1).collect();
    let (_, sigma) = mean_std(&values);
    if sigma <= f64::EPSILON {
        return None; // perfectly flat → no change.
    }

    let mut best_level = 0.0_f64; // normalized |Δmean|/σ
    let mut best_level_k = 0usize;
    let mut best_level_before = 0.0;
    let mut best_level_after = 0.0;

    let mut best_trend = 0.0_f64; // normalized |Δslope|*horizon/σ
    let mut best_trend_k = 0usize;
    let mut best_trend_before = 0.0;
    let mut best_trend_after = 0.0;

    for k in MIN_SEGMENT..=(n - MIN_SEGMENT) {
        let left = &points[..k];
        let right = &points[k..];

        // --- Level statistic: difference of segment means / σ.
        let ml = mean_y(left);
        let mr = mean_y(right);
        let level_norm = (mr - ml).abs() / sigma;
        if level_norm > best_level {
            best_level = level_norm;
            best_level_k = k;
            best_level_before = ml;
            best_level_after = mr;
        }

        // --- Trend statistic: change in fitted slope across the split.
        // Each segment is fit by its own line. The slope difference is
        // turned into a value delta over the shorter segment's time span
        // (a "where would the trend put us" horizon) and normalized by σ.
        // It is credited only when BOTH segments are well-explained by
        // their lines — `well_fit` ≈ 1 when the per-segment residual std
        // is tiny vs σ, and decays toward 0 for a series whose halves are
        // themselves noisy (e.g. a sawtooth), so a bend must be real.
        let lf = fit(left);
        let rf = fit(right);
        let span_l = (left[left.len() - 1].0 - left[0].0).abs();
        let span_r = (right[right.len() - 1].0 - right[0].0).abs();
        let horizon = span_l.min(span_r).max(f64::EPSILON);
        let resid_var = (lf.sse + rf.sse) / n as f64;
        let resid_std = resid_var.sqrt();
        // 1 when the two-line fit is near-perfect; → 0 as residual scatter
        // approaches σ (no real structure beyond the overall variance).
        let well_fit = (1.0 - resid_std / sigma).clamp(0.0, 1.0);
        let trend_norm = (rf.slope - lf.slope).abs() * horizon / sigma * well_fit;
        if trend_norm > best_trend {
            best_trend = trend_norm;
            best_trend_k = k;
            best_trend_before = ml;
            best_trend_after = mr;
        }
    }

    // Decide kind by the dominant normalized statistic, then apply the
    // min-magnitude guard to the chosen one.
    let (kind, k, before, after, magnitude) = if best_level >= best_trend {
        (
            ChangeKind::Level,
            best_level_k,
            best_level_before,
            best_level_after,
            best_level,
        )
    } else {
        (
            ChangeKind::Trend,
            best_trend_k,
            best_trend_before,
            best_trend_after,
            best_trend,
        )
    };

    if magnitude < min_magnitude {
        return None; // flat / noisy-but-stationary.
    }

    // confidence: a smooth squash of the magnitude into [0, 1]. At the
    // guard floor it's modest; it saturates toward 1 for large shifts.
    let confidence = (magnitude / (magnitude + 2.0)).clamp(0.0, 1.0);
    let at_ms = (points[k].0 * 1000.0) as u64;

    Some(ChangePoint {
        metric: String::new(),
        scope: None,
        kind,
        at_ms,
        before,
        after,
        magnitude,
        confidence,
    })
}

/// Run the detector across every tracked signal in the history. Pure +
/// cheap (means/slopes over the in-memory ring) — safe inside the advisor
/// tick. Each signal contributes at most one change point; a signal with
/// too few points or no data is skipped.
///
/// Tracked signals:
/// * `disk_used_pct` (Option),
/// * `rss_bytes`,
/// * `query_p95_ms` (the global query p95),
/// * `error_log_count` (the per-window ERROR count — a sustained-rate proxy),
/// * per-collection `recall` (from `index_health`).
pub fn compute_change_points(
    history: &SnapshotHistory,
    min_points: usize,
    min_magnitude: f64,
) -> Vec<ChangePoint> {
    let mut out = Vec::new();
    let ts: Vec<f64> = history.series(|s| s.taken_at_ms as f64 / 1000.0);

    // disk_used_pct (Option) — drop snapshots missing the signal.
    let disk = paired(&ts, &history.series(|s| s.disk_used_pct.unwrap_or(f64::NAN)));
    push_named(&mut out, "disk_used_pct", None, &disk, min_points, min_magnitude);

    // rss_bytes (Option).
    let rss = paired(
        &ts,
        &history.series(|s| s.rss_bytes.map(|b| b as f64).unwrap_or(f64::NAN)),
    );
    push_named(&mut out, "rss_bytes", None, &rss, min_points, min_magnitude);

    // query p95 (always present on the snapshot; 0 when no queries).
    let p95 = paired(&ts, &history.series(|s| s.query.p95_ms));
    push_named(&mut out, "query_p95_ms", None, &p95, min_points, min_magnitude);

    // error_log_count (per-window ERROR count — a sustained error-rate
    // proxy; the counter-rate isn't a stable per-snapshot series here).
    let errs = paired(&ts, &history.series(|s| s.error_log_count as f64));
    push_named(
        &mut out,
        "error_log_count",
        None,
        &errs,
        min_points,
        min_magnitude,
    );

    // Per-collection recall (Option, from index_health).
    let mut colls: Vec<String> = Vec::new();
    for snap in history.iter() {
        for ih in &snap.index_health {
            if ih.recall.is_some() && !colls.contains(&ih.collection) {
                colls.push(ih.collection.clone());
            }
        }
    }
    for coll in colls {
        let pts: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|s| {
                let recall = s
                    .index_health
                    .iter()
                    .find(|ih| ih.collection == coll)
                    .and_then(|ih| ih.recall)?;
                Some((s.taken_at_ms as f64 / 1000.0, recall))
            })
            .collect();
        push_named(
            &mut out,
            "recall",
            Some(coll.clone()),
            &pts,
            min_points,
            min_magnitude,
        );
    }

    out
}

/// Detect on `points` and, on a hit, stamp the metric/scope and push it.
fn push_named(
    out: &mut Vec<ChangePoint>,
    metric: &str,
    scope: Option<String>,
    points: &[(f64, f64)],
    min_points: usize,
    min_magnitude: f64,
) {
    if let Some(mut cp) = detect_change_point(points, min_points, min_magnitude) {
        cp.metric = metric.to_owned();
        cp.scope = scope;
        out.push(cp);
    }
}

/// Zip a timestamp series with a value series, dropping NaN points (a
/// snapshot where the signal was unavailable).
fn paired(ts: &[f64], values: &[f64]) -> Vec<(f64, f64)> {
    ts.iter()
        .zip(values.iter())
        .filter(|&(_, &v)| !v.is_nan())
        .map(|(&t, &v)| (t, v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::snapshot::{
        GlobalSnapshot, HealthSnapshot, OpsSnapshot, QuerySnapshot, SnapshotHistory,
        DEFAULT_HISTORY_CAP,
    };
    use std::collections::HashMap;

    /// Build an `n`-point `(ts_secs, value)` series spaced `step` apart from
    /// `t0`, with `value(i)`.
    fn series(n: usize, t0: f64, step: f64, value: impl Fn(usize) -> f64) -> Vec<(f64, f64)> {
        (0..n).map(|i| (t0 + i as f64 * step, value(i))).collect()
    }

    /// A minimal all-`None` snapshot at `taken_at_ms` for history tests.
    fn empty_snapshot(taken_at_ms: i64) -> OpsSnapshot {
        OpsSnapshot {
            taken_at_ms,
            global: GlobalSnapshot { active_collections: 0, total_records: 0 },
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
                p95_ms: 0.0,
                slow_count: 0,
                recent_latencies_ms: Vec::new(),
            },
            rss_bytes: None,
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
            counters: HashMap::new(),
            error_log_count: 0,
            warn_log_count: 0,
            slo_statuses: Vec::new(),
            synthetic_summaries: Vec::new(),
            index_health: Vec::new(),
            forecasts: Vec::new(),
            change_points: Vec::new(),
        }
    }

    #[test]
    fn clean_level_shift_is_detected() {
        // 50 for the first half, 90 for the second half (40 points).
        let pts = series(40, 0.0, 10.0, |i| if i < 20 { 50.0 } else { 90.0 });
        let cp = detect_change_point(&pts, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE)
            .expect("level change");
        assert_eq!(cp.kind, ChangeKind::Level);
        assert!((cp.before - 50.0).abs() < 1.0, "before {}", cp.before);
        assert!((cp.after - 90.0).abs() < 1.0, "after {}", cp.after);
        // The split should land at the 20th point: ts = 20*10 = 200s.
        assert_eq!(cp.at_ms, 200_000);
        assert!(cp.magnitude >= DEFAULT_MIN_MAGNITUDE);
        assert!(cp.confidence > 0.0 && cp.confidence <= 1.0);
    }

    #[test]
    fn flat_series_yields_none() {
        let pts = series(40, 0.0, 10.0, |_| 42.0);
        assert!(detect_change_point(&pts, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE).is_none());
    }

    #[test]
    fn noisy_stationary_series_below_min_magnitude_yields_none() {
        // Small sawtooth around 50, no regime shift → no change point.
        let pts = series(40, 0.0, 10.0, |i| if i % 2 == 0 { 49.0 } else { 51.0 });
        let cp = detect_change_point(&pts, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE);
        assert!(cp.is_none(), "stationary noise must not fire: {cp:?}");
    }

    #[test]
    fn trend_break_flat_then_rising_is_detected() {
        // Flat at 50 for the first half, then rising steeply.
        let pts = series(40, 0.0, 10.0, |i| {
            if i < 20 {
                50.0
            } else {
                50.0 + 5.0 * (i - 20) as f64
            }
        });
        let cp = detect_change_point(&pts, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE)
            .expect("trend change");
        assert_eq!(cp.kind, ChangeKind::Trend);
        assert!(cp.magnitude >= DEFAULT_MIN_MAGNITUDE);
    }

    #[test]
    fn warmup_too_few_points_yields_none() {
        let pts = series(10, 0.0, 10.0, |i| if i < 5 { 50.0 } else { 90.0 });
        assert!(detect_change_point(&pts, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE).is_none());
    }

    #[test]
    fn compute_change_points_finds_disk_level_shift() {
        let mut h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        for i in 0..40usize {
            let mut s = empty_snapshot(i as i64 * 10_000);
            s.disk_used_pct = Some(if i < 20 { 40.0 } else { 80.0 });
            h.push(s);
        }
        let cps = compute_change_points(&h, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE);
        let disk = cps
            .iter()
            .find(|c| c.metric == "disk_used_pct")
            .expect("disk change point");
        assert_eq!(disk.kind, ChangeKind::Level);
        assert!((disk.before - 40.0).abs() < 1.0);
        assert!((disk.after - 80.0).abs() < 1.0);
    }

    #[test]
    fn compute_change_points_empty_history_is_empty() {
        let h = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        assert!(compute_change_points(&h, DEFAULT_MIN_POINTS, DEFAULT_MIN_MAGNITUDE).is_empty());
    }
}
