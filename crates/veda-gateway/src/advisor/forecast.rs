//! Capacity forecasting — predictive ETAs from the snapshot-history time
//! series.
//!
//! Pure math over the in-memory [`SnapshotHistory`] (no network, safe to
//! run inside the advisor tick). For each tracked signal we fit a
//! least-squares line over `(timestamp_secs, value)` points and, when the
//! fit is trustworthy and trending toward a threshold, project the time to
//! cross it as an ETA.
//!
//! The guards mirror [`crate::advisor::anomaly`]: a **warm-up** (min
//! points, default 20) and a **min R²** (default 0.5) so a forecast NEVER
//! fires on a flat or noisy trend. `eta_ms` is `None` whenever the slope is
//! flat or points away from the threshold, or the fit is too poor.
//!
//! The `forecast_metric` output is attached to the snapshot
//! (`OpsSnapshot::forecasts`); rules in [`crate::advisor::rules`] read those
//! and fire when an ETA falls inside their window.

use serde::{Deserialize, Serialize};

use crate::advisor::snapshot::SnapshotHistory;

/// Default warm-up: minimum series length before we trust a trend.
pub const DEFAULT_MIN_POINTS: usize = 20;
/// Default minimum coefficient of determination (R²) for a usable fit.
pub const DEFAULT_MIN_R2: f64 = 0.5;

/// Disk-full threshold (% used) the disk forecast projects against. We aim
/// at 95% rather than a literal 100% — writes/seals fail before a disk is
/// truly full, so the operationally-relevant ETA is the near-full mark.
pub const DISK_FULL_PCT: f64 = 95.0;
/// Recall target the per-collection recall forecast projects a downward
/// crossing against (matches the index-advisor `recall_min` default).
pub const RECALL_TARGET: f64 = 0.90;

/// A least-squares linear fit `value = slope_per_sec * t + intercept`
/// over `(t_secs, value)` points, with its goodness-of-fit `r2` in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrendFit {
    pub slope_per_sec: f64,
    pub intercept: f64,
    pub r2: f64,
}

/// A capacity forecast for one metric: its current value, the fitted
/// trend, the threshold being projected against, and the ETA (ms epoch) to
/// cross it — `None` when the trend is flat, points away from the
/// threshold, or the fit is too poor to trust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    /// Metric key, e.g. `disk_used_pct`, `rss_bytes`, `recall`.
    pub metric: String,
    /// Optional scope target (e.g. a collection name for recall).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub current: f64,
    pub slope_per_sec: f64,
    pub r2: f64,
    pub threshold: f64,
    /// Projected epoch-ms at which `value` crosses `threshold`; `None` when
    /// not trending toward it or the fit is below the min-R² floor.
    pub eta_ms: Option<i64>,
}

/// Least-squares fit of `value` against `t_secs`. Returns `None` when
/// there are fewer than two points or `t` has zero variance (a vertical
/// line — slope undefined). `r2` is the coefficient of determination;
/// when `value` has zero variance (a perfectly flat series) the line is
/// exact, so `r2 = 1.0` with `slope_per_sec = 0.0`.
pub fn linear_fit(points: &[(f64, f64)]) -> Option<TrendFit> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / nf;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / nf;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for &(x, y) in points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 0.0 {
        return None; // no spread in time → slope undefined.
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    // R² = explained variance / total variance. A flat `value` series has
    // syy == 0 and is fit exactly by a zero-slope line → R² = 1.
    let r2 = if syy <= 0.0 {
        1.0
    } else {
        (sxy * sxy) / (sxx * syy)
    };
    Some(TrendFit {
        slope_per_sec: slope,
        intercept,
        r2: r2.clamp(0.0, 1.0),
    })
}

/// Forecast a single metric from its `(ts_secs, value)` series.
///
/// `series` is oldest-first. `threshold` is the level we project a
/// crossing against; `now_ms` is the reference time for the ETA. Returns
/// `None` (no forecast) when:
/// * fewer than `min_points` samples (warm-up),
/// * the fit's R² is below `min_r2` (noisy),
/// * the time series has no spread (`linear_fit` → None).
///
/// Otherwise a [`Forecast`] is returned; its `eta_ms` is `Some` only when
/// the trend is heading toward the threshold from the current side — i.e.
/// the projected crossing is in the future. A flat slope (or one pointing
/// away) yields `eta_ms = None` while still reporting the fit.
pub fn forecast_metric(
    metric: &str,
    scope: Option<String>,
    series: &[(f64, f64)],
    threshold: f64,
    now_ms: i64,
    min_points: usize,
    min_r2: f64,
) -> Option<Forecast> {
    if series.len() < min_points {
        return None; // warm-up.
    }
    let fit = linear_fit(series)?;
    if fit.r2 < min_r2 {
        return None; // too noisy to trust.
    }
    let current = series.last()?.1;
    let now_secs = now_ms as f64 / 1000.0;

    // Project the crossing time only when the trend actually moves toward
    // the threshold from the current value. `eta_secs` solves
    // `current + slope*(t - now) = threshold`.
    let eta_ms = eta_to_threshold(current, fit.slope_per_sec, threshold, now_secs);

    Some(Forecast {
        metric: metric.to_owned(),
        scope,
        current,
        slope_per_sec: fit.slope_per_sec,
        r2: fit.r2,
        threshold,
        eta_ms,
    })
}

/// Time (epoch ms) at which `current` reaches `threshold` under a linear
/// `slope_per_sec`, or `None` when the slope is flat or directed away from
/// the threshold (the crossing is in the past / never).
fn eta_to_threshold(
    current: f64,
    slope_per_sec: f64,
    threshold: f64,
    now_secs: f64,
) -> Option<i64> {
    let gap = threshold - current;
    // Already at/over the threshold on the approaching side → no future ETA.
    // A (near-)zero slope never crosses.
    if slope_per_sec.abs() < f64::EPSILON {
        return None;
    }
    // The slope must point in the same direction as the gap: rising toward
    // a higher threshold, or falling toward a lower one.
    if gap == 0.0 || gap.signum() != slope_per_sec.signum() {
        return None;
    }
    let dt_secs = gap / slope_per_sec;
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return None;
    }
    let eta_secs = now_secs + dt_secs;
    Some((eta_secs * 1000.0) as i64)
}

/// Build the capacity forecasts for the whole history at `now_ms`. Pure +
/// cheap (least-squares over the in-memory ring) — safe to call inside the
/// advisor tick. Each forecast is included only when its fit clears the
/// warm-up + min-R² guards; a forecast with `eta_ms = None` (flat / away
/// from the threshold) is still surfaced so the UI can show the trend.
///
/// Tracked signals:
/// * `disk_used_pct` → [`DISK_FULL_PCT`] (upward crossing),
/// * `mem_available_bytes` → 0 (downward — memory exhaustion),
/// * per-collection `recall` (from `index_health`) → [`RECALL_TARGET`]
///   (downward crossing).
///
/// Snapshots missing a signal contribute no point for it; a metric whose
/// series stays below `min_points` simply produces no forecast.
pub fn compute_forecasts(
    history: &SnapshotHistory,
    now_ms: i64,
    min_points: usize,
    min_r2: f64,
) -> Vec<Forecast> {
    let mut out = Vec::new();

    // Timestamps (seconds) shared by the per-snapshot signal series.
    let ts: Vec<f64> = history.series(|s| s.taken_at_ms as f64 / 1000.0);

    // disk_used_pct (Option) → upward toward DISK_FULL_PCT.
    let disk = paired(
        &ts,
        &history.series(|s| s.disk_used_pct.unwrap_or(f64::NAN)),
    );
    if let Some(f) = forecast_metric(
        "disk_used_pct",
        None,
        &disk,
        DISK_FULL_PCT,
        now_ms,
        min_points,
        min_r2,
    ) {
        out.push(f);
    }

    // mem_available_bytes (Option) → downward toward 0 (exhaustion).
    let mem = paired(
        &ts,
        &history.series(|s| s.mem_available_bytes.map(|b| b as f64).unwrap_or(f64::NAN)),
    );
    if let Some(f) = forecast_metric(
        "mem_available_bytes",
        None,
        &mem,
        0.0,
        now_ms,
        min_points,
        min_r2,
    ) {
        out.push(f);
    }

    // Per-collection recall (Option, from index_health) → downward toward
    // RECALL_TARGET. Collect the set of collection names seen across the
    // history (stable order), then build a `(ts_secs, recall)` series per
    // collection from snapshots that carried a recall sample for it.
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
        if let Some(f) = forecast_metric(
            "recall",
            Some(coll.clone()),
            &pts,
            RECALL_TARGET,
            now_ms,
            min_points,
            min_r2,
        ) {
            out.push(f);
        }
    }

    out
}

/// Zip a timestamp series with a value series, dropping any point whose
/// value is NaN (a snapshot where the signal was unavailable).
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

    /// Build a `(ts_secs, value)` series of `n` points spaced `step_secs`
    /// apart starting at `t0_secs`, with `value(i)`.
    fn series(
        n: usize,
        t0_secs: f64,
        step_secs: f64,
        value: impl Fn(usize) -> f64,
    ) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| (t0_secs + i as f64 * step_secs, value(i)))
            .collect()
    }

    #[test]
    fn linear_fit_recovers_known_line() {
        // y = 2x + 3.
        let pts = series(10, 0.0, 1.0, |i| 2.0 * i as f64 + 3.0);
        let fit = linear_fit(&pts).expect("fit");
        assert!(
            (fit.slope_per_sec - 2.0).abs() < 1e-9,
            "slope {}",
            fit.slope_per_sec
        );
        assert!(
            (fit.intercept - 3.0).abs() < 1e-9,
            "intercept {}",
            fit.intercept
        );
        assert!((fit.r2 - 1.0).abs() < 1e-9, "r2 {}", fit.r2);
    }

    #[test]
    fn linear_fit_flat_series_is_perfect_zero_slope() {
        let pts = series(10, 100.0, 5.0, |_| 42.0);
        let fit = linear_fit(&pts).expect("fit");
        assert!(fit.slope_per_sec.abs() < 1e-12);
        assert!((fit.r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_fit_guards() {
        // < 2 points → None.
        assert!(linear_fit(&[(1.0, 2.0)]).is_none());
        assert!(linear_fit(&[]).is_none());
        // zero variance in x → None.
        let same_x: Vec<(f64, f64)> = (0..5).map(|i| (10.0, i as f64)).collect();
        assert!(linear_fit(&same_x).is_none());
    }

    #[test]
    fn rising_disk_series_yields_finite_eta_and_high_r2() {
        // Disk rising 0.5%/step over 1000s steps from 50%, 30 points.
        // slope = 0.5% / 1000s = 5e-4 %/s. Threshold 100%.
        let now_secs = 1_000_000.0;
        let pts = series(30, now_secs - 29.0 * 1000.0, 1000.0, |i| {
            50.0 + 0.5 * i as f64
        });
        let f = forecast_metric(
            "disk_used_pct",
            None,
            &pts,
            100.0,
            (now_secs * 1000.0) as i64,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        )
        .expect("forecast");
        assert!(f.r2 > 0.99, "r2 {}", f.r2);
        // current ~ 50 + 0.5*29 = 64.5; gap 35.5 / slope 5e-4 = 71000s.
        let eta = f.eta_ms.expect("eta present");
        let dt_secs = (eta as f64 / 1000.0) - now_secs;
        assert!((dt_secs - 71_000.0).abs() < 2_000.0, "dt_secs {dt_secs}");
        assert!(eta > (now_secs * 1000.0) as i64, "eta in the future");
    }

    #[test]
    fn flat_series_has_no_eta() {
        let now_ms = 1_000_000_000i64;
        let pts = series(30, 0.0, 1000.0, |_| 50.0);
        let f = forecast_metric(
            "disk_used_pct",
            None,
            &pts,
            100.0,
            now_ms,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        )
        .expect("forecast (flat fit is perfect → returned)");
        assert!(f.eta_ms.is_none(), "flat slope → no eta");
    }

    #[test]
    fn noisy_series_below_min_r2_is_none() {
        // Sawtooth with no real trend → low R².
        let pts = series(40, 0.0, 1000.0, |i| if i % 2 == 0 { 40.0 } else { 60.0 });
        let f = forecast_metric(
            "disk_used_pct",
            None,
            &pts,
            100.0,
            1_000_000_000,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        );
        assert!(f.is_none(), "noisy series must not forecast: {f:?}");
    }

    #[test]
    fn too_few_points_is_none() {
        let pts = series(10, 0.0, 1000.0, |i| 50.0 + i as f64);
        let f = forecast_metric(
            "disk_used_pct",
            None,
            &pts,
            100.0,
            1_000_000_000,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        );
        assert!(f.is_none(), "warm-up must suppress");
    }

    #[test]
    fn falling_series_away_from_threshold_has_no_eta() {
        // Falling disk usage (away from the 100% ceiling) → no upward ETA.
        let pts = series(30, 0.0, 1000.0, |i| 80.0 - 0.5 * i as f64);
        let f = forecast_metric(
            "disk_used_pct",
            None,
            &pts,
            100.0,
            1_000_000_000,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        )
        .expect("forecast");
        assert!(f.eta_ms.is_none(), "falling away → no eta toward ceiling");
    }

    #[test]
    fn falling_recall_toward_lower_threshold_yields_eta() {
        // Recall decaying from 0.99 by 0.002/step toward a 0.90 target.
        let now_secs = 2_000_000.0;
        let pts = series(30, now_secs - 29.0 * 1000.0, 1000.0, |i| {
            0.99 - 0.002 * i as f64
        });
        let f = forecast_metric(
            "recall",
            Some("docs".into()),
            &pts,
            0.90,
            (now_secs * 1000.0) as i64,
            DEFAULT_MIN_POINTS,
            DEFAULT_MIN_R2,
        )
        .expect("forecast");
        assert_eq!(f.scope.as_deref(), Some("docs"));
        let eta = f.eta_ms.expect("downward crossing eta");
        assert!(eta > (now_secs * 1000.0) as i64, "eta in the future");
    }
}
