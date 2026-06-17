//! SLO (Service-Level-Objective) engine for the ops advisor.
//!
//! Pure types + a bucketed rolling-counter state with SLI /
//! error-budget / burn-rate math and JSON persistence. This module is
//! deliberately self-contained: no HTTP, no `AppState`, no snapshot
//! wiring — those land in a follow-up task. Everything here is unit
//! testable by constructing [`SloState`] directly and feeding it
//! synthetic records.
//!
//! ## Model
//!
//! An [`SloDef`] declares an objective ("SLI >= `target_pct` over
//! `window_secs`") plus two burn-rate alert windows (`fast_burn`,
//! `slow_burn`). Observations are recorded into [`SloState`], a
//! per-SLO ring of fixed-width time [`Bucket`]s, each holding `good`
//! and `total` counts. [`SloState::status`] sums the relevant windows
//! and computes the SLI, error-budget, and burn rates.
//!
//! ## Math (the heart of the module)
//!
//! For a window with `(good, total)` and `bad = total - good`, and an
//! objective `target_pct` with `allowed_bad = 1 - target_pct/100`:
//!
//! * `sli_pct = good / total * 100`  (`None` when `total == 0`).
//! * `observed_bad_fraction = bad / total`.
//! * `consumed = observed_bad_fraction / allowed_bad`  — fraction of
//!   the error budget used over the window.
//! * `error_budget_remaining_pct = (1 - consumed) * 100`, clamped to
//!   `[-1000, 100]` (negative = over budget). `None` when `total == 0`
//!   or `allowed_bad <= 0`.
//! * `burn_rate = observed_bad_fraction / allowed_bad`  — how many
//!   times faster than the budget-exhausting rate the window is
//!   burning (`1.0` = exactly on budget, `>1` = too fast). `None` when
//!   that window's `total == 0` or `allowed_bad <= 0`.
//! * `compliant = sli_pct.map_or(true, |s| s >= target_pct)`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Default bucket width (seconds) for [`SloState::new`].
pub const DEFAULT_BUCKET_SECS: u64 = 60;

/// Per-SLO cap on retained buckets. With 60s buckets this is 30 days
/// of history (44_640 = 30 * 24 * 60). Oldest buckets are dropped once
/// the cap is exceeded; the newest are always retained.
pub const MAX_BUCKETS_PER_SLO: usize = 44_640;

/// Floor for `error_budget_remaining_pct` (sanity clamp for badly
/// over-budget windows).
const REMAINING_PCT_FLOOR: f64 = -1000.0;

/// What an SLO measures. The "good event" definition differs per kind;
/// the rolling counter itself is kind-agnostic (it just stores
/// good/total), so wiring an SLI source to the right counter is a
/// Task-2 concern.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SloKind {
    /// "good" = request faster than `threshold_ms`.
    Latency { threshold_ms: f64 },
    /// "good" = successful (non-error) request.
    Availability,
    /// Complement of [`SloKind::Availability`] — same counter, framed
    /// as errors.
    ErrorRate,
    /// "good" = synthetic probe succeeded.
    Synthetic { probe_id: String },
}

/// A burn-rate alert window: observe the burn rate over `window_secs`
/// and alert when it exceeds `threshold`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BurnConfig {
    pub window_secs: u64,
    pub threshold: f64,
}

/// Default fast-burn window: 5 minutes, 14.4x.
pub fn default_fast_burn() -> BurnConfig {
    BurnConfig {
        window_secs: 300,
        threshold: 14.4,
    }
}

/// Default slow-burn window: 1 hour, 6.0x.
pub fn default_slow_burn() -> BurnConfig {
    BurnConfig {
        window_secs: 3_600,
        threshold: 6.0,
    }
}

/// A user-defined service-level objective.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloDef {
    /// Stable id. May be omitted on create (`POST /v1/admin/slos`), in which
    /// case the gateway assigns `slo-<uuid>`.
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub kind: SloKind,
    /// Objective: `SLI >= target_pct` (e.g. `99.0`).
    pub target_pct: f64,
    /// Compliance window in seconds (e.g. `2_592_000` = 30d).
    pub window_secs: u64,
    #[serde(default = "default_fast_burn")]
    pub fast_burn: BurnConfig,
    #[serde(default = "default_slow_burn")]
    pub slow_burn: BurnConfig,
}

/// Point-in-time SLO evaluation. All ratio fields are `None` when the
/// relevant window had no observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloStatus {
    pub slo_id: String,
    /// `good / total * 100` over the compliance window; `None` if no data.
    pub sli_pct: Option<f64>,
    pub target_pct: f64,
    pub good: u64,
    pub total: u64,
    /// `(1 - consumed) * 100`, clamped; `None` if no data / no budget.
    pub error_budget_remaining_pct: Option<f64>,
    /// Burn rate over `fast_burn.window_secs`.
    pub burn_rate_fast: Option<f64>,
    /// Burn rate over `slow_burn.window_secs`.
    pub burn_rate_slow: Option<f64>,
    /// `sli_pct.map_or(true, |s| s >= target_pct)`.
    pub compliant: bool,
    /// The def's `fast_burn.threshold` (copied for rule convenience).
    pub fast_burn_threshold: f64,
    /// The def's `slow_burn.threshold`.
    pub slow_burn_threshold: f64,
    /// `burn_rate_fast >= fast_burn_threshold` (false when no data).
    pub breaching_fast: bool,
    /// `burn_rate_slow >= slow_burn_threshold` (false when no data).
    pub breaching_slow: bool,
    /// `error_budget_remaining_pct <= 0` (false when no data).
    pub budget_exhausted: bool,
}

/// A fixed-width time bucket of observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Bucket {
    start_ms: u64,
    good: u64,
    total: u64,
}

/// Bucketed rolling counter, keyed by SLO id. Buckets are kept sorted
/// by `start_ms` and pruned to [`MAX_BUCKETS_PER_SLO`] per SLO.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloState {
    bucket_secs: u64,
    buckets: HashMap<String, Vec<Bucket>>,
}

impl SloState {
    /// New, empty state with the given bucket width (seconds).
    pub fn new(bucket_secs: u64) -> Self {
        let bucket_secs = if bucket_secs == 0 {
            DEFAULT_BUCKET_SECS
        } else {
            bucket_secs
        };
        Self {
            bucket_secs,
            buckets: HashMap::new(),
        }
    }

    /// Record `good`/`total` observations for `slo_id` at `now_ms`,
    /// folding them into the current bucket (creating it if needed).
    /// Buckets stay sorted by `start_ms` and are pruned to the cap.
    pub fn record(&mut self, slo_id: &str, good: u64, total: u64, now_ms: u64) {
        let width_ms = self.bucket_secs * 1_000;
        let start_ms = now_ms - now_ms % width_ms;
        let v = self.buckets.entry(slo_id.to_owned()).or_default();

        // Common case: appending to / updating the latest bucket.
        match v.last_mut() {
            Some(last) if last.start_ms == start_ms => {
                last.good += good;
                last.total += total;
            }
            Some(last) if start_ms > last.start_ms => {
                v.push(Bucket {
                    start_ms,
                    good,
                    total,
                });
            }
            _ => {
                // Out-of-order (or empty): find or insert keeping sorted.
                match v.binary_search_by(|b| b.start_ms.cmp(&start_ms)) {
                    Ok(i) => {
                        v[i].good += good;
                        v[i].total += total;
                    }
                    Err(i) => v.insert(
                        i,
                        Bucket {
                            start_ms,
                            good,
                            total,
                        },
                    ),
                }
            }
        }

        // Prune oldest, keep newest, cap retained buckets.
        if v.len() > MAX_BUCKETS_PER_SLO {
            let excess = v.len() - MAX_BUCKETS_PER_SLO;
            v.drain(0..excess);
        }
    }

    /// Sum `(good, total)` over buckets whose `start_ms` falls within
    /// the last `window_secs` ending at `now_ms`.
    pub fn window_counts(&self, slo_id: &str, window_secs: u64, now_ms: u64) -> (u64, u64) {
        let cutoff = now_ms.saturating_sub(window_secs * 1_000);
        let mut good = 0u64;
        let mut total = 0u64;
        if let Some(v) = self.buckets.get(slo_id) {
            for b in v.iter().filter(|b| b.start_ms >= cutoff) {
                good += b.good;
                total += b.total;
            }
        }
        (good, total)
    }

    /// Compute the [`SloStatus`] for `def` as of `now_ms`.
    pub fn status(&self, def: &SloDef, now_ms: u64) -> SloStatus {
        let (good, total) = self.window_counts(&def.id, def.window_secs, now_ms);
        let allowed_bad = 1.0 - def.target_pct / 100.0;

        let sli_pct = if total == 0 {
            None
        } else {
            Some(good as f64 / total as f64 * 100.0)
        };

        let error_budget_remaining_pct = if total == 0 || allowed_bad <= 0.0 {
            None
        } else {
            let bad = total - good;
            let observed_bad_fraction = bad as f64 / total as f64;
            let consumed = observed_bad_fraction / allowed_bad;
            let remaining = (1.0 - consumed) * 100.0;
            Some(remaining.clamp(REMAINING_PCT_FLOOR, 100.0))
        };

        let burn_rate_fast = self.burn_rate(def, def.fast_burn.window_secs, now_ms, allowed_bad);
        let burn_rate_slow = self.burn_rate(def, def.slow_burn.window_secs, now_ms, allowed_bad);

        let compliant = sli_pct.map_or(true, |s| s >= def.target_pct);

        let fast_burn_threshold = def.fast_burn.threshold;
        let slow_burn_threshold = def.slow_burn.threshold;
        let breaching_fast = burn_rate_fast.map_or(false, |r| r >= fast_burn_threshold);
        let breaching_slow = burn_rate_slow.map_or(false, |r| r >= slow_burn_threshold);
        let budget_exhausted = error_budget_remaining_pct.map_or(false, |p| p <= 0.0);

        SloStatus {
            slo_id: def.id.clone(),
            sli_pct,
            target_pct: def.target_pct,
            good,
            total,
            error_budget_remaining_pct,
            burn_rate_fast,
            burn_rate_slow,
            compliant,
            fast_burn_threshold,
            slow_burn_threshold,
            breaching_fast,
            breaching_slow,
            budget_exhausted,
        }
    }

    /// Burn rate over `window_secs`: `observed_bad_fraction /
    /// allowed_bad`. `None` if the window had no observations or the
    /// objective leaves no error budget.
    fn burn_rate(
        &self,
        def: &SloDef,
        window_secs: u64,
        now_ms: u64,
        allowed_bad: f64,
    ) -> Option<f64> {
        if allowed_bad <= 0.0 {
            return None;
        }
        let (good, total) = self.window_counts(&def.id, window_secs, now_ms);
        if total == 0 {
            return None;
        }
        let bad = total - good;
        let observed_bad_fraction = bad as f64 / total as f64;
        Some(observed_bad_fraction / allowed_bad)
    }

    /// Persist state to `path` as pretty JSON. Failure is logged, not
    /// surfaced — the in-memory copy is authoritative. Mirrors
    /// [`crate::advisor::event::EventStore::save`].
    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    tracing::warn!(error = %e, path = %path.display(), "slo_state: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "slo_state: serialize failed"),
        }
    }

    /// Load state from `path`. A missing or unparseable file yields an
    /// empty state with the default bucket width (logged).
    pub fn load(path: &Path) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Self::new(DEFAULT_BUCKET_SECS),
        };
        match serde_json::from_slice(&bytes) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "slo_state: parse failed, starting empty"
                );
                Self::new(DEFAULT_BUCKET_SECS)
            }
        }
    }
}

/// Built-in default SLOs. SLOs are user-defined, so this is empty; the
/// `default_fast_burn` / `default_slow_burn` constructors provide the
/// sensible burn-window defaults.
pub fn catalog_default_slos() -> Vec<SloDef> {
    Vec::new()
}

/// A thin, persisted registry of user-defined [`SloDef`]s. Mirrors
/// [`crate::advisor::event::EventStore`] persistence: a single JSON file
/// at `{data_dir}/_slos.json`, write-through on mutation by the caller,
/// missing/unparseable file → empty.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SloRegistry {
    defs: Vec<SloDef>,
}

impl SloRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self { defs: Vec::new() }
    }

    /// All defs, in insertion order.
    pub fn list(&self) -> &[SloDef] {
        &self.defs
    }

    /// Find a def by id.
    pub fn get(&self, id: &str) -> Option<&SloDef> {
        self.defs.iter().find(|d| d.id == id)
    }

    /// Insert a new def or replace an existing one with the same id.
    pub fn upsert(&mut self, def: SloDef) {
        match self.defs.iter_mut().find(|d| d.id == def.id) {
            Some(existing) => *existing = def,
            None => self.defs.push(def),
        }
    }

    /// Remove the def with `id`. Returns whether it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.defs.len();
        self.defs.retain(|d| d.id != id);
        self.defs.len() != before
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
                    tracing::warn!(error = %e, path = %path.display(), "slos: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "slos: serialize failed"),
        }
    }

    /// Load from `path`. A missing or unparseable file yields an empty
    /// registry (logged on parse error).
    pub fn load(path: &Path) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Self::new(),
        };
        match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "slos: parse failed, starting empty"
                );
                Self::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: an availability SLO with target 99.0 and the default
    /// burn windows, 30d compliance window.
    fn avail_slo(id: &str) -> SloDef {
        SloDef {
            id: id.to_owned(),
            name: format!("{id} availability"),
            kind: SloKind::Availability,
            target_pct: 99.0,
            window_secs: 2_592_000,
            fast_burn: default_fast_burn(),
            slow_burn: default_slow_burn(),
        }
    }

    const APPROX: f64 = 1e-9;

    #[test]
    fn record_then_window_counts_excludes_old_buckets() {
        let mut st = SloState::new(60);
        // Three records spread over ~30 minutes (now = 1_800_000 ms).
        st.record("s", 10, 10, 0); // bucket start 0
        st.record("s", 20, 20, 600_000); // bucket start 600_000 (10m)
        st.record("s", 30, 30, 1_800_000); // bucket start 1_800_000 (30m)

        // Window of 15 minutes ending at 30m → cutoff = 900_000.
        // Only the 30m bucket qualifies.
        let (good, total) = st.window_counts("s", 900, 1_800_000);
        assert_eq!((good, total), (30, 30));

        // Window of 25 minutes → cutoff = 300_000 → 10m + 30m buckets.
        let (good, total) = st.window_counts("s", 1_500, 1_800_000);
        assert_eq!((good, total), (50, 50));

        // Wide window → all three buckets.
        let (good, total) = st.window_counts("s", 100_000, 1_800_000);
        assert_eq!((good, total), (60, 60));
    }

    #[test]
    fn records_into_same_bucket_accumulate() {
        let mut st = SloState::new(60);
        st.record("s", 1, 2, 1_000);
        st.record("s", 3, 4, 30_000); // same 60s bucket (start 0)
        let (good, total) = st.window_counts("s", 100_000, 60_000);
        assert_eq!((good, total), (4, 6));
    }

    #[test]
    fn sli_pct_basic_and_empty() {
        let mut st = SloState::new(60);
        st.record("s", 990, 1000, 1_000);
        let status = st.status(&avail_slo("s"), 2_000);
        assert!((status.sli_pct.unwrap() - 99.0).abs() < APPROX);

        // No data → None, and compliant defaults to true.
        let empty = SloState::new(60);
        let status = empty.status(&avail_slo("s"), 2_000);
        assert_eq!(status.sli_pct, None);
        assert!(status.compliant);
    }

    #[test]
    fn error_budget_remaining_at_various_burns() {
        // target 99.0 → allowed_bad = 0.01.
        // 10 bad / 1000 = 1% bad → consumed ~1.0 → remaining ~0%.
        let mut st = SloState::new(60);
        st.record("s", 990, 1000, 1_000);
        let status = st.status(&avail_slo("s"), 2_000);
        assert!(
            status.error_budget_remaining_pct.unwrap().abs() < 1e-6,
            "got {:?}",
            status.error_budget_remaining_pct
        );

        // 5 bad / 1000 = 0.5% bad → consumed 0.5 → remaining ~50%.
        let mut st = SloState::new(60);
        st.record("s", 995, 1000, 1_000);
        let status = st.status(&avail_slo("s"), 2_000);
        assert!((status.error_budget_remaining_pct.unwrap() - 50.0).abs() < 1e-6);

        // 0 bad → remaining 100%.
        let mut st = SloState::new(60);
        st.record("s", 1000, 1000, 1_000);
        let status = st.status(&avail_slo("s"), 2_000);
        assert!((status.error_budget_remaining_pct.unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn error_budget_none_when_no_data_or_no_budget() {
        let empty = SloState::new(60);
        let status = empty.status(&avail_slo("s"), 2_000);
        assert_eq!(status.error_budget_remaining_pct, None);

        // target 100.0 → allowed_bad <= 0 → None.
        let mut st = SloState::new(60);
        st.record("s", 990, 1000, 1_000);
        let mut def = avail_slo("s");
        def.target_pct = 100.0;
        let status = st.status(&def, 2_000);
        assert_eq!(status.error_budget_remaining_pct, None);
        assert_eq!(status.burn_rate_fast, None);
        assert_eq!(status.burn_rate_slow, None);
    }

    #[test]
    fn burn_rate_fast_and_slow() {
        // target 99.0 → allowed_bad = 0.01.
        // Fast window = 300s. Put 144 bad / 1000 inside it → 14.4% bad
        // → burn = 0.144 / 0.01 = 14.4.
        // Slow window = 3600s. Add an older bucket so the slow window
        // sees 60 bad / 1000 total → 6.0% bad → burn = 6.0. We achieve
        // the distinct slow ratio by NOT putting the fast bucket's data
        // in a separate-from-slow position; instead build two SLOs.

        // Fast burn check.
        let mut st = SloState::new(60);
        let now = 10_000_000u64;
        st.record("s", 856, 1000, now); // 144 bad in current bucket
        let status = st.status(&avail_slo("s"), now);
        assert!(
            (status.burn_rate_fast.unwrap() - 14.4).abs() < 1e-6,
            "fast burn got {:?}",
            status.burn_rate_fast
        );

        // Slow burn check: place 60 bad / 1000 in a bucket inside the
        // 3600s window but OUTSIDE the 300s fast window.
        let mut st = SloState::new(60);
        let now = 10_000_000u64;
        // ~20 minutes ago → outside 300s (5m) fast window, inside 3600s.
        let old = now - 1_200_000;
        st.record("s", 940, 1000, old); // 60 bad
        let status = st.status(&avail_slo("s"), now);
        assert_eq!(status.burn_rate_fast, None, "fast window should be empty");
        assert!(
            (status.burn_rate_slow.unwrap() - 6.0).abs() < 1e-6,
            "slow burn got {:?}",
            status.burn_rate_slow
        );
    }

    #[test]
    fn compliant_flag() {
        // 99.5% with target 99 → true.
        let mut st = SloState::new(60);
        st.record("s", 995, 1000, 1_000);
        assert!(st.status(&avail_slo("s"), 2_000).compliant);

        // 98% with target 99 → false.
        let mut st = SloState::new(60);
        st.record("s", 980, 1000, 1_000);
        assert!(!st.status(&avail_slo("s"), 2_000).compliant);

        // No data → true.
        let empty = SloState::new(60);
        assert!(empty.status(&avail_slo("s"), 2_000).compliant);
    }

    #[test]
    fn prune_cap_respected_newest_retained() {
        let mut st = SloState::new(60);
        let width = 60_000u64;
        // Record one observation per bucket across more than the cap.
        let extra = 100u64;
        let n = MAX_BUCKETS_PER_SLO as u64 + extra;
        for i in 0..n {
            // good=i marks the bucket so we can check newest survives.
            st.record("s", i, 1, i * width + 1);
        }
        let v = st.buckets.get("s").unwrap();
        assert_eq!(v.len(), MAX_BUCKETS_PER_SLO);
        // Newest bucket (start (n-1)*width) retained with good = n-1.
        assert_eq!(v.last().unwrap().good, n - 1);
        // Oldest retained bucket is at index `extra` (good = extra).
        assert_eq!(v.first().unwrap().good, extra);
    }

    #[test]
    fn save_load_round_trip_preserves_buckets_and_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("_slo_state.json");

        let mut st = SloState::new(60);
        st.record("a", 990, 1000, 1_000);
        st.record("a", 5, 10, 120_000); // a second bucket for "a"
        st.record("b", 1, 1, 1_000);
        st.save(&path);

        let loaded = SloState::load(&path);
        assert_eq!(loaded, st, "round-trip must be identical");

        // Spot-check counts survive.
        let (good, total) = loaded.window_counts("a", 1_000_000, 200_000);
        assert_eq!((good, total), (995, 1010));
    }

    #[test]
    fn load_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.json");
        let st = SloState::load(&path);
        assert_eq!(st.window_counts("anything", 1000, 1000), (0, 0));
    }

    #[test]
    fn catalog_default_slos_is_empty() {
        assert!(catalog_default_slos().is_empty());
    }

    #[test]
    fn status_breaching_flags_when_over_threshold() {
        // target 99 → allowed_bad 0.01. 200 bad / 1000 = 20% bad → burn
        // 0.2 / 0.01 = 20.0, comfortably above both the fast (14.4) and
        // slow (6.0) thresholds; budget consumed 20x → over (≤0).
        let mut st = SloState::new(60);
        let now = 10_000_000u64;
        st.record("s", 800, 1000, now); // 200 bad
        let status = st.status(&avail_slo("s"), now);
        assert!(status.breaching_fast, "burn {:?} >= {}", status.burn_rate_fast, status.fast_burn_threshold);
        assert!(status.breaching_slow, "burn {:?} >= {}", status.burn_rate_slow, status.slow_burn_threshold);
        assert!(status.budget_exhausted, "remaining {:?}", status.error_budget_remaining_pct);
        assert!((status.fast_burn_threshold - 14.4).abs() < APPROX);
        assert!((status.slow_burn_threshold - 6.0).abs() < APPROX);
    }

    #[test]
    fn status_not_breaching_when_under_threshold() {
        // 5 bad / 1000 = 0.5% → burn 0.5, well under both thresholds;
        // budget remaining ~50% so not exhausted.
        let mut st = SloState::new(60);
        let now = 10_000_000u64;
        st.record("s", 995, 1000, now);
        let status = st.status(&avail_slo("s"), now);
        assert!(!status.breaching_fast);
        assert!(!status.breaching_slow);
        assert!(!status.budget_exhausted);
    }

    #[test]
    fn status_flags_false_when_no_data() {
        let empty = SloState::new(60);
        let status = empty.status(&avail_slo("s"), 2_000);
        assert!(!status.breaching_fast);
        assert!(!status.breaching_slow);
        assert!(!status.budget_exhausted);
    }

    #[test]
    fn slo_registry_upsert_get_remove_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("_slos.json");

        let mut reg = SloRegistry::new();
        assert!(reg.list().is_empty());
        reg.upsert(avail_slo("a"));
        reg.upsert(avail_slo("b"));
        assert_eq!(reg.list().len(), 2);

        // Upsert with same id replaces, not appends.
        let mut a2 = avail_slo("a");
        a2.target_pct = 95.0;
        reg.upsert(a2);
        assert_eq!(reg.list().len(), 2);
        assert!((reg.get("a").unwrap().target_pct - 95.0).abs() < APPROX);

        reg.save(&path);
        let loaded = SloRegistry::load(&path);
        assert_eq!(loaded.list().len(), 2);
        assert!((loaded.get("a").unwrap().target_pct - 95.0).abs() < APPROX);

        let mut reg = loaded;
        assert!(reg.remove("a"));
        assert!(!reg.remove("nope"));
        assert!(reg.get("a").is_none());
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn slo_registry_load_missing_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.json");
        assert!(SloRegistry::load(&path).list().is_empty());
    }
}
