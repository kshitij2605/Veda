//! Persisted faithfulness-score series for the quality SLO engine.
//!
//! Phase A's grounding scorer ([`crate::agent::grounding`]) produces a
//! per-converse-turn [`faithfulness`](crate::agent::grounding::Grounding::faithfulness)
//! in `[0.0, 1.0]` and emits a `veda_agent_faithfulness` gauge, but it
//! keeps no series. This store mirrors
//! [`crate::advisor::recall_probe::RecallResults`] /
//! [`crate::advisor::synthetic::SyntheticResults`]: a bounded ring of
//! `{ ts_ms, faithfulness, agent }` samples, persisted to
//! `{data_dir}/_faithfulness_results.json`, written (brief lock, off the
//! network path) by `run_agent_turn` whenever a turn scored faithfulness.
//!
//! The advisor tick reads it via [`FaithfulnessResults::since`] to feed
//! the [`crate::advisor::slo::SloKind::Faithfulness`] SLI.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Cap on retained faithfulness samples (newest kept).
pub const MAX_FAITHFULNESS_SAMPLES: usize = 500;

/// Faithfulness-results file name under `data_dir`.
pub const FAITHFULNESS_RESULTS_FILE: &str = "_faithfulness_results.json";

/// One faithfulness measurement from an agent turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaithfulnessSample {
    pub ts_ms: u64,
    /// Faithfulness score in `[0.0, 1.0]`.
    pub faithfulness: f64,
    /// The agent id this turn ran as (the user-facing id, e.g.
    /// `veda-agent`), so a [`SloKind::Faithfulness`] can scope to it.
    pub agent: String,
}

/// A bounded, newest-last ring of faithfulness samples, persisted to
/// `{data_dir}/_faithfulness_results.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaithfulnessResults {
    /// newest-last ring of samples (across all agents).
    samples: Vec<FaithfulnessSample>,
}

impl FaithfulnessResults {
    /// Empty results.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Append a sample, pruning the oldest beyond
    /// [`MAX_FAITHFULNESS_SAMPLES`].
    pub fn record(&mut self, agent: &str, faithfulness: f64, ts_ms: u64) {
        self.samples.push(FaithfulnessSample {
            ts_ms,
            faithfulness,
            agent: agent.to_owned(),
        });
        if self.samples.len() > MAX_FAITHFULNESS_SAMPLES {
            let excess = self.samples.len() - MAX_FAITHFULNESS_SAMPLES;
            self.samples.drain(0..excess);
        }
    }

    /// Faithfulness scores recorded strictly after `since_ms`, optionally
    /// filtered to one agent (`None` = all agents).
    pub fn since(&self, agent_filter: Option<&str>, since_ms: u64) -> Vec<f64> {
        self.samples
            .iter()
            .filter(|s| s.ts_ms > since_ms)
            .filter(|s| agent_filter.is_none_or(|a| s.agent == a))
            .map(|s| s.faithfulness)
            .collect()
    }

    /// The most-recent up-to-`limit` faithfulness scores, oldest-first
    /// (chronological), across all agents. Used by the advisor to attach a
    /// drift-detection series to the snapshot. `limit == 0` → empty.
    pub fn recent(&self, limit: usize) -> Vec<f64> {
        if limit == 0 {
            return Vec::new();
        }
        let start = self.samples.len().saturating_sub(limit);
        self.samples[start..]
            .iter()
            .map(|s| s.faithfulness)
            .collect()
    }

    /// Total retained sample count (test/inspection aid).
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the store holds no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Persist to `path` as pretty JSON. Failure is logged, not surfaced.
    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    tracing::warn!(error = %e, path = %path.display(), "faithfulness_results: write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "faithfulness_results: serialize failed"),
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
                tracing::warn!(error = %e, path = %path.display(), "faithfulness_results: parse failed, starting empty");
                Self::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_since_filters_by_time_and_agent() {
        let mut r = FaithfulnessResults::new();
        r.record("a", 0.5, 100);
        r.record("a", 0.7, 200);
        r.record("b", 0.9, 300);

        // since 150, all agents → 0.7 (a@200) + 0.9 (b@300).
        let all = r.since(None, 150);
        assert_eq!(all.len(), 2);
        assert!(all.contains(&0.7) && all.contains(&0.9));

        // since 150, agent "a" only → just 0.7.
        let a_only = r.since(Some("a"), 150);
        assert_eq!(a_only, vec![0.7]);

        // since 0, agent "a" → both a samples (mean 0.6).
        let a_all = r.since(Some("a"), 0);
        assert_eq!(a_all, vec![0.5, 0.7]);

        // strict > boundary: since 200 excludes the 200 sample.
        assert_eq!(r.since(Some("a"), 200), Vec::<f64>::new());
    }

    #[test]
    fn ring_is_bounded_newest_kept() {
        let mut r = FaithfulnessResults::new();
        let n = MAX_FAITHFULNESS_SAMPLES as u64 + 30;
        for i in 0..n {
            r.record("a", 0.5, i);
        }
        assert_eq!(r.len(), MAX_FAITHFULNESS_SAMPLES);
        // Oldest dropped: nothing at ts < 30 survives.
        assert!(r.since(None, 0).len() == MAX_FAITHFULNESS_SAMPLES);
        assert_eq!(r.since(Some("a"), n - 2).len(), 1); // only the newest ts
    }

    #[test]
    fn recent_returns_newest_limit_oldest_first() {
        let mut r = FaithfulnessResults::new();
        for (i, v) in [0.9, 0.8, 0.5, 0.4].into_iter().enumerate() {
            r.record("a", v, i as u64);
        }
        // Newest 2, in chronological order.
        assert_eq!(r.recent(2), vec![0.5, 0.4]);
        // limit > len → whole series.
        assert_eq!(r.recent(10), vec![0.9, 0.8, 0.5, 0.4]);
        // limit 0 → empty.
        assert!(r.recent(0).is_empty());
    }

    #[test]
    fn save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FAITHFULNESS_RESULTS_FILE);
        let mut r = FaithfulnessResults::new();
        r.record("a", 0.6, 1);
        r.record("b", 0.8, 2);
        r.save(&path);

        let loaded = FaithfulnessResults::load(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.since(Some("b"), 0), vec![0.8]);
    }

    #[test]
    fn load_missing_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.json");
        assert!(FaithfulnessResults::load(&path).is_empty());
    }
}
