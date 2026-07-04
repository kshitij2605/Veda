//! Cron-style scheduled-job scheduler for Veda.
//!
//! ## Design
//!
//! Jobs are a **typed enum** (`JobKind`) — not arbitrary code or SQL —
//! so the attack surface is bounded to a fixed set of safe operations:
//! TTL sweep, reindex, compact, and evaluate. This is the key safety
//! difference from pg_cron.
//!
//! The scheduler tick loop mirrors `log_appliers::spawn_one`: a 60-second
//! `tokio::time::interval` with `MissedTickBehavior::Delay`.
//!
//! Jobs are persisted to `{data_dir}/_scheduled_jobs.json`, mirroring
//! `SavedObjectStore`.
//!
//! ## Job dispatch status
//!
//! - `JobKind::TtlSweep` — **stubbed**: logs "would run TTL sweep" + TODO hook.
//! - `JobKind::Reindex` — **stubbed**: logs "would run reindex" + TODO hook.
//! - `JobKind::Compact` — **stubbed**: logs "would run compact" + TODO hook.
//! - `JobKind::Evaluate` — runs every eval set's harness against its
//!   configured agent and persists the resulting `EvalRun`.
//! - `JobKind::TuningCampaign` — runs one bounded auto-tuning campaign
//!   (Phase 4 of the auto-tuning plan) via
//!   `rest::tuning::run_and_persist_campaign`.
//!
//! Wiring to real ops is deferred; the scheduler infrastructure, registry,
//! REST API, and persistence are the deliverable for this task.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use cron::Schedule;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use tokio::task::JoinHandle;

use crate::rest::state::AppState;

/// The fixed set of operations a scheduled job can trigger.
///
/// Using an enum (not arbitrary strings or code) means the scheduler
/// can never be used to execute unexpected operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    TtlSweep {
        collection: String,
    },
    Reindex {
        collection: String,
    },
    Compact {
        collection: String,
    },
    Evaluate {
        collection: String,
    },
    /// Phase 4 of the auto-tuning plan: run one bounded campaign.
    TuningCampaign {
        collection: String,
        eval_set_id: String,
        max_trials: usize,
        max_duration_secs: u64,
    },
}

/// A single scheduled job entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    /// Unique job name (used as map key).
    pub name: String,
    /// Standard 5- or 6-field cron expression (parsed by the `cron` crate).
    pub cron: String,
    /// The operation to perform.
    pub kind: JobKind,
    /// Unix timestamp (seconds) of the last successful run. None = never run.
    pub last_run: Option<i64>,
}

impl ScheduledJob {
    /// Returns `true` if this job is due to fire at `now`.
    ///
    /// "Due" means: the next scheduled time after `last_run` (or after the
    /// epoch, if this is the first run) is ≤ `now`.
    ///
    /// `now` is injected (not `Utc::now()`) so tests are deterministic.
    pub fn is_due(&self, now: DateTime<Utc>) -> bool {
        let Ok(schedule) = Schedule::from_str(&self.cron) else {
            return false;
        };
        // The reference point: the last time it ran, or the Unix epoch.
        let after = match self.last_run {
            Some(ts) => match Utc.timestamp_opt(ts, 0) {
                chrono::LocalResult::Single(dt) => dt,
                _ => Utc.timestamp_opt(0, 0).unwrap(),
            },
            None => Utc.timestamp_opt(0, 0).unwrap(),
        };
        // The next scheduled tick after `after`.
        match schedule.after(&after).next() {
            Some(next) => next <= now,
            None => false,
        }
    }
}

/// In-process registry for scheduled jobs.
///
/// Thread-safe via `Mutex<HashMap>`. Mirrored to disk after every mutation.
pub struct JobRegistry {
    jobs: Mutex<HashMap<String, ScheduledJob>>,
    path: PathBuf,
}

impl JobRegistry {
    /// Load from `{data_dir}/_scheduled_jobs.json`, starting empty if absent or unparseable.
    pub fn new(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("_scheduled_jobs.json");
        let jobs: HashMap<String, ScheduledJob> = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "scheduler: parse failed, starting empty"
                );
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        Self {
            jobs: Mutex::new(jobs),
            path,
        }
    }

    /// Register (or overwrite) a job. Returns an error if the cron expr is invalid.
    pub fn register(&self, job: ScheduledJob) -> Result<(), String> {
        // Validate cron expression before storing.
        Schedule::from_str(&job.cron)
            .map_err(|e| format!("invalid cron expression '{}': {e}", job.cron))?;
        let mut m = self.jobs.lock().unwrap();
        m.insert(job.name.clone(), job);
        drop(m);
        self.save()?;
        Ok(())
    }

    /// List all registered jobs (snapshot, unordered).
    pub fn list(&self) -> Vec<ScheduledJob> {
        self.jobs.lock().unwrap().values().cloned().collect()
    }

    /// Remove a job by name. Returns `true` if it was present.
    pub fn remove(&self, name: &str) -> Result<bool, String> {
        let removed = self.jobs.lock().unwrap().remove(name).is_some();
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Return the names of all jobs whose next-fire time is ≤ `now`.
    pub fn due_jobs(&self, now: DateTime<Utc>) -> Vec<String> {
        self.jobs
            .lock()
            .unwrap()
            .values()
            .filter(|j| j.is_due(now))
            .map(|j| j.name.clone())
            .collect()
    }

    /// Update `last_run` for a job by name (in-memory only; caller must call `save`
    /// separately, or use `update_last_run_and_save`).
    pub fn set_last_run(&self, name: &str, ts: i64) {
        let mut m = self.jobs.lock().unwrap();
        if let Some(job) = m.get_mut(name) {
            job.last_run = Some(ts);
        }
    }

    /// Update `last_run` for a job by name and persist immediately.
    pub fn update_last_run_and_save(&self, name: &str, ts: i64) -> Result<(), String> {
        self.set_last_run(name, ts);
        self.save()
    }

    /// Persist the current registry to disk (blocking). Mirrors `SavedObjectStore::write_through`.
    pub fn save(&self) -> Result<(), String> {
        let path = self.path.clone();
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("scheduler: create_dir_all failed: {e}"))?;
        }
        let snapshot: HashMap<String, ScheduledJob> = self.jobs.lock().unwrap().clone();
        let bytes = serde_json::to_vec_pretty(&snapshot)
            .map_err(|e| format!("scheduler: serialize failed: {e}"))?;
        std::fs::write(&path, bytes)
            .map_err(|e| format!("scheduler: write failed to {}: {e}", path.display()))
    }

    /// Reload the registry from disk, replacing the in-memory state.
    pub fn load(&self) -> Result<(), String> {
        let bytes =
            std::fs::read(&self.path).map_err(|e| format!("scheduler: read failed: {e}"))?;
        let loaded: HashMap<String, ScheduledJob> = serde_json::from_slice(&bytes)
            .map_err(|e| format!("scheduler: deserialize failed: {e}"))?;
        *self.jobs.lock().unwrap() = loaded;
        Ok(())
    }
}

/// Dispatch a job kind.
///
/// `TtlSweep` calls `SegmentManager::sweep_expired` on all shards of the
/// named collection using the real wall-clock time at dispatch.
/// All other kinds remain stubs — their wiring is deferred to later phases.
async fn dispatch(kind: &JobKind, state: &AppState) {
    match kind {
        JobKind::TtlSweep { collection } => {
            let now_secs = Utc::now().timestamp();
            let mut total_swept = 0usize;
            {
                let mut mgrs = state.inner.segment_managers.write().await;
                if let Some(shards) = mgrs.get_mut(collection.as_str()) {
                    for sm in shards.values_mut() {
                        total_swept += sm.sweep_expired(now_secs);
                    }
                }
            }
            tracing::info!(
                collection = %collection,
                swept = total_swept,
                now_secs,
                "scheduler: TTL sweep complete"
            );
        }
        JobKind::Reindex { collection } => {
            tracing::info!(
                collection = %collection,
                "scheduler: would run reindex — TODO: wire to collections::reindex_collection"
            );
        }
        JobKind::Compact { collection } => {
            tracing::info!(
                collection = %collection,
                "scheduler: would run compact — TODO: wire to segment_manager compaction"
            );
        }
        JobKind::Evaluate { collection } => {
            // Was a stub ("would run evaluate — TODO"). Now runs every
            // eval set whose `agent_id` targets this collection's default
            // agent — a coarse but real dispatch; a per-collection eval-set
            // index is a natural follow-up, not required for this fix.
            let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
            for set in sets {
                let _ = crate::rest::rag_eval::run_eval_set_for_job(state, &set).await;
            }
            tracing::info!(collection = %collection, "scheduler: evaluate job dispatched");
        }
        JobKind::TuningCampaign {
            collection,
            eval_set_id,
            max_trials,
            max_duration_secs,
        } => {
            // Job-triggered campaigns (scheduled/corpus-drift) use a
            // fixed default base count; num_bases is a user-facing tuning
            // knob exposed only on the on-demand HTTP trigger
            // (`StartCampaignRequest`), not grown onto JobKind's schema.
            const DEFAULT_NUM_BASES: usize = 3;
            let _ = crate::rest::tuning::run_and_persist_campaign(
                state,
                collection,
                eval_set_id,
                crate::tuning::campaign::CampaignBudget {
                    max_trials: *max_trials,
                    max_duration_secs: *max_duration_secs,
                },
                DEFAULT_NUM_BASES,
            )
            .await;
        }
    }
}

/// Spawn the scheduler tick loop. Ticks every 60 seconds, finds due jobs,
/// dispatches them, and updates `last_run`. Mirrors `log_appliers::spawn_one`.
pub fn spawn_scheduler(state: AppState) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let now = Utc::now();
            let due = state.inner.job_registry.due_jobs(now);
            for name in due {
                // Fetch kind (avoid holding lock across await).
                let kind = {
                    let m = state.inner.job_registry.jobs.lock().unwrap();
                    m.get(&name).map(|j| j.kind.clone())
                };
                if let Some(kind) = kind {
                    dispatch(&kind, &state).await;
                    let ts = Utc::now().timestamp();
                    if let Err(e) = state.inner.job_registry.update_last_run_and_save(&name, ts) {
                        tracing::warn!(error = %e, job = %name, "scheduler: persist last_run failed");
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    // ── helpers ────────────────────────────────────────────────────────────

    fn every_minute_job(last_run: Option<i64>) -> ScheduledJob {
        ScheduledJob {
            name: "test-job".into(),
            cron: "0 * * * * *".into(), // every minute (second=0)
            kind: JobKind::Compact {
                collection: "col".into(),
            },
            last_run,
        }
    }

    // ── Test 1: parse a valid cron expr; reject an invalid one ─────────────

    #[test]
    fn valid_cron_parses() {
        assert!(
            Schedule::from_str("0 * * * * *").is_ok(),
            "every-minute should parse"
        );
        assert!(
            Schedule::from_str("0 0 * * * *").is_ok(),
            "every-hour should parse"
        );
        assert!(
            Schedule::from_str("0 0 0 * * *").is_ok(),
            "every-day should parse"
        );
    }

    #[test]
    fn invalid_cron_rejected() {
        assert!(
            Schedule::from_str("not-a-cron").is_err(),
            "garbage string should not parse"
        );
        assert!(
            Schedule::from_str("99 * * * *").is_err(),
            "out-of-range fields should not parse"
        );
    }

    // ── Test 2: is_due returns correct values ──────────────────────────────

    #[test]
    fn is_due_true_when_next_fire_has_passed() {
        // last_run 2 minutes ago → next fire was 1 minute ago → due
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 12, 2, 0).unwrap();
        let last_run = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        let job = every_minute_job(Some(last_run.timestamp()));
        assert!(
            job.is_due(now),
            "job last run 2 min ago with 1-min schedule should be due"
        );
    }

    #[test]
    fn is_due_false_when_next_fire_is_in_future() {
        // last_run 30 seconds ago → next fire is 30 seconds from now → not due
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 30).unwrap();
        let last_run = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        let job = every_minute_job(Some(last_run.timestamp()));
        assert!(
            !job.is_due(now),
            "job last run 30s ago with 1-min schedule should NOT be due"
        );
    }

    #[test]
    fn is_due_true_on_first_run_never_run() {
        // Never run → next fire after epoch → way in the past → due
        let now = Utc::now();
        let job = every_minute_job(None);
        assert!(
            job.is_due(now),
            "job that has never run should be due immediately"
        );
    }

    #[test]
    fn is_due_invalid_cron_returns_false() {
        let job = ScheduledJob {
            name: "bad".into(),
            cron: "not-a-cron".into(),
            kind: JobKind::Compact {
                collection: "col".into(),
            },
            last_run: None,
        };
        assert!(
            !job.is_due(Utc::now()),
            "job with invalid cron should never be due"
        );
    }

    // ── Test 3: registry register / list / remove + save/load roundtrip ────

    #[test]
    fn registry_register_list_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = JobRegistry::new(tmp.path());

        let job = every_minute_job(None);
        reg.register(job.clone()).unwrap();

        let list = reg.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "test-job");

        let removed = reg.remove("test-job").unwrap();
        assert!(removed);
        assert!(reg.list().is_empty());

        let removed_again = reg.remove("test-job").unwrap();
        assert!(!removed_again, "remove of non-existent job returns false");
    }

    #[test]
    fn registry_rejects_invalid_cron() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = JobRegistry::new(tmp.path());

        let bad_job = ScheduledJob {
            name: "bad".into(),
            cron: "not-valid".into(),
            kind: JobKind::Compact {
                collection: "col".into(),
            },
            last_run: None,
        };
        assert!(
            reg.register(bad_job).is_err(),
            "registry should reject invalid cron"
        );
    }

    #[test]
    fn registry_save_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = JobRegistry::new(tmp.path());

        let job = ScheduledJob {
            name: "daily-compact".into(),
            cron: "0 0 0 * * *".into(),
            kind: JobKind::Compact {
                collection: "my-col".into(),
            },
            last_run: Some(1_700_000_000),
        };
        reg.register(job).unwrap();

        // Load into a fresh registry from the same path.
        let reg2 = JobRegistry::new(tmp.path());
        let list = reg2.list();
        assert_eq!(list.len(), 1);
        let loaded = &list[0];
        assert_eq!(loaded.name, "daily-compact");
        assert_eq!(loaded.cron, "0 0 0 * * *");
        assert_eq!(loaded.last_run, Some(1_700_000_000));
        match &loaded.kind {
            JobKind::Compact { collection } => assert_eq!(collection, "my-col"),
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    #[test]
    fn due_jobs_returns_correct_subset() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = JobRegistry::new(tmp.path());

        // due_job: every minute, last_run 2 minutes ago
        // now=12:02:30 → next fire after 12:00:00 is 12:01:00 ≤ now → DUE
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 12, 2, 30).unwrap();
        let past = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();

        reg.register(ScheduledJob {
            name: "due-job".into(),
            cron: "0 * * * * *".into(),
            kind: JobKind::Compact {
                collection: "a".into(),
            },
            last_run: Some(past.timestamp()),
        })
        .unwrap();

        // not_due: every minute, last_run at 12:02:00
        // next fire after 12:02:00 is 12:03:00 > 12:02:30 → NOT DUE
        let recent = Utc.with_ymd_and_hms(2024, 1, 1, 12, 2, 0).unwrap();
        reg.register(ScheduledJob {
            name: "not-due-job".into(),
            cron: "0 * * * * *".into(),
            kind: JobKind::Compact {
                collection: "b".into(),
            },
            last_run: Some(recent.timestamp()),
        })
        .unwrap();

        let due = reg.due_jobs(now);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0], "due-job");
    }

    // ── Test 4: TuningCampaign job dispatch (real /reindex + eval-set
    // scoring wiring, Task 4.3) ─────────────────────────────────────────

    /// Builds an `AppState` with a text-modality embedding model registered
    /// (mock endpoint — `EmbeddingClient::embed_text` falls back to a
    /// deterministic vector, no real embedding service needed) and
    /// `collection` created with one seeded record. The registered model is
    /// required so the campaign's tier3-varying base (which triggers a real
    /// `/reindex` with `re_embed: true`) doesn't 424 on a missing embedder.
    ///
    /// This is a small local fixture rather than reusing `rag_eval.rs`'s
    /// private `test_state_with_llm_and_records` (see Task 4.3 notes): the
    /// eval set used below has zero cases, so the agent-turn/LLM path is
    /// never exercised and no LLM mock is needed here.
    async fn test_state_with_tuning_fixture(collection: &str) -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());

        {
            let mut registry = state.embedding_registry().write().await;
            registry.register(veda_core::embedding::EmbeddingModelConfig {
                name: "mock-text".into(),
                endpoint: "http://mock:8080/embed".into(),
                api_key: String::new(),
                supported_modalities: vec![veda_core::embedding::Modality::Text],
                output_dimension: 2,
                max_input_size_bytes: 1024 * 1024,
                timeout_seconds: 10,
                batch_size: 8,
            });
        }

        let req = serde_json::from_value(serde_json::json!({
            "name": collection,
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "text", "type": "string" }
            ],
            "seed_records": [{ "v": [1.0, 0.0], "text": "some text" }],
        }))
        .expect("fixture create-collection request should deserialize");
        let resp = crate::rest::collections::create_collection(
            axum::extract::State(state.clone()),
            crate::tenant::Tenant(crate::tenant::TenantId::default_tenant()),
            axum::Json(req),
        )
        .await;
        use axum::response::IntoResponse;
        let status = resp.into_response().status();
        assert!(
            status.is_success(),
            "fixture collection create should succeed, got {status}"
        );

        state
    }

    #[tokio::test]
    async fn tuning_campaign_job_calls_reindex_once_per_distinct_base_not_per_trial() {
        // "mycol" under the default tenant is stored as "default__mycol" —
        // `JobKind::TuningCampaign`'s `collection` field (like the other
        // `JobKind` variants, e.g. `TtlSweep`) is the already-scoped storage
        // key, not a user-facing name, so the fixture and the job below
        // both use the scoped form.
        let state = test_state_with_tuning_fixture("mycol").await;

        let eval_set = crate::rest::rag_eval::EvalSet {
            id: "eval-1".to_owned(),
            name: "fixture".to_owned(),
            agent_id: None,
            cases: vec![], // an empty set is enough to exercise the loop's plumbing
        };
        crate::rest::rag_eval::save_set_pub(&state.inner.saved_objects, &eval_set).await;

        let job = ScheduledJob {
            name: "campaign-1".to_owned(),
            cron: "0 * * * * *".to_owned(),
            kind: JobKind::TuningCampaign {
                collection: "default__mycol".to_owned(),
                eval_set_id: "eval-1".to_owned(),
                max_trials: 4,
                max_duration_secs: 60,
            },
            last_run: None,
        };
        dispatch(&job.kind, &state).await;

        let campaigns = crate::tuning::campaign::list_campaigns(&state.inner.saved_objects).await;
        assert_eq!(campaigns.len(), 1);
        assert_eq!(
            campaigns[0].status,
            crate::tuning::campaign::CampaignStatus::BudgetExhausted
        );
    }
}
