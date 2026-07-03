//! Persisted campaign/trial data model. Mirrors `rag_eval.rs`'s
//! `EvalSet`/`EvalRun` SavedObject persistence pattern exactly — no new
//! storage subsystem.

use serde::{Deserialize, Serialize};

use crate::rest::saved_objects::{SavedObject, SavedObjectStore};
use crate::tuning::config::PipelineConfig;

pub const TYPE_TUNING_CAMPAIGN: &str = "tuning_campaign";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CampaignStatus {
    Running,
    Converged,
    BudgetExhausted,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignBudget {
    pub max_trials: usize,
    pub max_duration_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialObjectives {
    pub recall: f64,
    pub faithfulness: f64,
    pub latency_p50_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub id: String,
    pub config: PipelineConfig,
    pub objectives: TrialObjectives,
    pub is_pareto_optimal: bool,
    pub started_at: i64,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sibling_collection: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningCampaign {
    pub id: String,
    pub collection: String,
    pub eval_set_id: String,
    pub status: CampaignStatus,
    pub budget: CampaignBudget,
    #[serde(default)]
    pub trials: Vec<Trial>,
    /// Trial ids on the current global Pareto front.
    #[serde(default)]
    pub pareto_front: Vec<String>,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
}

pub async fn save_campaign(store: &SavedObjectStore, campaign: &TuningCampaign) {
    let obj = SavedObject {
        id: campaign.id.clone(),
        kind: TYPE_TUNING_CAMPAIGN.to_owned(),
        name: format!("campaign {} on {}", &campaign.id, &campaign.collection),
        body: serde_json::to_value(campaign).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    };
    store.put(obj).await;
}

pub async fn get_campaign(store: &SavedObjectStore, id: &str) -> Option<TuningCampaign> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_TUNING_CAMPAIGN {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

pub async fn list_campaigns(store: &SavedObjectStore) -> Vec<TuningCampaign> {
    store
        .list_by_type(TYPE_TUNING_CAMPAIGN)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::saved_objects::SavedObjectStore;

    #[tokio::test]
    async fn campaign_persists_and_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SavedObjectStore::new(tmp.path());

        let campaign = TuningCampaign {
            id: "camp-1".to_owned(),
            collection: "mycol".to_owned(),
            eval_set_id: "eval-1".to_owned(),
            status: CampaignStatus::Running,
            budget: CampaignBudget {
                max_trials: 50,
                max_duration_secs: 3600,
            },
            trials: vec![],
            pareto_front: vec![],
            created_at: 1_700_000_000,
            completed_at: None,
        };
        save_campaign(&store, &campaign).await;

        let fetched = get_campaign(&store, "camp-1")
            .await
            .expect("must round-trip");
        assert_eq!(fetched.collection, "mycol");
        assert_eq!(fetched.status, CampaignStatus::Running);
    }

    #[tokio::test]
    async fn list_campaigns_filters_by_type() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SavedObjectStore::new(tmp.path());
        let c = TuningCampaign {
            id: "camp-1".to_owned(),
            collection: "mycol".to_owned(),
            eval_set_id: "eval-1".to_owned(),
            status: CampaignStatus::Converged,
            budget: CampaignBudget {
                max_trials: 10,
                max_duration_secs: 60,
            },
            trials: vec![],
            pareto_front: vec![],
            created_at: 1,
            completed_at: Some(2),
        };
        save_campaign(&store, &c).await;
        let all = list_campaigns(&store).await;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "camp-1");
    }
}

#[cfg(test)]
mod campaign_loop_tests {
    use super::*;
    use crate::tuning::config::PipelineConfig;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn run_campaign_produces_a_valid_pareto_front() {
        let bases = vec![PipelineConfig::default()]; // 1 outer base for this test
        let budget = CampaignBudget {
            max_trials: 10,
            max_duration_secs: 3600,
        };

        let campaign = run_campaign("mycol", "eval-1", budget, bases, |config| async move {
            // A synthetic objective landscape: higher ef_search ->
            // better recall but worse latency (a real tradeoff, so the
            // front should have more than one point).
            let ef = config.tier1.ef_search as f64;
            TrialObjectives {
                recall: (ef / 200.0).min(1.0),
                faithfulness: 0.9,
                latency_p50_ms: ef * 0.4,
            }
        })
        .await;

        assert_eq!(campaign.collection, "mycol");
        assert!(!campaign.trials.is_empty());
        assert!(!campaign.pareto_front.is_empty());
        assert!(matches!(
            campaign.status,
            CampaignStatus::Converged | CampaignStatus::BudgetExhausted
        ));
        // Every id on the front must correspond to a real trial, and be
        // flagged is_pareto_optimal on that trial.
        for front_id in &campaign.pareto_front {
            let t = campaign
                .trials
                .iter()
                .find(|t| &t.id == front_id)
                .expect("front id must exist in trials");
            assert!(t.is_pareto_optimal);
        }
    }

    #[tokio::test]
    async fn run_campaign_calls_evaluate_trial_once_per_trial_not_per_base_rebuild() {
        // The core cost-correctness property: the injected evaluator is
        // called once per trial (however many trials the budget allows),
        // NOT once per base times some multiplier — this test's fake
        // evaluator just counts calls; the real /reindex-triggering
        // wiring (Task 4.3) is what actually enforces "one reindex per
        // distinct base," which this pure-loop test doesn't touch.
        let bases = vec![PipelineConfig::default()];
        let budget = CampaignBudget {
            max_trials: 5,
            max_duration_secs: 3600,
        };
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let campaign = run_campaign("mycol", "eval-1", budget, bases, move |_config| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                TrialObjectives {
                    recall: 0.9,
                    faithfulness: 0.9,
                    latency_p50_ms: 10.0,
                }
            }
        })
        .await;

        assert_eq!(call_count.load(Ordering::SeqCst), campaign.trials.len());
        assert!(campaign.trials.len() <= 5, "must respect max_trials budget");
    }

    #[tokio::test]
    async fn run_campaign_respects_max_trials_budget() {
        let bases = vec![PipelineConfig::default()];
        let budget = CampaignBudget {
            max_trials: 3,
            max_duration_secs: 3600,
        };
        let campaign = run_campaign("mycol", "eval-1", budget, bases, |_c| async move {
            TrialObjectives {
                recall: 0.5,
                faithfulness: 0.5,
                latency_p50_ms: 20.0,
            }
        })
        .await;
        assert!(campaign.trials.len() <= 3);
    }
}

use crate::tuning::search::{MultiObjectiveSearch, Nsga2Search};

/// Convert a `Trial`'s objectives to the maximize-everything convention
/// `MultiObjectiveSearch` expects (see `search.rs`'s doc comment):
/// recall and faithfulness are already "higher is better"; latency is
/// negated so "higher (less negative) is better" too.
fn objectives_vec(o: &TrialObjectives) -> Vec<f64> {
    vec![o.recall, o.faithfulness, -o.latency_p50_ms]
}

/// Run one bounded campaign: for each distinct tier-2/3 base in
/// `outer_bases`, run an inner NSGA-II search over tier-1 params;
/// aggregate every trial (across all bases) into one global Pareto
/// front. `evaluate_trial` is injected — the HTTP-facing caller (Task
/// 4.3) passes a closure that actually triggers `/reindex` on a base
/// change and runs the eval-set harness; this pure loop never touches
/// either directly, which is what makes it unit-testable.
pub async fn run_campaign<F, Fut>(
    collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    outer_bases: Vec<PipelineConfig>,
    mut evaluate_trial: F,
) -> TuningCampaign
where
    F: FnMut(PipelineConfig) -> Fut,
    Fut: std::future::Future<Output = TrialObjectives>,
{
    let started_at = now_ms();
    let mut trials: Vec<Trial> = Vec::new();
    // Tier-1 search bounds: a conservative, fixed range (a smarter
    // corpus-size-derived range is a natural follow-up, not required
    // for v1). Order matches PipelineConfig::to_param_vec's tier-1
    // slice: [ef_search, fusion_mode_idx, dense_weight, rerank, top_k].
    let tier1_bounds = vec![
        (16.0, 256.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (5.0, 50.0),
    ];

    'outer: for base in &outer_bases {
        let mut search = Nsga2Search::new(3, tier1_bounds.clone(), 8);
        let mut no_improve_streak = 0usize;
        let mut best_front_size = 0usize;

        loop {
            if trials.len() >= budget.max_trials {
                break 'outer;
            }
            if elapsed_secs(started_at) >= budget.max_duration_secs {
                break 'outer;
            }
            let batch = search.propose(4.min(budget.max_trials - trials.len()));
            if batch.is_empty() {
                break;
            }
            for tier1_params in batch {
                if trials.len() >= budget.max_trials {
                    break 'outer;
                }
                let mut config = base.clone();
                apply_tier1_params(&mut config, &tier1_params);

                let trial_start = now_ms();
                let objectives = evaluate_trial(config.clone()).await;
                let duration_ms = (now_ms() - trial_start).max(0) as u64;

                search.record(tier1_params, objectives_vec(&objectives));
                trials.push(Trial {
                    id: format!("trial-{}", uuid::Uuid::new_v4()),
                    config,
                    objectives,
                    is_pareto_optimal: false, // recomputed globally below
                    started_at: trial_start,
                    duration_ms,
                    sibling_collection: None,
                });
            }

            let front_size = search.pareto_front().len();
            if front_size <= best_front_size {
                no_improve_streak += 1;
            } else {
                no_improve_streak = 0;
                best_front_size = front_size;
            }
            if no_improve_streak >= 3 {
                break; // this base has converged; move to the next base
            }
        }
    }

    let pareto_ids = mark_and_collect_global_pareto_front(&mut trials);
    let completed_at = now_ms();
    let status = if trials.len() >= budget.max_trials {
        CampaignStatus::BudgetExhausted
    } else {
        CampaignStatus::Converged
    };

    TuningCampaign {
        id: format!("camp-{}", uuid::Uuid::new_v4()),
        collection: collection.to_owned(),
        eval_set_id: eval_set_id.to_owned(),
        status,
        budget,
        trials,
        pareto_front: pareto_ids,
        created_at: started_at,
        completed_at: Some(completed_at),
    }
}

fn apply_tier1_params(config: &mut PipelineConfig, params: &[f64]) {
    config.tier1.ef_search = params[0].round().max(1.0) as usize;
    config.tier1.fusion_mode = if params[1] < 0.5 {
        crate::tuning::config::FusionMode::Rrf
    } else {
        crate::tuning::config::FusionMode::Weighted
    };
    config.tier1.dense_weight = params[2];
    config.tier1.rerank = params[3] >= 0.5;
    config.tier1.top_k = params[4].round().max(1.0) as usize;
}

/// Global non-dominated sort over every trial's objectives (maximize
/// recall, faithfulness; minimize latency_p50_ms), across ALL bases —
/// this is what makes the search genuinely joint, not per-base-greedy.
/// Sets `is_pareto_optimal` on each trial and returns the front's ids.
fn mark_and_collect_global_pareto_front(trials: &mut [Trial]) -> Vec<String> {
    let dominates = |a: &TrialObjectives, b: &TrialObjectives| {
        a.recall >= b.recall
            && a.faithfulness >= b.faithfulness
            && a.latency_p50_ms <= b.latency_p50_ms
            && (a.recall > b.recall
                || a.faithfulness > b.faithfulness
                || a.latency_p50_ms < b.latency_p50_ms)
    };
    let objectives: Vec<TrialObjectives> = trials.iter().map(|t| t.objectives.clone()).collect();
    let mut front_ids = Vec::new();
    for (i, trial) in trials.iter_mut().enumerate() {
        let is_optimal = !objectives
            .iter()
            .enumerate()
            .any(|(j, other)| j != i && dominates(other, &objectives[i]));
        trial.is_pareto_optimal = is_optimal;
        if is_optimal {
            front_ids.push(trial.id.clone());
        }
    }
    front_ids
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
fn elapsed_secs(since_ms: i64) -> u64 {
    ((now_ms() - since_ms).max(0) / 1000) as u64
}
