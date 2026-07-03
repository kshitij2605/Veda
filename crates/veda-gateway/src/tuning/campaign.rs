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
