//! Opt-in, per-collection auto-optimization configuration: when enabled
//! (default `false`) and the dedicated answer-gen LLM is configured,
//! `ingest_common::ingest_markdown` (a later task) automatically
//! regenerates this collection's eval set and starts a tuning campaign
//! after every ingestion that adds data. Persists via the same
//! `SavedObjectStore` campaigns/eval-sets already use — no new storage
//! subsystem.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::rest::saved_objects::{SavedObject, SavedObjectStore};
use crate::tenant::{self, Tenant};

pub const TYPE_AUTO_OPTIMIZATION_CONFIG: &str = "auto_optimization_config";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TuningMode {
    /// Multi-objective (recall, faithfulness, latency) Pareto search —
    /// today's existing tuning-engine behavior.
    #[default]
    Quality,
    /// Same evaluation loop, but reports the single lowest-latency trial
    /// among those clearing a recall floor (implemented in a later task).
    Performance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoOptimizationConfig {
    pub collection: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_questions: Option<usize>,
    #[serde(default)]
    pub mode: TuningMode,
}

pub async fn get_auto_optimization_config(
    store: &SavedObjectStore,
    scoped_collection: &str,
) -> Option<AutoOptimizationConfig> {
    let obj = store.get(scoped_collection).await?;
    if obj.kind != TYPE_AUTO_OPTIMIZATION_CONFIG {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

pub async fn save_auto_optimization_config(store: &SavedObjectStore, cfg: &AutoOptimizationConfig) {
    let obj = SavedObject {
        id: cfg.collection.clone(),
        kind: TYPE_AUTO_OPTIMIZATION_CONFIG.to_owned(),
        name: format!("auto-optimization config for {}", cfg.collection),
        body: serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    };
    store.put(obj).await;
}

pub async fn get_auto_optimization(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
) -> Json<AutoOptimizationConfig> {
    let scoped = tenant::scope(&tenant, &collection);
    let cfg = get_auto_optimization_config(&state.inner.saved_objects, &scoped)
        .await
        .unwrap_or(AutoOptimizationConfig {
            collection: collection.clone(),
            enabled: false,
            target_questions: None,
            mode: TuningMode::default(),
        });
    Json(cfg)
}

#[derive(Deserialize)]
pub struct SetAutoOptimizationRequest {
    pub enabled: bool,
    #[serde(default)]
    pub target_questions: Option<usize>,
    #[serde(default)]
    pub mode: TuningMode,
}

pub async fn set_auto_optimization(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<SetAutoOptimizationRequest>,
) -> Result<Json<AutoOptimizationConfig>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &collection);
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{collection}' not found"),
                }),
            ));
        }
    }
    let cfg = AutoOptimizationConfig {
        collection: scoped.clone(),
        enabled: req.enabled,
        target_questions: req.target_questions,
        mode: req.mode,
    };
    save_auto_optimization_config(&state.inner.saved_objects, &cfg).await;
    Ok(Json(AutoOptimizationConfig {
        collection: collection.clone(),
        ..cfg
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::state::AppState;

    fn test_state() -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        AppState::new(tmp.path())
    }

    #[tokio::test]
    async fn save_and_get_round_trips() {
        let state = test_state();
        let cfg = AutoOptimizationConfig {
            collection: "default__mycol".to_owned(),
            enabled: true,
            target_questions: Some(30),
            mode: TuningMode::Performance,
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg).await;
        let loaded = get_auto_optimization_config(&state.inner.saved_objects, "default__mycol")
            .await
            .expect("config should round-trip");
        assert_eq!(loaded, cfg);
    }

    #[tokio::test]
    async fn get_returns_disabled_default_when_never_set() {
        let state = test_state();
        let loaded =
            get_auto_optimization_config(&state.inner.saved_objects, "default__never_configured")
                .await;
        assert!(
            loaded.is_none(),
            "no config saved yet -> None, callers default to disabled"
        );
    }

    #[tokio::test]
    async fn saving_the_same_collection_twice_overwrites_not_duplicates() {
        let state = test_state();
        let cfg1 = AutoOptimizationConfig {
            collection: "default__mycol".to_owned(),
            enabled: false,
            target_questions: None,
            mode: TuningMode::Quality,
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg1).await;
        let cfg2 = AutoOptimizationConfig {
            enabled: true,
            ..cfg1
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg2).await;
        let loaded = get_auto_optimization_config(&state.inner.saved_objects, "default__mycol")
            .await
            .unwrap();
        assert!(
            loaded.enabled,
            "the second save must overwrite the first, not create a duplicate"
        );
    }
}
