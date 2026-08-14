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
) -> Result<Json<AutoOptimizationConfig>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = resolve_config_key(&state, &tenant, &collection).await?;
    let cfg = get_auto_optimization_config(&state.inner.saved_objects, &scoped)
        .await
        .unwrap_or(AutoOptimizationConfig {
            collection: collection.clone(),
            enabled: false,
            target_questions: None,
            mode: TuningMode::default(),
        });
    // Always answer with the name the caller used, not the alias target.
    Ok(Json(AutoOptimizationConfig { collection, ..cfg }))
}

/// Resolve the storage key for a collection's auto-optimization config.
///
/// The config must be keyed by the *resolved* collection, because that is
/// the name `ingest_markdown` hands to `maybe_trigger_auto_pipeline` when it
/// looks the config up after an ingestion. Keying it by the caller's name
/// instead meant an aliased collection — exactly what applying a tier-2/3
/// tuning trial produces — never fired its own auto-optimization: the config
/// sat under the pre-swap name and the lookup never found it.
///
/// A missing collection is a 404, matching every other collection-scoped
/// endpoint. `GET` used to answer 200 with a synthesized default for any
/// name at all, so a typo looked like a real, un-opted-in collection while
/// `PUT` on that same name correctly refused.
async fn resolve_config_key(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    collection: &str,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(tenant, collection);
    let scoped = state.inner.aliases.resolve(&scoped).await;
    let schemas = state.inner.schemas.read().await;
    if !schemas.contains_key(&scoped) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{collection}' not found"),
            }),
        ));
    }
    Ok(scoped)
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
    let scoped = resolve_config_key(&state, &tenant, &collection).await?;
    {
        let schemas = state.inner.schemas.read().await;
        // `resolve_config_key` already confirmed the collection exists;
        // re-fetch here (not folded into that helper) because `GET` shares
        // it and must keep reading an external collection's (always
        // disabled) config. Auto-optimization can never run against a
        // collection with no local segments to reindex, so storing the
        // setting would be a knob that provably never applies.
        if let Some(schema) = schemas.get(&scoped) {
            super::collections::reject_if_external(schema, &collection)?;
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

/// Called from `ingest_common::ingest_markdown`'s tail after every
/// successful document ingestion. Fire-and-forget: the caller spawns
/// this via `tokio::spawn` and never awaits it, so it can never block or
/// fail the ingest HTTP response. Best-effort throughout — any failure
/// (disabled, no LLM, generation produced zero cases, campaign failed)
/// is logged and this simply returns.
pub async fn maybe_trigger_auto_pipeline(
    state: AppState,
    tenant: crate::tenant::TenantId,
    scoped_collection: String,
) {
    let Some(cfg) =
        get_auto_optimization_config(&state.inner.saved_objects, &scoped_collection).await
    else {
        return;
    };
    if !cfg.enabled {
        return;
    }
    if state.inner.answer_gen_config.read().await.is_none() {
        tracing::warn!(
            collection = %scoped_collection,
            "auto-optimization enabled but ANSWER_GEN_BASE_URL not configured; skipping"
        );
        return;
    }

    let display_name = tenant::display_name(&scoped_collection).to_owned();

    // Always regenerate fresh, sized to the NOW-grown corpus (never
    // incrementally append — diversity/difficulty balance has to be
    // recomputed against the whole corpus each time).
    let target_questions = match cfg.target_questions {
        Some(n) => n,
        None => match super::rag_eval::load_source_chunks(&state, &scoped_collection).await {
            Ok(chunks) => {
                let groups = super::rag_eval::document_groups(&chunks);
                (groups.len() * 2).max(4).min(chunks.len().max(1))
            }
            Err(_) => return,
        },
    };

    let req = super::rag_eval::GenerateEvalSetRequest {
        name: Some(format!("auto-eval-{display_name}")),
        num_cases: target_questions,
    };
    // A DETERMINISTIC id (not the usual fresh `eval-{uuid}`) — this is
    // what makes "re-trigger every time it grows" a genuine regeneration
    // rather than an ever-accumulating pile of orphaned auto-generated
    // sets from every past ingestion. `SavedObjectStore.put` upserts by
    // id, so saving under this same id each time overwrites the prior
    // auto-generated set for this collection.
    let preferred_id = format!("auto-eval-{display_name}");
    let generated = match super::rag_eval::generate_eval_set_for_collection(
        &state,
        &tenant,
        &display_name,
        req,
        Some(preferred_id),
    )
    .await
    {
        Ok(set) => set,
        Err((status, err)) => {
            tracing::warn!(
                collection = %scoped_collection,
                status = %status,
                error = %err.0.error,
                "auto-optimization: eval-set generation failed; skipping campaign"
            );
            return;
        }
    };
    if generated.cases.is_empty() {
        tracing::warn!(collection = %scoped_collection, "auto-optimization: generated eval set has 0 cases; skipping campaign");
        return;
    }

    let scoped_eval_set_id = tenant::scope(&tenant, &generated.id);
    let budget = crate::tuning::campaign::CampaignBudget {
        max_trials: 40,
        max_duration_secs: 3600,
    };
    let campaign = crate::rest::tuning::run_and_persist_campaign_with_mode(
        &state,
        &scoped_collection,
        &scoped_eval_set_id,
        budget,
        3,
        cfg.mode.clone(),
    )
    .await;
    match campaign {
        Ok(c) => tracing::info!(
            collection = %scoped_collection,
            campaign_id = %c.id,
            trial_count = c.trials.len(),
            "auto-optimization: campaign completed"
        ),
        Err(e) => {
            tracing::warn!(collection = %scoped_collection, error = %e, "auto-optimization: campaign failed")
        }
    }
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

    async fn seed_collection_with_records(state: &AppState, name: &str, n: usize) -> String {
        use crate::tenant::TenantId;
        let tenant = TenantId::default_tenant();
        let scoped = tenant::scope(&tenant, name);
        let schema = veda_core::schema::CollectionSchema::builder(scoped.clone())
            .add_field(veda_core::schema::FieldDefinition::new(
                "content",
                veda_core::types::FieldType::String,
            ))
            .shard_count(1)
            .replication_factor(1)
            .build()
            .unwrap();
        {
            let mut schemas = state.inner.schemas.write().await;
            schemas.insert(scoped.clone(), schema.clone());
        }
        let mut managers = std::collections::HashMap::new();
        let tmp = tempfile::tempdir().unwrap();
        let mut sm = veda_storage::segment_manager::SegmentManager::new(
            schema,
            0,
            tmp.path(),
            512 * 1024 * 1024,
        )
        .unwrap();
        for i in 0..n {
            let mut record = veda_core::record::Record::new();
            record.set(
                "content",
                veda_core::record::FieldValue::String(format!("chunk number {i}")),
            );
            sm.insert(record).unwrap();
        }
        managers.insert(0u32, sm);
        state
            .inner
            .segment_managers
            .write()
            .await
            .insert(scoped.clone(), managers);
        scoped
    }

    #[tokio::test]
    async fn maybe_trigger_auto_pipeline_is_a_noop_when_disabled() {
        let state = test_state();
        let scoped = seed_collection_with_records(&state, "trig_off", 3).await;
        let tenant = crate::tenant::TenantId::default_tenant();

        // No auto-optimization config saved at all -> disabled by default.
        maybe_trigger_auto_pipeline(state.clone(), tenant, scoped.clone()).await;

        let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
        assert!(
            sets.is_empty(),
            "disabled auto-optimization must not generate any eval set"
        );
    }

    #[tokio::test]
    async fn maybe_trigger_auto_pipeline_is_a_noop_without_an_answer_gen_endpoint() {
        let state = test_state();
        let scoped = seed_collection_with_records(&state, "trig_no_llm", 3).await;
        let tenant = crate::tenant::TenantId::default_tenant();
        save_auto_optimization_config(
            &state.inner.saved_objects,
            &AutoOptimizationConfig {
                collection: scoped.clone(),
                enabled: true,
                target_questions: None,
                mode: TuningMode::Quality,
            },
        )
        .await;

        // Enabled, but no ANSWER_GEN_* configured on this test AppState -> still a no-op.
        maybe_trigger_auto_pipeline(state.clone(), tenant, scoped.clone()).await;

        let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
        assert!(
            sets.is_empty(),
            "enabled-but-no-LLM must not fail loudly nor generate anything"
        );
    }
}
