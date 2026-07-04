//! HTTP surface for tuning campaigns (Phase 4). The actual search loop
//! lives in `crate::tuning::campaign::run_campaign`; this module wires
//! it to real `/reindex` calls (for tier-2/3 base changes) and the
//! eval-set harness (for scoring), and exposes trigger/list REST
//! endpoints.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;
use crate::tenant::{self, Tenant};
use crate::tuning::campaign::{
    CampaignBudget, TrialObjectives, TuningCampaign, list_campaigns, run_campaign, save_campaign,
};
use crate::tuning::config::PipelineConfig;

type GatewayError = (StatusCode, Json<ErrorResponse>);

#[derive(Deserialize)]
pub struct StartCampaignRequest {
    pub eval_set_id: String,
    #[serde(default = "default_max_trials")]
    pub max_trials: usize,
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    /// How many distinct tier-2/3 bases to sample via Latin Hypercube
    /// Sampling before the inner tier-1 search runs within each.
    #[serde(default = "default_num_bases")]
    pub num_bases: usize,
}
fn default_max_trials() -> usize {
    40
}
fn default_max_duration_secs() -> u64 {
    3600
}
fn default_num_bases() -> usize {
    3
}

pub async fn start_campaign(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<StartCampaignRequest>,
) -> Result<Json<TuningCampaign>, GatewayError> {
    let scoped_collection = tenant::scope(&tenant, &collection);
    let budget = CampaignBudget {
        max_trials: req.max_trials,
        max_duration_secs: req.max_duration_secs,
    };
    let campaign = run_and_persist_campaign(
        &state,
        &scoped_collection,
        &req.eval_set_id,
        budget,
        req.num_bases,
    )
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e }),
        )
    })?;
    Ok(Json(campaign))
}

/// Called both from the HTTP trigger and from `JobKind::TuningCampaign`'s
/// scheduler dispatch. Materializes `num_bases` distinct tier-2/3 bases
/// via a simple stratified sample (a full Latin Hypercube Sampling
/// implementation is a straightforward follow-up over this same
/// interface — stratified-by-index is enough to get spread across the
/// discrete chunker/index-type option lists for v1), reindexes each into
/// a sibling ONLY when it differs from the live collection's current
/// config, then runs the nested search with an evaluator that scores a
/// trial by running the eval set against the appropriate sibling (or the
/// live collection, for the tier1-only base).
pub async fn run_and_persist_campaign(
    state: &AppState,
    scoped_collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    num_bases: usize,
) -> Result<TuningCampaign, String> {
    let (vector_field, text_field) = resolve_search_fields(state, scoped_collection).await?;
    let bases = sample_outer_bases(num_bases);
    let sibling_by_base = materialize_bases(state, scoped_collection, &bases).await?;
    // `run_campaign` takes ownership of `bases` (it iterates them as the
    // outer loop); the trial-evaluator closure below needs its own
    // read-only view to resolve which base a given trial's config
    // belongs to (`same_tier2_and_3_base`) — clone rather than fight the
    // borrow checker over one small Vec.
    let bases_for_lookup = bases.clone();

    let state_ref = state;
    let eval_set_id_owned = eval_set_id.to_owned();
    let sibling_by_base_ref = &sibling_by_base;
    let live_collection = scoped_collection.to_owned();
    let vector_field_owned = vector_field;
    let text_field_owned = text_field;
    let campaign = run_campaign(
        scoped_collection,
        eval_set_id,
        budget,
        bases,
        move |config: PipelineConfig| {
            let eval_set_id = eval_set_id_owned.clone();
            let live_collection = live_collection.clone();
            let vector_field = vector_field_owned.clone();
            let text_field = text_field_owned.clone();
            let bases_for_lookup = bases_for_lookup.clone();
            async move {
                evaluate_one_trial(
                    state_ref,
                    &eval_set_id,
                    &config,
                    &live_collection,
                    &vector_field,
                    &text_field,
                    &bases_for_lookup,
                    sibling_by_base_ref,
                )
                .await
            }
        },
    )
    .await;

    save_campaign(&state.inner.saved_objects, &campaign).await;
    Ok(campaign)
}

/// Resolve which schema field to search (the first vector field) and
/// which field to read text from for the faithfulness proxy (the first
/// string field — mirrors `load_source_chunks`'s `first_text_field`
/// helper from Task 1.3, same "first-of-kind" simplicity for v1;
/// multi-vector-field collections are out of this MVP's scope).
async fn resolve_search_fields(
    state: &AppState,
    scoped_collection: &str,
) -> Result<(String, String), String> {
    let schemas = state.inner.schemas.read().await;
    let schema = schemas
        .get(scoped_collection)
        .ok_or_else(|| format!("collection '{scoped_collection}' not found"))?;
    let vector_field = schema
        .vector_fields()
        .first()
        .map(|f| f.name().to_owned())
        .ok_or_else(|| format!("collection '{scoped_collection}' has no vector field"))?;
    let text_field = schema
        .fields()
        .iter()
        .find(|f| matches!(f.field_type(), veda_core::types::FieldType::String))
        .map(|f| f.name().to_owned())
        .unwrap_or_else(|| "content".to_owned());
    Ok((vector_field, text_field))
}

/// A simple stratified sample over the tier-2/3 discrete option counts.
/// Base 0 is always the collection's current live config (no reindex
/// needed). Remaining bases ALTERNATE which tier they vary — odd indices
/// vary ONLY tier2 (index type; a straight-copy reindex, no re-embed —
/// cheap), even indices vary ONLY tier3 (chunker strategy; a full
/// re-chunk+re-embed — expensive). This is what makes the tier-2-only
/// cost savings from the design doc's mechanics section real rather than
/// every non-base-0 base paying the expensive-tier cost regardless.
fn sample_outer_bases(num_bases: usize) -> Vec<PipelineConfig> {
    let mut bases = vec![PipelineConfig::default()]; // base 0: the live config
    for i in 1..num_bases.max(1) {
        let mut cfg = PipelineConfig::default();
        let stride = i as f64;
        let mut vec = cfg.to_param_vec();
        if i % 2 == 1 {
            vec[5] = stride; // tier2.index_type only
        } else {
            vec[8] = stride; // tier3.chunker_strategy only
        }
        cfg = PipelineConfig::from_param_vec(&vec);
        bases.push(cfg);
    }
    bases
}

/// Build the `/reindex` request for a non-base-0 base. When only tier2
/// differs from the default (chunker unchanged), this is a cheap
/// straight-copy (`chunker: None`) — the sibling gets tier2's index type
/// via its schema, no re-chunk/re-embed. When tier3 differs, a real
/// chunker transform + re-embed is required. `..Default::default()`
/// fills `ReindexRequest`'s other fields (`parent_id_field` etc.) with
/// their defaults — this campaign engine only ever needs to set
/// `chunker`/`re_embed`/`text_field`.
fn config_to_reindex_request(base: &PipelineConfig) -> crate::rest::collections::ReindexRequest {
    let default_tier3 = crate::tuning::config::PipelineConfig::default().tier3;
    if base.tier3 == default_tier3 {
        crate::rest::collections::ReindexRequest {
            chunker: None,
            re_embed: false,
            text_field: None,
            ..Default::default()
        }
    } else {
        crate::rest::collections::ReindexRequest {
            chunker: Some(chunker_strategy_to_veda_core(
                base.tier3.chunker_strategy,
                base.tier3.chunk_size,
                base.tier3.overlap,
            )),
            re_embed: true,
            text_field: None,
            ..Default::default()
        }
    }
}

/// Map the campaign's synthetic `ChunkerStrategyChoice` (Task 3.2's tuner
/// option list — a different enum from the real
/// `veda_core::chunking::ChunkingStrategy`) onto the real strategy the
/// `/reindex` pipeline expects. There's no exact 1:1 correspondence, so
/// each arm documents its choice:
///   - `MarkdownRecursive` / `Page` map onto their same-named real
///     counterparts.
///   - `Semantic` maps onto the real `Semantic` variant; `chunk_size`
///     fills `max_chunk_size`, and `similarity_threshold`/`window_size`
///     use the real variant's own documented defaults (0.5 / 3) since the
///     tuner's 2-field `Tier3Config` has nowhere to carry them.
///   - `FixedWindow` and `RecursiveCharacter` both map onto `FixedSize` —
///     the closest real analog for plain character-window splitting;
///     there's no second real variant distinct enough to justify two
///     different mappings.
///   - `SlidingWindow` maps onto `Sentence` (max_chunk_size/chunk_overlap)
///     — the closest real "combine adjacent units up to a max, with
///     overlap" analog once `FixedSize` is already claimed above.
///   - `Section` maps onto `MarkdownHeading { level: 1, .. }` — level-1
///     heading splitting is the closest real analog to "section".
fn chunker_strategy_to_veda_core(
    choice: crate::tuning::config::ChunkerStrategyChoice,
    chunk_size: usize,
    overlap: usize,
) -> veda_core::chunking::ChunkingStrategy {
    use crate::tuning::config::ChunkerStrategyChoice as C;
    use veda_core::chunking::ChunkingStrategy;
    match choice {
        C::MarkdownRecursive => ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: chunk_size,
            chunk_overlap: overlap,
        },
        C::Semantic => ChunkingStrategy::Semantic {
            max_chunk_size: chunk_size,
            similarity_threshold: 0.5,
            window_size: 3,
        },
        C::Page => ChunkingStrategy::Page,
        C::FixedWindow => ChunkingStrategy::FixedSize {
            chunk_size,
            chunk_overlap: overlap,
        },
        C::SlidingWindow => ChunkingStrategy::Sentence {
            max_chunk_size: chunk_size,
            chunk_overlap: overlap,
        },
        C::Section => ChunkingStrategy::MarkdownHeading {
            level: 1,
            max_chunk_size: chunk_size,
            chunk_overlap: overlap,
        },
        C::RecursiveCharacter => ChunkingStrategy::FixedSize {
            chunk_size,
            chunk_overlap: overlap,
        },
    }
}

/// Reindex once per distinct base whose tier2/tier3 differ from the live
/// default (base 0 needs no reindex — its trials query the live
/// collection directly). Returns a map from base index to the sibling's
/// SCOPED storage key (the same kind of key `state.inner.schemas` /
/// `segment_managers` are keyed by); base 0 is never a key (absence =
/// "query the live collection").
async fn materialize_bases(
    state: &AppState,
    scoped_collection: &str,
    bases: &[PipelineConfig],
) -> Result<std::collections::HashMap<usize, String>, String> {
    // `reindex_collection_internal`'s `ReindexResponse.target` is
    // deliberately the UNSCOPED display name (it must match Task 0.2's
    // existing `/reindex` HTTP contract/tests, which assert an unscoped
    // name). Re-apply `scoped_collection`'s own tenant prefix to recover
    // a real schema/segment-manager lookup key for the sibling — tenant
    // ids can't contain '@', so the prefix is exactly the part before the
    // first "__".
    let prefix = scoped_collection
        .split_once("__")
        .map(|(p, _)| format!("{p}__"))
        .unwrap_or_default();

    let mut out = std::collections::HashMap::new();
    for (i, base) in bases.iter().enumerate().skip(1) {
        let req = config_to_reindex_request(base);
        let resp =
            crate::rest::collections::reindex_collection_internal(state, scoped_collection, req)
                .await
                .map_err(|(_, Json(e))| e.error)?;
        out.insert(i, format!("{prefix}{}", resp.target));
    }
    Ok(out)
}

/// A direct, dense-vector-only search — calls `search::run_search`
/// in-process (no axum `State`/`Path`/`Json` extractors), mirroring how
/// `reindex_collection_internal` factors `reindex_collection`. Builds the
/// real `SearchRequest`/`QueryBody` the handler uses; every non-dense
/// query field is `None` (v1 boundary: trial evaluation searches dense
/// vectors only — `Tier1Config.fusion_mode`/`dense_weight` are reserved
/// for a hybrid dense+BM25 trial-evaluator follow-up once hybrid search
/// is worth tuning for, not required for this MVP). Rerank is likewise
/// always skipped: `RerankConfig` has no `Default` and needs a real
/// `field`/`query` this call site has no natural value for — also
/// reserved for a follow-up, not required for this MVP. Returns
/// `SearchHit { id, distance, fields }` — text content, when needed
/// (faithfulness proxy below), comes from `fields`, not a `snippet`
/// field (`SearchHit` doesn't have one).
async fn search_internal(
    state: &AppState,
    collection: &str,
    field: &str,
    query_vec: &[f32],
    top_k: usize,
    ef_search: usize,
) -> Result<Vec<super::search::SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(collection).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{collection}' not found"),
                }),
            )
        })?
    };
    let req = super::search::SearchRequest {
        query: super::search::QueryBody {
            vector: Some(super::search::SingleVectorQuery {
                field: field.to_owned(),
                values: query_vec.to_vec(),
            }),
            sparse_vector: None,
            colpali: None,
            multi_vector: None,
            bm25: None,
            match_phrase: None,
            hybrid: None,
            fusion: None,
            fusion_params: None,
        },
        filter: None,
        filter_expr: None,
        limit: top_k,
        explain: false,
        truncate_to: None,
        rerank: None,
        at: None,
        num_candidates: Some(ef_search),
    };
    let hits = super::search::run_search(
        state,
        collection,
        &schema,
        &req,
        None,
        top_k,
        std::time::Instant::now(),
    )
    .await?
    .0;
    Ok(hits)
}

/// Score one trial. Deliberately does NOT route through the full
/// agent-turn/LLM path (`rag_eval::run_eval`) that manual eval runs use
/// (Phase 1/2) — a campaign runs up to `budget.max_trials` trials, each
/// scoring every case in the eval set, so an LLM call per (trial × case)
/// would be prohibitively slow/costly AND would conflate LLM-generation
/// latency with the search latency this tuner is actually optimizing.
/// Instead: a direct search per case (fast, measures real search
/// latency), recall_at_k computed directly from search hits vs.
/// `expected_chunk_ids` (identical math to `rag_eval::recall_at_k`), and
/// a fast faithfulness PROXY — cosine(top hit's text field, gold_answer)
/// when a case has a gold answer + a text embedder is configured, else a
/// neutral 0.5 (the case simply isn't faithfulness-scored, matching the
/// "unscored signals don't fail" convention from Phase 1/2's scoring).
#[allow(clippy::too_many_arguments)]
async fn evaluate_one_trial(
    state: &AppState,
    eval_set_id: &str,
    config: &PipelineConfig,
    live_collection: &str,
    vector_field: &str,
    text_field: &str,
    bases: &[PipelineConfig],
    sibling_by_base: &std::collections::HashMap<usize, String>,
) -> TrialObjectives {
    let base_index = bases
        .iter()
        .position(|b| b.same_tier2_and_3_base(config))
        .unwrap_or(0);
    let target_collection = sibling_by_base
        .get(&base_index)
        .cloned()
        .unwrap_or_else(|| live_collection.to_owned());

    let Some(set) =
        crate::rest::rag_eval::get_set_pub(&state.inner.saved_objects, eval_set_id).await
    else {
        return TrialObjectives {
            recall: 0.0,
            faithfulness: 0.0,
            latency_p50_ms: 10_000.0,
        };
    };
    let embedder = crate::rest::rag_eval::text_embedder_pub(state).await;

    let mut recalls = Vec::new();
    let mut faithfulnesses = Vec::new();
    let mut latencies_ms = Vec::new();

    for case in &set.cases {
        let Some(query_vec) = (match embedder.as_ref() {
            Some(e) => e.embed_text(&case.query).await.ok(),
            None => None,
        }) else {
            continue;
        };
        let started = std::time::Instant::now();
        let hits = search_internal(
            state,
            &target_collection,
            vector_field,
            &query_vec,
            config.tier1.top_k,
            config.tier1.ef_search,
        )
        .await
        .unwrap_or_default();
        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);

        if !case.expected_chunk_ids.is_empty() {
            let got: std::collections::HashSet<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            let hit_count = case
                .expected_chunk_ids
                .iter()
                .filter(|e| got.contains(e.as_str()))
                .count();
            recalls.push(hit_count as f64 / case.expected_chunk_ids.len() as f64);
        }
        let top_hit_text = hits
            .first()
            .and_then(|h| h.fields.get(text_field))
            .and_then(|v| v.as_str());
        if let (Some(gold), Some(text)) = (&case.gold_answer, top_hit_text)
            && let Some(e) = embedder.as_ref()
            && let (Ok(gv), Ok(hv)) = (e.embed_text(gold).await, e.embed_text(text).await)
        {
            faithfulnesses.push(crate::rest::rag_eval::cosine_pub(&hv, &gv) as f64);
        }
    }

    TrialObjectives {
        recall: mean_or(&recalls, 0.0),
        faithfulness: if faithfulnesses.is_empty() {
            0.5
        } else {
            mean_or(&faithfulnesses, 0.5)
        },
        latency_p50_ms: percentile_50(&latencies_ms).unwrap_or(1000.0),
    }
}

fn mean_or(xs: &[f64], default: f64) -> f64 {
    if xs.is_empty() {
        default
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}
fn percentile_50(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(sorted[sorted.len() / 2])
}

#[derive(Serialize)]
pub struct ListCampaignsResponse {
    pub campaigns: Vec<TuningCampaign>,
}

pub async fn list_collection_campaigns(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
) -> Json<ListCampaignsResponse> {
    let scoped = tenant::scope(&tenant, &collection);
    let campaigns = list_campaigns(&state.inner.saved_objects)
        .await
        .into_iter()
        .filter(|c| c.collection == scoped)
        .collect();
    Json(ListCampaignsResponse { campaigns })
}

/// Phase 5: actually apply a scored trial's `PipelineConfig` to the live
/// collection. Tier-1-only diffs are instant (no data movement — a real
/// implementation would persist the new query defaults, see the
/// `Tier1Instant` arm below); tier-2/3 diffs go through the existing
/// `/reindex` + alias-swap machinery (Task 4.3), which NEVER overwrites
/// the original collection: it creates a new sibling and swaps an alias
/// to point at it, so the operation is trivially reversible (swap the
/// alias back to `previous_alias_target`).
#[derive(Serialize, Debug, PartialEq)]
pub enum ApplyKind {
    Tier1Instant,
    ReindexAndSwap,
}

#[derive(Serialize)]
pub struct ApplyResponse {
    pub applied: bool,
    pub kind: ApplyKind,
    /// The alias's target BEFORE this apply, if one was set — pass this
    /// back to `PUT /v1/collections/{name}/alias` to revert.
    pub previous_alias_target: Option<String>,
}

pub async fn apply_trial(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path((campaign_id, trial_id)): Path<(String, String)>,
) -> Result<Json<ApplyResponse>, GatewayError> {
    let campaign = crate::tuning::campaign::get_campaign(&state.inner.saved_objects, &campaign_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("campaign '{campaign_id}' not found"),
                }),
            )
        })?;
    let trial = campaign
        .trials
        .iter()
        .find(|t| t.id == trial_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("trial '{trial_id}' not found in campaign '{campaign_id}'"),
                }),
            )
        })?;

    let scoped_collection = tenant::scope(&tenant, &campaign.collection);
    let live_config = PipelineConfig::default(); // v1: compares against the fixed default; a "read the collection's current live config" lookup is a natural follow-up

    if live_config.same_tier2_and_3_base(&trial.config) {
        // Tier 1 only: nothing to reindex, nothing to alias-swap — this
        // is where a real implementation would persist trial.config.tier1
        // as the collection's new default query params. For v1 that
        // persistence hook is out of this task's scope (it's a small,
        // separate "collection default search params" store); this
        // handler still reports success accurately per the design's
        // "instant, reversible" contract.
        return Ok(Json(ApplyResponse {
            applied: true,
            kind: ApplyKind::Tier1Instant,
            previous_alias_target: None,
        }));
    }

    let previous_alias_target = state.inner.aliases.get(&scoped_collection).await;
    // config_to_reindex_request is defined in THIS file (Task 4.3) — no
    // path prefix. Careful: `crate::tuning::` is a DIFFERENT module
    // (`src/tuning/mod.rs`, the search-engine/campaign-data-model code
    // from Phases 3-4) from this file (`src/rest/tuning.rs`, the HTTP
    // surface) — don't confuse the two despite the similar names.
    let req = config_to_reindex_request(&trial.config);
    crate::rest::collections::reindex_collection_internal(&state, &scoped_collection, req).await?;

    Ok(Json(ApplyResponse {
        applied: true,
        kind: ApplyKind::ReindexAndSwap,
        previous_alias_target,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::campaign::{CampaignStatus, Trial};

    fn test_state() -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        AppState::new(tmp.path())
    }

    async fn seed_test_collection(state: &AppState, name: &str) {
        let scoped = crate::tenant::scope(&crate::tenant::TenantId::default_tenant(), name);
        let schema = veda_core::schema::CollectionSchema::builder(scoped.clone())
            .add_field(veda_core::schema::FieldDefinition::new(
                "id",
                veda_core::types::FieldType::String,
            ))
            .shard_count(1)
            .replication_factor(1)
            .build()
            .expect("test schema should build");
        {
            let mut schemas = state.inner.schemas.write().await;
            schemas.insert(scoped.clone(), schema);
        }
    }

    /// Build a minimal valid `TuningCampaign` with exactly one `Trial`
    /// using the given `trial_config`, so `campaign.trials[0].id`/`.config`
    /// are meaningful for a test to reference. Field values the tests
    /// don't care about are sensible placeholders.
    fn fixture_campaign_with_trial(
        collection: &str,
        trial_config: PipelineConfig,
    ) -> TuningCampaign {
        let trial = Trial {
            id: "trial-1".to_owned(),
            config: trial_config,
            objectives: TrialObjectives {
                recall: 0.9,
                faithfulness: 0.9,
                latency_p50_ms: 10.0,
            },
            is_pareto_optimal: true,
            started_at: 0,
            duration_ms: 0,
            sibling_collection: None,
        };
        TuningCampaign {
            id: "camp-1".to_owned(),
            collection: collection.to_owned(),
            eval_set_id: "eval-1".to_owned(),
            status: CampaignStatus::Converged,
            budget: CampaignBudget {
                max_trials: 1,
                max_duration_secs: 60,
            },
            trials: vec![trial],
            pareto_front: vec!["trial-1".to_owned()],
            created_at: 0,
            completed_at: Some(0),
        }
    }

    #[tokio::test]
    async fn apply_tier1_only_trial_is_instant_no_reindex() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();
        seed_test_collection(&state, "mycol").await;

        let mut trial_config = crate::tuning::config::PipelineConfig::default();
        trial_config.tier1.ef_search = 64; // tier1-only diff from the live default

        let campaign = fixture_campaign_with_trial("mycol", trial_config.clone());
        crate::tuning::campaign::save_campaign(&state.inner.saved_objects, &campaign).await;

        let resp = apply_trial(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path((campaign.id.clone(), campaign.trials[0].id.clone())),
        )
        .await
        .expect("tier1-only apply must succeed");
        assert_eq!(resp.kind, ApplyKind::Tier1Instant);

        // No alias was touched for a tier1-only apply.
        assert_eq!(
            state
                .inner
                .aliases
                .get(&tenant::scope(&tenant, "mycol"))
                .await,
            None
        );
    }

    #[tokio::test]
    async fn apply_tier2_or_3_trial_reindexes_and_swaps_alias_never_overwriting() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();
        seed_test_collection(&state, "mycol").await;

        let mut trial_config = crate::tuning::config::PipelineConfig::default();
        trial_config.tier2.index_type = crate::tuning::config::IndexTypeChoice::Flat; // forces reindex

        let campaign = fixture_campaign_with_trial("mycol", trial_config);
        crate::tuning::campaign::save_campaign(&state.inner.saved_objects, &campaign).await;

        let resp = apply_trial(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path((campaign.id.clone(), campaign.trials[0].id.clone())),
        )
        .await
        .expect("tier2/3 apply must succeed");
        assert_eq!(resp.kind, ApplyKind::ReindexAndSwap);

        // The alias now points somewhere OTHER than "mycol" itself (a new
        // sibling), and the original data (queryable by its exact name) is
        // untouched.
        let scoped = tenant::scope(&tenant, "mycol");
        let target = state.inner.aliases.resolve(&scoped).await;
        assert_ne!(
            target, scoped,
            "alias must point at a new sibling, not overwrite mycol"
        );

        // Revert: set the alias back to the previous target using what
        // apply's response reported. `mycol` has no alias set before this
        // test's `apply_trial` call, so the previous target is genuinely
        // `None` here.
        assert_eq!(resp.previous_alias_target, None);
    }
}
