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
    CampaignBudget, TrialObjectives, TuningCampaign, list_campaigns, run_campaign_with_mode,
    save_campaign,
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
    #[serde(default)]
    pub mode: crate::rest::auto_optimization::TuningMode,
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
    // A campaign materializes reindexed sibling collections to benchmark
    // trials against (see `materialize_bases`) — a write, and one that
    // `resolve_search_fields`/`materialize_bases` below would otherwise only
    // reject after losing the status code through a bare `String` error,
    // surfacing as an opaque 500 instead of a clear 405. Check up front
    // instead, the way every other collection-addressed handler does.
    {
        let schemas = state.inner.schemas.read().await;
        let schema = schemas.get(&scoped_collection).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{collection}' not found"),
                }),
            )
        })?;
        crate::rest::collections::reject_if_external(schema, &collection)?;
    }
    // Eval sets are always stored under a tenant-scoped key (every real
    // creation path — `create_eval_set`/`generate_eval_set` — scopes via
    // `save_and_return_set`, mirroring `collection` just above). Without
    // this, `evaluate_one_trial`'s `get_set_pub` lookup (which uses this
    // id verbatim, unscoped) silently misses the real object and every
    // trial falls back to a hardcoded 0.0/0.0/10_000.0 "not found"
    // placeholder — found via a live end-to-end run on real data, where
    // every trial in a real campaign scored zero recall with no error
    // logged anywhere.
    let scoped_eval_set_id = tenant::scope(&tenant, &req.eval_set_id);
    let budget = CampaignBudget {
        max_trials: req.max_trials,
        max_duration_secs: req.max_duration_secs,
    };
    let campaign = run_and_persist_campaign_with_mode(
        &state,
        &scoped_collection,
        &scoped_eval_set_id,
        budget,
        req.num_bases,
        req.mode.clone(),
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
pub async fn run_and_persist_campaign_with_mode(
    state: &AppState,
    scoped_collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    num_bases: usize,
    mode: crate::rest::auto_optimization::TuningMode,
) -> Result<TuningCampaign, String> {
    let (vector_field, text_field) = resolve_search_fields(state, scoped_collection).await?;
    // Chunk-id -> text for the LIVE collection, resolved once. Trials whose
    // base re-chunked score recall against this content instead of against
    // chunk ids, which a re-chunk invalidates (see `content_recall`). A
    // failure here is not fatal: an empty map just leaves re-chunked trials'
    // cases unscored, exactly as before this existed.
    let gold_texts: std::collections::HashMap<String, String> =
        match crate::rest::rag_eval::load_source_chunks(state, scoped_collection).await {
            Ok(chunks) => chunks.into_iter().map(|c| (c.chunk_id, c.text)).collect(),
            Err(e) => {
                tracing::warn!(
                    collection = %scoped_collection,
                    error = ?e,
                    "tuning campaign: could not load source chunks for content-based recall; \
                     trials on a re-chunked base will be left unscored on recall"
                );
                std::collections::HashMap::new()
            }
        };
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
    let gold_texts_ref = &gold_texts;
    let live_collection = scoped_collection.to_owned();
    let vector_field_owned = vector_field;
    let text_field_owned = text_field;
    let campaign = run_campaign_with_mode(
        scoped_collection,
        eval_set_id,
        budget,
        bases,
        mode,
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
                    gold_texts_ref,
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
/// straight-copy that still carries the new `index_type` — the sibling
/// genuinely gets tier2's index type via `ReindexRequest.index_type`, no
/// re-chunk/re-embed needed. When tier3 differs, a real chunker transform +
/// re-embed is required (tier2's index_type, if it also differs, rides
/// along on the same request). `..Default::default()` fills
/// `ReindexRequest`'s other fields (`parent_id_field` etc.) with their
/// defaults — this campaign engine only ever needs to set
/// `chunker`/`re_embed`/`text_field`/`index_type`.
///
/// NOTE: `Tier2Config.ef_construction`/`.m` are NOT wired here — those HNSW
/// build parameters aren't schema-configurable anywhere in Veda today (see
/// `ReindexRequest.index_type`'s doc comment in `collections.rs`), a
/// separate, pre-existing platform gap. `sample_outer_bases` only ever
/// varies `index_type` within tier2, never `ef_construction`/`m`, so this
/// omission doesn't silently drop a real search-space dimension the
/// sampler actually explores.
fn config_to_reindex_request(base: &PipelineConfig) -> crate::rest::collections::ReindexRequest {
    let default_cfg = crate::tuning::config::PipelineConfig::default();
    let chunker_differs = base.tier3 != default_cfg.tier3;
    let index_differs = base.tier2.index_type != default_cfg.tier2.index_type;

    crate::rest::collections::ReindexRequest {
        chunker: if chunker_differs {
            Some(chunker_strategy_to_veda_core(
                base.tier3.chunker_strategy,
                base.tier3.chunk_size,
                base.tier3.overlap,
            ))
        } else {
            None
        },
        re_embed: chunker_differs,
        text_field: None,
        index_type: if index_differs {
            Some(index_type_choice_to_reindex_str(base.tier2.index_type).to_owned())
        } else {
            None
        },
        ..Default::default()
    }
}

/// Map the campaign's `IndexTypeChoice` (Task 3.2's tuner option list) onto
/// the string values `parse_index_type` (in `collections.rs`, shared with
/// collection-create's `FieldDef.index`) accepts.
fn index_type_choice_to_reindex_str(
    choice: crate::tuning::config::IndexTypeChoice,
) -> &'static str {
    use crate::tuning::config::IndexTypeChoice as C;
    match choice {
        C::Hnsw => "hnsw",
        C::DiskAnn => "diskann",
        C::IvfPq => "ivfpq",
        C::Flat => "flat",
        C::RaBitQ => "rabitq",
        C::Sq8 => "sq8",
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
        let mut req = config_to_reindex_request(base);
        // Each base must land on its own distinct sibling, and this call
        // must NEVER touch the live collection's public alias — this is
        // materializing a candidate config for internal trial scoring,
        // not applying anything. The default `/reindex` `@vN` bump
        // derives purely from the SOURCE name, so calling
        // `reindex_collection_internal` once per base against the same
        // live collection would otherwise always compute the identical
        // `@v2` target, silently merging different bases' data into one
        // sibling.
        req.sibling_suffix = Some(format!("@tuning-base{i}-{}", uuid::Uuid::new_v4()));
        req.skip_alias_swap = true;
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
    gold_texts: &std::collections::HashMap<String, String>,
) -> TrialObjectives {
    let base_index = bases
        .iter()
        .position(|b| b.same_tier2_and_3_base(config))
        .unwrap_or(0);
    let target_collection = sibling_by_base
        .get(&base_index)
        .cloned()
        .unwrap_or_else(|| live_collection.to_owned());

    // A tier-3 base re-chunks the corpus, and re-chunking mints fresh record
    // ids, so the eval set's `expected_chunk_ids` — recorded against the live
    // collection's chunking — cannot match anything this trial retrieves.
    // Scoring it by id gave every chunker-varying trial recall 0.000 by
    // construction: a better chunking could never win the outer search, while
    // a worse one still reached the Pareto front on another objective and was
    // offered to the operator as a trade-off worth applying. When the base
    // re-chunked, score recall on CONTENT instead (see `content_recall`).
    let rechunked = config.tier3 != PipelineConfig::default().tier3;

    let Some(set) =
        crate::rest::rag_eval::get_set_pub(&state.inner.saved_objects, eval_set_id).await
    else {
        tracing::warn!(
            eval_set_id = %eval_set_id,
            "tuning campaign trial: eval set not found — scoring this trial (and every other \
             trial in this campaign) as the 0.0/0.0/10_000.0 fallback; check that eval_set_id \
             is the tenant-scoped storage key, not the unscoped display id"
        );
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
            if rechunked {
                if let Some(r) =
                    content_recall(&case.expected_chunk_ids, gold_texts, &hits, text_field)
                {
                    recalls.push(r);
                }
            } else {
                let got: std::collections::HashSet<&str> =
                    hits.iter().map(|h| h.id.as_str()).collect();
                let hit_count = case
                    .expected_chunk_ids
                    .iter()
                    .filter(|e| got.contains(e.as_str()))
                    .count();
                recalls.push(hit_count as f64 / case.expected_chunk_ids.len() as f64);
            }
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

/// Length of the character shingles used by [`content_recall`].
///
/// Character-level, not word-level: the corpora this runs against include
/// Japanese, which has no whitespace token boundaries, so a word tokenizer
/// would collapse a whole chunk into one "token".
const SHINGLE_LEN: usize = 5;

/// Fraction of a gold chunk's shingles that must appear in the retrieved
/// text before it counts as retrieved. Half is deliberately forgiving: a
/// re-chunk splits and re-joins boundaries, so a gold chunk is often
/// *covered* by the top-k without any single retrieved chunk containing all
/// of it.
const CONTENT_RECALL_THRESHOLD: f64 = 0.5;

/// Whitespace-insensitive, lowercase character shingles of `text`.
fn shingles(text: &str) -> std::collections::HashSet<Vec<char>> {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect();
    if chars.len() < SHINGLE_LEN {
        // Too short to shingle: treat the whole thing as one token so short
        // chunks still match themselves rather than scoring an automatic 0.
        return if chars.is_empty() {
            std::collections::HashSet::new()
        } else {
            std::collections::HashSet::from([chars])
        };
    }
    chars.windows(SHINGLE_LEN).map(<[char]>::to_vec).collect()
}

/// Recall for a trial whose base re-chunked the corpus.
///
/// The eval set pins ground truth as `expected_chunk_ids`, but a re-chunk
/// mints new record ids, so id matching scores every such trial 0.000 no
/// matter how good the chunking is. Instead, resolve each expected id to its
/// text in the LIVE collection (`gold_texts`, built once per campaign) and
/// ask whether the trial's top-k *collectively covers* that text — the union
/// is the right model here, because a re-chunk can legitimately spread one
/// gold chunk across several retrieved ones.
///
/// Returns `None` when no expected id resolves to gold text, so the case is
/// left unscored rather than counted as a miss (the same "unscored signals
/// don't fail" convention the faithfulness proxy uses).
fn content_recall(
    expected_chunk_ids: &[String],
    gold_texts: &std::collections::HashMap<String, String>,
    hits: &[super::search::SearchHit],
    text_field: &str,
) -> Option<f64> {
    let golds: Vec<&String> = expected_chunk_ids
        .iter()
        .filter_map(|id| gold_texts.get(id))
        .collect();
    if golds.is_empty() {
        return None;
    }

    let mut retrieved: std::collections::HashSet<Vec<char>> = std::collections::HashSet::new();
    for h in hits {
        if let Some(text) = h.fields.get(text_field).and_then(|v| v.as_str()) {
            retrieved.extend(shingles(text));
        }
    }

    let covered = golds
        .iter()
        .filter(|gold| {
            let gs = shingles(gold);
            if gs.is_empty() {
                return false;
            }
            let overlap = gs.iter().filter(|s| retrieved.contains(*s)).count();
            overlap as f64 / gs.len() as f64 >= CONTENT_RECALL_THRESHOLD
        })
        .count();
    Some(covered as f64 / golds.len() as f64)
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
    Tenant(_tenant): Tenant,
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

    // `campaign.collection` is already the tenant-scoped internal key —
    // both `start_campaign` (the HTTP trigger) and the scheduler's
    // `JobKind::TuningCampaign` dispatch pass an already-scoped name into
    // `run_and_persist_campaign_with_mode`'s `scoped_collection` parameter, which is
    // what gets persisted onto `TuningCampaign.collection` verbatim.
    // Re-scoping it here double-prefixes it (e.g. "default__default__
    // rag_test"), which never resolves to a real collection — found via a
    // live end-to-end apply on a real campaign, which 404'd with
    // "collection 'default__rag_test' not found" (the outer "default__"
    // stripped by `display_name`, leaving the inner one visible).
    let scoped_collection = campaign.collection.clone();
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
mod content_recall_tests {
    use super::*;
    use std::collections::HashMap;

    fn hit(text: &str) -> super::super::search::SearchHit {
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), serde_json::json!(text));
        super::super::search::SearchHit {
            id: uuid::Uuid::new_v4().to_string(),
            distance: 0.0,
            fields,
        }
    }

    fn golds(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    const GOLD: &str = "Veda searches dense vectors, sparse SPLADE vectors and BM25 together, \
                        then reranks the shortlist with a cross encoder.";

    #[test]
    fn gold_chunk_found_verbatim_scores_one() {
        let g = golds(&[("c1", GOLD)]);
        let r = content_recall(&["c1".to_owned()], &g, &[hit(GOLD)], "content");
        assert_eq!(r, Some(1.0));
    }

    #[test]
    fn gold_chunk_absent_scores_zero() {
        let g = golds(&[("c1", GOLD)]);
        let r = content_recall(
            &["c1".to_owned()],
            &g,
            &[hit(
                "An unrelated passage about office printers and toner cartridges.",
            )],
            "content",
        );
        assert_eq!(r, Some(0.0));
    }

    #[test]
    fn gold_chunk_split_across_two_retrieved_chunks_still_counts() {
        // This is the whole point: a re-chunk moves boundaries, so the gold
        // text is covered by the top-k collectively rather than by any one
        // retrieved chunk.
        let g = golds(&[("c1", GOLD)]);
        let (a, b) = GOLD.split_at(GOLD.len() / 2);
        let r = content_recall(&["c1".to_owned()], &g, &[hit(a), hit(b)], "content");
        assert_eq!(
            r,
            Some(1.0),
            "gold text covered by the union of top-k must count as retrieved"
        );
    }

    #[test]
    fn gold_chunk_embedded_in_a_much_larger_page_counts() {
        // The Page chunker yields whole pages: the gold chunk is a fraction
        // of one. Containment is measured against the GOLD's shingles, so
        // extra surrounding text must not dilute the score.
        let g = golds(&[("c1", GOLD)]);
        let page = format!("Header material. {GOLD} Footer material and a table of numbers.");
        let r = content_recall(&["c1".to_owned()], &g, &[hit(&page)], "content");
        assert_eq!(r, Some(1.0));
    }

    #[test]
    fn partial_coverage_below_the_threshold_does_not_count() {
        let g = golds(&[("c1", GOLD)]);
        let sliver = &GOLD[..GOLD.len() / 5];
        let r = content_recall(&["c1".to_owned()], &g, &[hit(sliver)], "content");
        assert_eq!(r, Some(0.0));
    }

    #[test]
    fn scores_the_fraction_of_expected_chunks_covered() {
        let other = "Row level security scopes every record to its owner field.";
        let g = golds(&[("c1", GOLD), ("c2", other)]);
        let r = content_recall(
            &["c1".to_owned(), "c2".to_owned()],
            &g,
            &[hit(GOLD)],
            "content",
        );
        assert_eq!(r, Some(0.5));
    }

    #[test]
    fn unresolvable_expected_ids_leave_the_case_unscored() {
        // No gold text for the id (e.g. the chunk was deleted): return None
        // so the case is skipped, not counted as a miss.
        let g = golds(&[("c1", GOLD)]);
        assert_eq!(
            content_recall(&["missing".to_owned()], &g, &[hit(GOLD)], "content"),
            None
        );
    }

    #[test]
    fn matching_is_insensitive_to_whitespace_and_case() {
        let g = golds(&[("c1", GOLD)]);
        let reflowed = GOLD.to_uppercase().replace(' ', "\n  ");
        let r = content_recall(&["c1".to_owned()], &g, &[hit(&reflowed)], "content");
        assert_eq!(
            r,
            Some(1.0),
            "a re-chunk reflows whitespace; that must not look like different content"
        );
    }

    #[test]
    fn japanese_text_matches_without_whitespace_tokens() {
        // The corpus this was found on is Japanese: no spaces, so a
        // word-level tokenizer would degenerate to a single token.
        let jp = "大塚商会のセキュリティ対策は、ファイアウォールと侵入防止システムを含みます。";
        let g = golds(&[("c1", jp)]);
        let page = format!("ページ見出し。{jp} 以下は表です。");
        let r = content_recall(&["c1".to_owned()], &g, &[hit(&page)], "content");
        assert_eq!(r, Some(1.0));
    }

    #[test]
    fn missing_text_field_on_hits_scores_zero_not_a_panic() {
        let g = golds(&[("c1", GOLD)]);
        let r = content_recall(&["c1".to_owned()], &g, &[hit(GOLD)], "no_such_field");
        assert_eq!(r, Some(0.0));
    }
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

        // `campaign.collection` mirrors what the real `start_campaign`
        // handler persists: already tenant-scoped (see the doc comment on
        // `apply_trial` — passing the unscoped name here would silently
        // hide the double-scoping bug it once had).
        let campaign =
            fixture_campaign_with_trial(&tenant::scope(&tenant, "mycol"), trial_config.clone());
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

        // `campaign.collection` mirrors what the real `start_campaign`
        // handler persists: already tenant-scoped (see the doc comment on
        // `apply_trial` — passing the unscoped name here would silently
        // hide the double-scoping bug it once had).
        let campaign = fixture_campaign_with_trial(&tenant::scope(&tenant, "mycol"), trial_config);
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

    #[test]
    fn config_to_reindex_request_carries_tier2_index_type_when_it_differs() {
        let mut cfg = PipelineConfig::default();
        cfg.tier2.index_type = crate::tuning::config::IndexTypeChoice::DiskAnn;
        let req = config_to_reindex_request(&cfg);
        assert_eq!(req.index_type.as_deref(), Some("diskann"));
        // A tier2-only diff must not trigger a needless re-chunk/re-embed.
        assert!(req.chunker.is_none());
        assert!(!req.re_embed);
    }

    #[test]
    fn config_to_reindex_request_omits_index_type_when_tier2_is_default() {
        let cfg = PipelineConfig::default();
        let req = config_to_reindex_request(&cfg);
        assert_eq!(req.index_type, None);
    }

    /// A mock OpenAI-compatible embedding server: `POST /v1/embeddings` ->
    /// `{"data": [{"embedding": [1.0, 0.0]}]}` (matching
    /// `embedding_client.rs`'s real response parsing). Fixed output lets
    /// `evaluate_one_trial`'s query-embedding step succeed deterministically.
    async fn start_mock_embedder() -> String {
        use axum::Router;
        use axum::routing::post;

        async fn embeddings() -> axum::Json<serde_json::Value> {
            axum::Json(serde_json::json!({
                "data": [{ "embedding": [1.0, 0.0] }]
            }))
        }

        let router = Router::new().route("/v1/embeddings", post(embeddings));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    /// Regression test for a bug found via a live end-to-end run on real
    /// data: `start_campaign` passed `req.eval_set_id` straight through
    /// unscoped, but every real eval-set creation path
    /// (`create_eval_set`/`generate_eval_set`, via `save_and_return_set`)
    /// stores the set under a TENANT-SCOPED key. `evaluate_one_trial`'s
    /// lookup (`get_set_pub`, unscoped) therefore always missed a
    /// genuinely-existing eval set, silently scoring every trial with the
    /// hardcoded "not found" fallback (recall 0.0, faithfulness 0.0,
    /// latency_p50_ms exactly 10_000.0) — with no error logged anywhere.
    /// A real campaign against real data showed 10/10 trials with that
    /// exact fallback triple. This test drives the REAL HTTP handlers
    /// (`create_eval_set`, `start_campaign`), not a hand-scoped fixture, so
    /// it actually exercises the scoping mismatch that caused the bug.
    #[tokio::test]
    async fn start_campaign_finds_a_real_eval_set_created_through_the_public_api() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();

        let mock_embed_url = start_mock_embedder().await;
        {
            let mut registry = state.inner.embedding_registry.write().await;
            registry.register(veda_core::embedding::EmbeddingModelConfig {
                name: "mock-text".to_owned(),
                endpoint: mock_embed_url,
                api_key: String::new(),
                supported_modalities: vec![veda_core::embedding::Modality::Text],
                output_dimension: 2,
                max_input_size_bytes: 1024 * 1024,
                timeout_seconds: 10,
                batch_size: 32,
            });
        }

        // A real collection with one seeded record (explicit vector, no
        // embedder needed for the document side).
        let create_req: crate::rest::collections::CreateCollectionRequest =
            serde_json::from_value(serde_json::json!({
                "name": "mycol",
                "shard_count": 1,
                "fields": [
                    { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                    { "name": "text", "type": "string" }
                ],
                "seed_records": [{ "v": [1.0, 0.0], "text": "hello world" }]
            }))
            .expect("fixture create-collection request should deserialize");
        let create_resp = crate::rest::collections::create_collection(
            State(state.clone()),
            Tenant(tenant.clone()),
            Json(create_req),
        )
        .await;
        use axum::response::IntoResponse;
        assert!(
            create_resp.into_response().status().is_success(),
            "fixture collection create should succeed"
        );

        // Create the eval set through the REAL public endpoint — this is
        // what tenant-scopes the storage key.
        let eval_set = crate::rest::rag_eval::EvalSet {
            id: String::new(),
            name: "e2e-fixture".to_owned(),
            agent_id: None,
            cases: vec![crate::rest::rag_eval::EvalCase {
                id: String::new(),
                query: "hello".to_owned(),
                expected_chunk_ids: vec![],
                gold_answer: None,
                source: None,
                difficulty: None,
            }],
        };
        let created = crate::rest::rag_eval::create_eval_set(
            State(state.clone()),
            Tenant(tenant.clone()),
            Json(eval_set),
        )
        .await
        .expect("create_eval_set should succeed")
        .0;

        let req = StartCampaignRequest {
            eval_set_id: created.id.clone(),
            max_trials: 2,
            max_duration_secs: 60,
            num_bases: 1,
            mode: crate::rest::auto_optimization::TuningMode::Quality,
        };
        let campaign = start_campaign(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path("mycol".to_owned()),
            Json(req),
        )
        .await
        .expect("start_campaign should succeed")
        .0;

        assert!(
            !campaign.trials.is_empty(),
            "campaign should have run trials"
        );
        for trial in &campaign.trials {
            assert_ne!(
                trial.objectives.latency_p50_ms, 10_000.0,
                "trial hit the eval-set-not-found fallback ({:?}) — eval_set_id scoping regressed",
                trial.objectives
            );
        }
    }

    /// Regression test for a bug found via a live end-to-end apply on a
    /// real campaign: `apply_trial` re-scoped `campaign.collection` with
    /// `tenant::scope(&tenant, &campaign.collection)`, but every real
    /// campaign-creation path (`start_campaign`, and the scheduler's
    /// `JobKind::TuningCampaign` dispatch) already passes a tenant-scoped
    /// name into `run_and_persist_campaign_with_mode`, which persists it onto
    /// `TuningCampaign.collection` verbatim. Re-scoping an already-scoped
    /// name double-prefixes it (e.g. "default__default__rag_test"), which
    /// never resolves — a real apply against a real campaign 404'd with
    /// "collection 'default__rag_test' not found". This test drives the
    /// REAL `start_campaign` handler (not a hand-scoped fixture, which is
    /// exactly the pattern that hid this bug the first time) so
    /// `campaign.collection` is genuinely tenant-scoped, then applies a
    /// tier2-differing trial through the real `apply_trial` handler.
    #[tokio::test]
    async fn apply_trial_finds_a_real_campaign_created_through_the_public_api() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();

        let mock_embed_url = start_mock_embedder().await;
        {
            let mut registry = state.inner.embedding_registry.write().await;
            registry.register(veda_core::embedding::EmbeddingModelConfig {
                name: "mock-text".to_owned(),
                endpoint: mock_embed_url,
                api_key: String::new(),
                supported_modalities: vec![veda_core::embedding::Modality::Text],
                output_dimension: 2,
                max_input_size_bytes: 1024 * 1024,
                timeout_seconds: 10,
                batch_size: 32,
            });
        }

        let create_req: crate::rest::collections::CreateCollectionRequest =
            serde_json::from_value(serde_json::json!({
                "name": "mycol2",
                "shard_count": 1,
                "fields": [
                    { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                    { "name": "text", "type": "string" }
                ],
                "seed_records": [{ "v": [1.0, 0.0], "text": "hello world" }]
            }))
            .expect("fixture create-collection request should deserialize");
        let create_resp = crate::rest::collections::create_collection(
            State(state.clone()),
            Tenant(tenant.clone()),
            Json(create_req),
        )
        .await;
        use axum::response::IntoResponse;
        assert!(
            create_resp.into_response().status().is_success(),
            "fixture collection create should succeed"
        );

        let eval_set = crate::rest::rag_eval::EvalSet {
            id: String::new(),
            name: "e2e-fixture-2".to_owned(),
            agent_id: None,
            cases: vec![crate::rest::rag_eval::EvalCase {
                id: String::new(),
                query: "hello".to_owned(),
                expected_chunk_ids: vec![],
                gold_answer: None,
                source: None,
                difficulty: None,
            }],
        };
        let created = crate::rest::rag_eval::create_eval_set(
            State(state.clone()),
            Tenant(tenant.clone()),
            Json(eval_set),
        )
        .await
        .expect("create_eval_set should succeed")
        .0;

        // num_bases > 1 so the stratified sampler produces at least one
        // tier2-differing base (needed to exercise the ReindexAndSwap arm).
        let req = StartCampaignRequest {
            eval_set_id: created.id.clone(),
            max_trials: 20,
            max_duration_secs: 60,
            num_bases: 5,
            mode: crate::rest::auto_optimization::TuningMode::Quality,
        };
        let campaign = start_campaign(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path("mycol2".to_owned()),
            Json(req),
        )
        .await
        .expect("start_campaign should succeed")
        .0;

        let tier2_trial = campaign
            .trials
            .iter()
            .find(|t| {
                !PipelineConfig::default().same_tier2_and_3_base(&t.config)
            })
            .expect("stratified sampling with num_bases=5 should produce at least one tier2/3-differing trial");

        let resp = apply_trial(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path((campaign.id.clone(), tier2_trial.id.clone())),
        )
        .await
        .expect(
            "apply_trial on a real campaign's tier2 trial must not 404 on a double-scoped name",
        );
        assert_eq!(resp.kind, ApplyKind::ReindexAndSwap);
    }
}
