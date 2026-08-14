use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use veda_core::metrics as m;
use veda_core::retrieval_semconv as rsc;
use veda_core::types::DistanceMetric;
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::filter::ast::FilterExpr;
use veda_query::fusion::{FieldQuery, FusionStrategy, MultiVectorQuery};
use veda_query::search::{SearchTimings, VectorSearchRequest};

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::auth::Claims;
use crate::rls;
use crate::tenant::{self, Tenant};

// ── Request types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: QueryBody,
    pub filter: Option<String>,
    /// Pre-parsed filter expression set by RLS injection. Not deserialized from JSON.
    #[serde(skip)]
    pub filter_expr: Option<FilterExpr>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub explain: bool,
    /// Matryoshka truncation length. When set, computes distances on the
    /// first N components only. Must be ≤ the field's dimension and (if the
    /// field declared `mrl_dims`) one of the declared lengths. Bypasses
    /// pre-built ANN indexes, so latency is higher.
    pub truncate_to: Option<u32>,
    /// Optional cross-encoder rerank stage. When set, the search retrieves
    /// `limit * 4` candidates (oversampling), reranks them via the
    /// configured `RERANK_ENDPOINT`, and returns the top `limit`.
    pub rerank: Option<super::rerank::RerankConfig>,
    /// Phase F MVCC snapshot timestamp (u128 decimal). When the
    /// collection has MVCC on, hits whose record isn't visible
    /// at this `read_ts` are filtered out before the response.
    /// Default = TSO.now() at the start of the search.
    /// Collections without MVCC ignore the value.
    #[serde(default)]
    pub at: Option<String>,
    /// Per-segment candidate budget (HNSW `ef`). Each sealed-segment ANN
    /// search returns up to `max(limit, num_candidates)` candidates to the
    /// cross-segment merge, so a fragmented collection does not drop true
    /// neighbours via per-segment `limit`-truncation. `None` uses the index
    /// default (200). Higher = better recall at slightly higher merge cost.
    #[serde(default)]
    pub num_candidates: Option<usize>,
}

fn default_limit() -> usize {
    10
}

/// Classify the query body into a coarse `query_type` string for the
/// query log / correlation analyzer, plus the primary vector field name
/// (used to resolve the backing index type). Lexical-only queries
/// (bm25 / match_phrase) have no vector field, so the field is `None`.
fn classify_query(body: &QueryBody) -> (&'static str, Option<String>) {
    if let Some(ref v) = body.vector {
        ("dense", Some(v.field.clone()))
    } else if let Some(ref s) = body.sparse_vector {
        ("sparse", Some(s.field.clone()))
    } else if let Some(ref cp) = body.colpali {
        ("colpali", Some(cp.field.clone()))
    } else if let Some(ref mv) = body.multi_vector {
        ("multi_vector", mv.first().map(|f| f.field.clone()))
    } else if let Some(ref h) = body.hybrid {
        ("hybrid", Some(h.dense.field.clone()))
    } else if body.bm25.is_some() {
        ("bm25", None)
    } else if body.match_phrase.is_some() {
        ("match_phrase", None)
    } else {
        ("unknown", None)
    }
}

/// Lowercase string form of a field's index type, for the query log.
fn index_type_str(it: veda_core::types::IndexType) -> &'static str {
    use veda_core::types::IndexType as I;
    match it {
        I::Hnsw => "hnsw",
        I::DiskAnn => "diskann",
        I::IvfPq => "ivf_pq",
        I::Flat => "flat",
        I::RaBitQ => "rabitq",
        I::RaBitQRerank => "rabitq_rerank",
        I::HnswRaBitQ => "hnsw_rabitq",
        I::Cagra => "cagra",
        I::RaBitQExtRerank => "rabitq_ext_rerank",
        I::Sq4 => "sq4",
        I::Sq8 => "sq8",
        I::SuffixArray => "suffix_array",
        I::Inverted => "inverted",
        I::Range => "range",
        I::Bitmap => "bitmap",
        I::Hash => "hash",
        I::None => "none",
    }
}

#[derive(Debug, Deserialize)]
pub struct QueryBody {
    /// Single dense-vector query.
    pub vector: Option<SingleVectorQuery>,
    /// Sparse-vector query.
    pub sparse_vector: Option<SparseVectorQuery>,
    /// ColPali / MaxSim multi-vector query against a single MultiVector field.
    pub colpali: Option<MultiVectorColpaliQuery>,
    /// Multi-vector fusion query (combines results across multiple dense fields).
    pub multi_vector: Option<Vec<MultiVectorField>>,
    /// Built-in BM25 full-text query against a `bm25 = true` string field.
    pub bm25: Option<Bm25Query>,
    /// Phrase / proximity query. Matches docs where the analyzed query terms
    /// appear in order with at most `slop` extra positions between consecutive
    /// terms (slop=0 = exact adjacency). Ranked by BM25 score.
    pub match_phrase: Option<MatchPhraseQuery>,
    /// Hybrid query: combines a dense vector leg and a BM25 leg with RRF.
    /// Use this instead of multi_vector when you want dense + BM25 fusion.
    pub hybrid: Option<HybridQuery>,
    /// Fusion strategy name (for multi_vector).
    pub fusion: Option<String>,
    /// Fusion-specific parameters.
    pub fusion_params: Option<FusionParams>,
}

#[derive(Debug, Deserialize)]
pub struct Bm25Query {
    /// String field with `bm25 = true` in the schema.
    pub field: String,
    /// Raw user-facing query text.
    pub query: String,
    /// Pruning aggressiveness for the block-max traversal. Absent or `1.0`
    /// is rank-safe: results are identical to exhaustive scoring. Values
    /// above `1.0` prune harder and may omit true top-k results. Values
    /// below `1.0` are rejected — they would only slow the query down
    /// without changing the answer.
    #[serde(default)]
    pub prune_theta: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct MatchPhraseQuery {
    /// String field with `bm25 = true` in the schema.
    pub field: String,
    /// The phrase text to search for.
    pub query: String,
    /// Maximum total extra positions allowed between consecutive query terms.
    /// `0` (default) = strict adjacency; `1` = one intervening token allowed, etc.
    #[serde(default)]
    pub slop: u32,
}

#[derive(Debug, Deserialize)]
pub struct HybridQuery {
    pub dense: SingleVectorQuery,
    pub bm25: Bm25Query,
    /// Weight on the dense leg (0..1). Default 0.5 = balanced.
    #[serde(default = "default_dense_weight")]
    pub dense_weight: f32,
    /// Fusion strategy. `"rrf"` (default) uses Reciprocal Rank Fusion —
    /// rank-based, no score calibration needed. `"weighted_sum"` does
    /// min-max normalization on each leg, then `dense_weight * dense_n +
    /// (1 - dense_weight) * bm25_n`. Use weighted_sum when you want
    /// linear control over the relative contribution of each leg.
    #[serde(default = "default_fusion")]
    pub fusion: String,
}

fn default_dense_weight() -> f32 {
    0.5
}

fn default_fusion() -> String {
    "rrf".to_owned()
}

#[derive(Debug, Deserialize)]
pub struct SingleVectorQuery {
    pub field: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct SparseVectorQuery {
    pub field: String,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct MultiVectorColpaliQuery {
    pub field: String,
    pub rows: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct MultiVectorField {
    pub field: String,
    pub values: Vec<f32>,
    #[serde(default = "default_weight")]
    pub weight: f32,
}

fn default_weight() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct FusionParams {
    pub k: Option<u32>,
}

// ── Response types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ExplainTiming {
    pub filter_ms: f64,
    pub ann_search_ms: f64,
    pub scoring_ms: f64,
    pub fusion_ms: f64,
    pub total_ms: f64,
}

/// Per-leg telemetry for hybrid retrieval. Set only when the query
/// is hybrid AND `req.explain == true`. Powers the Phase 5.6
/// retrieval Sankey in the dashboard.
#[derive(Debug, Serialize)]
pub struct HybridExplain {
    pub fusion: String,
    pub dense_weight: f32,
    pub bm25_weight: f32,
    /// Top doc-ids in rank order from each leg, before fusion.
    pub dense_top: Vec<String>,
    pub bm25_top: Vec<String>,
    /// Final fused / re-ranked top-K doc-ids in display order.
    pub final_top: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub count: usize,
    pub took_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<ExplainTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_explain: Option<HybridExplain>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: String,
    pub distance: f32,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

// ── Handler ──────────────────────────────────────────────────────────────

#[instrument(
    name = "search",
    skip(state, req),
    fields(
        collection = %name,
        top_k = req.limit,
        has_filter = req.filter.is_some(),
        // APM Phase 5.1: retrieval-quality + db.* attributes, recorded
        // below once the per-request `QueryMeta` and result count are known.
        // Declared Empty so the keys exist on the span even when a phase
        // returns early; `tracing_opentelemetry` maps them to OTel attrs.
        db.system = tracing::field::Empty,
        db.collection = tracing::field::Empty,
        retrieval.query_type = tracing::field::Empty,
        retrieval.top_k = tracing::field::Empty,
        retrieval.result_count = tracing::field::Empty,
        retrieval.reranked = tracing::field::Empty,
        retrieval.has_filter = tracing::field::Empty,
        retrieval.index_type = tracing::field::Empty,
        retrieval.filter_ms = tracing::field::Empty,
        retrieval.ann_search_ms = tracing::field::Empty,
        retrieval.scoring_ms = tracing::field::Empty,
        retrieval.fusion_ms = tracing::field::Empty,
        retrieval.total_ms = tracing::field::Empty,
    )
)]
pub async fn search(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    Json(mut req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    let name = state.inner.aliases.resolve(&name).await;
    // Cross-encoder rerank oversampling: if rerank is configured, fetch
    // `limit * 4` initial candidates so the reranker has enough to choose from.
    let final_limit = req.limit;
    let rerank_cfg = req.rerank.clone();
    if rerank_cfg.is_some() {
        req.limit = (req.limit * 4).max(20);
    }
    let start = Instant::now();

    // APM Phase 1: request-time query-log metadata. Classification and
    // filter/rerank flags are known up front (from `req`); `index_type`
    // is filled in once the schema is loaded. Threaded into the query
    // log on BOTH the success and error paths (exactly one write each).
    let display = crate::tenant::display_name(&name).to_owned();
    let (query_type, query_field) = classify_query(&req.query);
    let mut meta = super::query_log::QueryMeta {
        query_type: query_type.to_owned(),
        has_filter: req.filter.is_some() || req.filter_expr.is_some(),
        reranked: rerank_cfg.is_some(),
        top_k: final_limit,
        index_type: None,
        shard_ids: Vec::new(),
    };

    // Look up the schema to get default metric
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            )
        })?
    };

    // Resolve the backing index type of the queried vector field
    // (best-effort; lexical-only queries have no vector field).
    if let Some(field_name) = query_field.as_deref()
        && let Some(field) = schema.get_field(field_name)
    {
        meta.index_type = Some(index_type_str(field.index_type()).to_owned());
    }

    // RLS: if the collection has an owner field, inject an owner-equality
    // predicate so non-Admin callers only see their own records.
    if let Some(owner_field) = schema.rls_owner_field() {
        let is_admin = claims
            .as_ref()
            .map(|axum::Extension(c)| rls::is_admin(c))
            .unwrap_or(true); // auth disabled -> no restriction
        if !is_admin {
            let user = claims
                .as_ref()
                .map(|axum::Extension(c)| c.sub.clone())
                .unwrap_or_default();
            let user_filter = req
                .filter
                .as_deref()
                .and_then(|s| veda_query::filter::parser::FilterParser::parse(s).ok());
            // Inject the trusted owner predicate at the AST level.
            // Never re-serialize to a string: a quote in the username would
            // break out of the string literal and bypass RLS.
            req.filter_expr = rls::inject_owner_filter(user_filter, owner_field, &user, false);
        }
    }
    // RLS may have injected a filter where none existed — reflect that.
    meta.has_filter = req.filter.is_some() || req.filter_expr.is_some();

    // External collections are read in place: no segments, no index.
    // Checked after tenant scoping, alias resolution, AND row-level-
    // security injection — in that order, deliberately:
    //   - after scoping/resolution, so the freshness cache below is keyed
    //     by the same resolved name every other handler uses (keying it
    //     earlier would give two aliases of one collection independent
    //     caches with independent staleness);
    //   - after RLS injection, so a non-admin caller's owner predicate is
    //     already sitting in `req.filter_expr` before `execute_external`
    //     builds its predicate from `effective_filter(req)` (which reads
    //     `filter_expr` first). Checking `schema.external()` any earlier
    //     would return before RLS ever ran, silently serving every row of
    //     an RLS-configured external collection to every caller.
    if let Some(src) = schema.external() {
        return execute_external(&state, &name, src, &req)
            .await
            .map(|results| {
                let count = results.len();
                let took_ms = start.elapsed().as_millis() as u64;
                Json(SearchResponse {
                    results,
                    count,
                    took_ms,
                    explain: None,
                    hybrid_explain: None,
                })
            });
    }

    // APM Phase 5.1: record the always-known retrieval / db.* span
    // attributes from the Phase-1 `QueryMeta` (reusing the values already
    // computed — query_type, top_k, reranked, has_filter, index_type — plus
    // the user-facing collection name). `result_count` and the per-phase
    // timing breakdown are recorded after the search runs. These span attrs
    // are independent of the response body's `explain` flag.
    let span = tracing::Span::current();
    span.record(rsc::DB_SYSTEM, rsc::DB_SYSTEM_VEDA);
    span.record(rsc::DB_COLLECTION, display.as_str());
    span.record(rsc::RETRIEVAL_QUERY_TYPE, meta.query_type.as_str());
    span.record(rsc::RETRIEVAL_TOP_K, meta.top_k);
    span.record(rsc::RETRIEVAL_RERANKED, meta.reranked);
    span.record(rsc::RETRIEVAL_HAS_FILTER, meta.has_filter);
    if let Some(ref it) = meta.index_type {
        span.record(rsc::RETRIEVAL_INDEX_TYPE, it.as_str());
    }

    // Run the fallible search body. A single match below logs EXACTLY
    // one query-log entry per request: `record_success` on Ok,
    // `record_error` on Err — mutually exclusive, never both. Both
    // writes are unconditional (not gated on trace sampling).
    match run_search(&state, &name, &schema, &req, rerank_cfg, final_limit, start).await {
        Ok((results, explain_timing, mut hybrid_explain, took_ms)) => {
            let count = results.len();
            // APM Phase 5.1: result_count is always known; the per-phase
            // timing breakdown is only computed when the caller requested
            // `explain`, so record it conditionally (the span attrs are
            // otherwise independent of the response body).
            span.record(rsc::RETRIEVAL_RESULT_COUNT, count);
            if let Some(ref t) = explain_timing {
                span.record(rsc::RETRIEVAL_FILTER_MS, t.filter_ms);
                span.record(rsc::RETRIEVAL_ANN_SEARCH_MS, t.ann_search_ms);
                span.record(rsc::RETRIEVAL_SCORING_MS, t.scoring_ms);
                span.record(rsc::RETRIEVAL_FUSION_MS, t.fusion_ms);
                span.record(rsc::RETRIEVAL_TOTAL_MS, t.total_ms);
            }
            let doc_ids: Vec<String> = results.iter().map(|h| h.id.clone()).collect();
            super::query_log::record_success(&state, &name, &display, &meta, doc_ids, took_ms);
            if let Some(ref mut hx) = hybrid_explain {
                hx.final_top = results.iter().map(|h| h.id.clone()).collect();
            }
            Ok(Json(SearchResponse {
                results,
                count,
                took_ms,
                explain: explain_timing,
                hybrid_explain,
            }))
        }
        Err((status, body)) => {
            let took_ms = start.elapsed().as_millis() as u64;
            super::query_log::record_error(
                &state,
                &name,
                &display,
                &meta,
                error_kind_for_status(status),
                took_ms,
            );
            Err((status, body))
        }
    }
}

/// Coarse, low-cardinality error label derived from the HTTP status of a
/// failed search. Keeps the query log's `error_kind` enumerable for the
/// correlation analyzer (rather than free-form error strings).
fn error_kind_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "bad_request",
        StatusCode::NOT_FOUND => "not_found",
        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable",
        StatusCode::INTERNAL_SERVER_ERROR => "internal",
        _ => "other",
    }
}

/// The fallible core of the search handler. Produces the response
/// components (`hits`, explain timings, took_ms) or a gateway error. All
/// `?` early-returns funnel here so the caller can log exactly one
/// query-log entry. `req` is already RLS-resolved by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_search(
    state: &AppState,
    name: &str,
    schema: &veda_core::schema::CollectionSchema,
    req: &SearchRequest,
    rerank_cfg: Option<super::rerank::RerankConfig>,
    final_limit: usize,
    start: Instant,
) -> Result<
    (
        Vec<SearchHit>,
        Option<ExplainTiming>,
        Option<HybridExplain>,
        u64,
    ),
    (StatusCode, Json<ErrorResponse>),
> {
    // Phase F: resolve the snapshot timestamp once up front so
    // the executor request structs carry it and apply
    // visibility-at-read_ts filtering inside the search walk.
    // Non-MVCC collections leave this as None (no filtering).
    let mvcc_active_for_collection = {
        let managers = state.inner.segment_managers.read().await;
        managers
            .get(name)
            .map(|m| m.values().any(|sm| sm.has_mvcc()))
            .unwrap_or(false)
    };
    let effective_read_ts: Option<u128> = if mvcc_active_for_collection {
        Some(match req.at.as_deref() {
            Some(s) => super::time_parse::parse_read_ts(s)
                .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?,
            None => {
                let hlc = state.inner.tso.now().await.unwrap_or_default();
                ((hlc.physical_ns as u128) << 32) | (hlc.logical as u128)
            }
        })
    } else {
        None
    };
    // Phase H: cross-region reader gating. Sentinel 0 means
    // gating is off; once the operator advances safe_ts above 0,
    // reads ahead of it are rejected.
    if let Some(ts) = effective_read_ts {
        let safe = *state.inner.replica_safe_ts.read().await;
        if safe > 0 && ts > safe {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: format!(
                        "replica behind: read_ts={ts} exceeds replica_safe_ts={safe} by {}",
                        ts - safe
                    ),
                }),
            ));
        }
    }

    let executors = state.inner.query_executors.read().await;
    let executor = executors.get(name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "no query executor for collection".into(),
            }),
        )
    })?;

    let mut explain_timing: Option<ExplainTiming> = None;
    let mut hybrid_explain: Option<HybridExplain> = None;

    let results = if let Some(ref vec_query) = req.query.vector {
        // Single-vector search.
        // If truncate_to is set, validate against the field's declared MRL
        // ladder (if any) and dimension.
        if let Some(n) = req.truncate_to {
            let field = schema.get_field(&vec_query.field).ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("field '{}' not found", vec_query.field),
                    }),
                )
            })?;
            if let veda_core::types::FieldType::Vector {
                dimension,
                mrl_dims,
                ..
            } = field.field_type()
            {
                if n > *dimension {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!("truncate_to={n} exceeds field dimension {dimension}"),
                        }),
                    ));
                }
                if let Some(allowed) = mrl_dims
                    && !allowed.contains(&n)
                {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!(
                                "truncate_to={n} is not in field's declared mrl_dims {allowed:?}"
                            ),
                        }),
                    ));
                }
            } else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "truncate_to only applies to dense vector fields".into(),
                    }),
                ));
            }
        }
        validate_dense_query(schema, vec_query, req.truncate_to)?;
        let metric = resolve_metric(schema, &vec_query.field);
        let (hits, timings) =
            execute_single_vector(executor, vec_query, req, metric, effective_read_ts)?;
        if req.explain {
            let total = timings.filter_ms + timings.ann_search_ms + timings.scoring_ms;
            explain_timing = Some(ExplainTiming {
                filter_ms: timings.filter_ms,
                ann_search_ms: timings.ann_search_ms,
                scoring_ms: timings.scoring_ms,
                fusion_ms: 0.0,
                total_ms: total,
            });
        }
        hits
    } else if let Some(ref cp) = req.query.colpali {
        // MaxSim multi-vector search.
        let metric = resolve_colpali_metric(schema, &cp.field);
        let request = veda_query::search::MultiVectorSearchRequest {
            field: cp.field.clone(),
            query_rows: cp.rows.clone(),
            k: req.limit,
            metric,
            filter: req.filter.clone(),
            filter_expr: req.filter_expr.clone(),
            read_ts: effective_read_ts,
        };
        let (response, timings) = executor
            .search_multi_vector_colpali(&request)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("colpali search failed: {e}"),
                    }),
                )
            })?;
        if req.explain {
            let total = timings.filter_ms + timings.ann_search_ms + timings.scoring_ms;
            explain_timing = Some(ExplainTiming {
                filter_ms: timings.filter_ms,
                ann_search_ms: timings.ann_search_ms,
                scoring_ms: timings.scoring_ms,
                fusion_ms: 0.0,
                total_ms: total,
            });
        }
        response
            .results
            .into_iter()
            .map(|r| SearchHit {
                id: r.id.to_string(),
                distance: r.distance,
                fields: serde_json::Map::new(),
            })
            .collect()
    } else if let Some(ref sparse_query) = req.query.sparse_vector {
        // Sparse-vector search.
        let metric = resolve_sparse_metric(schema, &sparse_query.field);
        let (hits, timings) =
            execute_sparse_vector(executor, sparse_query, req, metric, effective_read_ts)?;
        if req.explain {
            let total = timings.filter_ms + timings.ann_search_ms + timings.scoring_ms;
            explain_timing = Some(ExplainTiming {
                filter_ms: timings.filter_ms,
                ann_search_ms: timings.ann_search_ms,
                scoring_ms: timings.scoring_ms,
                fusion_ms: 0.0,
                total_ms: total,
            });
        }
        hits
    } else if let Some(ref multi) = req.query.multi_vector {
        // Multi-vector fusion search
        let fusion_field = multi.first().map(|m| m.field.as_str()).unwrap_or("");
        let metric = resolve_metric(schema, fusion_field);
        if req.explain {
            let phase_start = Instant::now();
            let hits = execute_multi_vector(executor, multi, req, metric)?;
            let fusion_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
            explain_timing = Some(ExplainTiming {
                filter_ms: 0.0,
                ann_search_ms: 0.0,
                scoring_ms: 0.0,
                fusion_ms,
                total_ms: fusion_ms,
            });
            hits
        } else {
            execute_multi_vector(executor, multi, req, metric)?
        }
    } else if let Some(ref bm25_query) = req.query.bm25 {
        // Built-in BM25 full-text search against a string field with
        // `bm25 = true` in the schema. Runs the same tokenizer at ingest
        // and query time; no Lucene/Tantivy bundling required.
        validate_bm25_query(bm25_query)?;
        let eff_filter = effective_filter(req);
        execute_bm25(
            state,
            name,
            bm25_query,
            req.limit,
            eff_filter.as_ref(),
            effective_read_ts,
        )
        .await?
    } else if let Some(ref phrase_query) = req.query.match_phrase {
        // Phrase / proximity search: requires the analyzed query terms to
        // appear in order with at most `slop` extra positions between them.
        let eff_filter = effective_filter(req);
        execute_match_phrase(
            state,
            name,
            phrase_query,
            req.limit,
            eff_filter.as_ref(),
            effective_read_ts,
        )
        .await?
    } else if let Some(ref hybrid_query) = req.query.hybrid {
        // Dense + BM25 hybrid via Reciprocal Rank Fusion. Each leg is
        // scored independently; RRF folds the rankings into one set.
        // dense_weight in [0, 1] biases the RRF weighting.
        validate_dense_query(schema, &hybrid_query.dense, req.truncate_to)?;
        validate_bm25_query(&hybrid_query.bm25)?;
        let metric = resolve_metric(schema, &hybrid_query.dense.field);
        let eff_filter = effective_filter(req);
        let (hits, legs) = execute_hybrid_dense_bm25(
            state,
            name,
            executor,
            hybrid_query,
            req,
            metric,
            req.limit,
            effective_read_ts,
            eff_filter.as_ref(),
        )
        .await?;
        if req.explain {
            let dense_w = hybrid_query.dense_weight.clamp(0.0, 1.0);
            hybrid_explain = Some(HybridExplain {
                fusion: hybrid_query.fusion.clone(),
                dense_weight: dense_w,
                bm25_weight: 1.0 - dense_w,
                dense_top: legs.dense_top,
                bm25_top: legs.bm25_top,
                final_top: hits.iter().map(|h| h.id.clone()).collect(),
            });
        }
        hits
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "query must contain 'vector', 'sparse_vector', 'colpali', 'multi_vector', 'bm25', 'match_phrase', or 'hybrid'".into(),
            }),
        ));
    };

    // Optional rerank stage.
    let results = if let Some(cfg) = rerank_cfg {
        apply_rerank(state, name, req, &cfg, results, final_limit).await?
    } else {
        results
    };

    // Phase F: MVCC visibility filtering now lives inside the
    // LocalShardSearcher (both growing-segment and sealed-
    // segment paths intersect candidates with
    // `visible_doc_ids_at` when request.read_ts is set). The
    // previous belt-and-suspenders post-filter here is gone.
    let _ = effective_read_ts; // already plumbed through the request

    // Populate fields on each hit by looking up the stored record. Without
    // this every SearchHit.fields is an empty Map, which is unusable for
    // RAG — callers would have to do N follow-up GET-by-id calls.
    // TTL: capture now once; populate_hit_fields drops hits whose record
    // has expired (no sweep required).
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let mut results = populate_hit_fields(state, name, schema, results, now_secs).await;

    // Annotate every hit with its screening class. `/search` is a DATA API:
    // it must never drop a hit, or recall measurement in the eval harness and
    // the tuning campaigns would silently change. Enforcement happens only on
    // LLM-facing paths (see `ToolCatalog::dispatch`).
    for hit in &mut results {
        let class = crate::agent::retrieval_guard::classify_hit(&hit.fields);
        let name = serde_json::to_value(class)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unscreened".to_owned());
        hit.fields
            .insert("_guardrail_class".to_owned(), serde_json::json!(name));
    }

    let took_ms = start.elapsed().as_millis() as u64;
    metrics::histogram!(m::SEARCH_DURATION, "collection" => crate::tenant::display_name(name).to_owned())
        .record(start.elapsed().as_secs_f64());

    Ok((results, explain_timing, hybrid_explain, took_ms))
}

/// Apply a cross-encoder rerank pass to search results.
async fn apply_rerank(
    state: &AppState,
    collection: &str,
    _req: &SearchRequest,
    cfg: &super::rerank::RerankConfig,
    results: Vec<SearchHit>,
    final_limit: usize,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    use veda_core::record::FieldValue;

    let query = cfg.query.clone();

    // Look up each candidate's text field.
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(collection).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{collection}' not found"),
            }),
        )
    })?;
    // O(1) per-candidate text lookup via the UUID index.
    // Use lookup_record_at so expired records are excluded from reranking too.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let mut texts: Vec<String> = Vec::with_capacity(results.len());
    let mut keep_indices: Vec<usize> = Vec::with_capacity(results.len());
    for (i, hit) in results.iter().enumerate() {
        let Ok(id) = hit.id.parse::<uuid::Uuid>() else {
            continue;
        };
        for sm in shard_map.values() {
            if let Some(rec) = sm.lookup_record_at(id, now_secs) {
                if let Some(FieldValue::String(s)) = rec.get(&cfg.field) {
                    texts.push(s.clone());
                    keep_indices.push(i);
                }
                break;
            }
        }
    }
    drop(managers);

    if texts.is_empty() {
        let mut r = results;
        r.truncate(final_limit);
        return Ok(r);
    }

    let scored = super::rerank::call_reranker(
        &query,
        &texts,
        cfg.top_k.min(final_limit),
        cfg.model.as_deref(),
        cfg.instruction.as_deref(),
    )
    .await?;

    // Map rerank indices (into texts/keep_indices) back to original SearchHits.
    let mut out: Vec<SearchHit> = Vec::with_capacity(scored.len());
    for (idx, score) in scored {
        if let Some(&orig) = keep_indices.get(idx)
            && let Some(h) = results.get(orig)
        {
            let mut hit = h.clone();
            hit.distance = -score; // lower = better
            out.push(hit);
        }
    }
    Ok(out)
}

/// Populate `SearchHit.fields` for each hit by looking up the stored record
/// in this collection's segment managers. Vector fields are skipped — they
/// inflate the response without helping downstream callers, who already
/// have the query vector. Scalar / string / int / float / blob_ref fields
/// are serialized as JSON.
async fn populate_hit_fields(
    state: &AppState,
    collection: &str,
    schema: &veda_core::schema::CollectionSchema,
    hits: Vec<SearchHit>,
    now_secs: i64,
) -> Vec<SearchHit> {
    use veda_core::types::FieldType;

    if hits.is_empty() {
        return hits;
    }
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return hits;
    };

    let vector_field_names: std::collections::HashSet<&str> = schema
        .fields()
        .iter()
        .filter(|f| {
            matches!(
                f.field_type(),
                FieldType::Vector { .. }
                    | FieldType::MultiVector { .. }
                    | FieldType::SparseVector { .. }
            )
        })
        .map(|f| f.name())
        .collect();

    // O(1) per-hit lookup using each shard's UUID → doc_id index. The
    // SegmentManager.lookup_record helper walks growing-segment HashMap
    // then sealed-segment per-segment HashMap, returning a fully
    // materialized Record. No more collection-wide scans.
    let mut out: Vec<SearchHit> = Vec::with_capacity(hits.len());
    for mut hit in hits {
        let Ok(id) = hit.id.parse::<uuid::Uuid>() else {
            continue;
        };
        let mut record: Option<veda_core::record::Record> = None;
        for sm in shard_map.values() {
            if let Some(rec) = sm.lookup_record_at(id, now_secs) {
                record = Some(rec);
                break;
            }
        }
        let Some(rec) = record else {
            continue;
        };

        let mut map = serde_json::Map::new();
        for field_def in schema.fields() {
            let name = field_def.name();
            if vector_field_names.contains(name) {
                continue;
            }
            if let Some(value) = rec.get(name) {
                // Use the same JSON conversion as the records endpoint
                // so hit.fields[k] is the unwrapped value (`"foo"`, `42`),
                // not the tagged FieldValue variant (`{"String":"foo"}`)
                // that serde_json::to_value produces on the enum directly.
                map.insert(name.to_string(), super::records::field_value_to_json(value));
            }
        }
        // The retrieval-guardrail/TTL fields are RESERVED, not
        // schema-declared (the WAL writer persists them even when no
        // `CollectionSchema` lists them) — the loop above never hydrates
        // them, which would leave `classify_hit` (see the annotation step
        // in `run_search`) with nothing to classify on and every hit would
        // read `unscreened` on every real collection. Mirrors the identical
        // fix in `agent/tools/builtins.rs::search`.
        for reserved in RESERVED_GUARD_FIELDS {
            if let Some(value) = rec.get(reserved) {
                map.insert(
                    (*reserved).to_owned(),
                    super::records::field_value_to_json(value),
                );
            }
        }
        hit.fields = map;
        out.push(hit);
    }
    out
}

/// The retrieval-guardrail/TTL fields, reserved rather than
/// schema-declared. Mirrors the identical inline list in
/// `agent/tools/builtins.rs` and `veda-storage/src/segment.rs`'s WAL writer.
const RESERVED_GUARD_FIELDS: [&str; 5] = [
    veda_core::record::EXPIRES_AT_FIELD,
    veda_core::record::GUARDRAIL_FLAGGED_FIELD,
    veda_core::record::GUARDRAIL_SCORE_FIELD,
    veda_core::record::GUARDRAIL_CATEGORY_FIELD,
    veda_core::record::INDEXED_AT_FIELD,
];

/// Search an external (read-in-place Parquet) collection. Called by
/// `search` right after tenant scoping, alias resolution and the schema
/// fetch — never before, so the freshness cache below is keyed by the same
/// resolved name every other handler uses. Keying it on the unresolved
/// name would give two aliases of one collection independent caches with
/// independent staleness: one could serve stale data indefinitely while
/// the other stayed fresh, with nothing reporting it.
async fn execute_external(
    state: &AppState,
    name: &str,
    src: &veda_core::schema::ExternalSource,
    req: &SearchRequest,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let bad_request = |error: String| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }));

    // Reject query types that need an index Parquet does not carry. Each
    // gets a distinct, accurate reason rather than one generic message.
    if req.query.bm25.is_some() {
        return Err(bad_request(format!(
            "bm25 search is not available on external collection '{name}': it requires \
             the inverted index Veda builds at ingest, and external data is read in place."
        )));
    }
    if req.query.match_phrase.is_some() {
        return Err(bad_request(format!(
            "match_phrase search is not available on external collection '{name}': it \
             requires the inverted index Veda builds at ingest, and external data is read \
             in place."
        )));
    }
    if req.query.sparse_vector.is_some() {
        return Err(bad_request(format!(
            "sparse search is not available on external collection '{name}': it requires \
             stored sparse vectors, and external data carries only the dense vector and \
             scalar columns read from the source Parquet file."
        )));
    }
    if req.query.hybrid.is_some() {
        return Err(bad_request(format!(
            "hybrid search is not available on external collection '{name}': it composes \
             a dense leg with a bm25 leg, and the bm25 leg requires the inverted index \
             Veda builds at ingest, which external data does not have."
        )));
    }

    let Some(ref vec_query) = req.query.vector else {
        return Err(bad_request(format!(
            "external collection '{name}' supports dense vector search and scalar \
             filters only"
        )));
    };

    // Freshness, then cache. This runs before every query — the dataset
    // lives entirely in memory, so an undetected source change would serve
    // deleted or superseded rows as current forever.
    let store = state.inner.blob_store.read().await.clone();
    let fresh_stamp = veda_storage::external_parquet::stamp(store.as_ref(), src).map_err(|e| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: format!(
                    "external source '{}' for collection '{name}' is unavailable: {e}",
                    src.uri
                ),
            }),
        )
    })?;

    let cached = state
        .inner
        .external_datasets
        .read()
        .await
        .get(name)
        .cloned();
    let dataset = match cached {
        Some(ds) if ds.source_stamp() == &fresh_stamp => ds,
        _ => match veda_storage::external_parquet::load(store.as_ref(), src) {
            Ok(loaded) => {
                let loaded = std::sync::Arc::new(loaded);
                state
                    .inner
                    .external_datasets
                    .write()
                    .await
                    .insert(name.to_owned(), loaded.clone());
                loaded
            }
            Err(e) => {
                // A failed reload must not leave a stale entry behind: the
                // next query has to retry the load, not silently keep
                // serving the last-good snapshot forever.
                state.inner.external_datasets.write().await.remove(name);
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: format!(
                            "external source '{}' for collection '{name}' is unavailable: {e}",
                            src.uri
                        ),
                    }),
                ));
            }
        },
    };

    let eff = effective_filter(req);
    let pred = eff.as_ref().map(|expr| {
        move |rec: &veda_core::record::Record| {
            veda_query::filter::record_eval::evaluate_record(expr, rec)
        }
    });
    let hits = dataset
        .search(
            &vec_query.values,
            req.limit,
            src.metric,
            pred.as_ref()
                .map(|p| p as &dyn Fn(&veda_core::record::Record) -> bool),
        )
        .map_err(|e| bad_request(e.to_string()))?;

    // `search` returns ids, not row indices, so build a lookup once per
    // query rather than scanning `ids()` per hit.
    let id_to_row: std::collections::HashMap<&str, usize> = dataset
        .ids()
        .iter()
        .enumerate()
        .map(|(row, id)| (id.as_str(), row))
        .collect();

    let results = hits
        .into_iter()
        .map(|(id, distance)| {
            let mut fields = serde_json::Map::new();
            if let Some(&row) = id_to_row.get(id.as_str()) {
                let record = dataset.record_at(row);
                for (field_name, value) in record.fields() {
                    fields.insert(
                        field_name.clone(),
                        super::records::field_value_to_json(value),
                    );
                }
            }
            SearchHit {
                id,
                distance,
                fields,
            }
        })
        .collect();

    Ok(results)
}

fn execute_single_vector(
    executor: &DistributedQueryExecutor,
    vec_query: &SingleVectorQuery,
    req: &SearchRequest,
    metric: DistanceMetric,
    read_ts: Option<u128>,
) -> Result<(Vec<SearchHit>, SearchTimings), (StatusCode, Json<ErrorResponse>)> {
    let request = VectorSearchRequest {
        field: vec_query.field.clone(),
        query_vector: vec_query.values.clone(),
        k: req.limit,
        metric,
        filter: req.filter.clone(),
        filter_expr: req.filter_expr.clone(),
        truncate_to: req.truncate_to,
        read_ts,
        ef_search: req.num_candidates,
    };

    let (response, timings) = executor.search(&request).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("search failed: {e}"),
            }),
        )
    })?;

    let hits = response
        .results
        .into_iter()
        .map(|r| SearchHit {
            id: r.id.to_string(),
            distance: r.distance,
            fields: serde_json::Map::new(),
        })
        .collect();

    Ok((hits, timings))
}

/// Resolve the *effective* scalar filter for a search request as an AST,
/// mirroring the precedence the dense path relies on: a pre-parsed
/// `filter_expr` (e.g. RLS owner-injection) wins; otherwise parse the raw
/// `filter` string. The lexical (BM25 / match_phrase / hybrid) paths apply
/// this predicate per-candidate via `evaluate_record`, since they bypass the
/// scalar-index filtering the vector executor does.
///
/// Because `filter_expr` carries the RLS-injected owner predicate, honoring
/// it here means BM25/hybrid searches now enforce row-level security exactly
/// like the dense path.
fn effective_filter(req: &SearchRequest) -> Option<FilterExpr> {
    if let Some(ref fe) = req.filter_expr {
        return Some(fe.clone());
    }
    req.filter
        .as_deref()
        .and_then(|s| veda_query::filter::parser::FilterParser::parse(s).ok())
}

/// How many candidate docs each posting-list scan should request when a
/// scalar filter is in play. We oversample so that, after dropping
/// candidates that fail `evaluate_record`, we still have a good chance of
/// filling `k` survivors. With no filter we just request `k`.
fn bm25_fetch_k(k: usize, has_filter: bool) -> usize {
    if has_filter {
        (k.max(1)).saturating_mul(10)
    } else {
        k.max(1)
    }
}

/// Run a BM25 full-text search across all sealed segments of every shard
/// in the collection. Each segment scores independently; we merge the
/// per-segment top-k by raw BM25 score (no further normalization) and
/// take the global top-k. Returns hits with `distance = -bm25_score`
/// so that lower-is-better stays consistent across query types.
///
/// When `filter_expr` is set, each candidate's resolved `Record` is checked
/// with `evaluate_record`; only survivors are kept. We oversample the
/// posting-list scan (see `bm25_fetch_k`) so filtering doesn't under-fill
/// `k`. When `read_ts` is set (MVCC), candidates not visible at that
/// timestamp are dropped, matching the dense path's visibility semantics.
async fn execute_bm25(
    state: &AppState,
    collection: &str,
    query: &Bm25Query,
    k: usize,
    filter_expr: Option<&FilterExpr>,
    read_ts: Option<u128>,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{collection}' not found"),
            }),
        ));
    };

    let fetch_k = bm25_fetch_k(k, filter_expr.is_some());
    // Absent means exact. Storage clamps defensively too, but the request
    // boundary is where an out-of-range value becomes a 400.
    let theta = query.prune_theta.unwrap_or(1.0);

    // (record_uuid, bm25_score) merged across shards / growing+sealed
    // segments. Growing segments maintain an in-memory Bm25Index updated
    // on insert so freshly-ingested rows are immediately searchable —
    // earlier this path was sealed-only, which made BM25 invisible to
    // anything below the seal threshold.
    let mut merged: Vec<(uuid::Uuid, f32)> = Vec::new();
    for sm in shard_map.values() {
        // Growing segment
        let gs = sm.growing_segment();
        if let Some(idx) = gs.bm25_index(&query.field) {
            let visible = read_ts.and_then(|ts| gs.visible_doc_ids_at(ts));
            let scored = idx.search_with_theta(&query.query, fetch_k, theta);
            for (doc_id, score) in scored {
                if gs.is_deleted(doc_id) {
                    continue;
                }
                if let Some(ref vis) = visible
                    && !vis.contains(doc_id)
                {
                    continue;
                }
                if let Some(rec) = gs.get_record(doc_id) {
                    if filter_expr.is_some_and(|fe| !veda_query::filter::evaluate_record(fe, rec)) {
                        continue;
                    }
                    merged.push((rec.id(), score));
                }
            }
        }
        // Sealed segments
        for handle in sm.sealed_segments() {
            let Some(idx) = handle.reader.bm25_index(&query.field) else {
                continue;
            };
            let visible = read_ts.and_then(|ts| handle.reader.visible_doc_ids_at(ts));
            let scored = idx.search_with_theta(&query.query, fetch_k, theta);
            for (doc_id, score) in scored {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Some(ref vis) = visible
                    && !vis.contains(doc_id)
                {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if filter_expr.is_some_and(|fe| !veda_query::filter::evaluate_record(fe, &rec))
                    {
                        continue;
                    }
                    merged.push((rec.id(), score));
                }
            }
        }
    }
    drop(managers);

    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(k);
    let hits: Vec<SearchHit> = merged
        .into_iter()
        .map(|(id, s)| SearchHit {
            id: id.to_string(),
            distance: -s,
            fields: serde_json::Map::new(),
        })
        .collect();
    Ok(hits)
}

/// Phrase / proximity search across all sealed and growing segments of a
/// collection. Each segment scores independently via `Bm25Index::search_phrase`;
/// results are merged, sorted by descending score, and truncated to `k`.
/// Returns hits with `distance = -bm25_score` (lower = better, consistent
/// with the bag-of-words BM25 leg).
async fn execute_match_phrase(
    state: &AppState,
    collection: &str,
    query: &MatchPhraseQuery,
    k: usize,
    filter_expr: Option<&FilterExpr>,
    read_ts: Option<u128>,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{collection}' not found"),
            }),
        ));
    };

    let fetch_k = bm25_fetch_k(k, filter_expr.is_some());

    let mut merged: Vec<(uuid::Uuid, f32)> = Vec::new();
    for sm in shard_map.values() {
        // Growing segment
        let gs = sm.growing_segment();
        if let Some(idx) = gs.bm25_index(&query.field) {
            let visible = read_ts.and_then(|ts| gs.visible_doc_ids_at(ts));
            let scored = idx.search_phrase(&query.query, query.slop, fetch_k);
            for (doc_id, score) in scored {
                if gs.is_deleted(doc_id) {
                    continue;
                }
                if let Some(ref vis) = visible
                    && !vis.contains(doc_id)
                {
                    continue;
                }
                if let Some(rec) = gs.get_record(doc_id) {
                    if filter_expr.is_some_and(|fe| !veda_query::filter::evaluate_record(fe, rec)) {
                        continue;
                    }
                    merged.push((rec.id(), score));
                }
            }
        }
        // Sealed segments
        for handle in sm.sealed_segments() {
            let Some(idx) = handle.reader.bm25_index(&query.field) else {
                continue;
            };
            let visible = read_ts.and_then(|ts| handle.reader.visible_doc_ids_at(ts));
            let scored = idx.search_phrase(&query.query, query.slop, fetch_k);
            for (doc_id, score) in scored {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Some(ref vis) = visible
                    && !vis.contains(doc_id)
                {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if filter_expr.is_some_and(|fe| !veda_query::filter::evaluate_record(fe, &rec))
                    {
                        continue;
                    }
                    merged.push((rec.id(), score));
                }
            }
        }
    }
    drop(managers);

    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(k);
    let hits: Vec<SearchHit> = merged
        .into_iter()
        .map(|(id, s)| SearchHit {
            id: id.to_string(),
            distance: -s,
            fields: serde_json::Map::new(),
        })
        .collect();
    Ok(hits)
}

/// Dense + BM25 hybrid. Two fusion strategies:
/// - `rrf` (default): `score(d) = sum_l (w_l / (k_rrf + rank_l(d)))`,
///   k_rrf = 60. Rank-based — no score-scale calibration needed.
/// - `weighted_sum`: min-max normalize each leg's scores to [0,1], then
///   `dense_w * dense_n + (1 - dense_w) * bm25_n`. Linear, intuitive
///   when you want explicit control over each leg's contribution.
///
/// Each leg runs the full top-`k` retrieval first; results are folded
/// together into at most `k` hits.
/// Per-leg ids exposed to the gateway-level explain plumbing so the
/// dashboard can draw the retrieval Sankey (Phase 5.6).
pub struct HybridLegs {
    pub dense_top: Vec<String>,
    pub bm25_top: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
async fn execute_hybrid_dense_bm25(
    state: &AppState,
    collection: &str,
    executor: &DistributedQueryExecutor,
    hybrid: &HybridQuery,
    req: &SearchRequest,
    metric: DistanceMetric,
    k: usize,
    read_ts: Option<u128>,
    filter_expr: Option<&FilterExpr>,
) -> Result<(Vec<SearchHit>, HybridLegs), (StatusCode, Json<ErrorResponse>)> {
    let dense_w = hybrid.dense_weight.clamp(0.0, 1.0);
    let bm25_w = 1.0 - dense_w;

    // Dense leg already applies the effective filter via the executor
    // request (req.filter / req.filter_expr). Pass the same effective
    // filter into the BM25 leg so both legs are scoped identically.
    let (dense_hits, _t) = execute_single_vector(executor, &hybrid.dense, req, metric, read_ts)?;
    let bm25_hits = execute_bm25(
        state,
        collection,
        &hybrid.bm25,
        k.max(req.limit),
        filter_expr,
        read_ts,
    )
    .await?;

    let dense_top: Vec<String> = dense_hits.iter().map(|h| h.id.clone()).collect();
    let bm25_top: Vec<String> = bm25_hits.iter().map(|h| h.id.clone()).collect();

    let scores: std::collections::HashMap<String, f32> = match hybrid.fusion.as_str() {
        "weighted_sum" => weighted_sum_fuse(&dense_hits, &bm25_hits, dense_w, bm25_w, metric),
        _ => rrf_fuse(&dense_hits, &bm25_hits, dense_w, bm25_w),
    };

    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(k);
    let hits: Vec<SearchHit> = ranked
        .into_iter()
        .map(|(id, s)| SearchHit {
            id,
            distance: -s, // lower = better
            fields: serde_json::Map::new(),
        })
        .collect();
    Ok((
        hits,
        HybridLegs {
            dense_top,
            bm25_top,
        },
    ))
}

/// Reciprocal-rank fusion.
fn rrf_fuse(
    dense_hits: &[SearchHit],
    bm25_hits: &[SearchHit],
    dense_w: f32,
    bm25_w: f32,
) -> std::collections::HashMap<String, f32> {
    const K_RRF: f32 = 60.0;
    let mut scores: std::collections::HashMap<String, f32> =
        std::collections::HashMap::with_capacity(dense_hits.len() + bm25_hits.len());
    for (rank, h) in dense_hits.iter().enumerate() {
        let s = dense_w / (K_RRF + (rank + 1) as f32);
        *scores.entry(h.id.clone()).or_insert(0.0) += s;
    }
    for (rank, h) in bm25_hits.iter().enumerate() {
        let s = bm25_w / (K_RRF + (rank + 1) as f32);
        *scores.entry(h.id.clone()).or_insert(0.0) += s;
    }
    scores
}

/// Weighted-sum fusion with per-leg min-max normalization.
///
/// Dense hits carry `distance` (lower = closer). For cosine / dot we map
/// to a similarity (higher = better) before normalizing; for euclidean /
/// manhattan we negate. BM25 hits store `distance = -bm25_score`, so we
/// just flip the sign back to recover the raw score.
fn weighted_sum_fuse(
    dense_hits: &[SearchHit],
    bm25_hits: &[SearchHit],
    dense_w: f32,
    bm25_w: f32,
    metric: DistanceMetric,
) -> std::collections::HashMap<String, f32> {
    // Dense → similarity, BM25 → raw score.
    let dense_sims: Vec<(String, f32)> = dense_hits
        .iter()
        .map(|h| {
            let sim = match metric {
                DistanceMetric::Cosine | DistanceMetric::DotProduct => 1.0 - h.distance,
                _ => -h.distance,
            };
            (h.id.clone(), sim)
        })
        .collect();
    let bm25_raw: Vec<(String, f32)> = bm25_hits
        .iter()
        .map(|h| (h.id.clone(), -h.distance))
        .collect();

    let mut scores: std::collections::HashMap<String, f32> =
        std::collections::HashMap::with_capacity(dense_sims.len() + bm25_raw.len());
    for (id, n) in min_max_normalize(&dense_sims) {
        *scores.entry(id).or_insert(0.0) += dense_w * n;
    }
    for (id, n) in min_max_normalize(&bm25_raw) {
        *scores.entry(id).or_insert(0.0) += bm25_w * n;
    }
    scores
}

/// Min-max normalize a list of (id, score) into [0, 1]. A degenerate
/// `min == max` case collapses to a constant 1.0 so all hits contribute
/// equally rather than zero (better matches what users intuitively
/// expect when a leg returned identical scores).
fn min_max_normalize(scored: &[(String, f32)]) -> Vec<(String, f32)> {
    if scored.is_empty() {
        return Vec::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (_, s) in scored {
        if *s < min {
            min = *s;
        }
        if *s > max {
            max = *s;
        }
    }
    let range = max - min;
    if range.abs() < 1e-9 {
        return scored.iter().map(|(id, _)| (id.clone(), 1.0)).collect();
    }
    scored
        .iter()
        .map(|(id, s)| (id.clone(), (s - min) / range))
        .collect()
}

fn execute_multi_vector(
    executor: &DistributedQueryExecutor,
    multi: &[MultiVectorField],
    req: &SearchRequest,
    metric: DistanceMetric,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let fusion_strategy = match req.query.fusion.as_deref().unwrap_or("rrf") {
        "rrf" => {
            let k = req
                .query
                .fusion_params
                .as_ref()
                .and_then(|p| p.k)
                .unwrap_or(60);
            FusionStrategy::Rrf { k }
        }
        "weighted_sum" => FusionStrategy::WeightedSum,
        "intersection" => FusionStrategy::Intersection,
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("unknown fusion strategy: '{other}'"),
                }),
            ));
        }
    };

    let queries: Vec<FieldQuery> = multi
        .iter()
        .map(|m| FieldQuery {
            field: m.field.clone(),
            vector: m.values.clone(),
            weight: m.weight,
        })
        .collect();

    let mvq = MultiVectorQuery {
        queries,
        fusion: fusion_strategy,
        limit: req.limit,
    };

    let fused = executor
        .search_multi_vector(&mvq, req.filter.as_deref(), metric)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("multi-vector search failed: {e}"),
                }),
            )
        })?;

    Ok(fused
        .into_iter()
        .map(|r| SearchHit {
            id: r.id.to_string(),
            distance: r.score,
            fields: serde_json::Map::new(),
        })
        .collect())
}

fn execute_sparse_vector(
    executor: &DistributedQueryExecutor,
    sparse_query: &SparseVectorQuery,
    req: &SearchRequest,
    metric: DistanceMetric,
    read_ts: Option<u128>,
) -> Result<(Vec<SearchHit>, veda_query::search::SearchTimings), (StatusCode, Json<ErrorResponse>)>
{
    let sv = veda_core::sparse::SparseVec::new(
        sparse_query.indices.clone(),
        sparse_query.values.clone(),
    )
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid sparse query: {e}"),
            }),
        )
    })?;

    let request = veda_query::search::SparseSearchRequest {
        field: sparse_query.field.clone(),
        query: sv,
        k: req.limit,
        metric,
        filter: req.filter.clone(),
        filter_expr: req.filter_expr.clone(),
        read_ts,
    };

    let (response, timings) = executor.search_sparse(&request).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("sparse search failed: {e}"),
            }),
        )
    })?;

    let hits = response
        .results
        .into_iter()
        .map(|r| SearchHit {
            id: r.id.to_string(),
            distance: r.distance,
            fields: serde_json::Map::new(),
        })
        .collect();
    Ok((hits, timings))
}

/// Resolve the distance metric for a multi-vector field, defaulting to Cosine.
fn resolve_colpali_metric(
    schema: &veda_core::schema::CollectionSchema,
    field_name: &str,
) -> DistanceMetric {
    schema
        .get_field(field_name)
        .and_then(|f| match f.field_type() {
            veda_core::types::FieldType::MultiVector { metric, .. } => Some(*metric),
            _ => None,
        })
        .unwrap_or(DistanceMetric::Cosine)
}

/// Resolve the distance metric for a sparse field, defaulting to Cosine.
fn resolve_sparse_metric(
    schema: &veda_core::schema::CollectionSchema,
    field_name: &str,
) -> DistanceMetric {
    schema
        .get_field(field_name)
        .and_then(|f| match f.field_type() {
            veda_core::types::FieldType::SparseVector { metric, .. } => Some(*metric),
            _ => None,
        })
        .unwrap_or(DistanceMetric::Cosine)
}

/// Validate the BM25 leg's `prune_theta` before dispatching to `execute_bm25`.
/// Values below `1.0` would only slow the query down without changing the
/// answer (see `Bm25Index::search_with_theta`), so reject them at the
/// request boundary rather than silently clamping.
fn validate_bm25_query(q: &Bm25Query) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(theta) = q.prune_theta
        && (!theta.is_finite() || theta < 1.0)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "prune_theta must be a finite value >= 1.0 (got {theta}); \
                     1.0 is exact, higher values prune more aggressively"
                ),
            }),
        ));
    }
    Ok(())
}

/// Resolve the distance metric for a field from the schema, defaulting to Cosine.
/// Validate a dense query leg against the schema before dispatching it.
///
/// Both failures here are caller mistakes that used to be invisible or
/// mis-classified:
///   - an unknown / non-vector field fell through to `resolve_metric`'s
///     `unwrap_or(Cosine)` default and searched nothing, so a typo'd field
///     name returned `200` with zero results instead of saying why;
///   - a wrong-length query vector reached the executor and came back as a
///     `500 search failed: Dimension mismatch`, telling clients to retry a
///     request that can never succeed and charging it to the error budget.
fn validate_dense_query(
    schema: &veda_core::schema::CollectionSchema,
    q: &SingleVectorQuery,
    truncate_to: Option<u32>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let bad_request = |error: String| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }));

    let field = schema
        .get_field(&q.field)
        .ok_or_else(|| bad_request(format!("field '{}' not found", q.field)))?;

    let veda_core::types::FieldType::Vector { dimension, .. } = field.field_type() else {
        return Err(bad_request(format!(
            "field '{}' is not a dense vector field",
            q.field
        )));
    };

    // MRL truncation is the one case where a shorter query vector is
    // legitimate — the truncated scan uses `min(truncate_to, q.len())`
    // components. `truncate_to` itself is validated at the call site.
    if truncate_to.is_none() && q.values.len() != *dimension as usize {
        return Err(bad_request(format!(
            "field '{}' has dimension {dimension}, got a query vector of length {}",
            q.field,
            q.values.len()
        )));
    }

    Ok(())
}

fn resolve_metric(
    schema: &veda_core::schema::CollectionSchema,
    field_name: &str,
) -> DistanceMetric {
    schema
        .get_field(field_name)
        .and_then(|f| match f.field_type() {
            veda_core::types::FieldType::Vector { metric, .. } => Some(*metric),
            _ => None,
        })
        .unwrap_or(DistanceMetric::Cosine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::collections::create_collection;

    fn test_state() -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        AppState::new(tmp.path())
    }

    /// Creates a collection with one "v" vector field (dim 2, cosine).
    /// `seed_records`, when non-empty, is bulk-loaded atomically at create
    /// time (mirrors `create_collection`'s Phase G behaviour) and
    /// synchronously builds the query executor, so `search` can find it
    /// immediately without racing the background compaction/rebuild task
    /// that a plain `/insert` spawns.
    async fn seed_test_collection(
        state: &AppState,
        name: &str,
        seed_records: Vec<serde_json::Value>,
    ) {
        let req = serde_json::from_value(serde_json::json!({
            "name": name,
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ],
            "seed_records": seed_records,
        }))
        .expect("fixture create-collection request should deserialize");
        let resp = create_collection(
            State(state.clone()),
            crate::tenant::Tenant(crate::tenant::TenantId::default_tenant()),
            Json(req),
        )
        .await;
        use axum::response::IntoResponse;
        let status = resp.into_response().status();
        assert!(
            status.is_success(),
            "seed_test_collection({name}) should succeed, got {status}"
        );
    }

    fn fixture_search_request() -> SearchRequest {
        serde_json::from_value(serde_json::json!({
            "query": { "vector": { "field": "v", "values": [1.0, 0.0] } },
            "limit": 10,
        }))
        .expect("fixture search request should deserialize")
    }

    #[tokio::test]
    async fn search_follows_an_alias_to_the_swapped_collection() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();
        seed_test_collection(&state, "mycol", vec![]).await;
        seed_test_collection(
            &state,
            "mycol@v2",
            vec![serde_json::json!({ "v": [1.0, 0.0] })],
        )
        .await;

        state
            .inner
            .aliases
            .set(
                crate::tenant::scope(&tenant, "mycol"),
                crate::tenant::scope(&tenant, "mycol@v2"),
            )
            .await;

        let results = search(
            State(state),
            crate::tenant::Tenant(tenant),
            Path("mycol".to_owned()),
            None,
            Json(fixture_search_request()),
        )
        .await
        .expect("search should succeed");
        assert!(
            !results.results.is_empty(),
            "search on the aliased name must hit mycol@v2's data, not the empty original"
        );
    }
}
