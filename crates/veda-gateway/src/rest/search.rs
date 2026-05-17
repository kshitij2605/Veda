use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use veda_core::metrics as m;
use veda_core::types::DistanceMetric;
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::fusion::{FieldQuery, FusionStrategy, MultiVectorQuery};
use veda_query::search::{SearchTimings, VectorSearchRequest};

use super::collections::ErrorResponse;
use super::state::AppState;

// ── Request types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: QueryBody,
    pub filter: Option<String>,
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
}

fn default_limit() -> usize {
    10
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
}

#[derive(Debug, Deserialize)]
pub struct HybridQuery {
    pub dense: SingleVectorQuery,
    pub bm25: Bm25Query,
    /// Weight on the dense leg (0..1). Default 0.5 = balanced.
    #[serde(default = "default_dense_weight")]
    pub dense_weight: f32,
}

fn default_dense_weight() -> f32 {
    0.5
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

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub count: usize,
    pub took_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<ExplainTiming>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: String,
    pub distance: f32,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

// ── Handler ──────────────────────────────────────────────────────────────

#[instrument(name = "search", skip(state, req), fields(collection = %name, top_k = req.limit, has_filter = req.filter.is_some()))]
pub async fn search(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(mut req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Cross-encoder rerank oversampling: if rerank is configured, fetch
    // `limit * 4` initial candidates so the reranker has enough to choose from.
    let final_limit = req.limit;
    let rerank_cfg = req.rerank.clone();
    if rerank_cfg.is_some() {
        req.limit = (req.limit * 4).max(20);
    }
    let start = Instant::now();

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

    let executors = state.inner.query_executors.read().await;
    let executor = executors.get(&name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "no query executor for collection".into(),
            }),
        )
    })?;

    let mut explain_timing: Option<ExplainTiming> = None;

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
            if let veda_core::types::FieldType::Vector { dimension, mrl_dims, .. } = field.field_type() {
                if n > *dimension {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: format!(
                                "truncate_to={n} exceeds field dimension {dimension}"
                            ),
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
        let metric = resolve_metric(&schema, &vec_query.field);
        let (hits, timings) = execute_single_vector(executor, vec_query, &req, metric)?;
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
        let metric = resolve_colpali_metric(&schema, &cp.field);
        let request = veda_query::search::MultiVectorSearchRequest {
            field: cp.field.clone(),
            query_rows: cp.rows.clone(),
            k: req.limit,
            metric,
            filter: req.filter.clone(),
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
        let metric = resolve_sparse_metric(&schema, &sparse_query.field);
        let (hits, timings) = execute_sparse_vector(executor, sparse_query, &req, metric)?;
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
        let metric = resolve_metric(&schema, fusion_field);
        if req.explain {
            let phase_start = Instant::now();
            let hits = execute_multi_vector(executor, multi, &req, metric)?;
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
            execute_multi_vector(executor, multi, &req, metric)?
        }
    } else if let Some(ref bm25_query) = req.query.bm25 {
        // Built-in BM25 full-text search against a string field with
        // `bm25 = true` in the schema. Runs the same tokenizer at ingest
        // and query time; no Lucene/Tantivy bundling required.
        execute_bm25(&state, &name, bm25_query, req.limit).await?
    } else if let Some(ref hybrid_query) = req.query.hybrid {
        // Dense + BM25 hybrid via Reciprocal Rank Fusion. Each leg is
        // scored independently; RRF folds the rankings into one set.
        // dense_weight in [0, 1] biases the RRF weighting.
        let metric = resolve_metric(&schema, &hybrid_query.dense.field);
        execute_hybrid_dense_bm25(
            &state,
            &name,
            executor,
            hybrid_query,
            &req,
            metric,
            req.limit,
        )
        .await?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "query must contain 'vector', 'sparse_vector', 'colpali', 'multi_vector', 'bm25', or 'hybrid'".into(),
            }),
        ));
    };

    // Optional rerank stage.
    let mut results = if let Some(cfg) = rerank_cfg {
        apply_rerank(&state, &name, &req, &cfg, results, final_limit).await?
    } else {
        results
    };

    // Populate fields on each hit by looking up the stored record. Without
    // this every SearchHit.fields is an empty Map, which is unusable for
    // RAG — callers would have to do N follow-up GET-by-id calls.
    populate_hit_fields(&state, &name, &schema, &mut results).await;

    let took_ms = start.elapsed().as_millis() as u64;
    metrics::histogram!(m::SEARCH_DURATION, "collection" => name.clone())
        .record(start.elapsed().as_secs_f64());
    let count = results.len();

    Ok(Json(SearchResponse {
        results,
        count,
        took_ms,
        explain: explain_timing,
    }))
}

/// Apply a cross-encoder rerank pass to search results.
async fn apply_rerank(
    state: &AppState,
    collection: &str,
    req: &SearchRequest,
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
    let mut texts: Vec<String> = Vec::with_capacity(results.len());
    let mut keep_indices: Vec<usize> = Vec::with_capacity(results.len());
    for (i, hit) in results.iter().enumerate() {
        let Ok(id) = hit.id.parse::<uuid::Uuid>() else {
            continue;
        };
        for sm in shard_map.values() {
            if let Some(rec) = sm.lookup_record(id) {
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
    hits: &mut [SearchHit],
) {
    use veda_core::types::FieldType;

    if hits.is_empty() {
        return;
    }
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(collection) else {
        return;
    };

    let vector_field_names: std::collections::HashSet<&str> = schema
        .fields()
        .iter()
        .filter(|f| matches!(f.field_type(), FieldType::Vector { .. } | FieldType::MultiVector { .. } | FieldType::SparseVector { .. }))
        .map(|f| f.name())
        .collect();

    // O(1) per-hit lookup using each shard's UUID → doc_id index. The
    // SegmentManager.lookup_record helper walks growing-segment HashMap
    // then sealed-segment per-segment HashMap, returning a fully
    // materialized Record. No more collection-wide scans.
    for hit in hits.iter_mut() {
        let Ok(id) = hit.id.parse::<uuid::Uuid>() else {
            continue;
        };
        let mut record: Option<veda_core::record::Record> = None;
        for sm in shard_map.values() {
            if let Some(rec) = sm.lookup_record(id) {
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
            if let Some(value) = rec.get(name)
                && let Ok(json) = serde_json::to_value(value)
            {
                map.insert(name.to_string(), json);
            }
        }
        hit.fields = map;
    }
}

fn execute_single_vector(
    executor: &DistributedQueryExecutor,
    vec_query: &SingleVectorQuery,
    req: &SearchRequest,
    metric: DistanceMetric,
) -> Result<(Vec<SearchHit>, SearchTimings), (StatusCode, Json<ErrorResponse>)> {
    let request = VectorSearchRequest {
        field: vec_query.field.clone(),
        query_vector: vec_query.values.clone(),
        k: req.limit,
        metric,
        filter: req.filter.clone(),
        truncate_to: req.truncate_to,
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

/// Run a BM25 full-text search across all sealed segments of every shard
/// in the collection. Each segment scores independently; we merge the
/// per-segment top-k by raw BM25 score (no further normalization) and
/// take the global top-k. Returns hits with `distance = -bm25_score`
/// so that lower-is-better stays consistent across query types.
async fn execute_bm25(
    state: &AppState,
    collection: &str,
    query: &Bm25Query,
    k: usize,
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

    // (record_uuid, bm25_score) merged across shards/segments. We do not
    // currently score against growing segments — the growing-segment BM25
    // story is a follow-up; sealed segments cover the post-ingest steady
    // state which is what the benchmark measures.
    let mut merged: Vec<(uuid::Uuid, f32)> = Vec::new();
    for sm in shard_map.values() {
        for handle in sm.sealed_segments() {
            let Some(idx) = handle.reader.bm25_index(&query.field) else {
                continue;
            };
            let scored = idx.search(&query.query, k.max(1));
            for (doc_id, score) in scored {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
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

/// Dense + BM25 hybrid via Reciprocal Rank Fusion. Each leg runs the
/// full top-`k` retrieval; results are folded together via
/// `RRF(d) = sum_l (w_l / (k_rrf + rank_l(d)))` with `k_rrf = 60`.
/// Returns at most `k` hits.
async fn execute_hybrid_dense_bm25(
    state: &AppState,
    collection: &str,
    executor: &DistributedQueryExecutor,
    hybrid: &HybridQuery,
    req: &SearchRequest,
    metric: DistanceMetric,
    k: usize,
) -> Result<Vec<SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    const K_RRF: f32 = 60.0;
    let dense_w = hybrid.dense_weight.clamp(0.0, 1.0);
    let bm25_w = 1.0 - dense_w;

    let (dense_hits, _t) = execute_single_vector(
        executor,
        &hybrid.dense,
        req,
        metric,
    )?;
    let bm25_hits = execute_bm25(state, collection, &hybrid.bm25, k.max(req.limit)).await?;

    // RRF: rank ids in each list, sum weighted reciprocal-rank.
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
    Ok(hits)
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
            ))
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

/// Resolve the distance metric for a field from the schema, defaulting to Cosine.
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
