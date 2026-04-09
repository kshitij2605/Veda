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
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
pub struct QueryBody {
    /// Single-vector query.
    pub vector: Option<SingleVectorQuery>,
    /// Multi-vector fusion query.
    pub multi_vector: Option<Vec<MultiVectorField>>,
    /// Fusion strategy name (for multi_vector).
    pub fusion: Option<String>,
    /// Fusion-specific parameters.
    pub fusion_params: Option<FusionParams>,
}

#[derive(Debug, Deserialize)]
pub struct SingleVectorQuery {
    pub field: String,
    pub values: Vec<f32>,
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

#[derive(Debug, Serialize)]
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
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
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
        // Single-vector search
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
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "query must contain 'vector' or 'multi_vector'".into(),
            }),
        ));
    };

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
