//! Adaptive query routing endpoint.
//!
//! `POST /v1/route` accepts a natural-language query and returns a
//! recommended retrieval strategy (sparse/dense/hybrid + rerank + suggested
//! `k`). Pure heuristic — the caller can ignore the recommendation; it's
//! just a fast first-pass hint.

use axum::Json;
use serde::Deserialize;

use veda_query::adaptive::{RouteRecommendation, route_query};

#[derive(Debug, Deserialize)]
pub struct RouteRequest {
    pub query: String,
}

pub async fn route(Json(req): Json<RouteRequest>) -> Json<RouteRecommendation> {
    Json(route_query(&req.query))
}
