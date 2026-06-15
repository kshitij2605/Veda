//! Admin REST API for Log Insights.
//!
//! Admin-only — registered under the same protected router as
//! `/v1/admin/insights`, with an extra `rbac::require_admin` route layer.
//!
//!   GET /v1/admin/logs/categories — semantic categories of recent
//!                                    WARN/ERROR log messages.
//!
//! Later phases (task 3) add a raw log-listing endpoint to this file.

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::logs::categorize::CategorizeOptions;
use crate::logs::LogCategory;
use crate::rest::state::AppState;

#[derive(Debug, Serialize)]
pub struct CategoriesResponse {
    pub categories: Vec<LogCategory>,
}

/// `GET /v1/admin/logs/categories` — cluster recent WARN+ERROR log
/// messages into semantic categories. Uses the configured text
/// embedding model when available, else an embedder-free fallback.
pub async fn list_categories(State(state): State<AppState>) -> Json<CategoriesResponse> {
    let registry = state.embedding_registry().read().await;
    let categories = state
        .log_categorizer()
        .categorize(state.log_buffer(), &registry, CategorizeOptions::default())
        .await;
    Json(CategoriesResponse { categories })
}
