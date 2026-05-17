use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::auth;

pub mod admin;
pub mod collections;
pub mod document;
pub mod embed_insert;
pub mod evaluate;
pub mod metrics_middleware;
pub mod records;
pub mod rerank;
pub mod search;
pub mod state;

/// Build the complete REST API router.
pub fn build_router(app_state: state::AppState) -> Router {
    // Public routes — no auth required.
    let public = Router::new()
        .route("/v1/admin/health", get(admin::health))
        .route("/v1/admin/readiness", get(admin::readiness))
        .route("/v1/auth/login", post(admin::login));

    // Protected routes — auth middleware applied when auth_enabled=true.
    let protected = Router::new()
        // Admin
        .route("/v1/admin/nodes", get(admin::list_nodes))
        .route("/v1/admin/metrics", get(admin::metrics))
        // Collections
        .route("/v1/collections", post(collections::create_collection))
        .route("/v1/collections", get(collections::list_collections))
        .route(
            "/v1/collections/{name}",
            get(collections::describe_collection),
        )
        .route(
            "/v1/collections/{name}",
            delete(collections::drop_collection),
        )
        // Search
        .route("/v1/collections/{name}/search", post(search::search))
        // Records
        .route("/v1/collections/{name}/insert", post(records::insert))
        .route(
            "/v1/collections/{name}/embed-insert",
            post(embed_insert::embed_insert),
        )
        .route(
            "/v1/collections/{name}/records/{id}",
            get(records::get_record),
        )
        .route(
            "/v1/collections/{name}/records",
            get(records::list_records).delete(records::delete_records),
        )
        // Data quality / duplicate evaluation
        .route(
            "/v1/collections/{name}/evaluate",
            get(evaluate::evaluate),
        )
        // Cross-encoder rerank
        .route(
            "/v1/collections/{name}/rerank",
            post(rerank::rerank),
        )
        // Document processing (OCR)
        .route(
            "/v1/collections/{name}/process-document",
            post(document::process_document),
        )
        .route(
            "/v1/collections/{name}/process-documents",
            post(document::process_documents_batch),
        )
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth::auth_middleware,
        ));

    public
        .merge(protected)
        .layer(axum::middleware::from_fn(metrics_middleware::request_metrics))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024)) // 256MB max body
        .with_state(app_state)
}
