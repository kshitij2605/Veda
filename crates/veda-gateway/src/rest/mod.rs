use axum::middleware;
use axum::routing::{delete, get, post};
use axum::Router;

use crate::auth;

pub mod admin;
pub mod collections;
pub mod document;
pub mod embed_insert;
pub mod records;
pub mod search;
pub mod state;

/// Build the complete REST API router.
pub fn build_router(app_state: state::AppState) -> Router {
    // Public routes — no auth required.
    let public = Router::new()
        .route("/v1/admin/health", get(admin::health))
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
            delete(records::delete_records),
        )
        // Document processing (OCR)
        .route(
            "/v1/collections/{name}/process-document",
            post(document::process_document),
        )
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth::auth_middleware,
        ));

    public.merge(protected).with_state(app_state)
}
