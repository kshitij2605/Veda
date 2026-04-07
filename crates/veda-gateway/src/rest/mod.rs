use axum::routing::{delete, get, post};
use axum::Router;

pub mod admin;
pub mod collections;
pub mod records;
pub mod search;
pub mod state;

/// Build the complete REST API router.
pub fn build_router(app_state: state::AppState) -> Router {
    Router::new()
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
            "/v1/collections/{name}/records/{id}",
            get(records::get_record),
        )
        .route(
            "/v1/collections/{name}/records",
            delete(records::delete_records),
        )
        // Admin
        .route("/v1/admin/health", get(admin::health))
        .route("/v1/admin/nodes", get(admin::list_nodes))
        // State
        .with_state(app_state)
}
