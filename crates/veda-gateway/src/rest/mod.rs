use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::auth;
use crate::tenant;

pub mod admin;
pub mod chunk_preview;
pub mod collections;
pub mod document;
pub mod embed;
pub mod embed_insert;
pub mod llm;
pub mod evaluate;
pub mod mcp;
pub mod metrics_middleware;
pub mod query_log;
pub mod records;
pub mod rerank;
pub mod route;
pub mod saved_objects;
pub mod search;
pub mod state;
pub mod txn;

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
        // Raft membership API (Phase 6).
        .route(
            "/v1/admin/cluster/join",
            post(admin::cluster_join),
        )
        .route(
            "/v1/admin/cluster/nodes/{id}",
            delete(admin::cluster_remove),
        )
        // Submit a Noop command — drives the Raft log for test
        // harnesses (snapshot soak, load gen).
        .route(
            "/v1/admin/cluster/noop",
            post(admin::cluster_noop),
        )
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
        // Coverage rollup over the per-collection query log
        .route(
            "/v1/collections/{name}/coverage",
            get(query_log::coverage),
        )
        // One-shot text/image -> vector for the playground composer
        .route("/v1/embed", post(embed::embed))
        // LLM chat-completion proxy (cluster labels / NL→DSL / failure classifier)
        .route("/v1/llm/complete", post(llm::complete))
        // Chunker A/B preview (no storage / no embedding)
        .route("/v1/chunk/preview", post(chunk_preview::preview))
        // Transactions v0 (Raft-only; non-production, see
        // crates/veda-gateway/src/rest/txn.rs).
        .route("/v1/txn", post(txn::begin))
        .route("/v1/txn/{txn_id}/commit", post(txn::commit))
        .route("/v1/txn/{txn_id}/abort", post(txn::abort))
        // SavedObjects — Phase 6.5 follow-up: server-side persistence
        // for searches / atlas views / dashboards / etc. so they
        // survive a browser/localStorage wipe.
        .route("/v1/saved_objects", get(saved_objects::list))
        .route(
            "/v1/saved_objects/{id}",
            axum::routing::put(saved_objects::upsert)
                .delete(saved_objects::delete),
        )
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
        // Model Context Protocol (MCP) for AI agents
        .route("/v1/mcp", post(mcp::handle_mcp))
        // Adaptive query router (recommends retrieval strategy for a query)
        .route("/v1/route", post(route::route))
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
        ))
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            tenant::tenant_middleware,
        ));

    public
        .merge(protected)
        .layer(axum::middleware::from_fn(metrics_middleware::request_metrics))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024)) // 256MB max body
        .with_state(app_state)
}
