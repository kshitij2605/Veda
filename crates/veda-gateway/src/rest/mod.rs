use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{delete, get, post};
use tower_http::trace::TraceLayer;

use crate::auth;
use crate::rbac;
use crate::tenant;

pub mod a2a;
pub mod admin;
pub mod agents;
pub mod aggregate;
pub mod alias;
pub mod changepoint;
pub mod chunk_preview;
pub mod collections;
pub mod connectors;
pub mod correlations;
pub mod crawl4ai;
pub mod depmap;
pub mod document;
pub mod embed;
pub mod embed_insert;
pub mod evaluate;
pub mod forecast;
pub mod incidents;
pub mod index_advisor;
pub mod ingest_common;
pub mod ingest_logs;
pub mod ingest_url;
pub mod insights;
pub mod jobs;
pub mod llm;
pub mod llm_traces;
pub mod log_rate_analysis;
pub mod logs;
pub mod mcp;
pub mod media;
pub mod metrics_middleware;
#[cfg(feature = "profiling")]
pub mod pprof;
pub mod query_log;
pub mod rag_eval;
pub mod records;
pub mod rerank;
pub mod resources;
pub mod route;
pub mod saved_objects;
pub mod search;
pub mod searxng;
pub mod slos;
pub mod state;
pub mod synthetics;
pub mod time_parse;
pub mod tuning;
pub mod txn;
pub mod url_guard;

/// Ops-advisor admin routes, gated by an extra `require_admin` route
/// layer (the shared auth middleware in `build_router` still runs first,
/// populating the `Claims` that `require_admin` reads). State is bound by
/// the caller's `.with_state` at the end of `build_router`.
fn insights_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/insights", get(insights::list_insights))
        .route("/v1/admin/insights/history", get(insights::list_history))
        .route("/v1/admin/insights/rules", get(insights::list_rules))
        .route(
            "/v1/admin/insights/rules/{id}",
            axum::routing::put(insights::put_rule),
        )
        .route(
            "/v1/admin/insights/channels",
            get(insights::get_channels).put(insights::put_channels),
        )
        .route("/v1/admin/insights/run", post(insights::run_now))
        .route(
            "/v1/admin/insights/{event_id}/ack",
            post(insights::ack_event),
        )
        .route(
            "/v1/admin/insights/{event_id}/triage",
            post(insights::triage_event),
        )
        .route(
            "/v1/admin/insights/{event_id}/rca",
            get(insights::event_rca),
        )
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// SLO admin routes, gated by an extra `require_admin` route layer (same
/// pattern as [`insights_router`]).
fn slos_router() -> Router<state::AppState> {
    Router::new()
        .route(
            "/v1/admin/slos",
            get(slos::list_slos).post(slos::create_slo),
        )
        .route(
            "/v1/admin/slos/{id}",
            get(slos::get_slo)
                .put(slos::put_slo)
                .delete(slos::delete_slo),
        )
        .route("/v1/admin/slos/{id}/status", get(slos::get_slo_status))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Synthetic self-check probe admin routes, gated by an extra
/// `require_admin` route layer (same pattern as [`slos_router`]).
fn synthetics_router() -> Router<state::AppState> {
    Router::new()
        .route(
            "/v1/admin/synthetics",
            get(synthetics::list_synthetics).post(synthetics::create_synthetic),
        )
        .route(
            "/v1/admin/synthetics/{id}",
            get(synthetics::get_synthetic)
                .put(synthetics::put_synthetic)
                .delete(synthetics::delete_synthetic),
        )
        .route(
            "/v1/admin/synthetics/{id}/results",
            get(synthetics::get_synthetic_results),
        )
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Data-source connectors admin routes, gated by an extra `require_admin`
/// route layer (same pattern as [`synthetics_router`]). Tenant-scoped.
fn connectors_router() -> Router<state::AppState> {
    Router::new()
        .route(
            "/v1/admin/connectors",
            get(connectors::list_connectors).post(connectors::create_connector),
        )
        .route(
            "/v1/admin/connectors/{id}",
            get(connectors::get_connector)
                .put(connectors::put_connector)
                .delete(connectors::delete_connector),
        )
        .route(
            "/v1/admin/connectors/{id}/run",
            post(connectors::run_connector),
        )
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Log Insights admin routes, gated by an extra `require_admin` route
/// layer (same pattern as [`insights_router`]). Task 3 extends this with
/// a raw log-listing endpoint.
fn logs_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/logs", get(logs::list_logs))
        .route("/v1/admin/logs/categories", get(logs::list_categories))
        .route("/v1/admin/slow-queries", get(logs::list_slow_queries))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Node-resources admin route, gated by an extra `require_admin` route
/// layer (same pattern as [`insights_router`]).
fn resources_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/resources", get(resources::get_resources))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Slow-query / error correlation analyzer admin route, gated by an
/// extra `require_admin` route layer (same pattern as [`logs_router`]).
fn correlations_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/correlations", get(correlations::correlations))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Self-service dependency-map admin route, gated by an extra
/// `require_admin` route layer (same pattern as [`correlations_router`]).
fn depmap_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/depmap", get(depmap::depmap))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// ANN index-advisor admin route, gated by an extra `require_admin` route
/// layer (same pattern as [`depmap_router`]).
fn index_advisor_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/index-advisor", get(index_advisor::index_advisor))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Capacity-forecast admin route, gated by an extra `require_admin` route
/// layer (same pattern as [`index_advisor_router`]).
fn forecast_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/forecast", get(forecast::forecast))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Change-point admin route, gated by an extra `require_admin` route
/// layer (same pattern as [`forecast_router`]).
fn changepoint_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/change-points", get(changepoint::change_points))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Log-rate-analysis admin route (semantic spike root-cause attribution),
/// gated by an extra `require_admin` route layer (same pattern as
/// [`changepoint_router`]).
fn log_rate_analysis_router() -> Router<state::AppState> {
    Router::new()
        .route(
            "/v1/admin/log-rate-analysis",
            get(log_rate_analysis::log_rate_analysis),
        )
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Semantic incident-correlation admin route, gated by an extra
/// `require_admin` route layer (same pattern as [`index_advisor_router`]).
fn incidents_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/incidents", get(incidents::list_incidents))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Searchable semantic LLM trace store admin routes, gated by an extra
/// `require_admin` route layer (same pattern as [`incidents_router`]).
fn llm_traces_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/llm-traces", get(llm_traces::search_traces))
        .route(
            "/v1/admin/llm-traces/clusters",
            get(llm_traces::cluster_traces),
        )
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// On-demand CPU-profiling admin route, gated by an extra
/// `require_admin` route layer (same pattern as [`resources_router`]).
/// Compiled only under the `profiling` cargo feature.
#[cfg(feature = "profiling")]
fn pprof_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/pprof", get(pprof::pprof))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// PLAT-2: core operator/admin routes, gated by a `require_admin`
/// route layer (same pattern as [`insights_router`] et al.). Mounted
/// into the protected router below.
fn core_admin_router() -> Router<state::AppState> {
    Router::new()
        .route("/v1/admin/nodes", get(admin::list_nodes))
        .route("/v1/admin/metrics", get(admin::metrics))
        .route(
            "/v1/admin/region_topology",
            get(admin::get_region_topology).put(admin::put_region_topology),
        )
        .route(
            "/v1/admin/region_topology/promote",
            post(admin::promote_region),
        )
        .route(
            "/v1/admin/replica/safe_ts",
            get(admin::get_replica_safe_ts).post(admin::set_replica_safe_ts),
        )
        .route("/v1/admin/cluster/join", post(admin::cluster_join))
        .route(
            "/v1/admin/cluster/nodes/{id}",
            delete(admin::cluster_remove),
        )
        .route("/v1/admin/cluster/noop", post(admin::cluster_noop))
        .route(
            "/v1/admin/collections/{name}/mvcc_checkpoint",
            post(admin::mvcc_checkpoint),
        )
        .route(
            "/v1/admin/jobs",
            get(jobs::list_jobs).post(jobs::create_job),
        )
        .route("/v1/admin/jobs/{name}", delete(jobs::delete_job))
        .route_layer(middleware::from_fn(rbac::require_admin))
}

/// Build the complete REST API router.
pub fn build_router(app_state: state::AppState) -> Router {
    // Public routes — no auth required.
    let public = Router::new()
        .route("/v1/admin/health", get(admin::health))
        .route("/v1/admin/readiness", get(admin::readiness))
        .route("/v1/auth/login", post(admin::login));

    // Protected routes — auth middleware applied when auth_enabled=true.
    let protected = Router::new()
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
        // Phase G: atomic re-chunk swap. Creates name@vN and
        // points the alias at it.
        .route(
            "/v1/collections/{name}/reindex",
            post(collections::reindex_collection),
        )
        // Phase 4 (auto-tuning): trigger / list bounded search-space
        // campaigns for a collection (recommend-only; actually applying a
        // result still goes through /reindex, see Phase 5).
        .route(
            "/v1/collections/{name}/tuning-campaigns",
            get(tuning::list_collection_campaigns).post(tuning::start_campaign),
        )
        // Phase 5: safe-apply a scored trial — tier1-only diffs are
        // instant, tier2/3 diffs go through /reindex + alias-swap (never
        // overwrites the original collection).
        .route(
            "/v1/tuning-campaigns/{campaign_id}/trials/{trial_id}/apply",
            post(tuning::apply_trial),
        )
        // Force-seal the growing segment(s) so the configured vector index
        // type builds (small datasets otherwise never reach the seal
        // threshold and always brute-force with Flat).
        .route(
            "/v1/collections/{name}/seal",
            post(collections::seal_collection),
        )
        // Force-merge all sealed segments per shard down to one, then rebuild
        // executors — operator-facing recall/latency optimize for a fragmented
        // bulk load (akin to a Lucene force-merge).
        .route(
            "/v1/collections/{name}/optimize",
            post(collections::optimize_collection),
        )
        // Phase G: read / set the alias for a collection.
        .route(
            "/v1/admin/collections/{name}/alias",
            get(collections::get_alias).put(collections::set_alias),
        )
        // Search
        .route("/v1/collections/{name}/search", post(search::search))
        // Coverage rollup over the per-collection query log
        .route("/v1/collections/{name}/coverage", get(query_log::coverage))
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
            axum::routing::put(saved_objects::upsert).delete(saved_objects::delete),
        )
        // Records
        .route("/v1/collections/{name}/insert", post(records::insert))
        .route(
            "/v1/collections/{name}/embed-insert",
            post(embed_insert::embed_insert),
        )
        // Logs-as-a-collection ingest — same auth/tenant as /insert
        // (a WRITE, not admin-only). Parses raw log lines and inserts
        // them through the insert / embed-insert path.
        .route(
            "/v1/collections/{name}/ingest-logs",
            post(ingest_logs::ingest_logs),
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
        .route("/v1/collections/{name}/evaluate", get(evaluate::evaluate))
        // Analytics aggregations (Feature E)
        .route(
            "/v1/collections/{name}/aggregate",
            get(aggregate::aggregate),
        )
        .route(
            "/v1/collections/{name}/histogram",
            get(aggregate::histogram),
        )
        .route(
            "/v1/collections/{name}/field-stats",
            get(aggregate::field_stats),
        )
        // Cross-encoder rerank
        .route("/v1/collections/{name}/rerank", post(rerank::rerank))
        // Model Context Protocol (MCP) for AI agents — Streamable HTTP
        // transport: POST carries a JSON-RPC message; GET (server->client
        // SSE) is unsupported (stateless) and returns 405.
        .route("/v1/mcp", get(mcp::handle_mcp_get).post(mcp::handle_mcp))
        // Adaptive query router (recommends retrieval strategy for a query)
        .route("/v1/route", post(route::route))
        // Veda Agents — CRUD + tool catalog/execute + converse + conversations
        .route(
            "/v1/agents",
            get(agents::list_agents).post(agents::create_agent),
        )
        .route(
            "/v1/agents/{id}",
            get(agents::get_agent)
                .put(agents::put_agent)
                .delete(agents::delete_agent),
        )
        .route("/v1/agents/{id}/converse", post(agents::converse))
        // Streaming converse over Server-Sent Events (additive; blocking
        // /converse above is unchanged). Same auth/tenant/RLS.
        .route(
            "/v1/agents/{id}/converse/stream",
            post(agents::converse_stream),
        )
        // Agent-to-Agent (A2A) — discovery card + sync JSON-RPC message/send
        .route("/v1/a2a/{agent_id}/card", get(a2a::agent_card))
        .route("/v1/a2a/{agent_id}", post(a2a::message_send))
        .route(
            "/v1/agent-tools",
            get(agents::list_tools).post(agents::create_tool),
        )
        .route("/v1/agent-tools/catalog", get(agents::catalog))
        .route(
            "/v1/agent-tools/{id}",
            get(agents::get_tool)
                .put(agents::put_tool)
                .delete(agents::delete_tool),
        )
        .route("/v1/agent-tools/{id}/execute", post(agents::execute_tool))
        .route("/v1/conversations", get(agents::list_conversations))
        .route(
            "/v1/conversations/{id}",
            get(agents::get_conversation).delete(agents::delete_conversation),
        )
        .route(
            "/v1/conversations/{id}/feedback",
            post(agents::submit_feedback),
        )
        // RAG/agent eval harness + production-trace flywheel. Tenant/RLS-
        // scoped like /converse (not admin-only): eval runs under the
        // caller's identity.
        .route(
            "/v1/eval-sets",
            get(rag_eval::list_eval_sets).post(rag_eval::create_eval_set),
        )
        .route(
            "/v1/eval-sets/{id}",
            get(rag_eval::get_eval_set)
                .put(rag_eval::put_eval_set)
                .delete(rag_eval::delete_eval_set),
        )
        .route("/v1/eval-sets/{id}/run", post(rag_eval::run_eval))
        .route("/v1/eval-sets/{id}/runs", get(rag_eval::list_runs))
        .route(
            "/v1/eval-sets/{id}/cases/from-conversation",
            post(rag_eval::case_from_conversation),
        )
        .route(
            "/v1/collections/{name}/eval-sets/generate",
            post(rag_eval::generate_eval_set),
        )
        .route(
            "/v1/collections/{name}/eval-set-size-recommendation",
            get(rag_eval::eval_set_size_recommendation),
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
        // Web ingestion (crawl4ai) — crawl URL(s) -> markdown -> chunk/embed/
        // insert. Tenant/RLS-scoped like process-document; opt-in (503 when
        // VEDA_CRAWL4AI_ENDPOINT is unset).
        .route(
            "/v1/collections/{name}/ingest-url",
            post(ingest_url::ingest_url),
        )
        // Media ingestion — store an image/audio/video file as a blob, embed
        // it via the omni multimodal model (one shared vector space), and
        // insert one record. Tenant/RLS-scoped like process-document.
        .route(
            "/v1/collections/{name}/ingest-media",
            post(media::ingest_media_route),
        )
        // Ops advisor ("Insights") — admin-only (extra require_admin
        // route layer on top of the shared auth middleware below).
        .merge(insights_router())
        // SLOs — admin-only (extra require_admin route layer).
        .merge(slos_router())
        // Synthetic self-check probes — admin-only (extra require_admin).
        .merge(synthetics_router())
        // Data-source connectors — admin-only (extra require_admin).
        .merge(connectors_router())
        // Log Insights — admin-only (extra require_admin route layer).
        .merge(logs_router())
        // Node resources — admin-only (extra require_admin route layer).
        .merge(resources_router())
        // Slow-query / error correlations — admin-only (extra require_admin).
        .merge(correlations_router())
        // Self-service dependency map — admin-only (extra require_admin).
        .merge(depmap_router())
        // ANN index advisor — admin-only (extra require_admin).
        .merge(index_advisor_router())
        // Capacity forecasts — admin-only (extra require_admin).
        .merge(forecast_router())
        // Change-point detection — admin-only (extra require_admin).
        .merge(changepoint_router())
        // Semantic spike root-cause attribution — admin-only (extra require_admin).
        .merge(log_rate_analysis_router())
        // Semantic incident correlation — admin-only (extra require_admin).
        .merge(incidents_router())
        // Searchable semantic LLM trace store — admin-only (extra require_admin).
        .merge(llm_traces_router())
        // PLAT-2 (2026-06-21): core admin routes (nodes / metrics /
        // region topology / replica watermark / raft membership /
        // mvcc checkpoint / scheduled jobs). Previously added to
        // `protected` with only the shared auth middleware, so any
        // AUTHENTICATED user (even a Viewer) could reach them. Now
        // behind a `require_admin` route layer like the other admin
        // sub-routers. (Auth-disabled deployments still pass: with
        // no Claims in extensions, `require_role` is a no-op.)
        .merge(core_admin_router());

    // On-demand CPU profiling — admin-only, only present under the
    // `profiling` cargo feature (no route, no `pprof` dep otherwise).
    #[cfg(feature = "profiling")]
    let protected = protected.merge(pprof_router());

    let protected = protected
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
        .layer(axum::middleware::from_fn(
            metrics_middleware::request_metrics,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024)) // 256MB max body
        .with_state(app_state)
}
