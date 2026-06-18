use std::path::PathBuf;
use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusBuilder;

use crate::logs::LogBuffer;
use crate::rest::{build_router, state::AppState};

/// Configuration for the Veda API server.
pub struct ServerConfig {
    pub listen_addr: String,
    pub data_dir: PathBuf,
    /// Shared Log Insights capture buffer. The binary builds this and
    /// installs a [`crate::logs::CaptureLayer`] over the same `Arc`
    /// before calling [`start_server`], so captured events land on
    /// `AppState.log_buffer`.
    pub log_buffer: Arc<LogBuffer>,
}

/// Start the HTTP server and block until shutdown.
pub async fn start_server(config: ServerConfig) -> std::io::Result<()> {
    // Install Prometheus metrics recorder (global).
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let app_state = AppState::with_prometheus_and_log_buffer(
        &config.data_dir,
        prometheus_handle,
        config.log_buffer.clone(),
    );
    app_state.load_embedding_from_env().await;
    app_state.load_splade_from_env().await;
    app_state.load_llm_from_env().await;
    app_state.load_auth_from_env().await;
    app_state.load_raft_from_env().await;
    app_state.load_meta_leased_tso().await;
    let _orphan_handle = app_state.spawn_orphan_resolver();
    app_state.load_raft_rest_peers_from_env().await;
    app_state.spawn_raft_metrics_publisher().await;
    app_state.spawn_raft_reconcile().await;
    app_state.load_blob_store_from_env().await;
    let recovered = app_state.recover_collections_from_disk().await;
    if recovered > 0 {
        tracing::info!(collections = recovered, "recovered collections from data_dir on startup");
    }
    // Phase C: optional per-shard log appliers (VEDA_LOG_APPLIER=1).
    // Single-node default leaves them off — the writer's
    // insert_via_log applies inline. Multi-node follower nodes
    // turn them on; the inline apply is then skipped (see
    // log_appliers::applier_enabled).
    let _appliers = crate::log_appliers::spawn_for_all_known_collections(app_state.clone()).await;

    // Cron-style scheduled-job scheduler tick loop (60s interval).
    let _scheduler = crate::scheduler::spawn_scheduler(app_state.clone());
    tracing::info!("scheduler: tick loop started (60s interval)");

    // Ops-advisor tick loop (default 30s; VEDA_ADVISOR_INTERVAL_SECS).
    let _advisor = crate::advisor::spawn_advisor(app_state.clone());
    tracing::info!("advisor: tick loop started");

    // Proxy recall-probe loop — a SEPARATE, much-slower task (default
    // 600s; VEDA_RECALL_PROBE_INTERVAL_SECS). Runs the expensive
    // ANN-vs-exact brute-force scans OFF the advisor tick so they never
    // stall the advisor's rules/anomaly/notify pipeline.
    let _recall_prober = crate::advisor::spawn_recall_prober(app_state.clone());
    tracing::info!("advisor: recall-probe loop started");

    // Semantic incident-correlation loop — a SEPARATE, slow task (default
    // 300s; VEDA_INCIDENT_INTERVAL_SECS). Embeds + clusters ops events OFF
    // the advisor tick so the embedding network calls never stall the
    // advisor's rules/anomaly/notify pipeline.
    let _incident_correlator = crate::advisor::spawn_incident_correlator(app_state.clone());
    tracing::info!("advisor: incident-correlation loop started");

    // Optional NATS streaming ingestion worker.
    if let Ok(url) = std::env::var("VEDA_STREAMING_NATS")
        && !url.is_empty()
    {
        match crate::streaming::spawn_ingestion_worker(app_state.clone(), url.clone()).await {
            Ok(()) => tracing::info!(url = %url, "streaming ingestion worker started"),
            Err(e) => tracing::error!(
                url = %url,
                error = %e,
                "streaming ingestion worker failed to start"
            ),
        }
    }

    let router = build_router(app_state);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    tracing::info!("Veda listening on {}", config.listen_addr);
    axum::serve(listener, router).await?;
    Ok(())
}
