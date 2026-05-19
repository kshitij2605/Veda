use std::path::PathBuf;

use metrics_exporter_prometheus::PrometheusBuilder;

use crate::rest::{build_router, state::AppState};

/// Configuration for the Veda API server.
pub struct ServerConfig {
    pub listen_addr: String,
    pub data_dir: PathBuf,
}

/// Start the HTTP server and block until shutdown.
pub async fn start_server(config: ServerConfig) -> std::io::Result<()> {
    // Install Prometheus metrics recorder (global).
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let app_state = AppState::with_prometheus(&config.data_dir, prometheus_handle);
    app_state.load_embedding_from_env().await;
    app_state.load_splade_from_env().await;
    app_state.load_llm_from_env().await;
    app_state.load_auth_from_env().await;
    app_state.load_raft_from_env().await;
    app_state.load_raft_rest_peers_from_env().await;
    app_state.spawn_raft_metrics_publisher().await;
    app_state.spawn_raft_reconcile().await;
    app_state.load_blob_store_from_env().await;
    let recovered = app_state.recover_collections_from_disk().await;
    if recovered > 0 {
        tracing::info!(collections = recovered, "recovered collections from data_dir on startup");
    }

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
