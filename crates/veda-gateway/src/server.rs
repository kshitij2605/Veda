use std::path::PathBuf;

use crate::rest::{build_router, state::AppState};

/// Configuration for the Veda API server.
pub struct ServerConfig {
    pub listen_addr: String,
    pub data_dir: PathBuf,
}

/// Start the HTTP server and block until shutdown.
pub async fn start_server(config: ServerConfig) -> std::io::Result<()> {
    let app_state = AppState::new(&config.data_dir);
    app_state.load_embedding_from_env().await;
    let router = build_router(app_state);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    tracing::info!("Veda listening on {}", config.listen_addr);
    axum::serve(listener, router).await?;
    Ok(())
}
