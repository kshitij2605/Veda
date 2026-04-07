use veda_gateway::server::{ServerConfig, start_server};

#[tokio::main]
async fn main() {
    // Load .env file if present.
    if let Ok(contents) = std::fs::read_to_string(".env") {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                // SAFETY: called before any threads are spawned.
                unsafe { std::env::set_var(key.trim(), value.trim()) };
            }
        }
    }

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .json()
        .init();

    // Parse command line args (simple, no clap needed yet)
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args
        .get(1)
        .map(|s| std::path::PathBuf::from(s))
        .unwrap_or_else(|| std::path::PathBuf::from("./veda-data"));

    let listen_addr = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "0.0.0.0:6340".to_string());

    // Print banner
    eprintln!("╔═══════════════════════════════════════╗");
    eprintln!("║         Veda Vector Database           ║");
    eprintln!("║         v0.1.0                         ║");
    eprintln!("╠═══════════════════════════════════════╣");
    eprintln!("║  REST API: http://{}  ║", listen_addr);
    eprintln!("║  Data dir: {:?}", data_dir);
    eprintln!("╚═══════════════════════════════════════╝");

    tracing::info!("Starting Veda v0.1.0");
    tracing::info!(data_dir = ?data_dir, listen_addr = %listen_addr);

    // Create data directory
    std::fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    let config = ServerConfig {
        listen_addr,
        data_dir,
    };

    if let Err(e) = start_server(config).await {
        tracing::error!(error = %e, "Server failed");
        std::process::exit(1);
    }
}
