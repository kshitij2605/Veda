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

    // Initialize tracing with optional OpenTelemetry export.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info".into());

    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok().filter(|s| !s.is_empty());

    if let Some(ref endpoint) = otel_endpoint {
        use opentelemetry::trace::TracerProvider as _;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let service_name = std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "veda".into());

        // Trace exporter
        let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .build()
            .expect("failed to create OTLP trace exporter");

        let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(trace_exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", service_name.clone()),
            ]))
            .build();

        let tracer = tracer_provider.tracer("veda");
        let otel_trace_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        // Log exporter
        let log_exporter = opentelemetry_otlp::LogExporter::builder()
            .with_tonic()
            .build()
            .expect("failed to create OTLP log exporter");

        let log_provider = opentelemetry_sdk::logs::LoggerProvider::builder()
            .with_batch_exporter(log_exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", service_name),
            ]))
            .build();

        let otel_log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&log_provider);

        let fmt_layer = tracing_subscriber::fmt::layer().json();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otel_trace_layer)
            .with(otel_log_layer)
            .init();

        eprintln!("OpenTelemetry export enabled to {endpoint}");
    } else {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let fmt_layer = tracing_subscriber::fmt::layer().json();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }

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
