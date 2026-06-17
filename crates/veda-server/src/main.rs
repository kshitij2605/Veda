use veda_gateway::server::{ServerConfig, start_server};

/// Map the standard OTel sampler env vars onto an `opentelemetry_sdk` `Sampler`.
///
/// `name` is `OTEL_TRACES_SAMPLER`, `arg` is `OTEL_TRACES_SAMPLER_ARG`.
/// Unknown / unset names fall back to the OTel default
/// (`parentbased_traceidratio`); a missing or unparseable ratio arg
/// defaults to `1.0`, so the unset-env case samples everything — i.e. no
/// change from the previous SDK-default behaviour.
fn resolve_sampler(
    name: Option<String>,
    arg: Option<String>,
) -> opentelemetry_sdk::trace::Sampler {
    use opentelemetry_sdk::trace::Sampler;

    let ratio = arg
        .as_deref()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(1.0);

    match name.as_deref().map(str::trim) {
        Some("always_on") => Sampler::AlwaysOn,
        Some("always_off") => Sampler::AlwaysOff,
        Some("traceidratio") => Sampler::TraceIdRatioBased(ratio),
        Some("parentbased_always_on") => Sampler::ParentBased(Box::new(Sampler::AlwaysOn)),
        Some("parentbased_always_off") => Sampler::ParentBased(Box::new(Sampler::AlwaysOff)),
        // "parentbased_traceidratio" and any unknown/unset name → default.
        _ => Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(ratio))),
    }
}

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

    // Log Insights capture buffer. The same `Arc` feeds both the
    // tracing CaptureLayer (below) and `AppState.log_buffer` (via
    // ServerConfig) so events the layer captures are queryable. The
    // capture threshold defaults to WARN; override with
    // `VEDA_LOG_CAPTURE_LEVEL` (ERROR|WARN|INFO|DEBUG|TRACE).
    let log_buffer = std::sync::Arc::new(veda_gateway::logs::LogBuffer::with_default_capacity());
    let capture_level = match std::env::var("VEDA_LOG_CAPTURE_LEVEL")
        .ok()
        .map(|s| s.trim().to_ascii_uppercase())
        .as_deref()
    {
        Some("ERROR") => tracing::Level::ERROR,
        Some("WARN") => tracing::Level::WARN,
        Some("INFO") => tracing::Level::INFO,
        Some("DEBUG") => tracing::Level::DEBUG,
        Some("TRACE") => tracing::Level::TRACE,
        _ => tracing::Level::WARN,
    };

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

        let sampler = resolve_sampler(
            std::env::var("OTEL_TRACES_SAMPLER").ok(),
            std::env::var("OTEL_TRACES_SAMPLER_ARG").ok(),
        );

        let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_sampler(sampler)
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
            .with(veda_gateway::logs::CaptureLayer::new(log_buffer.clone(), capture_level))
            .init();

        // Enable W3C Trace Context propagation for incoming traceparent headers.
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );

        eprintln!("OpenTelemetry export enabled to {endpoint}");
    } else {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let fmt_layer = tracing_subscriber::fmt::layer().json();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(veda_gateway::logs::CaptureLayer::new(log_buffer.clone(), capture_level))
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
        log_buffer,
    };

    if let Err(e) = start_server(config).await {
        tracing::error!(error = %e, "Server failed");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::resolve_sampler;

    fn dbg(name: Option<&str>, arg: Option<&str>) -> String {
        format!(
            "{:?}",
            resolve_sampler(name.map(str::to_owned), arg.map(str::to_owned))
        )
    }

    #[test]
    fn default_when_unset_is_parentbased_ratio_1() {
        assert_eq!(dbg(None, None), "ParentBased(TraceIdRatioBased(1.0))");
    }

    #[test]
    fn garbage_name_falls_back_to_default() {
        assert_eq!(
            dbg(Some("not-a-sampler"), None),
            "ParentBased(TraceIdRatioBased(1.0))"
        );
    }

    #[test]
    fn always_on_and_off() {
        assert_eq!(dbg(Some("always_on"), None), "AlwaysOn");
        assert_eq!(dbg(Some("always_off"), None), "AlwaysOff");
    }

    #[test]
    fn traceidratio_uses_arg() {
        assert_eq!(dbg(Some("traceidratio"), Some("0.1")), "TraceIdRatioBased(0.1)");
    }

    #[test]
    fn parentbased_traceidratio_uses_arg() {
        assert_eq!(
            dbg(Some("parentbased_traceidratio"), Some("0.1")),
            "ParentBased(TraceIdRatioBased(0.1))"
        );
    }

    #[test]
    fn ratio_arg_defaults_to_one_when_missing_or_garbage() {
        assert_eq!(
            dbg(Some("traceidratio"), None),
            "TraceIdRatioBased(1.0)"
        );
        assert_eq!(
            dbg(Some("traceidratio"), Some("garbage")),
            "TraceIdRatioBased(1.0)"
        );
    }
}
