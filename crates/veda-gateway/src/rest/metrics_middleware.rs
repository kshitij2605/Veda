use std::time::Instant;

use axum::extract::Request;
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use opentelemetry::propagation::Extractor;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use veda_core::metrics as m;

/// Adapter that lets the OTel propagator read from axum HeaderMap.
struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Middleware that records HTTP request count and duration for all endpoints.
pub async fn request_metrics(req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    // Skip metrics for health/metrics/readiness endpoints to avoid noise.
    if path == "/v1/admin/health" || path == "/v1/admin/metrics" || path == "/v1/admin/readiness" {
        return next.run(req).await;
    }

    // Extract W3C traceparent from incoming headers and attach to current span.
    let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(req.headers()))
    });
    tracing::Span::current().set_parent(parent_cx);

    let start = Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16().to_string();

    metrics::counter!(m::REQUESTS_TOTAL, "method" => method.clone(), "endpoint" => path.clone(), "status" => status.clone())
        .increment(1);
    metrics::histogram!(m::REQUEST_DURATION, "method" => method, "endpoint" => path)
        .record(start.elapsed().as_secs_f64());

    // Record errors for 4xx/5xx responses.
    let status_code = response.status().as_u16();
    if status_code >= 400 {
        let error_type = if status_code >= 500 { "server_error" } else { "client_error" };
        metrics::counter!(m::ERRORS_TOTAL, "type" => error_type).increment(1);
    }

    response
}
