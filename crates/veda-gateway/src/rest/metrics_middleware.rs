use std::time::Instant;

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use veda_core::metrics as m;

/// Middleware that records HTTP request count and duration for all endpoints.
pub async fn request_metrics(req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    // Skip metrics for health/metrics/readiness endpoints to avoid noise.
    if path == "/v1/admin/health" || path == "/v1/admin/metrics" || path == "/v1/admin/readiness" {
        return next.run(req).await;
    }

    let start = Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16().to_string();

    metrics::counter!(m::REQUESTS_TOTAL, "method" => method.clone(), "endpoint" => path.clone(), "status" => status)
        .increment(1);
    metrics::histogram!(m::REQUEST_DURATION, "method" => method, "endpoint" => path)
        .record(start.elapsed().as_secs_f64());

    response
}
