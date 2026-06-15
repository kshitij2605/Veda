//! Log Insights — in-process log capture.
//!
//! This module provides the foundation for the Log Insights feature:
//! a bounded ring [`LogBuffer`] of recent [`LogRecord`]s, fed by a
//! `tracing_subscriber` [`CaptureLayer`]. Later phases add semantic
//! categorization, a REST surface, alerting, and logs-as-a-collection
//! ingest on top of this buffer.

pub mod buffer;
pub mod categorize;
pub mod layer;

pub use buffer::{LogBuffer, LogRecord, DEFAULT_CAPACITY};
pub use categorize::{LogCategorizer, LogCategory};
pub use layer::CaptureLayer;

/// Best-effort read of the current OpenTelemetry trace/span ids from
/// the active tracing span.
///
/// Returns `(trace_id, span_id)` as lower-hex strings when a *valid*
/// (non-zero) trace context is active, else `(None, None)`. This is a
/// no-op (returns `None`) when no OpenTelemetry layer is installed —
/// e.g. when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, or in unit tests
/// — so callers never need a live OTLP endpoint.
///
/// Never panics; safe to call from synchronous, non-blocking contexts.
pub fn current_trace_ids() -> (Option<String>, Option<String>) {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let cx = tracing::Span::current().context();
    let span = cx.span();
    let sc = span.span_context();
    if sc.is_valid() {
        (Some(sc.trace_id().to_string()), Some(sc.span_id().to_string()))
    } else {
        (None, None)
    }
}
