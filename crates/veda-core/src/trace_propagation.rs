//! Outbound W3C Trace Context propagation.
//!
//! Veda is a client of several HTTP services (LLM, embedding, rerank
//! endpoints). To stitch those downstream calls into the same trace as
//! the inbound request, we inject the current span's OTel context as
//! `traceparent` / `tracestate` headers before sending.
//!
//! This is a safe no-op when no OTLP exporter / propagator is
//! configured: the global propagator defaults to a no-op that writes
//! nothing, and an empty/invalid current context produces no headers.

use opentelemetry::propagation::Injector;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

/// Adapter letting the OTel `TextMapPropagator` write into a reqwest
/// `HeaderMap`.
pub struct HeaderInjector<'a>(pub &'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, val);
        }
    }
}

/// Inject the current span's OTel context into `headers` as W3C
/// `traceparent` / `tracestate`.
///
/// No-op when there is no active sampled context or no propagator is
/// configured.
pub fn inject_current_context(headers: &mut HeaderMap) {
    let cx = current_context();
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&cx, &mut HeaderInjector(headers));
    });
}

/// The current OTel context, bridged from the active `tracing` span when
/// `tracing-opentelemetry` is installed.
fn current_context() -> opentelemetry::Context {
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    tracing::Span::current().context()
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::Context;
    use opentelemetry::trace::{
        SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState,
    };
    use reqwest::header::HeaderMap;

    fn set_w3c_propagator() {
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );
    }

    #[test]
    fn injects_well_formed_traceparent_under_known_context() {
        set_w3c_propagator();

        let trace_id = TraceId::from_hex("0af7651916cd43dd8448eb211c80319c").unwrap();
        let span_id = SpanId::from_hex("b7ad6b7169203331").unwrap();
        let span_cx = SpanContext::new(
            trace_id,
            span_id,
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        );
        let cx = Context::new().with_remote_span_context(span_cx);
        let _guard = cx.attach();

        let mut headers = HeaderMap::new();
        opentelemetry::global::get_text_map_propagator(|p| {
            p.inject_context(&Context::current(), &mut HeaderInjector(&mut headers));
        });

        let tp = headers
            .get("traceparent")
            .expect("traceparent set")
            .to_str()
            .unwrap();
        // W3C format: 00-<32 hex>-<16 hex>-<2 hex>
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent has 4 fields: {tp}");
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[1], "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(parts[2].len(), 16);
        assert_eq!(parts[2], "b7ad6b7169203331");
        assert_eq!(parts[3].len(), 2);
    }

    #[test]
    fn no_context_does_not_panic_and_sets_no_traceparent() {
        set_w3c_propagator();
        // No active span context: empty current context.
        let mut headers = HeaderMap::new();
        inject_current_context(&mut headers);
        assert!(headers.get("traceparent").is_none());
    }
}
