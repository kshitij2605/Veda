//! A `tracing_subscriber::Layer` that captures events into a
//! [`LogBuffer`].
//!
//! [`CaptureLayer`] records every event at or above a configurable
//! minimum level (default `WARN`) into the shared ring buffer. The
//! `on_event` hook is fully synchronous, non-blocking and panic-free:
//! it visits the event's fields into a JSON map, reads wall-clock time
//! and a best-effort OTel trace id, then does one brief `push`.

use std::fmt::Debug;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

use super::buffer::{LogBuffer, LogRecord};

/// Tracing layer that copies events into a shared [`LogBuffer`].
pub struct CaptureLayer {
    buffer: Arc<LogBuffer>,
    min_level: Level,
}

impl CaptureLayer {
    /// Create a layer capturing events at or above `min_level` into
    /// `buffer`.
    pub fn new(buffer: Arc<LogBuffer>, min_level: Level) -> Self {
        Self { buffer, min_level }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        // `tracing::Level` orders so that ERROR < WARN < INFO < ...,
        // i.e. more-severe is "less". We keep events at or above the
        // threshold, which is `level <= min_level`.
        if meta.level() > &self.min_level {
            return;
        }

        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);

        let (trace_id, span_id) = super::current_trace_ids();

        let rec = LogRecord {
            ts_ms: now_ms(),
            level: meta.level().to_string(),
            target: meta.target().to_string(),
            message: visitor.message.unwrap_or_default(),
            fields: visitor.fields,
            trace_id,
            span_id,
        };
        self.buffer.push(rec);
    }
}

/// Collects an event's fields into a JSON map, pulling out `message`
/// separately so it lands in [`LogRecord::message`].
#[derive(Default)]
struct FieldVisitor {
    message: Option<String>,
    fields: serde_json::Map<String, serde_json::Value>,
}

impl FieldVisitor {
    fn insert(&mut self, field: &Field, value: serde_json::Value) {
        if field.name() == "message"
            && let serde_json::Value::String(s) = &value
        {
            self.message = Some(s.clone());
            return;
        }
        self.fields.insert(field.name().to_string(), value);
    }
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        // The `message` field arrives via the Debug path.
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
            // Strip the quotes that Debug adds for plain string
            // messages, matching how fmt layers render them.
            if let Some(m) = &self.message
                && m.starts_with('"')
                && m.ends_with('"')
                && m.len() >= 2
            {
                self.message = Some(m[1..m.len() - 1].to_string());
            }
            return;
        }
        self.insert(field, serde_json::Value::String(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, serde_json::Value::String(value.to_string()));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, serde_json::Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, serde_json::Value::from(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, serde_json::Value::Bool(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.insert(field, serde_json::Value::from(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn captures_warn_and_error_with_fields_skips_info() {
        let buffer = Arc::new(LogBuffer::new(100));
        let layer = CaptureLayer::new(buffer.clone(), Level::WARN);
        let subscriber = Registry::default().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(answer = 42, "boom");
            tracing::error!("bad");
            tracing::info!("noise");
        });

        let got = buffer.recent(10, None, None);
        // INFO must be excluded by the default WARN threshold.
        assert_eq!(got.len(), 2, "only WARN+ERROR should be captured");

        // newest-first => error then warn.
        let err = &got[0];
        assert_eq!(err.level, "ERROR");
        assert_eq!(err.message, "bad");
        assert_eq!(err.target, module_path!());

        let warn = &got[1];
        assert_eq!(warn.level, "WARN");
        assert_eq!(warn.message, "boom");
        assert_eq!(warn.target, module_path!());
        assert_eq!(
            warn.fields.get("answer"),
            Some(&serde_json::Value::from(42))
        );

        // trace_id is best-effort: with no OTel layer it is None, but
        // the record is still captured and nothing panicked.
        assert!(warn.trace_id.is_none());
        assert!(warn.span_id.is_none());
    }

    #[test]
    fn min_level_error_excludes_warn() {
        let buffer = Arc::new(LogBuffer::new(100));
        let layer = CaptureLayer::new(buffer.clone(), Level::ERROR);
        let subscriber = Registry::default().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!("w");
            tracing::error!("e");
        });

        let got = buffer.recent(10, None, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].level, "ERROR");
    }
}
