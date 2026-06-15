//! Bounded in-process ring buffer of captured log records.
//!
//! [`LogBuffer`] holds the most recent [`LogRecord`]s emitted by the
//! tracing layer (see [`super::layer::CaptureLayer`]). It is a simple
//! bounded `VecDeque` behind a `std::sync::Mutex` — pushes take a brief
//! lock, do no I/O and never await, so they're safe to call from the
//! synchronous `on_event` hot path.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

/// Default capacity of the ring buffer when none is specified.
pub const DEFAULT_CAPACITY: usize = 2000;

/// A single captured log event.
#[derive(Debug, Clone, Serialize)]
pub struct LogRecord {
    /// Unix milliseconds (wall clock) when the event was captured.
    pub ts_ms: i64,
    /// Level string: `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`.
    pub level: String,
    /// Event target (usually the emitting module path).
    pub target: String,
    /// The `message` field of the event (empty if none).
    pub message: String,
    /// All remaining structured fields on the event.
    pub fields: serde_json::Map<String, serde_json::Value>,
    /// OTel trace id (32-hex) if a valid trace context was active.
    pub trace_id: Option<String>,
    /// OTel span id (16-hex) if a valid trace context was active.
    pub span_id: Option<String>,
}

/// A bounded, newest-wins ring of [`LogRecord`]s.
pub struct LogBuffer {
    cap: usize,
    inner: Mutex<VecDeque<LogRecord>>,
}

impl LogBuffer {
    /// Create a buffer holding up to `cap` records (clamped to >= 1).
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            cap,
            inner: Mutex::new(VecDeque::with_capacity(cap)),
        }
    }

    /// Create a buffer with [`DEFAULT_CAPACITY`].
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    /// Push a record, evicting the oldest if at capacity. Brief lock,
    /// no I/O, no await. A poisoned lock is recovered from (we only
    /// ever mutate a `VecDeque`, so the data is never left invalid).
    pub fn push(&self, rec: LogRecord) {
        let mut q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if q.len() >= self.cap {
            q.pop_front();
        }
        q.push_back(rec);
    }

    /// Number of records currently buffered.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// True when no records are buffered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return up to `limit` records, newest-first. Optional filters:
    /// `min_level` keeps only records at or above the given severity
    /// (e.g. `min_level = "WARN"` excludes INFO/DEBUG/TRACE);
    /// `contains` keeps only records whose message or target contains
    /// the substring (case-insensitive).
    pub fn recent(
        &self,
        limit: usize,
        min_level: Option<&str>,
        contains: Option<&str>,
    ) -> Vec<LogRecord> {
        let min_rank = min_level.map(level_rank);
        let needle = contains.map(|s| s.to_lowercase());
        let q = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        q.iter()
            .rev()
            .filter(|r| match min_rank {
                Some(min) => level_rank(&r.level) >= min,
                None => true,
            })
            .filter(|r| match &needle {
                Some(n) => {
                    r.message.to_lowercase().contains(n)
                        || r.target.to_lowercase().contains(n)
                }
                None => true,
            })
            .take(limit)
            .cloned()
            .collect()
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// Map a level string to a severity rank where higher = more severe.
/// Unknown strings rank lowest so they're only kept when no
/// `min_level` filter is applied.
fn level_rank(level: &str) -> u8 {
    match level.to_ascii_uppercase().as_str() {
        "ERROR" => 4,
        "WARN" => 3,
        "INFO" => 2,
        "DEBUG" => 1,
        "TRACE" => 0,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(level: &str, message: &str) -> LogRecord {
        LogRecord {
            ts_ms: 0,
            level: level.to_string(),
            target: "test".to_string(),
            message: message.to_string(),
            fields: serde_json::Map::new(),
            trace_id: None,
            span_id: None,
        }
    }

    #[test]
    fn push_is_bounded_and_drops_oldest() {
        let buf = LogBuffer::new(3);
        for i in 0..5 {
            buf.push(rec("INFO", &format!("m{i}")));
        }
        assert_eq!(buf.len(), 3);
        // Newest-first: m4, m3, m2 retained; m0/m1 dropped.
        let got = buf.recent(10, None, None);
        let msgs: Vec<_> = got.iter().map(|r| r.message.as_str()).collect();
        assert_eq!(msgs, vec!["m4", "m3", "m2"]);
    }

    #[test]
    fn recent_is_newest_first_and_respects_limit() {
        let buf = LogBuffer::new(10);
        buf.push(rec("INFO", "a"));
        buf.push(rec("INFO", "b"));
        buf.push(rec("INFO", "c"));
        let got = buf.recent(2, None, None);
        let msgs: Vec<_> = got.iter().map(|r| r.message.as_str()).collect();
        assert_eq!(msgs, vec!["c", "b"]);
    }

    #[test]
    fn min_level_filter_excludes_lower_levels() {
        let buf = LogBuffer::new(10);
        buf.push(rec("INFO", "info"));
        buf.push(rec("WARN", "warn"));
        buf.push(rec("ERROR", "err"));
        let got = buf.recent(10, Some("WARN"), None);
        let msgs: Vec<_> = got.iter().map(|r| r.message.as_str()).collect();
        assert_eq!(msgs, vec!["err", "warn"]);
    }

    #[test]
    fn contains_substring_filter_works() {
        let buf = LogBuffer::new(10);
        buf.push(rec("INFO", "connection refused"));
        buf.push(rec("INFO", "all good"));
        buf.push(rec("INFO", "Connection reset"));
        let got = buf.recent(10, None, Some("connection"));
        let msgs: Vec<_> = got.iter().map(|r| r.message.as_str()).collect();
        // case-insensitive; newest-first
        assert_eq!(msgs, vec!["Connection reset", "connection refused"]);
    }
}
