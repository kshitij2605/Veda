//! Pure log-line parsing for logs-as-a-collection ingest.
//!
//! [`parse_log_line`] turns one raw log line (either a JSON object or a
//! plain text line) into a normalized [`ParsedLog`] with a millisecond
//! timestamp, an upper-cased level, a message, and any remaining
//! top-level scalar fields. It is a pure function — no I/O — and never
//! panics on weird input.

use chrono::DateTime;
use serde_json::{Map, Value};

/// One parsed log line, normalized for insertion into a logs collection.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLog {
    /// Event time in unix milliseconds.
    pub ts_ms: i64,
    /// Upper-cased severity (e.g. `INFO`, `WARN`, `ERROR`).
    pub level: String,
    /// The human-readable message body.
    pub message: String,
    /// Remaining top-level scalar fields (JSON lines only).
    pub fields: Map<String, Value>,
}

/// Parse a single raw log line.
///
/// - **JSON object line:** extracts `message` (keys `message`/`msg`/`body`),
///   `level` (`level`/`severity`/`lvl`, upper-cased), and `ts_ms`
///   (`timestamp`/`ts`/`time`/`@timestamp` — RFC3339 string or epoch
///   ms/secs number). Remaining top-level scalar keys go into `fields`.
/// - **Plain line:** best-effort scan for a leading RFC3339 timestamp and a
///   level token; the rest is the message.
/// - Anything unrecognised → whole line is the message, `default_level`,
///   `now_ms`.
///
/// `default_level` is used when no level can be determined; it is
/// upper-cased. `now_ms` is the fallback timestamp.
pub fn parse_log_line(line: &str, default_level: &str, now_ms: i64) -> ParsedLog {
    let default_level_uc = default_level.to_ascii_uppercase();

    // ── JSON object line ────────────────────────────────────────────
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line.trim()) {
        return parse_json_object(obj, &default_level_uc, now_ms);
    }

    // ── Plain line ──────────────────────────────────────────────────
    parse_plain_line(line, &default_level_uc, now_ms)
}

fn parse_json_object(obj: Map<String, Value>, default_level_uc: &str, now_ms: i64) -> ParsedLog {
    let message = first_string(&obj, &["message", "msg", "body"]).unwrap_or_default();

    let level = first_string(&obj, &["level", "severity", "lvl"])
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_else(|| default_level_uc.to_owned());

    let ts_ms = ["timestamp", "ts", "time", "@timestamp"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(value_to_ts_ms))
        .unwrap_or(now_ms);

    // Remaining top-level scalar keys → fields (skip the keys we consumed
    // for message/level/ts, and skip non-scalar values).
    let consumed = [
        "message", "msg", "body", "level", "severity", "lvl", "timestamp", "ts", "time",
        "@timestamp",
    ];
    let mut fields = Map::new();
    for (k, v) in obj {
        if consumed.contains(&k.as_str()) {
            continue;
        }
        if is_scalar(&v) {
            fields.insert(k, v);
        }
    }

    ParsedLog { ts_ms, level, message, fields }
}

fn parse_plain_line(line: &str, default_level_uc: &str, now_ms: i64) -> ParsedLog {
    let trimmed = line.trim();
    let mut tokens = trimmed.split_whitespace();

    // Try a leading RFC3339 timestamp.
    let mut ts_ms = now_ms;
    let mut rest_start = 0usize; // byte offset into `trimmed` after consumed prefix
    if let Some(first) = tokens.clone().next() {
        if let Some(ms) = rfc3339_to_ms(first) {
            ts_ms = ms;
            // advance past the timestamp token.
            rest_start = first.len();
            let _ = tokens.next();
        }
    }

    // Try a level token at the (new) front of the remaining tokens.
    let mut level = default_level_uc.to_owned();
    let remainder = trimmed[rest_start..].trim_start();
    let mut msg_start_in_remainder = 0usize;
    if let Some(tok) = remainder.split_whitespace().next() {
        if let Some(lvl) = match_level(tok) {
            level = lvl;
            msg_start_in_remainder = tok.len();
        }
    }

    let message = remainder[msg_start_in_remainder..].trim().to_owned();
    let message = if message.is_empty() {
        // No usable message after stripping ts/level — fall back to the
        // whole original line so we never lose content.
        trimmed.to_owned()
    } else {
        message
    };

    ParsedLog { ts_ms, level, message, fields: Map::new() }
}

/// Match a bare level token (case-insensitive). Returns the canonical
/// upper-cased level, or `None` if the token isn't a recognised level.
fn match_level(tok: &str) -> Option<String> {
    match tok.to_ascii_uppercase().as_str() {
        "DEBUG" => Some("DEBUG".into()),
        "INFO" => Some("INFO".into()),
        "WARN" => Some("WARN".into()),
        "WARNING" => Some("WARNING".into()),
        "ERROR" => Some("ERROR".into()),
        "TRACE" => Some("TRACE".into()),
        _ => None,
    }
}

/// First present, non-empty string value among `keys`.
fn first_string(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(Value::String(s)) = obj.get(*k) {
            if !s.is_empty() {
                return Some(s.clone());
            }
        }
    }
    None
}

/// A scalar JSON value (string / number / bool / null) — arrays and
/// objects are dropped from `fields`.
fn is_scalar(v: &Value) -> bool {
    matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null)
}

/// Convert a JSON timestamp value to unix millis. Accepts an RFC3339
/// string or an epoch number (ms if large, secs otherwise).
fn value_to_ts_ms(v: &Value) -> Option<i64> {
    match v {
        Value::String(s) => rfc3339_to_ms(s),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(epoch_to_ms(i))
            } else {
                n.as_f64().map(|f| epoch_to_ms(f as i64))
            }
        }
        _ => None,
    }
}

/// Heuristic: values >= 1e12 are already milliseconds; smaller positive
/// values are treated as seconds. (1e12 ms ≈ year 2001; 1e12 s is far in
/// the future, so this disambiguates real-world log timestamps cleanly.)
fn epoch_to_ms(n: i64) -> i64 {
    const MS_THRESHOLD: i64 = 1_000_000_000_000; // 1e12
    if n.abs() >= MS_THRESHOLD {
        n
    } else {
        n.saturating_mul(1000)
    }
}

/// Parse an RFC3339 / ISO-8601 timestamp to unix millis, else `None`.
fn rfc3339_to_ms(s: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000_000; // fixed "now" in ms

    #[test]
    fn json_line_extracts_level_message_ts_and_fields() {
        let line = r#"{"level":"error","msg":"disk full","ts":"2026-06-15T10:00:00Z","host":"h1"}"#;
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.level, "ERROR");
        assert_eq!(p.message, "disk full");
        assert_ne!(p.ts_ms, NOW, "ts should be parsed, not the now fallback");
        assert_eq!(p.fields.get("host").and_then(|v| v.as_str()), Some("h1"));
        // Consumed keys must not leak into fields.
        assert!(!p.fields.contains_key("level"));
        assert!(!p.fields.contains_key("msg"));
        assert!(!p.fields.contains_key("ts"));
    }

    #[test]
    fn json_line_epoch_ms_ts_parses() {
        let line = r#"{"message":"hi","timestamp":1718445600000}"#;
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.ts_ms, 1_718_445_600_000);
        assert_eq!(p.message, "hi");
    }

    #[test]
    fn json_line_epoch_secs_ts_parses() {
        let line = r#"{"message":"hi","time":1718445600}"#;
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.ts_ms, 1_718_445_600_000);
    }

    #[test]
    fn json_line_missing_ts_uses_now() {
        let line = r#"{"message":"no ts here","level":"warn"}"#;
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.ts_ms, NOW);
        assert_eq!(p.level, "WARN");
    }

    #[test]
    fn json_line_missing_level_uses_default() {
        let line = r#"{"message":"hello"}"#;
        let p = parse_log_line(line, "debug", NOW);
        assert_eq!(p.level, "DEBUG", "default level should be upper-cased");
    }

    #[test]
    fn plain_line_with_ts_and_level() {
        let line = "2026-06-15T10:00:00Z WARN cache miss for key=42";
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.level, "WARN");
        assert!(p.message.contains("cache miss"), "got: {}", p.message);
        assert!(!p.message.contains("WARN"), "level token should be stripped");
        assert_ne!(p.ts_ms, NOW, "ts should be parsed, not now");
    }

    #[test]
    fn garbage_line_is_all_message() {
        let line = "just some text";
        let p = parse_log_line(line, "INFO", NOW);
        assert_eq!(p.message, "just some text");
        assert_eq!(p.level, "INFO");
        assert_eq!(p.ts_ms, NOW);
        assert!(p.fields.is_empty());
    }

    #[test]
    fn malformed_jsonish_never_panics() {
        for line in [
            "{not really json",
            "{\"unterminated\": ",
            "[]",
            "null",
            "123",
            "",
            "   ",
            "{}",
        ] {
            let p = parse_log_line(line, "INFO", NOW);
            // Non-object JSON / garbage → treated as plain; never panics.
            assert!(!p.level.is_empty());
        }
    }

    #[test]
    fn json_array_and_nested_are_dropped_from_fields() {
        let line = r#"{"message":"m","arr":[1,2],"obj":{"a":1},"n":5}"#;
        let p = parse_log_line(line, "INFO", NOW);
        assert!(!p.fields.contains_key("arr"));
        assert!(!p.fields.contains_key("obj"));
        assert_eq!(p.fields.get("n").and_then(|v| v.as_i64()), Some(5));
    }
}
