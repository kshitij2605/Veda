//! Veda metric name constants for the `metrics` crate.
//!
//! All metric recording uses `metrics::counter!()`, `metrics::histogram!()`,
//! and `metrics::gauge!()` macros directly in handlers, referencing these constants.

pub const REQUESTS_TOTAL: &str = "veda_requests_total";
pub const REQUEST_DURATION: &str = "veda_request_duration_seconds";
pub const SEARCH_DURATION: &str = "veda_search_duration_seconds";
pub const INSERT_TOTAL: &str = "veda_insert_total";
pub const INSERT_DURATION: &str = "veda_insert_duration_seconds";
pub const EMBEDDING_DURATION: &str = "veda_embedding_duration_seconds";
pub const BLOB_OP_DURATION: &str = "veda_blob_operation_duration_seconds";
pub const SEGMENTS_TOTAL: &str = "veda_segments_total";
pub const INDEX_BUILD_DURATION: &str = "veda_index_build_duration_seconds";
pub const COMPACTION_DURATION: &str = "veda_compaction_duration_seconds";
pub const ERRORS_TOTAL: &str = "veda_errors_total";
pub const ACTIVE_COLLECTIONS: &str = "veda_active_collections";
pub const RECORDS_TOTAL: &str = "veda_records_total";
pub const COMPACTION_IN_PROGRESS: &str = "veda_compaction_in_progress";
pub const INDEX_REBUILD_IN_PROGRESS: &str = "veda_index_rebuild_in_progress";
pub const SEAL_DURATION: &str = "veda_seal_duration_seconds";

// ── LLM / agent APM metrics ───────────────────────────────────────────────

/// Counter of LLM tokens, labelled `type="input"|"output"` and `model`.
pub const LLM_TOKENS_TOTAL: &str = "veda_llm_tokens_total";
/// Histogram of end-to-end agent run wall-clock duration in seconds.
pub const AGENT_RUN_DURATION: &str = "veda_agent_run_duration_seconds";
/// Counter of estimated LLM spend, labelled `model`. Only emitted for
/// models with a known price. The `metrics` crate's counters are u64, so
/// the value is accumulated in **micro-USD** (1e-6 USD); divide by 1e6 in
/// dashboards to get dollars. See [`cost_micro_usd`].
pub const LLM_COST_USD_TOTAL: &str = "veda_llm_cost_usd_total";
/// Histogram of time-to-first-token (seconds) for streaming agent runs,
/// labelled `model`. Only emitted on the streaming path; the blocking
/// `/converse` path leaves TTFT unset and emits nothing here.
pub const LLM_TTFT_SECONDS: &str = "veda_llm_ttft_seconds";
/// Counter of user feedback on agent conversations, labelled
/// `rating="up"|"down"`.
pub const LLM_FEEDBACK_TOTAL: &str = "veda_llm_feedback_total";

/// Built-in fallback price table: USD per 1,000 tokens, `(input, output)`.
/// Deliberately tiny — covers a few common OpenAI-compatible models. Any
/// model not listed (and not overridden via env) is treated as unpriced.
const DEFAULT_PRICES: &[(&str, f64, f64)] = &[
    ("gpt-4o", 0.005, 0.015),
    ("gpt-4o-mini", 0.00015, 0.0006),
    ("gpt-4.1", 0.002, 0.008),
    ("gpt-4.1-mini", 0.0004, 0.0016),
    ("o3", 0.002, 0.008),
];

/// Resolve the `(usd_per_1k_input, usd_per_1k_output)` price for `model`.
///
/// Resolution order:
/// 1. `VEDA_LLM_PRICES` env var — JSON `{"model":{"input":x,"output":y}}`.
///    Malformed JSON is ignored (falls through), never panics.
/// 2. The built-in [`DEFAULT_PRICES`] table (exact, case-insensitive match).
///
/// Returns `None` for an unpriced model: callers MUST still emit token
/// metrics but skip cost. Pure + testable (env is read fresh each call so
/// tests can set it).
pub fn model_price(model: &str) -> Option<(f64, f64)> {
    if let Ok(raw) = std::env::var("VEDA_LLM_PRICES")
        && let Ok(map) = serde_json::from_str::<serde_json::Value>(&raw)
        && let Some(entry) = map.get(model)
    {
        let input = entry.get("input").and_then(|v| v.as_f64());
        let output = entry.get("output").and_then(|v| v.as_f64());
        if let (Some(i), Some(o)) = (input, output) {
            return Some((i, o));
        }
    }
    DEFAULT_PRICES
        .iter()
        .find(|(name, _, _)| name.eq_ignore_ascii_case(model))
        .map(|(_, i, o)| (*i, *o))
}

/// Compute estimated USD cost from token counts and a `(in_per_1k,
/// out_per_1k)` price. Pure helper, no I/O.
pub fn token_cost_usd(prompt_tokens: u32, completion_tokens: u32, price: (f64, f64)) -> f64 {
    (prompt_tokens as f64 / 1000.0) * price.0 + (completion_tokens as f64 / 1000.0) * price.1
}

/// USD cost expressed in **micro-USD** (1e-6 USD) as a `u64`, suitable for
/// incrementing the [`LLM_COST_USD_TOTAL`] counter (the `metrics` crate has
/// no float counter). Saturates rather than panicking.
pub fn cost_micro_usd(prompt_tokens: u32, completion_tokens: u32, price: (f64, f64)) -> u64 {
    (token_cost_usd(prompt_tokens, completion_tokens, price) * 1_000_000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-mutating tests share a process; serialise via a mutex so the
    // VEDA_LLM_PRICES var doesn't race across parallel test threads.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn default_price_table_known_model() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("VEDA_LLM_PRICES") };
        let p = model_price("gpt-4o-mini").expect("known model");
        assert_eq!(p, (0.00015, 0.0006));
        // Case-insensitive.
        assert!(model_price("GPT-4O-MINI").is_some());
    }

    #[test]
    fn unpriced_model_returns_none() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("VEDA_LLM_PRICES") };
        assert!(model_price("some-local-llama-7b").is_none());
    }

    #[test]
    fn env_override_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                "VEDA_LLM_PRICES",
                r#"{"my-model":{"input":0.1,"output":0.2},"gpt-4o-mini":{"input":9.9,"output":9.9}}"#,
            );
        }
        assert_eq!(model_price("my-model"), Some((0.1, 0.2)));
        // Env override beats the built-in table for the same name.
        assert_eq!(model_price("gpt-4o-mini"), Some((9.9, 9.9)));
        // Model not in env nor table → None even with env set.
        assert!(model_price("nope-model").is_none());
        unsafe { std::env::remove_var("VEDA_LLM_PRICES") };
    }

    #[test]
    fn malformed_env_falls_through_to_table() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("VEDA_LLM_PRICES", "{not json") };
        // Falls through to the built-in table; no panic.
        assert_eq!(model_price("gpt-4o"), Some((0.005, 0.015)));
        unsafe { std::env::remove_var("VEDA_LLM_PRICES") };
    }

    #[test]
    fn cost_computation() {
        // 2000 input @ 0.005/1k = 0.01 ; 1000 output @ 0.015/1k = 0.015.
        let c = token_cost_usd(2000, 1000, (0.005, 0.015));
        assert!((c - 0.025).abs() < 1e-12, "got {c}");
        // Zero tokens → zero cost.
        assert_eq!(token_cost_usd(0, 0, (1.0, 1.0)), 0.0);
    }

    #[test]
    fn cost_micro_usd_scaling() {
        // 0.025 USD → 25_000 micro-USD.
        assert_eq!(cost_micro_usd(2000, 1000, (0.005, 0.015)), 25_000);
        assert_eq!(cost_micro_usd(0, 0, (1.0, 1.0)), 0);
    }
}
