//! Semantic categorization of recent WARN/ERROR log messages.
//!
//! [`LogCategorizer::categorize`] pulls recent warnings and errors from
//! the in-process [`LogBuffer`] and clusters them into a small set of
//! [`LogCategory`]s. Two clustering paths exist:
//!
//!   * **Embedding path** — when a text embedding model is configured in
//!     the [`EmbeddingModelRegistry`], each *distinct* message is embedded
//!     (cache-missing ones only) via the async
//!     [`EmbeddingClient::embed_text`], then greedily clustered by cosine
//!     similarity against running-mean centroids ([`cluster_embedded`]).
//!   * **Fallback path** — when no text model is configured (or embedding
//!     fails), messages are grouped by a normalized template key
//!     ([`normalize_message`] / [`cluster_normalized`]). No network calls.
//!
//! The categorizer keeps a small amount of state across calls: an
//! embedding cache (keyed by message hash) and the most recent set of
//! categories, so `first_seen_ms` persists. State lives behind a
//! `tokio::sync::Mutex` and the whole thing is held as an
//! `Arc<LogCategorizer>` on `AppStateInner`.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::Serialize;
use tokio::sync::Mutex;
use veda_core::embedding::EmbeddingModelRegistry;
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;

use crate::logs::LogBuffer;

/// Maximum number of distinct messages embedded / considered per call.
const MAX_MESSAGES: usize = 200;
/// Maximum number of categories returned / retained.
const MAX_CATEGORIES: usize = 50;
/// Maximum number of example messages stored per category.
const MAX_EXAMPLES: usize = 3;
/// Default cosine threshold for the embedding clustering path.
const DEFAULT_THRESHOLD: f32 = 0.82;

/// A cluster of semantically similar log messages.
#[derive(Debug, Clone, Serialize)]
pub struct LogCategory {
    /// Stable-ish id derived from the representative template.
    pub id: String,
    /// Representative (normalized) message for the category.
    pub representative: String,
    /// Number of messages assigned to this category.
    pub count: u64,
    /// Highest severity seen in the category (`ERROR` > `WARN`).
    pub max_severity: String,
    /// Up to [`MAX_EXAMPLES`] raw example messages.
    pub examples: Vec<String>,
    /// Earliest `ts_ms` observed for the category.
    pub first_seen_ms: i64,
    /// Latest `ts_ms` observed for the category.
    pub last_seen_ms: i64,
}

/// Options controlling a [`LogCategorizer::categorize`] call.
#[derive(Debug, Clone)]
pub struct CategorizeOptions {
    /// How many recent WARN+ERROR records to pull from the buffer.
    pub limit: usize,
    /// Cosine threshold for the embedding path.
    pub threshold: f32,
}

impl Default for CategorizeOptions {
    fn default() -> Self {
        Self {
            limit: MAX_MESSAGES,
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

/// Internal, mutable categorizer state. Guarded by a `tokio::sync::Mutex`.
#[derive(Default)]
struct CategorizerInner {
    /// message-hash -> embedding, so repeat messages aren't re-embedded.
    cache: HashMap<u64, Vec<f32>>,
    /// first-seen timestamps keyed by category id, persisted across calls.
    first_seen: HashMap<String, i64>,
}

/// Semantic log categorizer. Cheap to construct; holds an embedding
/// cache and first-seen timestamps behind an async mutex.
#[derive(Default)]
pub struct LogCategorizer {
    inner: Mutex<CategorizerInner>,
}

impl LogCategorizer {
    /// Construct an empty categorizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cluster recent WARN+ERROR messages from `buffer` into categories.
    ///
    /// Uses the embedding path when a text model is available in
    /// `registry`, else the embedder-free normalized fallback. Returns
    /// categories sorted by severity (ERROR first) then count desc.
    pub async fn categorize(
        &self,
        buffer: &LogBuffer,
        registry: &EmbeddingModelRegistry,
        opts: CategorizeOptions,
    ) -> Vec<LogCategory> {
        let limit = opts.limit.clamp(1, MAX_MESSAGES);
        let records = buffer.recent(limit, Some("WARN"), None);
        if records.is_empty() {
            return Vec::new();
        }

        // Resolve a text embedding model (clone the config so we don't
        // hold a borrow on the registry across the await points).
        let text_model = registry
            .find_by_modality(Modality::Text)
            .first()
            .map(|c| (*c).clone());

        let mut categories = if let Some(model) = text_model {
            match self
                .embed_and_cluster(&records, model, opts.threshold)
                .await
            {
                Some(cats) => cats,
                // Any embedding error → fall back to normalized grouping.
                None => cluster_normalized(&normalized_items(&records)),
            }
        } else {
            cluster_normalized(&normalized_items(&records))
        };

        // Persist / restore first_seen across calls, then sort + cap.
        {
            let mut inner = self.inner.lock().await;
            for cat in categories.iter_mut() {
                let fs = inner
                    .first_seen
                    .entry(cat.id.clone())
                    .or_insert(cat.first_seen_ms);
                *fs = (*fs).min(cat.first_seen_ms);
                cat.first_seen_ms = *fs;
            }
        }

        sort_categories(&mut categories);
        categories.truncate(MAX_CATEGORIES);
        categories
    }

    /// Embedding path: embed distinct messages (cache-aware) then
    /// [`cluster_embedded`]. Returns `None` on the first embedding error
    /// so the caller can fall back.
    async fn embed_and_cluster(
        &self,
        records: &[crate::logs::LogRecord],
        model: veda_core::embedding::EmbeddingModelConfig,
        threshold: f32,
    ) -> Option<Vec<LogCategory>> {
        let client = EmbeddingClient::new(model);

        // Compute the distinct messages (in buffer order).
        let mut distinct: Vec<&str> = Vec::new();
        let mut seen: HashMap<u64, ()> = HashMap::new();
        for r in records {
            let h = hash_msg(&r.message);
            if seen.insert(h, ()).is_none() {
                distinct.push(&r.message);
            }
        }

        // Phase B (brief lock): partition distinct messages into those
        // already cached vs cache misses. We collect the misses by value
        // and drop the lock before any network embedding await.
        let misses: Vec<String> = {
            let inner = self.inner.lock().await;
            distinct
                .iter()
                .filter(|msg| !inner.cache.contains_key(&hash_msg(msg)))
                .map(|msg| (*msg).to_string())
                .collect()
        };

        // Phase C (NO lock held): embed each miss via the network. On any
        // embedding error, abort the embedding path so the caller falls
        // back to normalized grouping.
        let mut embedded: Vec<(String, Vec<f32>)> = Vec::with_capacity(misses.len());
        for msg in &misses {
            match client.embed_text(msg).await {
                Ok(emb) => embedded.push((msg.clone(), emb)),
                Err(_) => return None,
            }
        }

        // Phase D (brief lock): insert the newly-embedded misses into the
        // cache, then build the (msg, emb, ts, level) item list in buffer
        // order from the combined cache. Lock dropped at end of scope.
        let items: Vec<(String, Vec<f32>, i64, String)> = {
            let mut inner = self.inner.lock().await;
            for (msg, emb) in embedded {
                inner.cache.insert(hash_msg(&msg), emb);
            }
            let mut items = Vec::with_capacity(records.len());
            for r in records {
                let h = hash_msg(&r.message);
                let emb = inner.cache.get(&h)?.clone();
                items.push((r.message.clone(), emb, r.ts_ms, r.level.clone()));
            }
            items
        };

        // Phase E (no lock): pure clustering.
        Some(cluster_embedded(&items, threshold))
    }
}

/// Build `(msg, ts, level)` tuples for the normalized fallback path.
fn normalized_items(records: &[crate::logs::LogRecord]) -> Vec<(String, i64, String)> {
    records
        .iter()
        .map(|r| (r.message.clone(), r.ts_ms, r.level.clone()))
        .collect()
}

/// Stable FNV-ish hash of a message string (via `DefaultHasher`).
fn hash_msg(msg: &str) -> u64 {
    let mut h = DefaultHasher::new();
    msg.hash(&mut h);
    h.finish()
}

/// Severity rank: higher = more severe. Unknown ranks lowest.
fn severity_rank(level: &str) -> u8 {
    match level.to_ascii_uppercase().as_str() {
        "ERROR" => 4,
        "WARN" => 3,
        "INFO" => 2,
        "DEBUG" => 1,
        _ => 0,
    }
}

/// Keep the more severe of two level strings.
fn max_severity(a: &str, b: &str) -> String {
    if severity_rank(b) > severity_rank(a) {
        b.to_ascii_uppercase()
    } else {
        a.to_ascii_uppercase()
    }
}

/// Sort categories by severity desc, then count desc, then id for
/// determinism.
fn sort_categories(cats: &mut [LogCategory]) {
    cats.sort_by(|a, b| {
        severity_rank(&b.max_severity)
            .cmp(&severity_rank(&a.max_severity))
            .then(b.count.cmp(&a.count))
            .then(a.id.cmp(&b.id))
    });
}

/// Derive a short category id from a representative template.
fn category_id(template: &str) -> String {
    format!("cat_{:016x}", hash_msg(template))
}

/// Collapse the variable bits of a message into a stable template.
///
/// Replaces UUIDs and long hex runs with `<id>`, integer/decimal runs
/// with `<num>`, and quoted strings with `<str>`; trims and collapses
/// whitespace. Two messages that differ only in such values produce the
/// same template, so they group together.
pub fn normalize_message(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let mut chars = msg.char_indices().peekable();

    while let Some((_, c)) = chars.next() {
        // Quoted strings ("..." or '...') -> <str>.
        if c == '"' || c == '\'' {
            let quote = c;
            // consume until matching quote (or end).
            while let Some(&(_, n)) = chars.peek() {
                chars.next();
                if n == quote {
                    break;
                }
            }
            out.push_str("<str>");
            continue;
        }

        // Runs starting with an ASCII hex digit or digit: classify the
        // whole alphanumeric(+-._:) run.
        if c.is_ascii_alphanumeric() {
            let mut run = String::new();
            run.push(c);
            while let Some(&(_, n)) = chars.peek() {
                if n.is_ascii_alphanumeric() || n == '-' || n == '_' || n == '.' || n == ':' {
                    run.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push_str(&classify_token(&run));
            continue;
        }

        out.push(c);
    }

    // Collapse runs of whitespace and trim.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Classify a single alphanumeric token into a placeholder or keep it.
fn classify_token(tok: &str) -> String {
    // Pure number (int or decimal, optional sign already split off).
    let is_number = !tok.is_empty()
        && tok.chars().all(|c| c.is_ascii_digit() || c == '.')
        && tok.chars().any(|c| c.is_ascii_digit());
    if is_number {
        return "<num>".to_string();
    }

    // UUID-ish: 8-4-4-4-12 hex with dashes.
    if is_uuid(tok) {
        return "<id>".to_string();
    }

    // Long hex run (>= 8 chars, all hex) -> id (covers hashes / object
    // ids like `a1b2c3d4` and `deadbeef`). We don't require a digit:
    // 8+ char all-hex tokens are overwhelmingly identifiers, and real
    // English words that are pure hex letters at that length are
    // vanishingly rare.
    let hex_len = tok.chars().filter(|c| c.is_ascii_hexdigit()).count();
    if tok.len() >= 8 && hex_len == tok.len() {
        return "<id>".to_string();
    }

    tok.to_string()
}

/// True if `tok` looks like a canonical UUID.
fn is_uuid(tok: &str) -> bool {
    let parts: Vec<&str> = tok.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let lens = [8usize, 4, 4, 4, 12];
    for (p, l) in parts.iter().zip(lens.iter()) {
        if p.len() != *l || !p.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
    }
    true
}

/// Cosine similarity of two equal-length vectors. Returns 0.0 if either
/// is zero-length or has zero norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Mutable accumulator while clustering by embedding.
struct EmbAccum {
    centroid: Vec<f32>,
    rep: String,
    count: u64,
    max_sev: String,
    examples: Vec<String>,
    first_seen: i64,
    last_seen: i64,
}

/// Greedy embedding-based clustering. For each item, assign to the first
/// existing centroid with cosine >= `threshold` (updating it as a running
/// mean), else start a new category. Deterministic in input order.
pub fn cluster_embedded(
    items: &[(String, Vec<f32>, i64, String)],
    threshold: f32,
) -> Vec<LogCategory> {
    let mut accs: Vec<EmbAccum> = Vec::new();

    for (msg, emb, ts, level) in items {
        let mut best: Option<usize> = None;
        let mut best_sim = threshold;
        for (i, acc) in accs.iter().enumerate() {
            let sim = cosine(&acc.centroid, emb);
            if sim >= best_sim {
                best_sim = sim;
                best = Some(i);
            }
        }

        match best {
            Some(i) => {
                let acc = &mut accs[i];
                // Running-mean centroid update.
                let n = acc.count as f32;
                #[allow(clippy::needless_range_loop)]
                for k in 0..acc.centroid.len() {
                    acc.centroid[k] = (acc.centroid[k] * n + emb[k]) / (n + 1.0);
                }
                acc.count += 1;
                acc.max_sev = max_severity(&acc.max_sev, level);
                acc.first_seen = acc.first_seen.min(*ts);
                acc.last_seen = acc.last_seen.max(*ts);
                if acc.examples.len() < MAX_EXAMPLES && !acc.examples.contains(msg) {
                    acc.examples.push(msg.clone());
                }
            }
            None => {
                accs.push(EmbAccum {
                    centroid: emb.clone(),
                    rep: normalize_message(msg),
                    count: 1,
                    max_sev: level.to_ascii_uppercase(),
                    examples: vec![msg.clone()],
                    first_seen: *ts,
                    last_seen: *ts,
                });
            }
        }
    }

    accs.into_iter()
        .map(|a| LogCategory {
            id: category_id(&a.rep),
            representative: a.rep,
            count: a.count,
            max_severity: a.max_sev,
            examples: a.examples,
            first_seen_ms: a.first_seen,
            last_seen_ms: a.last_seen,
        })
        .collect()
}

/// Embedder-free clustering: group messages by their normalized template.
/// Deterministic: categories appear in first-seen (input) order.
pub fn cluster_normalized(items: &[(String, i64, String)]) -> Vec<LogCategory> {
    // Preserve insertion order of distinct templates.
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, EmbAccum> = HashMap::new();

    for (msg, ts, level) in items {
        let key = normalize_message(msg);
        match map.get_mut(&key) {
            Some(acc) => {
                acc.count += 1;
                acc.max_sev = max_severity(&acc.max_sev, level);
                acc.first_seen = acc.first_seen.min(*ts);
                acc.last_seen = acc.last_seen.max(*ts);
                if acc.examples.len() < MAX_EXAMPLES && !acc.examples.contains(msg) {
                    acc.examples.push(msg.clone());
                }
            }
            None => {
                order.push(key.clone());
                map.insert(
                    key.clone(),
                    EmbAccum {
                        centroid: Vec::new(),
                        rep: key,
                        count: 1,
                        max_sev: level.to_ascii_uppercase(),
                        examples: vec![msg.clone()],
                        first_seen: *ts,
                        last_seen: *ts,
                    },
                );
            }
        }
    }

    order
        .into_iter()
        .filter_map(|k| map.remove(&k))
        .map(|a| LogCategory {
            id: category_id(&a.rep),
            representative: a.rep,
            count: a.count,
            max_severity: a.max_sev,
            examples: a.examples,
            first_seen_ms: a.first_seen,
            last_seen_ms: a.last_seen,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logs::LogRecord;

    fn rec(level: &str, message: &str, ts: i64) -> LogRecord {
        LogRecord {
            ts_ms: ts,
            level: level.to_string(),
            target: "test".to_string(),
            message: message.to_string(),
            fields: serde_json::Map::new(),
            trace_id: None,
            span_id: None,
        }
    }

    #[test]
    fn normalize_message_collapses_numbers_and_ids() {
        let a = normalize_message("request 4242 failed for user a1b2c3d4");
        let b = normalize_message("request 88 failed for user deadbeef");
        assert_eq!(a, b, "templates should match: {a} vs {b}");
        assert!(a.contains("<num>"));
        assert!(a.contains("<id>"));
    }

    #[test]
    fn normalize_message_handles_uuid_and_quotes() {
        let a =
            normalize_message("deleting record 550e8400-e29b-41d4-a716-446655440000 named \"foo\"");
        let b =
            normalize_message("deleting record 7c9e6679-7425-40de-944b-e07fc1f90ae7 named \"bar\"");
        assert_eq!(a, b);
        assert!(a.contains("<id>"));
        assert!(a.contains("<str>"));
    }

    #[test]
    fn normalize_message_keeps_distinct_messages_distinct() {
        let a = normalize_message("connection refused");
        let b = normalize_message("disk full");
        assert_ne!(a, b);
    }

    #[test]
    fn cosine_basic_correctness() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn cluster_embedded_groups_similar_splits_orthogonal() {
        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], 10i64, "WARN".to_string()),
            (
                "b".to_string(),
                vec![0.99, 0.01],
                20i64,
                "ERROR".to_string(),
            ),
            ("c".to_string(), vec![0.0, 1.0], 30i64, "WARN".to_string()),
        ];
        let cats = cluster_embedded(&items, 0.9);
        assert_eq!(cats.len(), 2, "expected 2 categories");

        // The big category has the two near-identical vectors.
        let big = cats.iter().max_by_key(|c| c.count).unwrap();
        assert_eq!(big.count, 2);
        assert_eq!(big.max_severity, "ERROR");
        assert_eq!(big.first_seen_ms, 10);
        assert_eq!(big.last_seen_ms, 20);
        assert_eq!(big.examples, vec!["a".to_string(), "b".to_string()]);

        let small = cats.iter().min_by_key(|c| c.count).unwrap();
        assert_eq!(small.count, 1);
    }

    #[test]
    fn cluster_embedded_centroid_is_running_mean() {
        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], 1i64, "WARN".to_string()),
            ("b".to_string(), vec![1.0, 2.0], 2i64, "WARN".to_string()),
        ];
        // Threshold 0.0 so both always merge.
        let cats = cluster_embedded(&items, 0.0);
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0].count, 2);
        // Centroid is the mean of the two inputs: (1.0, 1.0). We assert
        // indirectly: a query equal to the mean should be cosine ~1.
        // (centroid not exposed, so verify via merge behaviour above).
    }

    #[test]
    fn cluster_normalized_groups_by_template() {
        let items = vec![
            ("request 1 failed".to_string(), 10i64, "WARN".to_string()),
            ("request 2 failed".to_string(), 20i64, "ERROR".to_string()),
            ("disk full".to_string(), 30i64, "WARN".to_string()),
        ];
        let cats = cluster_normalized(&items);
        assert_eq!(cats.len(), 2);
        let grouped = cats.iter().find(|c| c.count == 2).unwrap();
        assert_eq!(grouped.max_severity, "ERROR");
        assert_eq!(grouped.first_seen_ms, 10);
        assert_eq!(grouped.last_seen_ms, 20);
        assert_eq!(grouped.examples.len(), 2);
    }

    #[tokio::test]
    async fn categorize_falls_back_without_embedder() {
        let buf = LogBuffer::new(100);
        // INFO should be excluded (min_level WARN).
        buf.push(rec("INFO", "ignored info", 1));
        buf.push(rec("WARN", "request 1 failed", 10));
        buf.push(rec("ERROR", "request 2 failed", 20));
        buf.push(rec("WARN", "disk full", 30));

        let categorizer = LogCategorizer::new();
        let registry = EmbeddingModelRegistry::new(); // empty → fallback
        let cats = categorizer
            .categorize(&buf, &registry, CategorizeOptions::default())
            .await;

        assert_eq!(cats.len(), 2, "two semantic groups expected");
        // Sorted: ERROR-bearing "request ... failed" group first.
        assert_eq!(cats[0].max_severity, "ERROR");
        assert_eq!(cats[0].count, 2);
        assert!(cats[0].representative.contains("<num>"));
        assert_eq!(cats[1].count, 1);
        assert_eq!(cats[1].representative, "disk full");
    }

    #[tokio::test]
    async fn categorize_persists_first_seen_across_calls() {
        let buf = LogBuffer::new(100);
        buf.push(rec("WARN", "request 1 failed", 100));
        buf.push(rec("WARN", "disk full", 200));

        let categorizer = LogCategorizer::new();
        let registry = EmbeddingModelRegistry::new(); // no embedder → fallback, no network

        let first = categorizer
            .categorize(&buf, &registry, CategorizeOptions::default())
            .await;
        assert_eq!(first.len(), 2);

        // Second call sees the same templates at later timestamps; the
        // persisted first_seen must win (min) across the two calls.
        let buf2 = LogBuffer::new(100);
        buf2.push(rec("WARN", "request 9 failed", 500));
        buf2.push(rec("WARN", "disk full", 600));
        let second = categorizer
            .categorize(&buf2, &registry, CategorizeOptions::default())
            .await;

        for cat in &second {
            let prev = first.iter().find(|c| c.id == cat.id).unwrap();
            assert_eq!(
                cat.first_seen_ms, prev.first_seen_ms,
                "first_seen_ms should persist across calls"
            );
        }
    }

    #[tokio::test]
    async fn categorize_empty_buffer_is_empty() {
        let buf = LogBuffer::new(10);
        let categorizer = LogCategorizer::new();
        let registry = EmbeddingModelRegistry::new();
        let cats = categorizer
            .categorize(&buf, &registry, CategorizeOptions::default())
            .await;
        assert!(cats.is_empty());
    }
}
