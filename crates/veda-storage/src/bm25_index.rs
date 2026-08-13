//! BM25 full-text index for scalar string fields.
//!
//! Tokenizes input with a simple Unicode-aware splitter: ASCII alphanumeric
//! runs become whitespace-style tokens (lowercased), and CJK characters
//! emit both unigrams *and* overlapping bigrams. The bigram trick keeps
//! lookup competitive for Japanese / Chinese / Korean without bundling a
//! morphological analyzer (Kuromoji / Jieba) — close enough to ES's
//! `kuromoji_tokenizer` for retrieval purposes, at the cost of larger
//! posting lists.
//!
//! BM25 parameters use the Lucene defaults: `k1 = 1.2`, `b = 0.75`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};

/// Lucene-default BM25 parameter k1.
pub const BM25_K1: f32 = 1.2;
/// Lucene-default BM25 parameter b.
pub const BM25_B: f32 = 0.75;

/// BM25 inverse document frequency, Lucene practical form with the `+1`
/// floor. `n` = document count, `df` = documents containing the term.
///
/// This is the single definition of IDF. `search`, block-max construction
/// and the posting cursor all call it, so an upper bound computed for
/// pruning can never drift from the score it is bounding.
#[inline]
pub(crate) fn bm25_idf(n: f32, df: f32) -> f32 {
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

/// Score contribution of one posting: `idf` for the term, `tf` within the
/// document, `dl` document length, `avg_dl` mean document length.
///
/// This is the single definition of the BM25 term score. See `bm25_idf`.
#[inline]
pub(crate) fn bm25_impact(idf: f32, tf: u32, dl: u32, avg_dl: f32) -> f32 {
    let tf_f = tf as f32;
    let denom = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl as f32 / avg_dl.max(1.0));
    idf * tf_f * (BM25_K1 + 1.0) / denom.max(1e-6)
}

/// Identifier for a tokenizer choice. Persisted with each index so a
/// query at read time uses the same tokenizer that built the postings.
///
/// Adding a new variant requires updating [`run_tokenizer`] to dispatch
/// to its implementation. External tokenizers (Kuromoji, Jieba,
/// SentencePiece) can be slotted in here behind feature flags without
/// touching the index format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    /// Built-in: lowercase ASCII word runs + unigrams and overlapping
    /// bigrams for CJK characters. Good default for mixed-script and
    /// Japanese / Chinese / Korean corpora without bundling Kuromoji.
    #[default]
    CjkBigram,
    /// Pure whitespace + punctuation split, lowercased. Cheaper and
    /// smaller postings; fine for ASCII / Latin-script workloads where
    /// CJK is rare.
    Whitespace,
    /// Lindera-backed morphological tokenizer (Kuromoji-compatible
    /// IPADIC). Activated only when the `lindera_tokenizer` cargo
    /// feature is enabled — otherwise this variant deserializes
    /// successfully but tokenization falls back to `CjkBigram` at
    /// runtime, with a one-line warning logged the first time it's
    /// touched. That keeps a segment serialized with `lindera` from
    /// breaking a binary built without the feature.
    Lindera,
}

impl TokenizerKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CjkBigram => "cjk_bigram",
            Self::Whitespace => "whitespace",
            Self::Lindera => "lindera",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cjk_bigram" | "cjk-bigram" | "default" => Some(Self::CjkBigram),
            "whitespace" | "ws" => Some(Self::Whitespace),
            "lindera" | "kuromoji" | "ja" => Some(Self::Lindera),
            _ => None,
        }
    }
}

/// Token-filter configuration persisted alongside the tokenizer.
///
/// All flags default to `false` so that older indexes (which lack this
/// field in their JSON) load cleanly and continue to behave identically
/// to before this feature was added.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AnalyzerConfig {
    /// Strip HTML tags and decode common HTML entities before tokenizing.
    #[serde(default)]
    pub strip_html: bool,
    /// Drop English stopwords after tokenizing.
    #[serde(default)]
    pub remove_stopwords: bool,
    /// Apply Snowball English stemming to each remaining token.
    #[serde(default)]
    pub stem_english: bool,
}

impl AnalyzerConfig {
    /// No filters — identical to the original behaviour. Used by `new()`.
    pub fn none() -> Self {
        Self::default()
    }

    /// Full English analyzer: HTML strip + stopword removal + English stemming.
    pub fn english() -> Self {
        Self {
            strip_html: true,
            remove_stopwords: true,
            stem_english: true,
        }
    }
}

/// English stopwords (ISO 639-1 "en", matching the Lucene/ES default set).
static ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "as", "up", "out", "so", "yet", "both", "either", "that", "this", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "their",
];

/// Per-segment BM25 inverted index.
///
/// `postings` is `term → Vec<(doc_id, term_freq, positions)>` sorted by
/// doc_id at save time so queries can short-circuit AND-style scoring
/// later. Positions are sorted ascending and record the raw tokenizer
/// output position (before stopword removal) so that phrase queries
/// correctly account for gaps left by removed stopwords — mirroring
/// Lucene's `positionIncrement` semantics.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Bm25Index {
    postings: HashMap<String, Vec<(u32, u32, Vec<u32>)>>,
    /// `doc_lengths[doc_id]` = token count for that doc.
    doc_lengths: Vec<u32>,
    /// Sum of all `doc_lengths` entries, kept incrementally so `avg_dl`
    /// is O(1).
    total_tokens: u64,
    /// Tokenizer used to build (and that must be used to query) this
    /// index. Persisted so a sealed segment can answer queries with the
    /// same tokenizer that built its postings even if the schema default
    /// is later changed.
    #[serde(default)]
    tokenizer: TokenizerKind,
    /// Token-filter configuration. Defaults to all-false (no filters) so
    /// older persisted indexes that lack this field load cleanly.
    #[serde(default)]
    analyzer: AnalyzerConfig,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a specific tokenizer. Use this for fields whose
    /// schema declares a non-default tokenizer.
    pub fn with_tokenizer(tokenizer: TokenizerKind) -> Self {
        Self {
            tokenizer,
            ..Self::default()
        }
    }

    /// Construct with a tokenizer and a full `AnalyzerConfig`.
    pub fn with_analyzer(tokenizer: TokenizerKind, analyzer: AnalyzerConfig) -> Self {
        Self {
            tokenizer,
            analyzer,
            ..Self::default()
        }
    }

    pub fn tokenizer(&self) -> TokenizerKind {
        self.tokenizer
    }

    /// Add `text` for `doc_id`. Caller is responsible for inserting in
    /// increasing doc_id order (the segment writer does this).
    pub fn insert(&mut self, doc_id: u32, text: &str) {
        let positional_tokens = run_analyzer_positional(self.tokenizer, &self.analyzer, text);
        // doc_length = number of raw tokenizer positions (including removed
        // stopwords), so BM25 length normalization stays consistent with the
        // positional counting used for phrase queries.
        let raw_len = run_tokenizer(self.tokenizer, text).len() as u32;

        // Aggregate term frequencies and collect positions for this doc.
        let mut term_data: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
        for (term, pos) in positional_tokens {
            let entry = term_data.entry(term).or_insert((0, Vec::new()));
            entry.0 += 1;
            entry.1.push(pos);
        }
        for (term, (tf, mut positions)) in term_data {
            positions.sort_unstable();
            self.postings
                .entry(term)
                .or_default()
                .push((doc_id, tf, positions));
        }

        while self.doc_lengths.len() <= doc_id as usize {
            self.doc_lengths.push(0);
        }
        self.doc_lengths[doc_id as usize] = raw_len;
        self.total_tokens += raw_len as u64;
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_lengths.len() as u32
    }

    pub fn avg_doc_length(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            0.0
        } else {
            (self.total_tokens as f64 / self.doc_lengths.len() as f64) as f32
        }
    }

    /// Rank docs by BM25 score against `query`. Returns `(doc_id, score)`
    /// pairs sorted by score desc, truncated to `k`. Docs with zero
    /// overlap on tokenized terms are omitted.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        let n = self.doc_count() as f32;
        if n == 0.0 || k == 0 {
            return Vec::new();
        }
        let avg_dl = self.avg_doc_length();
        let q_tokens = run_analyzer(self.tokenizer, &self.analyzer, query);

        // Dedup query terms — repeated tokens don't add new evidence in
        // standard BM25 (we already sum across all matching docs).
        let mut seen: std::collections::HashSet<&String> =
            std::collections::HashSet::with_capacity(q_tokens.len());
        let mut unique_q_tokens: Vec<&String> = Vec::with_capacity(q_tokens.len());
        for t in &q_tokens {
            if seen.insert(t) {
                unique_q_tokens.push(t);
            }
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in unique_q_tokens {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = bm25_idf(n, df);
            for &(doc_id, tf, _) in postings {
                let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0);
                *scores.entry(doc_id).or_insert(0.0) += bm25_impact(idf, tf, dl, avg_dl);
            }
        }
        let mut result: Vec<(u32, f32)> = scores.into_iter().collect();
        // Score descending, then doc_id ascending. The doc_id tie-break is
        // what makes the result a total order: without it, equal-scoring
        // docs come back in HashMap iteration order, which is neither
        // stable across runs nor comparable against a second implementation.
        result.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        result.truncate(k);
        result
    }

    /// Phrase / proximity search. Returns `(doc_id, bm25_score)` pairs for
    /// docs where the analyzed query terms appear in order with at most
    /// `slop` extra intervening positions between consecutive terms.
    ///
    /// `slop = 0` requires strict adjacency (`p_{i+1} == p_i + 1`).
    /// `slop > 0` allows the sum of gaps (`p_{i+1} - p_i - 1` for each
    /// consecutive pair) to be ≤ `slop`, with positions strictly increasing.
    ///
    /// Positions are the raw tokenizer positions (before stopword removal),
    /// so a removed stopword still occupies a position slot — a phrase
    /// that spans a stopword will NOT falsely match at slop=0 (Lucene
    /// positionIncrement semantics).
    ///
    /// Matched docs are ranked by the bag-of-words BM25 score (reusing
    /// the existing `search` scoring) and the top `k` are returned.
    pub fn search_phrase(&self, phrase: &str, slop: u32, k: usize) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Analyze the phrase to get the ordered query terms with their
        // relative positions. We use the positional form so the query's
        // own stopwords are handled consistently with index time.
        let q_terms_pos = run_analyzer_positional(self.tokenizer, &self.analyzer, phrase);
        if q_terms_pos.is_empty() {
            return Vec::new();
        }

        // Collect just the ordered terms for the candidate intersection.
        let q_terms: Vec<&str> = q_terms_pos.iter().map(|(t, _)| t.as_str()).collect();

        // Single-term phrase is equivalent to a BM25 search restricted to
        // docs containing that one term.
        if q_terms.len() == 1 {
            return self.search(phrase, k);
        }

        // Candidate docs: intersection of postings for ALL query terms.
        // Build a map: doc_id → per-term position lists (in q_terms order).
        let mut candidates: Option<HashMap<u32, Vec<Vec<u32>>>> = None;
        for (qi, term) in q_terms.iter().enumerate() {
            let Some(postings) = self.postings.get(*term) else {
                // Term not in index — no results possible.
                return Vec::new();
            };
            match candidates {
                None => {
                    let mut map: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
                    for (doc_id, _tf, positions) in postings {
                        let mut per_term = vec![Vec::new(); q_terms.len()];
                        per_term[qi] = positions.clone();
                        map.insert(*doc_id, per_term);
                    }
                    candidates = Some(map);
                }
                Some(ref mut map) => {
                    // Intersect: keep only docs already in the map.
                    let posting_ids: HashMap<u32, &Vec<u32>> =
                        postings.iter().map(|(id, _, pos)| (*id, pos)).collect();
                    map.retain(|doc_id, per_term| {
                        if let Some(pos) = posting_ids.get(doc_id) {
                            per_term[qi] = (*pos).clone();
                            true
                        } else {
                            false
                        }
                    });
                }
            }
        }

        let candidates = match candidates {
            Some(m) => m,
            None => return Vec::new(),
        };

        if candidates.is_empty() {
            return Vec::new();
        }

        // For each candidate, verify a positional phrase match.
        // We need to find positions p_0, p_1, ..., p_{m-1} (one per query
        // term in order) such that:
        //   p_0 < p_1 < ... < p_{m-1}
        //   sum_{i=0}^{m-2} (p_{i+1} - p_i - 1) <= slop
        //
        // Algorithm: iterate over all positions of the first term as
        // anchor; for each anchor greedily pick the smallest valid
        // position for each subsequent term.
        let mut matched: Vec<u32> = Vec::new();
        for (doc_id, per_term) in &candidates {
            if phrase_match_with_slop(per_term, slop) {
                matched.push(*doc_id);
            }
        }

        if matched.is_empty() {
            return Vec::new();
        }

        // Score matched docs with the standard BM25 bag-of-words score.
        let bm25_hits = self.search(phrase, usize::MAX);
        let score_map: HashMap<u32, f32> = bm25_hits.into_iter().collect();

        let mut result: Vec<(u32, f32)> = matched
            .into_iter()
            .map(|id| (id, *score_map.get(&id).unwrap_or(&0.0)))
            .collect();
        // Score descending, then doc_id ascending — see the identical
        // comparator in `search` for why the tie-break matters.
        result.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        result.truncate(k);
        result
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| VedaError::Storage(format!("bm25 save serialize: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let idx: Self = serde_json::from_slice(&bytes)
            .map_err(|e| VedaError::Storage(format!("bm25 load parse: {e}")))?;
        Ok(idx)
    }
}

/// Check whether the per-term position lists in `per_term` (indexed by
/// query-term order) contain a valid phrase match with at most `slop`
/// total extra positions.
///
/// For each anchor position of term 0, greedily advance each subsequent
/// term to the smallest position that is > the previous term's chosen
/// position, then accumulate the gap. If total gap ≤ slop, it's a match.
fn phrase_match_with_slop(per_term: &[Vec<u32>], slop: u32) -> bool {
    if per_term.is_empty() {
        return false;
    }
    // Anchors are positions of the first term.
    'anchor: for &anchor in &per_term[0] {
        let mut prev = anchor;
        let mut total_gap: u32 = 0;
        for pos_list in per_term.iter().skip(1) {
            // Find smallest position in this term's list that is > prev.
            // Binary search for the first element > prev.
            let idx = pos_list.partition_point(|&p| p <= prev);
            if idx >= pos_list.len() {
                // No valid position for this term after prev — try next anchor.
                continue 'anchor;
            }
            let next = pos_list[idx];
            let gap = next - prev - 1; // extra positions between consecutive terms
            total_gap += gap;
            if total_gap > slop {
                continue 'anchor;
            }
            prev = next;
        }
        return true;
    }
    false
}

/// Full analyzer pipeline: char filters → tokenizer → token filters.
///
/// This is the single entry point called at BOTH index time (in `insert`)
/// and query time (in `search`), guaranteeing symmetric analysis.
pub fn run_analyzer(kind: TokenizerKind, cfg: &AnalyzerConfig, text: &str) -> Vec<String> {
    run_analyzer_positional(kind, cfg, text)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// Positional analyzer pipeline. Returns `(term, position)` pairs where
/// `position` is the raw tokenizer output index BEFORE stopword removal.
///
/// This means that a removed stopword still consumes a position slot, so
/// phrase queries across a stopword boundary correctly detect the gap
/// (Lucene positionIncrement semantics). Stemming is applied after
/// stopword removal but positions are assigned before both filters.
pub fn run_analyzer_positional(
    kind: TokenizerKind,
    cfg: &AnalyzerConfig,
    text: &str,
) -> Vec<(String, u32)> {
    // 1. Char filter: strip HTML tags + decode common entities.
    let cleaned: String;
    let text = if cfg.strip_html {
        cleaned = strip_html(text);
        &cleaned
    } else {
        text
    };

    // 2. Tokenize via the selected tokenizer — produces raw tokens.
    let raw_tokens = run_tokenizer(kind, text);

    // 3. Assign positions (raw tokenizer index) and apply token filters,
    //    keeping the original position even when a token is dropped.
    let stemmer = if cfg.stem_english {
        Some(Stemmer::create(Algorithm::English))
    } else {
        None
    };

    let mut result: Vec<(String, u32)> = Vec::with_capacity(raw_tokens.len());
    for (pos, token) in raw_tokens.into_iter().enumerate() {
        // Stopword filter: drop the token but do NOT increment result — the
        // position slot is "consumed" by being absent in the output, which
        // is exactly what we want for phrase-gap detection.
        if cfg.remove_stopwords && ENGLISH_STOPWORDS.contains(&token.as_str()) {
            continue;
        }
        let term = if let Some(ref s) = stemmer {
            s.stem(&token).into_owned()
        } else {
            token
        };
        result.push((term, pos as u32));
    }

    result
}

/// Remove HTML tags (`<...>`) and decode the most common named and
/// numeric entities. Tags become spaces; entities become their plain-text
/// equivalents. The result is suitable for passing on to a tokenizer.
fn strip_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    let mut entity_buf = String::new();
    let mut in_entity = false;

    for c in text.chars() {
        match c {
            '<' => {
                if in_entity {
                    // Malformed entity — flush literally.
                    out.push('&');
                    out.push_str(&entity_buf);
                    entity_buf.clear();
                    in_entity = false;
                }
                in_tag = true;
            }
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if in_tag => { /* consume tag content */ }
            '&' => {
                in_entity = true;
                entity_buf.clear();
            }
            ';' if in_entity => {
                let decoded = decode_entity(&entity_buf);
                out.push_str(decoded);
                entity_buf.clear();
                in_entity = false;
            }
            _ if in_entity => {
                entity_buf.push(c);
                // Safety valve: if the buffer grows beyond a reasonable
                // entity length, treat as literal text.
                if entity_buf.len() > 10 {
                    out.push('&');
                    out.push_str(&entity_buf);
                    entity_buf.clear();
                    in_entity = false;
                }
            }
            _ => out.push(c),
        }
    }
    // Flush any dangling entity.
    if in_entity {
        out.push('&');
        out.push_str(&entity_buf);
    }
    out
}

/// Map a bare entity name (without `&` / `;`) to its replacement text.
fn decode_entity(name: &str) -> &'static str {
    match name {
        "amp" => "and",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" | "#39" => "'",
        "nbsp" => " ",
        _ => "",
    }
}

/// Dispatch to the selected tokenizer implementation. New variants are
/// plumbed by extending [`TokenizerKind`] and adding a branch here.
fn run_tokenizer(kind: TokenizerKind, text: &str) -> Vec<String> {
    match kind {
        TokenizerKind::CjkBigram => tokenize_cjk_bigram(text),
        TokenizerKind::Whitespace => tokenize_whitespace(text),
        TokenizerKind::Lindera => tokenize_lindera(text),
    }
}

/// Lindera-backed morphological tokenizer.
///
/// Two-layer activation:
/// 1. The `lindera_tokenizer` cargo feature pulls the lindera crate
///    into the binary. Without it, this function is a one-line stub
///    that warns and falls back to `tokenize_cjk_bigram`.
/// 2. Even with the feature on, the operator must supply a dict by
///    setting `VEDA_LINDERA_DICT_PATH` to a directory containing a
///    pre-built IPADIC / UniDic / etc. lindera dictionary. We
///    deliberately do not bundle one — auto-downloading dicts at
///    build time breaks sandboxed builds.
///
/// When neither layer is satisfied, the fallback path keeps segments
/// built or queried by feature-off binaries readable.
#[cfg(feature = "lindera_tokenizer")]
fn tokenize_lindera(text: &str) -> Vec<String> {
    use std::sync::OnceLock;
    static TOK: OnceLock<Option<lindera::tokenizer::Tokenizer>> = OnceLock::new();
    let tok_opt = TOK.get_or_init(|| {
        let path = match std::env::var("VEDA_LINDERA_DICT_PATH") {
            Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => {
                tracing::warn!(
                    "lindera_tokenizer feature is on but VEDA_LINDERA_DICT_PATH is unset; \
                     falling back to cjk_bigram. Point it at a lindera-compatible dict \
                     directory (e.g. IPADIC extracted from \
                     https://lindera.s3.ap-northeast-1.amazonaws.com/mecab-ipadic-2.7.0-20070801.tar.gz)."
                );
                return None;
            }
        };
        match lindera::tokenizer::Tokenizer::from_config(
            lindera::tokenizer::TokenizerConfig {
                dictionary: lindera::dictionary::DictionaryConfig {
                    kind: None,
                    path: Some(path),
                },
                user_dictionary: None,
                mode: lindera::mode::Mode::Normal,
            },
        ) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::error!(error = %e, "lindera dict load failed; falling back to cjk_bigram");
                None
            }
        }
    });
    let Some(tok) = tok_opt else {
        return tokenize_cjk_bigram(text);
    };
    match tok.tokenize(text) {
        Ok(toks) => toks
            .into_iter()
            .map(|t| t.text.to_lowercase())
            .filter(|s| !s.trim().is_empty())
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(not(feature = "lindera_tokenizer"))]
fn tokenize_lindera(text: &str) -> Vec<String> {
    use std::sync::Once;
    static WARN: Once = Once::new();
    WARN.call_once(|| {
        tracing::warn!(
            "bm25 field requested lindera tokenizer but the lindera_tokenizer \
             cargo feature is not enabled; falling back to cjk_bigram. \
             Build with `--features veda-storage/lindera_tokenizer` to enable."
        );
    });
    tokenize_cjk_bigram(text)
}

/// Built-in default: lowercase ASCII word runs + unigrams + overlapping
/// 2-char bigrams for CJK runs.
///
/// Two regimes mixed together:
/// - Latin / digits: split on Unicode word boundaries, lowercase ASCII.
/// - CJK (Hiragana, Katakana, CJK ideographs, Hangul): emit each
///   character as a unigram AND emit overlapping 2-char bigrams.
fn tokenize_cjk_bigram(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut cjk_run: Vec<char> = Vec::new();

    let flush_alnum = |current: &mut String, tokens: &mut Vec<String>| {
        if !current.is_empty() {
            tokens.push(current.to_lowercase());
            current.clear();
        }
    };
    let flush_cjk = |run: &mut Vec<char>, tokens: &mut Vec<String>| {
        // Unigrams
        for c in run.iter() {
            tokens.push(c.to_string());
        }
        // Overlapping bigrams — only useful when run length >= 2.
        if run.len() >= 2 {
            for window in run.windows(2) {
                let mut bg = String::with_capacity(6);
                bg.push(window[0]);
                bg.push(window[1]);
                tokens.push(bg);
            }
        }
        run.clear();
    };

    for c in text.chars() {
        if is_cjk(c) {
            flush_alnum(&mut current, &mut tokens);
            cjk_run.push(c);
        } else if c.is_alphanumeric() {
            flush_cjk(&mut cjk_run, &mut tokens);
            current.push(c);
        } else {
            flush_alnum(&mut current, &mut tokens);
            flush_cjk(&mut cjk_run, &mut tokens);
        }
    }
    flush_alnum(&mut current, &mut tokens);
    flush_cjk(&mut cjk_run, &mut tokens);

    tokens
}

/// Whitespace tokenizer: lowercase + split on Unicode word boundaries.
/// Skips the CJK bigram augmentation — cheaper and produces smaller
/// postings; recommended for ASCII / Latin-script workloads.
fn tokenize_whitespace(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.push(c);
        } else if !current.is_empty() {
            tokens.push(current.to_lowercase());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }
    tokens
}

/// True for Hiragana, Katakana, CJK Unified Ideographs (basic + Ext A),
/// CJK compatibility ideographs, and Hangul syllables. Caller-side
/// tokenizers may extend this if exotic scripts matter.
fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    (0x3040..=0x309F).contains(&cp)   // Hiragana
        || (0x30A0..=0x30FF).contains(&cp) // Katakana
        || (0x4E00..=0x9FFF).contains(&cp) // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&cp) // CJK Ext A
        || (0xF900..=0xFAFF).contains(&cp) // CJK Compatibility
        || (0xAC00..=0xD7AF).contains(&cp) // Hangul Syllables
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_tokenization_lowercases_and_splits() {
        let toks = tokenize_cjk_bigram("Hello, World! BM25 search.");
        assert_eq!(toks, vec!["hello", "world", "bm25", "search"]);
    }

    #[test]
    fn cjk_emits_unigrams_and_bigrams() {
        let toks = tokenize_cjk_bigram("ランサム");
        // Unigrams: ラ, ン, サ, ム; bigrams: ラン, ンサ, サム
        assert_eq!(toks.len(), 7);
        assert!(toks.contains(&"ラ".to_owned()));
        assert!(toks.contains(&"ラン".to_owned()));
        assert!(toks.contains(&"サム".to_owned()));
    }

    #[test]
    fn mixed_script_runs_tokenize_independently() {
        let toks = tokenize_cjk_bigram("Adobe Acrobat とPDF");
        assert!(toks.contains(&"adobe".to_owned()));
        assert!(toks.contains(&"acrobat".to_owned()));
        assert!(toks.contains(&"と".to_owned()));
        assert!(toks.contains(&"pdf".to_owned()));
    }

    #[test]
    fn bm25_ranks_exact_match_higher() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "The quick brown fox jumps over the lazy dog");
        idx.insert(1, "Rust is a systems programming language");
        idx.insert(2, "A rust-coloured fox in the forest");

        let hits = idx.search("rust fox", 3);
        assert!(!hits.is_empty(), "should return at least one hit");
        // Doc 2 has both "rust" and "fox" — should rank first.
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn bm25_handles_japanese_query() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "ランサムウェア対策の最新情報");
        idx.insert(1, "Adobe Acrobat の使い方ガイド");
        idx.insert(2, "サイバーセキュリティ概論");

        let hits = idx.search("ランサムウェア", 3);
        assert!(!hits.is_empty());
        // Doc 0 explicitly mentions ランサムウェア — should rank first.
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn lindera_variant_falls_back_when_feature_off() {
        // With the lindera feature off, the variant deserializes and
        // tokenization works (via the CJK-bigram fallback). The point
        // of this test is to lock in the "doesn't panic / doesn't
        // return empty" contract.
        let toks = run_tokenizer(TokenizerKind::Lindera, "ランサムウェア対策");
        assert!(!toks.is_empty(), "fallback should still produce tokens");
    }

    #[test]
    fn lindera_name_parses() {
        assert_eq!(
            TokenizerKind::from_name("lindera"),
            Some(TokenizerKind::Lindera)
        );
        assert_eq!(
            TokenizerKind::from_name("kuromoji"),
            Some(TokenizerKind::Lindera)
        );
        assert_eq!(TokenizerKind::from_name("ja"), Some(TokenizerKind::Lindera));
    }

    #[test]
    fn whitespace_tokenizer_skips_cjk_bigrams() {
        let toks = tokenize_whitespace("ランサム Adobe");
        // Each CJK char becomes a unigram, but NO bigrams. Plus "adobe".
        assert!(toks.contains(&"adobe".to_owned()));
        // No bigrams like "ラン" should appear.
        assert!(
            !toks
                .iter()
                .any(|t| t.chars().filter(|c| is_cjk(*c)).count() == 2)
        );
    }

    #[test]
    fn tokenizer_choice_is_persisted_across_save_load() {
        let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
        idx.insert(0, "Hello world");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bm25.idx");
        idx.save(&path).unwrap();
        let loaded = Bm25Index::load(&path).unwrap();
        assert_eq!(loaded.tokenizer(), TokenizerKind::Whitespace);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "alpha bravo charlie");
        idx.insert(1, "alpha alpha bravo");
        idx.insert(2, "delta echo");

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bm25.idx");
        idx.save(&path).unwrap();
        let loaded = Bm25Index::load(&path).unwrap();

        assert_eq!(loaded.doc_count(), 3);
        let h1 = idx.search("alpha", 3);
        let h2 = loaded.search("alpha", 3);
        assert_eq!(h1, h2);
    }

    // ── New tests for AnalyzerConfig / ES-parity features ──────────────

    #[test]
    fn stemming_matches_singular_and_plural() {
        // With English stemming, a doc containing "foxes" must be found by querying "fox".
        let mut idx =
            Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::english());
        idx.insert(0, "the quick brown foxes jumped");
        idx.insert(1, "a lazy dog");
        let hits = idx.search("fox", 10);
        assert!(
            hits.iter().any(|(id, _)| *id == 0),
            "querying 'fox' should match doc 0 ('foxes') via stemming; hits={hits:?}"
        );
    }

    #[test]
    fn stopwords_are_dropped() {
        let cfg = AnalyzerConfig::english();
        let toks = run_analyzer(TokenizerKind::Whitespace, &cfg, "the quick a fox");
        assert!(
            !toks.contains(&"the".to_string()) && !toks.contains(&"a".to_string()),
            "stopwords dropped: {toks:?}"
        );
        // "fox" → stemmed to "fox" by Snowball (already a stem).
        assert!(
            toks.iter().any(|t| t.starts_with("fox") || t == "fox"),
            "content words kept: {toks:?}"
        );
    }

    #[test]
    fn char_filter_strips_html_and_entities() {
        let cfg = AnalyzerConfig::english();
        let toks = run_analyzer(TokenizerKind::Whitespace, &cfg, "<p>Tom &amp; Jerry</p>");
        // HTML tags removed; &amp; mapped to "and" (a stopword → may be dropped).
        assert!(
            toks.iter().any(|t| t == "tom"),
            "html stripped, 'tom' present: {toks:?}"
        );
        assert!(
            toks.iter().any(|t| t == "jerri" || t == "jerry"),
            "'jerry' present (stemmed ok): {toks:?}"
        );
        assert!(
            !toks.iter().any(|t| t.contains('<') || t.contains("&amp")),
            "no markup/entities: {toks:?}"
        );
    }

    #[test]
    fn analyzer_config_persists_and_applies_at_query_time() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bm25.bin");
        let mut idx =
            Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::english());
        idx.insert(0, "running runners run");
        idx.save(&p).unwrap();
        let idx2 = Bm25Index::load(&p).unwrap();
        // The persisted analyzer (stemming) must apply to the QUERY too:
        // "run" stems to "run"; the indexed tokens are also stemmed → match.
        assert!(
            idx2.search("run", 10).iter().any(|(id, _)| *id == 0),
            "stemmed query must match stemmed indexed tokens after save/load"
        );
    }

    #[test]
    fn none_analyzer_is_backward_compatible() {
        // AnalyzerConfig::none() must produce the exact same tokens as the
        // old run_tokenizer path (no filters applied at all).
        let text = "The quick brown Fox";
        let old = run_tokenizer(TokenizerKind::Whitespace, text);
        let new = run_analyzer(TokenizerKind::Whitespace, &AnalyzerConfig::none(), text);
        assert_eq!(
            old, new,
            "none() analyzer must be byte-identical to the old tokenizer"
        );
    }

    // ── Feature B: phrase / proximity query tests ──────────────────────

    #[test]
    fn phrase_query_requires_adjacency() {
        let mut idx = Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::none());
        idx.insert(0, "the quick brown fox jumps");
        idx.insert(1, "the brown quick fox"); // same words, wrong order
        idx.insert(2, "quick brown"); // adjacent but missing 'fox'
        let hits = idx.search_phrase("quick brown fox", 0, 10); // slop=0 => exact adjacency
        let ids: Vec<u32> = hits.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&0), "doc 0 has the phrase adjacent");
        assert!(!ids.contains(&1), "doc 1 has the words but not adjacent");
        assert!(!ids.contains(&2), "doc 2 missing 'fox'");
    }

    #[test]
    fn proximity_query_allows_slop() {
        let mut idx = Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::none());
        idx.insert(0, "quick brown fox");
        idx.insert(1, "quick lazy brown fox"); // one extra word between quick and brown
        // slop=1 allows one intervening position between consecutive query terms.
        let ids: Vec<u32> = idx
            .search_phrase("quick brown", 1, 10)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert!(
            ids.contains(&0) && ids.contains(&1),
            "slop=1 matches both: {ids:?}"
        );
        let ids0: Vec<u32> = idx
            .search_phrase("quick brown", 0, 10)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert!(
            ids0.contains(&0) && !ids0.contains(&1),
            "slop=0 excludes the gapped doc: {ids0:?}"
        );
    }

    #[test]
    fn positions_survive_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bm25.bin");
        let mut idx = Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::none());
        idx.insert(0, "alpha beta gamma");
        idx.save(&p).unwrap();
        let idx2 = Bm25Index::load(&p).unwrap();
        assert!(
            idx2.search_phrase("beta gamma", 0, 10)
                .iter()
                .any(|(id, _)| *id == 0)
        );
    }

    #[test]
    fn stopword_gap_blocks_false_phrase() {
        // With stopword removal, positions must still increment for the removed word,
        // so a phrase spanning a removed stopword is NOT falsely adjacent.
        let mut idx =
            Bm25Index::with_analyzer(TokenizerKind::Whitespace, AnalyzerConfig::english());
        idx.insert(0, "quick the brown"); // 'the' is a stopword between quick and brown
        // 'quick brown' as an exact phrase (slop 0) should NOT match because 'the' occupied a position.
        let ids: Vec<u32> = idx
            .search_phrase("quick brown", 0, 10)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert!(
            !ids.contains(&0),
            "removed stopword must leave a position gap (Lucene positionIncrement): {ids:?}"
        );
    }

    #[test]
    fn equal_scores_are_ordered_by_doc_id() {
        let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
        // Identical documents inserted out of any helpful order produce
        // identical BM25 scores, so only the tie-break decides the order.
        for doc_id in [4u32, 0, 3, 1, 2] {
            idx.insert(doc_id, "alpha beta");
        }
        let hits = idx.search("alpha", 5);
        let ids: Vec<u32> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2, 3, 4],
            "ties must break by doc_id ascending"
        );
    }

    #[test]
    fn search_is_deterministic_across_repeated_calls() {
        let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
        for doc_id in 0..50u32 {
            idx.insert(doc_id, "alpha beta gamma");
        }
        let first = idx.search("alpha beta", 10);
        for _ in 0..20 {
            assert_eq!(idx.search("alpha beta", 10), first);
        }
    }

    #[test]
    fn impact_helper_matches_inline_formula() {
        // Guards against the helper drifting from the expression it replaced.
        let idf = bm25_idf(100.0, 10.0);
        let got = bm25_impact(idf, 3, 120, 100.0);
        let tf_f = 3.0f32;
        let denom = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * 120.0 / 100.0f32.max(1.0));
        let want = idf * tf_f * (BM25_K1 + 1.0) / denom.max(1e-6);
        assert_eq!(got, want);
    }
}
