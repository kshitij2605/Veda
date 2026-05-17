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

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};

/// Lucene-default BM25 parameter k1.
pub const BM25_K1: f32 = 1.2;
/// Lucene-default BM25 parameter b.
pub const BM25_B: f32 = 0.75;

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
}

impl TokenizerKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CjkBigram => "cjk_bigram",
            Self::Whitespace => "whitespace",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cjk_bigram" | "cjk-bigram" | "default" => Some(Self::CjkBigram),
            "whitespace" | "ws" => Some(Self::Whitespace),
            _ => None,
        }
    }
}

/// Per-segment BM25 inverted index.
///
/// `postings` is `term → Vec<(doc_id, term_freq)>` sorted by doc_id at
/// save time so queries can short-circuit AND-style scoring later. For
/// now it's bag-of-words scoring (OR-style sum across query terms).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Bm25Index {
    postings: HashMap<String, Vec<(u32, u32)>>,
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

    pub fn tokenizer(&self) -> TokenizerKind {
        self.tokenizer
    }

    /// Add `text` for `doc_id`. Caller is responsible for inserting in
    /// increasing doc_id order (the segment writer does this).
    pub fn insert(&mut self, doc_id: u32, text: &str) {
        let tokens = run_tokenizer(self.tokenizer, text);
        let len = tokens.len() as u32;

        // Aggregate term frequencies for this doc first.
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *term_counts.entry(t).or_insert(0) += 1;
        }
        for (term, tf) in term_counts {
            self.postings.entry(term).or_default().push((doc_id, tf));
        }

        while self.doc_lengths.len() <= doc_id as usize {
            self.doc_lengths.push(0);
        }
        self.doc_lengths[doc_id as usize] = len;
        self.total_tokens += len as u64;
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
        let q_tokens = run_tokenizer(self.tokenizer, query);

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
            // BM25 IDF with the +1 floor (Lucene practical form).
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc_id, tf) in postings {
                let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0) as f32;
                let tf_f = tf as f32;
                let denom = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_dl.max(1.0));
                let s = idf * tf_f * (BM25_K1 + 1.0) / denom.max(1e-6);
                *scores.entry(doc_id).or_insert(0.0) += s;
            }
        }
        let mut result: Vec<(u32, f32)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

/// Dispatch to the selected tokenizer implementation. New variants are
/// plumbed by extending [`TokenizerKind`] and adding a branch here.
fn run_tokenizer(kind: TokenizerKind, text: &str) -> Vec<String> {
    match kind {
        TokenizerKind::CjkBigram => tokenize_cjk_bigram(text),
        TokenizerKind::Whitespace => tokenize_whitespace(text),
    }
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
    fn whitespace_tokenizer_skips_cjk_bigrams() {
        let toks = tokenize_whitespace("ランサム Adobe");
        // Each CJK char becomes a unigram, but NO bigrams. Plus "adobe".
        assert!(toks.contains(&"adobe".to_owned()));
        // No bigrams like "ラン" should appear.
        assert!(!toks.iter().any(|t| t.chars().filter(|c| is_cjk(*c)).count() == 2));
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
}
