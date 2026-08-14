//! BM25 latency benchmark. `#[ignore]` so it never runs in normal
//! `cargo test` — matching the convention in
//! `crates/veda-index/tests/ann_benchmark.rs`.
//!
//! Run with:
//!   cargo test -p veda-storage --test bm25_bench -- --ignored --nocapture

use std::time::Instant;

use veda_storage::bm25_index::{Bm25Index, TokenizerKind};

/// Deterministic xorshift so runs are comparable across machines.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// Vocabulary with a Zipf-ish skew: term 0 is very common, later terms are
/// rare. This matters — uniform terms produce unrealistically short posting
/// lists and would understate both the baseline and any later improvement.
fn vocab(size: usize) -> Vec<String> {
    (0..size).map(|i| format!("t{i}")).collect()
}

fn build_corpus(docs: usize, vocab_size: usize, terms_per_doc: usize) -> Bm25Index {
    let words = vocab(vocab_size);
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..docs {
        let mut text = String::with_capacity(terms_per_doc * 6);
        for _ in 0..terms_per_doc {
            // Square the draw to skew toward low-numbered (common) terms.
            let r = rng.below(vocab_size as u32) as u64;
            let skewed = (r * r / vocab_size as u64) as usize;
            text.push_str(&words[skewed.min(vocab_size - 1)]);
            text.push(' ');
        }
        idx.insert(doc_id as u32, &text);
    }
    idx
}

fn queries(count: usize, vocab_size: usize, terms: usize) -> Vec<String> {
    let words = vocab(vocab_size);
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    (0..count)
        .map(|_| {
            (0..terms)
                .map(|_| {
                    let r = rng.below(vocab_size as u32) as u64;
                    let skewed = (r * r / vocab_size as u64) as usize;
                    words[skewed.min(vocab_size - 1)].clone()
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn run_case(label: &str, docs: usize, k: usize) {
    let idx = build_corpus(docs, 5_000, 120);
    let qs = queries(200, 5_000, 4);

    // Warm up so the first-touch page faults don't land in the measurement.
    for q in qs.iter().take(20) {
        std::hint::black_box(idx.search(q, k));
    }

    let start = Instant::now();
    let mut hits = 0usize;
    for q in &qs {
        hits += idx.search(q, k).len();
    }
    let elapsed = start.elapsed();
    let per_query_us = elapsed.as_secs_f64() * 1e6 / qs.len() as f64;

    println!(
        "{label}: docs={docs} k={k} queries={} total={:?} per_query={per_query_us:.1}us hits={hits}",
        qs.len(),
        elapsed
    );
}

#[test]
#[ignore]
fn bm25_baseline_10k() {
    run_case("baseline", 10_000, 10);
}

#[test]
#[ignore]
fn bm25_baseline_100k() {
    run_case("baseline", 100_000, 10);
}

#[test]
#[ignore]
fn bm25_baseline_100k_filtered_k() {
    // Mirrors the filtered path, where `bm25_fetch_k` requests 10 * k.
    run_case("baseline", 100_000, 100);
}

fn run_comparison(docs: usize, k: usize) {
    let plain = build_corpus(docs, 5_000, 120);
    let mut blocked = build_corpus(docs, 5_000, 120);

    let build_start = Instant::now();
    blocked.build_blocks();
    let build_ms = build_start.elapsed().as_secs_f64() * 1e3;

    let qs = queries(200, 5_000, 4);
    for q in qs.iter().take(20) {
        std::hint::black_box(plain.search(q, k));
        std::hint::black_box(blocked.search(q, k));
    }

    let start = Instant::now();
    for q in &qs {
        std::hint::black_box(plain.search(q, k));
    }
    let plain_us = start.elapsed().as_secs_f64() * 1e6 / qs.len() as f64;

    let start = Instant::now();
    let mut skipped = 0u64;
    let mut evaluated = 0u64;
    for q in &qs {
        let (hits, stats) = blocked.search_with_stats(q, k, 1.0);
        std::hint::black_box(hits);
        skipped += stats.blocks_skipped;
        evaluated += stats.docs_evaluated;
    }
    let blocked_us = start.elapsed().as_secs_f64() * 1e6 / qs.len() as f64;

    // Correctness alongside speed: a fast wrong answer is not a result.
    for q in qs.iter().take(50) {
        assert_eq!(blocked.search(q, k), plain.search(q, k), "query {q}");
    }

    println!(
        "compare: docs={docs} k={k} exhaustive={plain_us:.1}us bmw={blocked_us:.1}us \
         speedup={:.2}x build_blocks={build_ms:.1}ms blocks_skipped={skipped} docs_evaluated={evaluated}",
        plain_us / blocked_us.max(f64::MIN_POSITIVE)
    );
}

#[test]
#[ignore]
fn bm25_compare_10k() {
    run_comparison(10_000, 10);
}

#[test]
#[ignore]
fn bm25_compare_100k() {
    run_comparison(100_000, 10);
}

#[test]
#[ignore]
fn bm25_compare_100k_filtered_k() {
    run_comparison(100_000, 100);
}

/// Separate the three costs paid when a sealed segment's BM25 index is
/// opened: reading the file, parsing the JSON, and deriving block-max
/// metadata. `Bm25Index::load` does all three. The design doc claims the
/// derivation is cheap next to the parse — this measures whether that is
/// actually true, because the derivation is now paid on every segment open.
fn run_open_cost(docs: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.bm25");
    let idx = build_corpus(docs, 5_000, 120);
    idx.save(&path).unwrap();
    let bytes_on_disk = std::fs::metadata(&path).unwrap().len();

    let start = Instant::now();
    let bytes = std::fs::read(&path).unwrap();
    let read_ms = start.elapsed().as_secs_f64() * 1e3;

    let start = Instant::now();
    let mut parsed: Bm25Index = serde_json::from_slice(&bytes).unwrap();
    let parse_ms = start.elapsed().as_secs_f64() * 1e3;
    // `block_indexes` is #[serde(skip)], so a freshly parsed index has none.
    assert!(!parsed.has_blocks());

    let start = Instant::now();
    parsed.build_blocks();
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    assert!(parsed.has_blocks());

    let io_and_parse = read_ms + parse_ms;
    println!(
        "openbench: docs={docs} file={:.1}MB read={read_ms:.1}ms parse={parse_ms:.1}ms \
         build_blocks={build_ms:.1}ms build_over_read_parse={:.2}x",
        bytes_on_disk as f64 / 1_048_576.0,
        build_ms / io_and_parse.max(f64::MIN_POSITIVE)
    );
}

#[test]
#[ignore]
fn bm25_open_cost_10k() {
    run_open_cost(10_000);
}

#[test]
#[ignore]
fn bm25_open_cost_100k() {
    run_open_cost(100_000);
}
