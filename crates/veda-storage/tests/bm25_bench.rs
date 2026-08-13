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
