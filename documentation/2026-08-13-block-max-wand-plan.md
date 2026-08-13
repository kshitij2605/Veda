# Block-Max WAND for BM25 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BM25 search skip postings that provably cannot enter the top-k, with results identical to today's exhaustive scoring by default.

**Architecture:** A domain-agnostic Block-Max WAND core in `veda-storage/src/block_max.rs`, driven through a `PostingCursor` trait. `bm25_index.rs` supplies a cursor over its existing doc-sorted posting vectors plus per-term block metadata derived at segment open. Sealed segments prune; growing segments keep the existing exhaustive path and serve as the differential-test oracle.

**Tech Stack:** Rust 2024, `veda-storage` and `veda-gateway` crates, `std::collections::BinaryHeap`, `serde` (skip-only, no format change).

**Spec:** `documentation/2026-08-13-block-max-wand-design.md` (committed at `b89c405`).

## Global Constraints

- Toolchain is pinned to **1.96.0** via `rust-toolchain.toml`. Do not change it.
- **All work happens on `gpu-host-a`.** Edit files locally, `scp` each touched file to the matching path under `/home/mac/test/project-mdb`, then build and test over SSH. Cargo is at `/home/mac/.cargo/bin/cargo`.
- **Run every cargo command in the FOREGROUND.** Do not background builds or tests, and do not poll for them.
- **Never** run `pkill`/`kill -9` against `cargo` or `rustc`, and never kill a process you did not start. A slow build is slow, not hung.
- **Never** run an unscoped workspace-wide `cargo test` or `cargo build`. Always scope with `-p veda-storage` or `-p veda-gateway`.
- **Never** run `git clean`, `git checkout -- .`, or `git stash` — another person works in this checkout concurrently. `git add` explicit file paths only, never `git add .`.
- Commit messages use conventional-commit prefixes and **must never mention Claude, AI assistance, or include a `Co-Authored-By` line**. Verify with `git log` after each commit.
- Branch is `feat/block-max-wand`, already created from `main`.
- `BLOCK_SIZE` is exactly **128**.
- Default `theta` is exactly **1.0** and must be rank-safe.
- **The pruning comparison is strictly `<`** (`upper_bound < theta * threshold` prunes). A candidate whose upper bound *equals* the threshold must still be evaluated, because an exact score tie is resolved by lower `doc_id` and could legitimately displace the current k-th result. Using `<=` here silently breaks rank-safety.
- All BM25 scoring — in `search`, in block-max construction, and in the cursor — must go through the single shared helper `bm25_impact`. Never write the formula out a second time.
- Gates before every commit: `cargo fmt --all --check`, and `cargo clippy -p <crate> --all-targets -- -D warnings` for each crate touched.

---

### Task 1: Baseline benchmark

Capture what BM25 costs today, before anything changes. Without this number, no later task can show the feature works.

**Files:**
- Create: `crates/veda-storage/tests/bm25_bench.rs`

**Interfaces:**
- Consumes: `veda_storage::bm25_index::{Bm25Index, TokenizerKind}` (existing public API).
- Produces: nothing consumed by later tasks. Task 9 re-runs this same file.

- [ ] **Step 1: Write the benchmark**

Create `crates/veda-storage/tests/bm25_bench.rs`:

```rust
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
```

- [ ] **Step 2: Verify it compiles and the ignored tests are discovered**

Run: `cargo test -p veda-storage --test bm25_bench -- --list`
Expected: three test names listed, all as `: test`.

- [ ] **Step 3: Run the benchmark and record the numbers**

Run: `cargo test -p veda-storage --test bm25_bench -- --ignored --nocapture`
Expected: three `baseline: docs=... per_query=...us` lines.

**Copy the three `per_query` figures verbatim into your report file.** Task 9 compares against them. A benchmark whose output is not recorded is wasted work.

- [ ] **Step 4: Commit**

```bash
git add crates/veda-storage/tests/bm25_bench.rs
git commit -m "test: baseline latency benchmark for BM25 search"
```

---

### Task 2: Shared scoring helper and deterministic ordering

Two prerequisites for a testable rank-safety claim: one scoring expression that both paths call, and a total order on results so two implementations can be compared at all.

**Files:**
- Modify: `crates/veda-storage/src/bm25_index.rs` (the `search` body around lines 218-260, `search_phrase` result sort around line 366)

**Interfaces:**
- Produces:
  - `pub(crate) fn bm25_idf(n: f32, df: f32) -> f32`
  - `pub(crate) fn bm25_impact(idf: f32, tf: u32, dl: u32, avg_dl: f32) -> f32`

  Tasks 4 and 5 call both. They are the only place the BM25 formula appears.

- [ ] **Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` block in `crates/veda-storage/src/bm25_index.rs`:

```rust
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
    assert_eq!(ids, vec![0, 1, 2, 3, 4], "ties must break by doc_id ascending");
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib bm25_index -- equal_scores_are_ordered_by_doc_id search_is_deterministic_across_repeated_calls impact_helper_matches_inline_formula`
Expected: compile error — `bm25_idf` and `bm25_impact` are not defined. The two ordering tests will also fail once it compiles, because there is no tie-breaker today.

- [ ] **Step 3: Add the helpers**

Add near `BM25_K1` / `BM25_B` at the top of `crates/veda-storage/src/bm25_index.rs`:

```rust
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
```

- [ ] **Step 4: Rewrite the `search` inner loop to call the helpers**

In `Bm25Index::search`, replace the inline `idf` computation and the `denom` / `s` lines with calls. The loop body becomes:

```rust
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
```

- [ ] **Step 5: Add the tie-breaker to both sorts**

In `Bm25Index::search`, replace the final sort with:

```rust
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
```

Apply the identical sort to the `result.sort_by(...)` at the end of `search_phrase`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib bm25_index`
Expected: PASS, including all pre-existing bm25 tests.

- [ ] **Step 7: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/bm25_index.rs
git commit -m "refactor(bm25): single scoring helper and deterministic result order"
```

---

### Task 3: The Block-Max WAND core

Domain-agnostic. This file must not mention BM25.

**Files:**
- Create: `crates/veda-storage/src/block_max.rs`
- Modify: `crates/veda-storage/src/lib.rs` (add the module declaration)

**Interfaces:**
- Produces:
  - `pub const BLOCK_SIZE: usize = 128;`
  - `pub trait PostingCursor` with `doc`, `advance_to`, `next`, `block_max`, `block_last_doc`, `list_max`, `score`
  - `pub struct PruneStats { pub docs_evaluated: u64, pub blocks_skipped: u64 }`
  - `pub fn block_max_wand<C: PostingCursor>(cursors: Vec<C>, k: usize, theta: f32) -> (Vec<(u32, f32)>, PruneStats)`

  Task 5 implements `PostingCursor` for BM25 and calls `block_max_wand`.

- [ ] **Step 1: Write the failing tests**

Create `crates/veda-storage/src/block_max.rs` containing only the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Test double: an explicit posting list of `(doc_id, score)` with
    /// block metadata derived the same way a real cursor derives it.
    struct VecCursor {
        postings: Vec<(u32, f32)>,
        blocks: Vec<(u32, f32)>,
        list_max: f32,
        pos: usize,
    }

    impl VecCursor {
        fn new(mut postings: Vec<(u32, f32)>) -> Self {
            postings.sort_by_key(|p| p.0);
            let mut blocks = Vec::new();
            for chunk in postings.chunks(BLOCK_SIZE) {
                let last_doc = chunk.last().map(|p| p.0).unwrap_or(0);
                let max = chunk.iter().map(|p| p.1).fold(f32::MIN, f32::max);
                blocks.push((last_doc, max));
            }
            let list_max = blocks.iter().map(|b| b.1).fold(f32::MIN, f32::max);
            Self { postings, blocks, list_max, pos: 0 }
        }
    }

    impl PostingCursor for VecCursor {
        fn doc(&self) -> Option<u32> {
            self.postings.get(self.pos).map(|p| p.0)
        }
        fn advance_to(&mut self, target: u32) {
            self.pos += self.postings[self.pos..].partition_point(|p| p.0 < target);
        }
        fn next(&mut self) {
            self.pos += 1;
        }
        fn block_max(&self) -> f32 {
            self.blocks.get(self.pos / BLOCK_SIZE).map(|b| b.1).unwrap_or(f32::MIN)
        }
        fn block_last_doc(&self) -> Option<u32> {
            self.blocks.get(self.pos / BLOCK_SIZE).map(|b| b.0)
        }
        fn list_max(&self) -> f32 {
            self.list_max
        }
        fn score(&self) -> f32 {
            self.postings[self.pos].1
        }
    }

    /// Reference implementation: score everything, sort, truncate. This is
    /// the oracle every correctness test compares against.
    fn exhaustive(lists: &[Vec<(u32, f32)>], k: usize) -> Vec<(u32, f32)> {
        let mut acc: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for list in lists {
            for &(doc, s) in list {
                *acc.entry(doc).or_insert(0.0) += s;
            }
        }
        let mut out: Vec<(u32, f32)> = acc.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k);
        out
    }

    fn cursors_for(lists: &[Vec<(u32, f32)>]) -> Vec<VecCursor> {
        lists.iter().cloned().map(VecCursor::new).collect()
    }

    /// xorshift, so failures reproduce exactly.
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
    }

    #[test]
    fn empty_input_returns_empty() {
        let (hits, _) = block_max_wand(Vec::<VecCursor>::new(), 10, 1.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn zero_k_returns_empty() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 0, 1.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn single_list_matches_exhaustive() {
        let lists = vec![vec![(5u32, 1.0f32), (1, 3.0), (9, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, 1.0);
        assert_eq!(hits, exhaustive(&lists, 2));
    }

    #[test]
    fn disjoint_lists_sum_correctly() {
        let lists = vec![
            vec![(1u32, 1.0f32), (2, 1.0)],
            vec![(2u32, 5.0f32), (3, 1.0)],
        ];
        let (hits, _) = block_max_wand(cursors_for(&lists), 3, 1.0);
        assert_eq!(hits, exhaustive(&lists, 3));
        assert_eq!(hits[0], (2, 6.0), "doc 2 appears in both lists");
    }

    #[test]
    fn k_larger_than_corpus() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 100, 1.0);
        assert_eq!(hits, exhaustive(&lists, 100));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn exact_score_ties_are_not_pruned() {
        // Every doc scores exactly 1.0. A `<=` pruning comparison would
        // stop after the first k and miss lower doc_ids that must win the
        // tie-break; `<` keeps them in play.
        let lists = vec![(0u32..500).map(|d| (d, 1.0f32)).collect::<Vec<_>>()];
        let (hits, _) = block_max_wand(cursors_for(&lists), 5, 1.0);
        assert_eq!(hits, exhaustive(&lists, 5));
        assert_eq!(
            hits.iter().map(|h| h.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn randomized_differential_against_exhaustive() {
        let mut rng = Rng(0xDEAD_BEEF_0000_0001);
        for trial in 0..300 {
            let n_lists = 1 + (rng.next_u32() % 6) as usize;
            let n_docs = 1 + (rng.next_u32() % 2000);
            let lists: Vec<Vec<(u32, f32)>> = (0..n_lists)
                .map(|_| {
                    let len = 1 + (rng.next_u32() % n_docs) as usize;
                    let mut seen = std::collections::HashSet::new();
                    let mut list: Vec<(u32, f32)> = Vec::new();
                    for _ in 0..len {
                        let doc = rng.next_u32() % n_docs;
                        if seen.insert(doc) {
                            // Quantized scores so exact ties actually occur.
                            let s = (rng.next_u32() % 20) as f32 / 4.0;
                            list.push((doc, s));
                        }
                    }
                    list.sort_by_key(|p| p.0);
                    list
                })
                .collect();
            for k in [1usize, 3, 10, 50] {
                let (hits, _) = block_max_wand(cursors_for(&lists), k, 1.0);
                assert_eq!(hits, exhaustive(&lists, k), "trial {trial} k {k}");
            }
        }
    }

    #[test]
    fn blocks_are_actually_skipped() {
        // One very common low-scoring term and one rare high-scoring term.
        // Once the heap fills from the rare term, most of the common term's
        // blocks cannot reach the threshold and must be skipped wholesale.
        let common: Vec<(u32, f32)> = (0u32..20_000).map(|d| (d, 0.01)).collect();
        let rare: Vec<(u32, f32)> = (0u32..40).map(|d| (d * 500, 50.0)).collect();
        let lists = vec![common, rare];
        let (hits, stats) = block_max_wand(cursors_for(&lists), 10, 1.0);
        assert_eq!(hits, exhaustive(&lists, 10));
        assert!(
            stats.blocks_skipped > 0,
            "no blocks skipped — the pruning path is inert"
        );
    }

    #[test]
    fn theta_above_one_prunes_at_least_as_much() {
        let common: Vec<(u32, f32)> = (0u32..20_000).map(|d| (d, 0.01)).collect();
        let rare: Vec<(u32, f32)> = (0u32..40).map(|d| (d * 500, 50.0)).collect();
        let lists = vec![common, rare];
        let (_, exact) = block_max_wand(cursors_for(&lists), 10, 1.0);
        let (approx_hits, approx) = block_max_wand(cursors_for(&lists), 10, 4.0);
        assert!(approx.blocks_skipped >= exact.blocks_skipped);
        assert!(approx_hits.len() <= 10);
    }

    #[test]
    fn theta_below_one_is_clamped_to_exact() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0), (3, 3.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, 0.1);
        assert_eq!(hits, exhaustive(&lists, 2));
    }

    #[test]
    fn non_finite_theta_is_clamped_to_exact() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0), (3, 3.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, f32::NAN);
        assert_eq!(hits, exhaustive(&lists, 2));
    }
}
```

- [ ] **Step 2: Register the module**

Add to `crates/veda-storage/src/lib.rs`, alongside the existing `pub mod bm25_index;`:

```rust
pub mod block_max;
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib block_max`
Expected: compile errors — `BLOCK_SIZE`, `PostingCursor`, `PruneStats` and `block_max_wand` do not exist.

- [ ] **Step 4: Implement the core**

Prepend to `crates/veda-storage/src/block_max.rs`, above the test module:

```rust
//! Block-Max WAND: disjunctive top-k retrieval that skips whole blocks of
//! postings whose best possible contribution cannot reach the current
//! top-k threshold.
//!
//! Deliberately domain-agnostic. Everything specific to a scoring model
//! lives behind `PostingCursor`, so the same core serves BM25 postings and
//! sparse-vector postings without change.
//!
//! Reference: Ding & Suel, "Faster top-k document retrieval using
//! block-max indexes", SIGIR 2011.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Postings per block. 128 is the standard choice: large enough that the
/// per-block metadata is negligible against the postings it covers, small
/// enough that a block's maximum stays a tight bound on its members.
pub const BLOCK_SIZE: usize = 128;

/// A cursor over one term's postings, positioned at a document.
///
/// Implementors must visit postings in ascending `doc_id` order — the
/// algorithm's skipping is only sound if the lists are doc-sorted.
pub trait PostingCursor {
    /// Current doc_id, or `None` once the list is exhausted.
    fn doc(&self) -> Option<u32>;

    /// Advance to the first doc_id `>= target`, exhausting the cursor if
    /// there is none. Never moves backwards.
    fn advance_to(&mut self, target: u32);

    /// Advance past the current doc.
    fn next(&mut self);

    /// Upper bound on the contribution of any posting in the current
    /// block. Meaningless once exhausted.
    fn block_max(&self) -> f32;

    /// Last doc_id in the current block, used to skip the block wholesale.
    fn block_last_doc(&self) -> Option<u32>;

    /// Upper bound on the contribution of any posting in the whole list.
    /// Constant for the cursor's lifetime.
    fn list_max(&self) -> f32;

    /// Exact contribution of the current doc.
    fn score(&self) -> f32;
}

/// How much work the traversal avoided. Used by tests to prove the pruning
/// path is not inert, and by the benchmark to report effectiveness.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    /// Documents fully scored across all cursors.
    pub docs_evaluated: u64,
    /// Block alignments rejected without scoring any of their postings.
    pub blocks_skipped: u64,
}

/// Heap entry ordered so that `BinaryHeap::peek` yields the *worst* result
/// currently held — lowest score, and among equal scores the highest
/// doc_id, since the final order breaks ties by ascending doc_id.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    score: f32,
    doc: u32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then(self.doc.cmp(&other.doc))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Disjunctive top-k over `cursors`.
///
/// Returns `(doc_id, score)` sorted by score descending with ties broken by
/// doc_id ascending, truncated to `k`, plus traversal statistics.
///
/// `theta` scales the pruning threshold. `1.0` is rank-safe: the result is
/// identical to scoring every posting exhaustively. Values above `1.0`
/// prune more aggressively and may omit true top-k results. Values below
/// `1.0` and non-finite values are clamped to `1.0` — they could only make
/// the traversal slower without changing the result.
pub fn block_max_wand<C: PostingCursor>(
    mut cursors: Vec<C>,
    k: usize,
    theta: f32,
) -> (Vec<(u32, f32)>, PruneStats) {
    let mut stats = PruneStats::default();
    if k == 0 {
        return (Vec::new(), stats);
    }
    let theta = if theta.is_finite() && theta > 1.0 {
        theta
    } else {
        1.0
    };

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

    loop {
        cursors.retain(|c| c.doc().is_some());
        if cursors.is_empty() {
            break;
        }
        cursors.sort_by_key(|c| c.doc().unwrap_or(u32::MAX));

        // Until the heap is full nothing may be pruned, so the bound is
        // -inf rather than 0.0 — a corpus whose scores are all zero must
        // still fill k results.
        let bound = if heap.len() >= k {
            theta * heap.peek().map(|e| e.score).unwrap_or(0.0)
        } else {
            f32::NEG_INFINITY
        };

        // Pivot: the first cursor at which the cumulative list maxima could
        // still reach the threshold. `>=` (not `>`) because a candidate
        // that exactly ties the threshold can still win on doc_id.
        let mut acc = 0.0f32;
        let mut pivot = None;
        for (i, c) in cursors.iter().enumerate() {
            acc += c.list_max();
            if acc >= bound {
                pivot = Some(i);
                break;
            }
        }
        let Some(p) = pivot else {
            // No remaining document can reach the threshold.
            break;
        };
        let pivot_doc = match cursors[p].doc() {
            Some(d) => d,
            None => break,
        };

        let first_doc = match cursors[0].doc() {
            Some(d) => d,
            None => break,
        };

        if first_doc < pivot_doc {
            // Cursors trail the pivot: catch them up. Guaranteed progress,
            // since at least cursors[0] moves.
            for c in cursors[..p].iter_mut() {
                if c.doc().is_some_and(|d| d < pivot_doc) {
                    c.advance_to(pivot_doc);
                }
            }
            continue;
        }

        // Block-max check. `<` (not `<=`) preserves exact ties, as above.
        let block_sum: f32 = cursors[..=p].iter().map(|c| c.block_max()).sum();
        if block_sum < bound {
            stats.blocks_skipped += 1;
            let block_end = cursors[..=p]
                .iter()
                .filter_map(|c| c.block_last_doc())
                .min()
                .map_or(u32::MAX, |d| d.saturating_add(1));
            let next_after_pivot = cursors.get(p + 1).and_then(|c| c.doc()).unwrap_or(u32::MAX);
            // Never overshoot the next candidate, and always make progress.
            let target = block_end
                .min(next_after_pivot)
                .max(pivot_doc.saturating_add(1));
            for c in cursors[..=p].iter_mut() {
                if c.doc().is_some_and(|d| d < target) {
                    c.advance_to(target);
                }
            }
            continue;
        }

        // Score the pivot. Cursors are doc-sorted and cursors[0] is at the
        // pivot, so every cursor positioned there forms a prefix.
        let mut score = 0.0f32;
        for c in cursors.iter_mut() {
            if c.doc() == Some(pivot_doc) {
                score += c.score();
                c.next();
            } else {
                break;
            }
        }
        stats.docs_evaluated += 1;

        if heap.len() < k {
            heap.push(Entry { score, doc: pivot_doc });
        } else if let Some(worst) = heap.peek().copied()
            && (score > worst.score || (score == worst.score && pivot_doc < worst.doc))
        {
            heap.pop();
            heap.push(Entry { score, doc: pivot_doc });
        }
    }

    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|e| (e.doc, e.score)).collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    (out, stats)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib block_max`
Expected: all 11 tests PASS. `randomized_differential_against_exhaustive` is the one that matters — if it fails, the pivot or skip arithmetic is wrong. Do not weaken the test to make it pass.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/block_max.rs crates/veda-storage/src/lib.rs
git commit -m "feat(storage): block-max WAND core over a posting-cursor trait"
```

---

### Task 4: Block metadata for BM25 postings

**Files:**
- Modify: `crates/veda-storage/src/bm25_index.rs`

**Interfaces:**
- Consumes: `crate::block_max::BLOCK_SIZE`, `bm25_idf`, `bm25_impact` (Task 2).
- Produces:
  - `struct BlockIndex { blocks: Vec<(u32, f32)>, list_max: f32 }` (private)
  - field `block_indexes: HashMap<String, BlockIndex>` on `Bm25Index`, `#[serde(skip)]`
  - `pub fn build_blocks(&mut self)`
  - `pub fn has_blocks(&self) -> bool`

  Task 5 reads `block_indexes`; Task 6 calls `build_blocks`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-storage/src/bm25_index.rs`:

```rust
#[test]
fn build_blocks_populates_and_reports() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    assert!(!idx.has_blocks(), "empty index has no blocks");
    for doc_id in 0..300u32 {
        idx.insert(doc_id, "alpha beta");
    }
    assert!(!idx.has_blocks(), "blocks are not built implicitly");
    idx.build_blocks();
    assert!(idx.has_blocks());
}

#[test]
fn build_blocks_on_empty_index_is_a_noop() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    idx.build_blocks();
    assert!(!idx.has_blocks());
    assert!(idx.search("anything", 5).is_empty());
}

#[test]
fn block_max_bounds_every_member_posting() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    // Varied lengths and term frequencies so impacts differ within blocks.
    for doc_id in 0..1000u32 {
        let repeats = (doc_id % 7) as usize + 1;
        let filler = (doc_id % 13) as usize;
        let mut text = "alpha ".repeat(repeats);
        text.push_str(&"filler ".repeat(filler));
        text.push_str("beta gamma");
        idx.insert(doc_id, &text);
    }
    idx.build_blocks();

    let n = idx.doc_count() as f32;
    let avg_dl = idx.avg_doc_length();
    for (term, postings) in &idx.postings {
        let bi = idx
            .block_indexes
            .get(term)
            .unwrap_or_else(|| panic!("no block index for {term}"));
        let idf = bm25_idf(n, postings.len() as f32);
        for (i, &(doc_id, tf, _)) in postings.iter().enumerate() {
            let dl = idx.doc_lengths[doc_id as usize];
            let impact = bm25_impact(idf, tf, dl, avg_dl);
            let (_, block_max) = bi.blocks[i / crate::block_max::BLOCK_SIZE];
            assert!(
                block_max >= impact,
                "term {term} posting {i}: block_max {block_max} < impact {impact}"
            );
            assert!(
                bi.list_max >= block_max,
                "term {term}: list_max {} < block_max {block_max}",
                bi.list_max
            );
        }
    }
}

#[test]
fn block_last_doc_is_the_final_doc_in_each_block() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..500u32 {
        idx.insert(doc_id, "alpha");
    }
    idx.build_blocks();
    let postings = &idx.postings["alpha"];
    let bi = &idx.block_indexes["alpha"];
    for (b, chunk) in postings.chunks(crate::block_max::BLOCK_SIZE).enumerate() {
        assert_eq!(bi.blocks[b].0, chunk.last().unwrap().0);
    }
}

#[test]
fn blocks_survive_a_second_build() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..200u32 {
        idx.insert(doc_id, "alpha beta");
    }
    idx.build_blocks();
    let first = idx.block_indexes["alpha"].blocks.clone();
    idx.build_blocks();
    assert_eq!(idx.block_indexes["alpha"].blocks, first, "rebuild must be idempotent");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib bm25_index -- build_blocks block_max_bounds block_last_doc blocks_survive`
Expected: compile error — `has_blocks`, `build_blocks` and `block_indexes` do not exist.

- [ ] **Step 3: Add the block index type and the field**

In `crates/veda-storage/src/bm25_index.rs`, add above `pub struct Bm25Index`:

```rust
/// Per-term block metadata: `(last_doc_id_in_block, max_impact_in_block)`,
/// one entry per `BLOCK_SIZE` postings, in the same order as the postings
/// themselves, plus the maximum across the whole list.
///
/// Derived data — never serialized. See `build_blocks`.
#[derive(Debug, Default, Clone)]
struct BlockIndex {
    blocks: Vec<(u32, f32)>,
    list_max: f32,
}
```

Add the field to `Bm25Index`:

```rust
    /// Block-max metadata, keyed by term. Empty until `build_blocks` runs,
    /// which happens only for read-only (sealed) indexes — a growing index
    /// invalidates these on every insert, so it keeps the exhaustive path.
    #[serde(skip)]
    block_indexes: HashMap<String, BlockIndex>,
```

`Bm25Index` already derives `Default`, so `#[serde(skip)]` needs no further support.

- [ ] **Step 4: Implement `build_blocks` and `has_blocks`**

Add to `impl Bm25Index`, after `avg_doc_length`:

```rust
    /// Whether block-max metadata is available, i.e. whether `search` will
    /// take the pruning path.
    pub fn has_blocks(&self) -> bool {
        !self.block_indexes.is_empty()
    }

    /// Derive per-term block-max metadata from the current postings.
    ///
    /// Call once, after the index is complete and will not be inserted into
    /// again. Idempotent. Cheap relative to the JSON parse that produced
    /// the postings, which is why none of this is persisted.
    ///
    /// Bounds are computed with the same `bm25_idf` / `bm25_impact` helpers
    /// that `search` uses, so a bound can never drift from the score it
    /// bounds.
    pub fn build_blocks(&mut self) {
        self.block_indexes.clear();
        let n = self.doc_count() as f32;
        if n == 0.0 {
            return;
        }
        let avg_dl = self.avg_doc_length();
        for (term, postings) in &self.postings {
            if postings.is_empty() {
                continue;
            }
            let idf = bm25_idf(n, postings.len() as f32);
            let mut blocks = Vec::with_capacity(postings.len().div_ceil(BLOCK_SIZE));
            let mut list_max = f32::MIN;
            for chunk in postings.chunks(BLOCK_SIZE) {
                let mut block_max = f32::MIN;
                for &(doc_id, tf, _) in chunk {
                    let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0);
                    let impact = bm25_impact(idf, tf, dl, avg_dl);
                    if impact > block_max {
                        block_max = impact;
                    }
                }
                if block_max > list_max {
                    list_max = block_max;
                }
                // `chunk` is non-empty by construction of `chunks`.
                blocks.push((chunk[chunk.len() - 1].0, block_max));
            }
            self.block_indexes
                .insert(term.clone(), BlockIndex { blocks, list_max });
        }
    }
```

Add the import at the top of the file:

```rust
use crate::block_max::BLOCK_SIZE;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib bm25_index`
Expected: PASS, including all pre-existing bm25 tests.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/bm25_index.rs
git commit -m "feat(bm25): derive per-term block-max metadata"
```

---

### Task 5: Wire BM25 search onto the pruning core

**Files:**
- Modify: `crates/veda-storage/src/bm25_index.rs`

**Interfaces:**
- Consumes: `block_max::{PostingCursor, PruneStats, block_max_wand, BLOCK_SIZE}`, `BlockIndex` (Task 4), `bm25_idf` / `bm25_impact` (Task 2).
- Produces:
  - `pub fn search_with_theta(&self, query: &str, k: usize, theta: f32) -> Vec<(u32, f32)>`
  - `pub fn search_with_stats(&self, query: &str, k: usize, theta: f32) -> (Vec<(u32, f32)>, PruneStats)`

  `search(query, k)` keeps its signature and delegates to `search_with_theta(query, k, 1.0)`, so no existing call site changes. Task 8 calls `search_with_theta`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-storage/src/bm25_index.rs`:

```rust
/// Build one corpus twice: with blocks (pruning path) and without
/// (exhaustive path). Same data, same struct — the only difference is
/// which traversal runs.
fn paired_indexes(docs: u32) -> (Bm25Index, Bm25Index) {
    let build = || {
        let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
        for doc_id in 0..docs {
            let a = (doc_id % 5) as usize + 1;
            let b = (doc_id % 11) as usize;
            let c = (doc_id % 3) as usize;
            let mut text = "alpha ".repeat(a);
            text.push_str(&"beta ".repeat(b));
            text.push_str(&"gamma ".repeat(c));
            text.push_str("delta");
            idx.insert(doc_id, &text);
        }
        idx
    };
    let plain = build();
    let mut blocked = build();
    blocked.build_blocks();
    (plain, blocked)
}

#[test]
fn pruned_search_matches_exhaustive_search() {
    let (plain, blocked) = paired_indexes(3000);
    assert!(!plain.has_blocks());
    assert!(blocked.has_blocks());
    for query in [
        "alpha",
        "beta",
        "alpha beta",
        "alpha beta gamma delta",
        "alpha alpha beta",
        "delta zzz_absent",
        "zzz_absent",
    ] {
        for k in [1usize, 5, 10, 100, 5000] {
            assert_eq!(
                blocked.search_with_theta(query, k, 1.0),
                plain.search(query, k),
                "query {query:?} k {k}"
            );
        }
    }
}

#[test]
fn pruned_search_matches_on_a_corpus_smaller_than_one_block() {
    let (plain, blocked) = paired_indexes(7);
    for k in [1usize, 3, 100] {
        assert_eq!(
            blocked.search_with_theta("alpha beta", k, 1.0),
            plain.search("alpha beta", k)
        );
    }
}

#[test]
fn pruning_actually_skips_blocks() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    // "common" is in every doc and scores low; "rare" is in a handful and
    // scores high. Once the heap fills from "rare", most of "common"'s
    // blocks cannot reach the threshold.
    for doc_id in 0..20_000u32 {
        if doc_id % 500 == 0 {
            idx.insert(doc_id, "common rare rare rare rare rare");
        } else {
            idx.insert(doc_id, "common");
        }
    }
    idx.build_blocks();
    let (_, stats) = idx.search_with_stats("common rare", 10, 1.0);
    assert!(
        stats.blocks_skipped > 0,
        "no blocks skipped — pruning is inert on a corpus built to force it"
    );
}

#[test]
fn search_without_blocks_uses_the_exhaustive_path() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..1000u32 {
        idx.insert(doc_id, "alpha beta");
    }
    // No build_blocks call — this is the growing-segment shape.
    let (hits, stats) = idx.search_with_stats("alpha", 10, 1.0);
    assert_eq!(hits, idx.search("alpha", 10));
    assert_eq!(stats, crate::block_max::PruneStats::default());
}

#[test]
fn theta_above_one_stays_within_k_and_returns_real_hits() {
    let (plain, blocked) = paired_indexes(3000);
    // Every genuine match, scored exhaustively, as the reference set.
    let all_matches: HashMap<u32, f32> = plain
        .search("alpha beta gamma", usize::MAX)
        .into_iter()
        .collect();

    let approx = blocked.search_with_theta("alpha beta gamma", 10, 8.0);
    assert!(approx.len() <= 10);
    assert!(!approx.is_empty(), "aggressive pruning must not empty the result");
    for (doc_id, score) in &approx {
        // Approximate mode may omit true top-k members. It must never
        // invent a doc that is not a match, nor report a wrong score.
        let expected = all_matches
            .get(doc_id)
            .unwrap_or_else(|| panic!("doc {doc_id} is not a match at all"));
        assert_eq!(score, expected, "doc {doc_id} scored wrongly");
    }
}

#[test]
fn search_delegates_to_theta_one() {
    let (_, blocked) = paired_indexes(500);
    assert_eq!(
        blocked.search("alpha beta", 20),
        blocked.search_with_theta("alpha beta", 20, 1.0)
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib bm25_index -- pruned_search pruning_actually search_without_blocks theta_above_one search_delegates`
Expected: compile error — `search_with_theta` and `search_with_stats` do not exist.

- [ ] **Step 3: Implement the cursor**

Add to `crates/veda-storage/src/bm25_index.rs`, after the `BlockIndex` definition:

```rust
/// A `PostingCursor` over one term's BM25 postings.
///
/// Borrows the index's posting vector and block metadata; holds no copies.
struct Bm25Cursor<'a> {
    postings: &'a [(u32, u32, Vec<u32>)],
    blocks: &'a [(u32, f32)],
    list_max: f32,
    doc_lengths: &'a [u32],
    idf: f32,
    avg_dl: f32,
    pos: usize,
}

impl PostingCursor for Bm25Cursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.postings.get(self.pos).map(|p| p.0)
    }

    fn advance_to(&mut self, target: u32) {
        // Postings are doc-sorted, so the first index at or past `target`
        // is a partition point over the remaining slice.
        self.pos += self.postings[self.pos.min(self.postings.len())..]
            .partition_point(|p| p.0 < target);
    }

    fn next(&mut self) {
        self.pos += 1;
    }

    fn block_max(&self) -> f32 {
        self.blocks
            .get(self.pos / BLOCK_SIZE)
            .map_or(f32::MIN, |b| b.1)
    }

    fn block_last_doc(&self) -> Option<u32> {
        self.blocks.get(self.pos / BLOCK_SIZE).map(|b| b.0)
    }

    fn list_max(&self) -> f32 {
        self.list_max
    }

    fn score(&self) -> f32 {
        let (doc_id, tf, _) = self.postings[self.pos];
        let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0);
        bm25_impact(self.idf, tf, dl, self.avg_dl)
    }
}
```

Add the imports at the top of the file:

```rust
use crate::block_max::{BLOCK_SIZE, PostingCursor, PruneStats, block_max_wand};
```

(replacing the `use crate::block_max::BLOCK_SIZE;` line added in Task 4).

- [ ] **Step 4: Split `search` into the two paths**

Rename the existing exhaustive body to a private method and add the dispatchers. In `impl Bm25Index`:

```rust
    /// Rank docs by BM25 score against `query`. Returns `(doc_id, score)`
    /// sorted by score descending, ties broken by doc_id ascending,
    /// truncated to `k`. Docs with zero overlap on tokenized terms are
    /// omitted.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        self.search_with_theta(query, k, 1.0)
    }

    /// `search` with explicit pruning aggressiveness.
    ///
    /// `theta` of `1.0` (the default) is rank-safe: the result is identical
    /// to exhaustive scoring. Higher values prune harder and may omit true
    /// top-k results. Values below `1.0` and non-finite values are clamped
    /// to `1.0`.
    pub fn search_with_theta(&self, query: &str, k: usize, theta: f32) -> Vec<(u32, f32)> {
        self.search_with_stats(query, k, theta).0
    }

    /// `search_with_theta` plus traversal statistics. Exists so tests can
    /// assert that pruning is actually happening rather than that results
    /// merely remain correct — a pruning path that silently never prunes
    /// passes every correctness test.
    pub fn search_with_stats(
        &self,
        query: &str,
        k: usize,
        theta: f32,
    ) -> (Vec<(u32, f32)>, PruneStats) {
        let n = self.doc_count() as f32;
        if n == 0.0 || k == 0 {
            return (Vec::new(), PruneStats::default());
        }
        let terms = self.unique_query_terms(query);
        if self.block_indexes.is_empty() {
            return (self.search_exhaustive(&terms, k), PruneStats::default());
        }

        let avg_dl = self.avg_doc_length();
        let mut cursors = Vec::with_capacity(terms.len());
        for term in &terms {
            let (Some(postings), Some(bi)) = (self.postings.get(*term), self.block_indexes.get(*term))
            else {
                continue;
            };
            if postings.is_empty() {
                continue;
            }
            cursors.push(Bm25Cursor {
                postings,
                blocks: &bi.blocks,
                list_max: bi.list_max,
                doc_lengths: &self.doc_lengths,
                idf: bm25_idf(n, postings.len() as f32),
                avg_dl,
                pos: 0,
            });
        }
        block_max_wand(cursors, k, theta)
    }
```

Extract the term-dedup logic currently inline in `search` into:

```rust
    /// Analyzed query terms with duplicates removed — repeated tokens add
    /// no new evidence in standard BM25.
    fn unique_query_terms(&self, query: &str) -> Vec<&String> {
        let q_tokens = run_analyzer(self.tokenizer, &self.analyzer, query);
        let mut seen: std::collections::HashSet<&String> =
            std::collections::HashSet::with_capacity(q_tokens.len());
        let mut unique = Vec::with_capacity(q_tokens.len());
        // Resolve each analyzed token to the owned `String` key stored in
        // `postings`, so the returned references borrow from `self` rather
        // than from the temporary token vector.
        for t in &q_tokens {
            if let Some((key, _)) = self.postings.get_key_value(t)
                && seen.insert(key)
            {
                unique.push(key);
            }
        }
        unique
    }
```

and rename the old accumulate-and-sort body to:

```rust
    /// Exhaustive scoring: every posting of every query term accumulated
    /// into a map, then sorted. Used by growing segments, which mutate too
    /// often for block metadata to stay valid, and as the oracle that the
    /// pruning path is tested against.
    fn search_exhaustive(&self, terms: &[&String], k: usize) -> Vec<(u32, f32)> {
        let n = self.doc_count() as f32;
        let avg_dl = self.avg_doc_length();
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in terms {
            let Some(postings) = self.postings.get(*term) else {
                continue;
            };
            let idf = bm25_idf(n, postings.len() as f32);
            for &(doc_id, tf, _) in postings {
                let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0);
                *scores.entry(doc_id).or_insert(0.0) += bm25_impact(idf, tf, dl, avg_dl);
            }
        }
        let mut result: Vec<(u32, f32)> = scores.into_iter().collect();
        result.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        result.truncate(k);
        result
    }
```

Note: `unique_query_terms` returning keys borrowed from `self.postings` means a query token absent from the index contributes no term at all. That is already the behaviour — `search_exhaustive` skips missing terms — and it is required for the cursor lifetimes to work.

- [ ] **Step 5: Update `search_phrase`'s use of the shared scoring**

`search_phrase` currently calls `self.search(phrase, usize::MAX)` at line ~360. Leave that call alone in this task — Task 7 replaces it. Confirm it still compiles against the new `search`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib bm25_index`
Expected: PASS, all tests including every pre-existing bm25 test.

`pruned_search_matches_exhaustive_search` is the primary correctness gate. If it fails, fix the implementation — never the assertion.

- [ ] **Step 7: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/bm25_index.rs
git commit -m "feat(bm25): prune sealed-segment search with block-max WAND"
```

---

### Task 6: Build blocks at segment open

**Files:**
- Modify: `crates/veda-storage/src/bm25_index.rs` (the `load` function, around line 379)
- Modify: `crates/veda-storage/src/segment.rs` (the sealed-writer close path, around line 589)

**Interfaces:**
- Consumes: `Bm25Index::build_blocks`, `Bm25Index::has_blocks` (Task 4).
- Produces: no new API. Sealed segments prune; growing segments do not.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-storage/src/bm25_index.rs`:

```rust
#[test]
fn load_builds_blocks_and_preserves_results() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.bm25");

    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..2000u32 {
        let a = (doc_id % 5) as usize + 1;
        let mut text = "alpha ".repeat(a);
        text.push_str("beta gamma");
        idx.insert(doc_id, &text);
    }
    let before = idx.search("alpha beta", 25);
    idx.save(&path).unwrap();

    let loaded = Bm25Index::load(&path).unwrap();
    assert!(loaded.has_blocks(), "load must leave the index ready to prune");
    assert_eq!(loaded.search("alpha beta", 25), before);
}

#[test]
fn saved_file_contains_no_block_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.bm25");
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..200u32 {
        idx.insert(doc_id, "alpha beta");
    }
    idx.build_blocks();
    idx.save(&path).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains("block_indexes") && !raw.contains("list_max"),
        "block metadata is derived and must never be persisted"
    );
}

#[test]
fn an_index_saved_before_this_feature_still_loads() {
    // A file written by the previous version has no block fields at all.
    // `#[serde(skip)]` means it must load unchanged and then build blocks.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.bm25");
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..300u32 {
        idx.insert(doc_id, "alpha beta");
    }
    idx.save(&path).unwrap();

    let loaded = Bm25Index::load(&path).unwrap();
    assert_eq!(loaded.doc_count(), 300);
    assert!(loaded.has_blocks());
}
```

`tempfile = "3"` is already a dev-dependency of `veda-storage` (`Cargo.toml:45`), and the existing `save_and_load_roundtrip` test at `bm25_index.rs:830` already writes to disk. Do not add any dependency.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib bm25_index -- load_builds_blocks saved_file_contains an_index_saved_before`
Expected: `load_builds_blocks_and_preserves_results` and `an_index_saved_before_this_feature_still_loads` FAIL on `has_blocks()` being false.

- [ ] **Step 3: Build blocks in `load`**

In `crates/veda-storage/src/bm25_index.rs`, change `load`:

```rust
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let mut idx: Self = serde_json::from_slice(&bytes)
            .map_err(|e| VedaError::Storage(format!("bm25 load parse: {e}")))?;
        // A loaded index is read-only by construction, so block metadata
        // stays valid for its lifetime. Deriving it here — rather than
        // persisting it — keeps the on-disk format unchanged, so indexes
        // written before this feature load with no migration.
        idx.build_blocks();
        Ok(idx)
    }
```

- [ ] **Step 4: Build blocks when a segment is sealed**

`SegmentWriter` builds its own `Bm25Index` by repeated `insert` and then saves at `segment.rs:589`. The in-memory copy used before the next open should prune too. In `crates/veda-storage/src/segment.rs`, at the loop that saves the indexes (around line 589), call `build_blocks` on each index before saving.

Read the surrounding code first — if the writer's indexes are consumed by value and never queried again after save, no change is needed here and the `load` path alone is sufficient. **State which of the two you found in your report.** Do not add a call that nothing uses.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib`
Expected: PASS.

- [ ] **Step 6: Run the wider storage suite**

Run: `cargo test -p veda-storage`
Expected: PASS. This covers segment seal/open round-trips that exercise the new `load` behaviour.

- [ ] **Step 7: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/bm25_index.rs crates/veda-storage/src/segment.rs
git commit -m "feat(bm25): derive block metadata when a sealed index is opened"
```

---

### Task 7: Stop `search_phrase` scoring the whole corpus

**Files:**
- Modify: `crates/veda-storage/src/bm25_index.rs` (`search_phrase`, around line 360)

**Interfaces:**
- Consumes: `bm25_idf`, `bm25_impact` (Task 2), `unique_query_terms` (Task 5).
- Produces: `fn score_docs(&self, query: &str, doc_ids: &[u32]) -> Vec<f32>` (private).

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-storage/src/bm25_index.rs`:

```rust
#[test]
fn score_docs_matches_full_search_scores() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..500u32 {
        let a = (doc_id % 4) as usize + 1;
        let mut text = "alpha ".repeat(a);
        text.push_str("beta gamma");
        idx.insert(doc_id, &text);
    }
    let full: HashMap<u32, f32> = idx.search("alpha beta", usize::MAX).into_iter().collect();
    let picked: Vec<u32> = vec![3, 17, 200, 499];
    let scores = idx.score_docs("alpha beta", &picked);
    assert_eq!(scores.len(), picked.len());
    for (doc_id, got) in picked.iter().zip(scores) {
        assert_eq!(got, full[doc_id], "doc {doc_id}");
    }
}

#[test]
fn score_docs_returns_zero_for_a_non_matching_doc() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    idx.insert(0, "alpha");
    idx.insert(1, "zzz");
    let scores = idx.score_docs("alpha", &[1]);
    assert_eq!(scores, vec![0.0]);
}

#[test]
fn phrase_scores_are_unchanged_by_the_rewrite() {
    let mut idx = Bm25Index::with_tokenizer(TokenizerKind::Whitespace);
    for doc_id in 0..200u32 {
        idx.insert(doc_id, "the quick brown fox jumps");
    }
    let hits = idx.search_phrase("quick brown", 0, 10);
    assert_eq!(hits.len(), 10);
    let bag: HashMap<u32, f32> = idx.search("quick brown", usize::MAX).into_iter().collect();
    for (doc_id, score) in &hits {
        assert_eq!(*score, bag[doc_id], "phrase score must stay the bag-of-words score");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-storage --lib bm25_index -- score_docs phrase_scores_are_unchanged`
Expected: compile error — `score_docs` does not exist.

- [ ] **Step 3: Implement `score_docs`**

Add to `impl Bm25Index`:

```rust
    /// BM25 bag-of-words score for each doc in `doc_ids`, in the same
    /// order. Docs that match none of the analyzed query terms score `0.0`.
    ///
    /// Exists so `search_phrase` can score only the docs that already
    /// passed positional matching, instead of scoring the entire corpus
    /// and discarding almost all of it.
    fn score_docs(&self, query: &str, doc_ids: &[u32]) -> Vec<f32> {
        let n = self.doc_count() as f32;
        if n == 0.0 {
            return vec![0.0; doc_ids.len()];
        }
        let avg_dl = self.avg_doc_length();
        let wanted: std::collections::HashSet<u32> = doc_ids.iter().copied().collect();
        let mut acc: HashMap<u32, f32> = HashMap::with_capacity(doc_ids.len());
        for term in self.unique_query_terms(query) {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let idf = bm25_idf(n, postings.len() as f32);
            for &(doc_id, tf, _) in postings {
                if !wanted.contains(&doc_id) {
                    continue;
                }
                let dl = *self.doc_lengths.get(doc_id as usize).unwrap_or(&0);
                *acc.entry(doc_id).or_insert(0.0) += bm25_impact(idf, tf, dl, avg_dl);
            }
        }
        doc_ids.iter().map(|d| *acc.get(d).unwrap_or(&0.0)).collect()
    }
```

- [ ] **Step 4: Use it in `search_phrase`**

Replace the block at `bm25_index.rs:360-366` — currently

```rust
        let bm25_hits = self.search(phrase, usize::MAX);
        let score_map: HashMap<u32, f32> = bm25_hits.into_iter().collect();

        let mut result: Vec<(u32, f32)> = matched
            .into_iter()
            .map(|id| (id, *score_map.get(&id).unwrap_or(&0.0)))
            .collect();
```

with

```rust
        // Score only the docs that passed positional matching. Scoring the
        // whole corpus here was unbounded work for a result set that is
        // usually tiny.
        let scores = self.score_docs(phrase, &matched);
        let mut result: Vec<(u32, f32)> = matched.into_iter().zip(scores).collect();
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p veda-storage --lib bm25_index`
Expected: PASS, including the pre-existing phrase tests `phrase_query_requires_adjacency`, `proximity_query_allows_slop`, `positions_survive_save_load` and `stopword_gap_blocks_false_phrase`.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/bm25_index.rs
git commit -m "perf(bm25): score only phrase-matched docs instead of the whole corpus"
```

---

### Task 8: Expose `prune_theta` on the search API

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs` (`Bm25Query` at line 138, `execute_bm25` at line 1034, dispatch at line 660, hybrid at line 1249)
- Test: `crates/veda-gateway/tests/bm25_prune_theta_test.rs` (create)

**Interfaces:**
- Consumes: `Bm25Index::search_with_theta` (Task 5).
- Produces: `prune_theta: Option<f32>` on `Bm25Query`; `fn validate_bm25_query(q: &Bm25Query) -> Result<(), (StatusCode, Json<ErrorResponse>)>`.

- [ ] **Step 1: Write the failing tests**

Create `crates/veda-gateway/tests/bm25_prune_theta_test.rs`.

**Convention:** this crate's integration tests are deliberately **self-contained per file** — there is no `tests/common/` module and you must not create one. The harness below follows `crates/veda-gateway/tests/bm25_filter_test.rs` exactly. `tempfile = "3"` is already a dev-dependency.

```rust
//! `prune_theta` on the BM25 leg of `/v1/collections/{name}/search`:
//! rejected below 1.0, accepted at or above it, and rank-safe by default.
//!
//! The collection is force-sealed before querying, because block metadata —
//! and therefore pruning — exists only on sealed segments. Pruning
//! *effectiveness* is measured in `veda-storage`; this file covers the
//! request plumbing and validation at both entry points into
//! `execute_bm25` (plain BM25 and the hybrid BM25 leg).
//!
//! Harness copied from `bm25_filter_test.rs` — gateway integration tests
//! are kept self-contained per file by convention.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

fn client() -> Client {
    Client::new()
}

/// Create a bm25 collection, insert `docs` records with varying term
/// frequencies, then force-seal so the sealed (pruning) path is exercised.
async fn create_seed_and_seal(base: &str, name: &str, docs: usize) {
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [{ "name": "message", "type": "string", "bm25": true }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let records: Vec<Value> = (0..docs)
        .map(|i| {
            let alpha = "alpha ".repeat(i % 5 + 1);
            let beta = "beta ".repeat(i % 11);
            json!({ "message": format!("{alpha}{beta}gamma") })
        })
        .collect();
    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "insert: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/{name}/seal"))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success(), "seal: {}", r.text().await.unwrap());
}

/// Issue a BM25 search, optionally with `prune_theta`. Returns the raw
/// response so the caller can assert on status as well as body.
async fn search(base: &str, name: &str, theta: Option<f32>) -> reqwest::Response {
    let mut bm25 = json!({ "field": "message", "query": "alpha beta" });
    if let Some(t) = theta {
        bm25["prune_theta"] = json!(t);
    }
    client()
        .post(format!("{base}/v1/collections/{name}/search"))
        .json(&json!({ "query": { "bm25": bm25 }, "limit": 10 }))
        .send()
        .await
        .unwrap()
}

/// `(id, distance)` pairs, which is what the caller actually compares.
fn hits(body: &Value) -> Vec<(String, f64)> {
    body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            (
                h["id"].as_str().unwrap().to_owned(),
                h["distance"].as_f64().unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn omitted_prune_theta_matches_explicit_one() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_default", 400).await;

    let a = search(&base, "prune_default", None).await;
    assert_eq!(a.status(), 200);
    let a: Value = a.json().await.unwrap();

    let b = search(&base, "prune_default", Some(1.0)).await;
    assert_eq!(b.status(), 200);
    let b: Value = b.json().await.unwrap();

    assert!(!hits(&a).is_empty(), "fixture produced no hits");
    assert_eq!(hits(&a), hits(&b), "theta 1.0 must equal the default");
}

#[tokio::test]
async fn prune_theta_below_one_is_rejected() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_low", 40).await;

    let r = search(&base, "prune_low", Some(0.5)).await;
    assert_eq!(r.status(), 400, "body: {}", r.text().await.unwrap());
}

#[tokio::test]
async fn prune_theta_above_one_is_accepted() {
    let (base, _tmp) = start_server().await;
    create_seed_and_seal(&base, "prune_high", 400).await;

    let r = search(&base, "prune_high", Some(4.0)).await;
    assert_eq!(r.status(), 200, "body: {}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    assert!(hits(&body).len() <= 10);
}

#[tokio::test]
async fn hybrid_validates_its_bm25_leg_too() {
    // execute_bm25 is reachable from two dispatch sites. Validation that
    // covers only the plain one leaves the hybrid path unguarded.
    let (base, _tmp) = start_server().await;

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "prune_hybrid",
            "fields": [
                { "name": "message", "type": "string", "bm25": true },
                { "name": "vec", "type": "vector", "dimension": 4 }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/prune_hybrid/search"))
        .json(&json!({
            "query": {
                "hybrid": {
                    "dense": { "field": "vec", "vector": [0.1, 0.2, 0.3, 0.4] },
                    "bm25": {
                        "field": "message",
                        "query": "alpha",
                        "prune_theta": 0.5
                    }
                }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "body: {}", r.text().await.unwrap());
}
```

Before writing the file, confirm the exact request shape for a hybrid dense leg and the exact response field names (`results`, `id`, `distance`) against `crates/veda-gateway/tests/bm25_filter_test.rs`. Request-shape guesses have cost real time on this project before — read, do not assume.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p veda-gateway --test bm25_prune_theta_test`
Expected: compile error — `prune_theta` is not a field of `Bm25Query`.

- [ ] **Step 3: Add the field**

In `crates/veda-gateway/src/rest/search.rs`, extend `Bm25Query`:

```rust
#[derive(Debug, Deserialize)]
pub struct Bm25Query {
    /// String field with `bm25 = true` in the schema.
    pub field: String,
    /// Raw user-facing query text.
    pub query: String,
    /// Pruning aggressiveness for the block-max traversal. Absent or `1.0`
    /// is rank-safe: results are identical to exhaustive scoring. Values
    /// above `1.0` prune harder and may omit true top-k results. Values
    /// below `1.0` are rejected — they could only make the query slower
    /// without changing the answer.
    #[serde(default)]
    pub prune_theta: Option<f32>,
}
```

- [ ] **Step 4: Add validation**

Add next to `validate_dense_query` (line 1531):

```rust
fn validate_bm25_query(q: &Bm25Query) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(theta) = q.prune_theta
        && (!theta.is_finite() || theta < 1.0)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "prune_theta must be a finite value >= 1.0 (got {theta}); \
                     1.0 is exact, higher values prune more aggressively"
                ),
            }),
        ));
    }
    Ok(())
}
```

- [ ] **Step 5: Call validation at both entry points**

`execute_bm25` is reached from two places. Add a call to each, matching how `validate_dense_query` is called at lines 569 and 691.

In the plain BM25 branch (around line 660), before `execute_bm25`:

```rust
    } else if let Some(ref bm25_query) = req.query.bm25 {
        validate_bm25_query(bm25_query)?;
```

In the hybrid branch, alongside the existing `validate_dense_query(schema, &hybrid_query.dense, req.truncate_to)?;` at line 691:

```rust
        validate_bm25_query(&hybrid_query.bm25)?;
```

- [ ] **Step 6: Thread the value into the index calls**

In `execute_bm25` (line 1034), replace both `idx.search(&query.query, fetch_k)` calls — the growing-segment one at line 1065 and the sealed one at line 1089 — with:

```rust
            let scored = idx.search_with_theta(&query.query, fetch_k, theta);
```

and add, just after `let fetch_k = bm25_fetch_k(k, filter_expr.is_some());`:

```rust
    // Absent means exact. Storage clamps defensively too, but the request
    // boundary is where an out-of-range value becomes a 400.
    let theta = query.prune_theta.unwrap_or(1.0);
```

The growing-segment index has no block metadata, so `search_with_theta` takes the exhaustive path there regardless of `theta` — that is intended, not a bug.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test -p veda-gateway --test bm25_prune_theta_test`
Expected: PASS.

- [ ] **Step 8: Run the gateway suites that touch BM25**

Run: `cargo test -p veda-gateway --test integration_test`
Expected: PASS.

Run: `cargo test -p veda-gateway --lib`
Expected: PASS.

- [ ] **Step 9: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-gateway --all-targets -- -D warnings
git add crates/veda-gateway/src/rest/search.rs crates/veda-gateway/tests/bm25_prune_theta_test.rs
git commit -m "feat(search): expose prune_theta on the BM25 query"
```

---

### Task 9: Measure the result and document it

The feature is not done until the win is a number. If there is no win, that is the finding — report it.

**Files:**
- Modify: `crates/veda-storage/tests/bm25_bench.rs`
- Modify: `CHANGELOG.md`
- Modify: `documentation/2026-08-13-block-max-wand-design.md` (append a "Measured result" section)

**Interfaces:**
- Consumes: `Bm25Index::build_blocks`, `Bm25Index::search_with_stats` (Tasks 4, 5).
- Produces: recorded measurements.

- [ ] **Step 1: Extend the benchmark to compare both paths**

Add to `crates/veda-storage/tests/bm25_bench.rs`:

```rust
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
        let (_, stats) = blocked.search_with_stats(q, k, 1.0);
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
```

- [ ] **Step 2: Run it and record the numbers**

Run: `cargo test -p veda-storage --test bm25_bench -- --ignored --nocapture`
Expected: three `compare:` lines plus the three original `baseline:` lines.

Record every field of all three `compare:` lines in your report file, and compare the `exhaustive` figures against the baselines captured in Task 1 — they should be close, since nothing about the exhaustive path changed. **A large discrepancy means Task 2 or Task 5 slowed the exhaustive path and must be investigated, not ignored.**

- [ ] **Step 3: Append the measured result to the design doc**

Add a `## Measured result` section at the end of `documentation/2026-08-13-block-max-wand-design.md` with the three `compare:` lines, the `build_blocks` cost, and one sentence stating whether the feature delivers at Veda's realistic corpus sizes. Report the numbers you actually got. If the speedup is below 1.2x at every size, say exactly that.

- [ ] **Step 4: Update the changelog**

Add an entry to `CHANGELOG.md` under the unreleased section, matching the surrounding style:

```markdown
- BM25 search on sealed segments now uses Block-Max WAND, skipping posting
  blocks that cannot enter the top-k. Results are unchanged by default;
  `prune_theta` on a `bm25` query opts into more aggressive pruning.
- `match_phrase` no longer scores the whole collection to rank its matches.
- BM25 results with identical scores are now ordered by document id, so
  repeated identical queries return identical orderings.
```

- [ ] **Step 5: Full verification**

Run: `cargo test -p veda-storage`
Expected: PASS.

Run: `cargo test -p veda-gateway`
Expected: PASS.

Run: `cargo fmt --all --check`
Expected: no output.

Run: `cargo clippy -p veda-storage -p veda-gateway --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/veda-storage/tests/bm25_bench.rs CHANGELOG.md documentation/2026-08-13-block-max-wand-design.md
git commit -m "test: measure block-max WAND against exhaustive BM25"
```

---

## Plan self-review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Pruning core in `block_max.rs`, `PostingCursor` trait, `BLOCK_SIZE = 128` | 3 |
| `PruneStats` for proving pruning is not inert | 3, 5 |
| BM25 adapter with impact from the shared helper | 2, 5 |
| Block metadata derived, never persisted | 4, 6 |
| Growing segments keep the exhaustive path | 5, 6 |
| `prune_theta` per request, default exact, `400` below 1.0 | 8 |
| Storage-level defensive clamp of theta | 3 |
| `score_docs` replacing the `usize::MAX` corpus scan | 7 |
| Deterministic ordering via doc_id tie-break | 2 |
| Baseline measured before the change | 1 |
| Differential test vs exhaustive | 3, 5 |
| Upper-bound property test | 4 |
| Blocks-skipped > 0 test | 3, 5 |
| `search_phrase` semantics preserved | 7 |
| Measured improvement reported | 9 |

**Type consistency:** `block_max_wand` returns `(Vec<(u32, f32)>, PruneStats)` in Tasks 3, 5 and 9. `build_blocks(&mut self)` and `has_blocks(&self) -> bool` are consistent across Tasks 4, 5 and 6. `search_with_theta(&str, usize, f32) -> Vec<(u32, f32)>` is consistent across Tasks 5 and 8. `score_docs(&str, &[u32]) -> Vec<f32>` appears only in Task 7. `bm25_idf` / `bm25_impact` signatures are fixed in Task 2 and used unchanged in Tasks 4, 5 and 7.

**Known open item for the implementer:** Task 6 Step 4 asks the implementer to read `segment.rs` around line 589 and decide whether the sealed writer's in-memory index is queried after save. This is a deliberate read-then-decide, not a placeholder — the plan states both outcomes and requires the finding to be reported.
