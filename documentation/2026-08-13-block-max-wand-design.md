# Block-Max WAND for BM25 — Design

**Date:** 2026-08-13
**Status:** Approved for implementation
**Scope:** Cycle A of a two-cycle effort. Cycle B (sparse/SPLADE inverted index) gets its own spec.

## Context

The 2026-08-12 competitive refresh (`research/2026-08-12-competitive-refresh/`) listed
"sparse execution is brute-force" as the most concrete, smallest-scope engineering gap it
found. Reading the code shows that single sentence covers two unrelated problems:

| Path | Location | State |
|---|---|---|
| BM25 full-text | `crates/veda-storage/src/bm25_index.rs:218` | Real inverted index, but scored term-at-a-time with **full accumulation** — every posting of every query term is scored into a `HashMap<u32, f32>`, then the whole accumulator is sorted. No pruning of any kind. |
| Sparse vectors / SPLADE | `crates/veda-query/src/search.rs:198` | **No inverted index at all.** A flat `Vec<SparseVec>` per segment (`sparse_storage.rs`) is scanned linearly, scoring every stored vector. Its own doc comment says "brute-force in v1". |

Block-Max WAND applies directly to the first. The second needs an inverted index built
before any pruning algorithm is meaningful — a storage-shaped problem, not an algorithmic
one. Splitting them keeps a storage-format change reviewable separately from an algorithm
change, and lets the pruning core be proven against a real consumer before a second
consumer depends on it.

**This spec covers Cycle A only: Block-Max WAND for BM25, plus the shared pruning core
that Cycle B will reuse.**

## Goals

1. BM25 search stops scoring postings that provably cannot enter the top-k.
2. Default behaviour is **rank-safe**: the returned results are the same as today's
   brute-force result.
3. The pruning core is reusable by Cycle B without modification to the core itself.
4. The win is **measured**, not asserted — a baseline is captured before the change.

## Non-goals

- The sparse/SPLADE inverted index (Cycle B).
- Persisting block metadata to disk (see "Why nothing is persisted").
- Changing BM25 scoring, tokenization, analysis, or phrase-match semantics.
- Changing fusion, filtering, MVCC, or delete handling.

## Current behaviour, precisely

`Bm25Index` (`crates/veda-storage/src/bm25_index.rs:126`):

```rust
postings: HashMap<String, Vec<(u32, u32, Vec<u32>)>>,  // term -> [(doc_id, tf, positions)]
doc_lengths: Vec<u32>,
total_tokens: u64,
tokenizer: TokenizerKind,
analyzer: AnalyzerConfig,
```

Facts established by reading the code, each of which the design depends on:

- **Postings are sorted by `doc_id` ascending.** `insert` appends in increasing doc_id
  order and the segment writer inserts in order (documented at `bm25_index.rs:174`).
  Document-at-a-time traversal is therefore possible with no re-sorting.
- **The whole index is resident in memory.** `segment.rs:796` calls `Bm25Index::load` at
  segment open, which `serde_json`-deserializes the entire structure. There is no mmap and
  no lazy posting access.
- **`search` is disjunctive (OR).** A doc scores on whatever subset of query terms it
  contains. BMW is a disjunctive top-k algorithm, so semantics match exactly.
- **Query terms are deduplicated** before scoring (`bm25_index.rs:228`).
- **Effective k is small.** `bm25_fetch_k` (`rest/search.rs:1015`) requests `k` with no
  filter and `10 * k` with one. Typical range 10–100, worst case ~1000.
- **Deletes, MVCC visibility and scalar filters are applied *after* BM25 returns**
  (`rest/search.rs:1065-1090`), with the 10× oversample compensating for post-filter
  losses. BM25 itself knows nothing about them.
- **Scoring uses fixed per-index constants.** `n = doc_count()` and `avg_dl` are fixed for
  a sealed index, so a term's `idf` and every posting's impact are computable at build
  time.

### Two defects found while reading

1. **`search_phrase` scores the entire corpus to rank a handful of docs.**
   `bm25_index.rs:360` calls `self.search(phrase, usize::MAX)`, builds a `HashMap` of every
   scored doc in the collection, then looks up only the docs that already passed positional
   phrase matching. This is unbounded work proportional to corpus size for a result set that
   is usually tiny.

2. **Result order is non-deterministic under equal scores.** Both `search` and
   `search_phrase` sort by score descending with `partial_cmp` and no tie-breaker, so two
   docs with identical BM25 scores may be returned in either order depending on `HashMap`
   iteration order. This is pre-existing, and it makes any differential test against a
   second implementation unreliable.

Both are addressed by this work — (1) because BMW makes the fix natural, (2) because a
rank-safe claim is untestable without deterministic ordering.

## Design

### Component 1 — the pruning core

New file: `crates/veda-storage/src/block_max.rs`. Contains the algorithm and nothing
domain-specific. It does not know what BM25 is.

```rust
/// Block size for block-max metadata. 128 is the standard choice in the
/// literature: large enough that per-block overhead is negligible against
/// the postings it covers, small enough that a block's max impact stays a
/// tight bound on its members.
pub const BLOCK_SIZE: usize = 128;

/// A cursor over one term's postings, positioned at a document.
///
/// Implementors guarantee postings are visited in ascending doc_id order.
pub trait PostingCursor {
    /// Current doc_id, or `None` when the list is exhausted.
    fn doc(&self) -> Option<u32>;

    /// Advance to the first doc_id >= `target`. Exhausts the cursor if no
    /// such doc exists.
    fn advance_to(&mut self, target: u32);

    /// Advance past the current doc.
    fn next(&mut self);

    /// Upper bound on the score any doc in the current block can receive
    /// from this term. Meaningless once exhausted.
    fn block_max(&self) -> f32;

    /// Last doc_id in the current block. Used to skip the whole block.
    fn block_last_doc(&self) -> Option<u32>;

    /// Upper bound on the score any doc in the *entire* list can receive
    /// from this term. Constant for the cursor's lifetime.
    fn list_max(&self) -> f32;

    /// Exact score contribution of the current doc for this term.
    fn score(&self) -> f32;
}

/// Disjunctive top-k over `cursors`, returning `(doc_id, score)` sorted by
/// score descending, ties broken by doc_id ascending, truncated to `k`.
///
/// `theta` scales the pruning threshold. `1.0` is rank-safe: only blocks
/// that provably cannot enter the top-k are skipped. Values above 1.0 prune
/// more aggressively and may drop results that belong in the true top-k.
pub fn block_max_wand<C: PostingCursor>(
    cursors: Vec<C>,
    k: usize,
    theta: f32,
) -> Vec<(u32, f32)>;
```

**Algorithm.** Standard Block-Max WAND (Ding & Suel, SIGIR 2011):

1. Maintain a min-heap of the best `k` `(score, doc_id)` seen. `threshold` is the heap's
   minimum score once full, `0.0` before that. Pruning decisions compare against
   `theta * threshold`.
2. Sort live cursors by current `doc()` ascending.
3. **Pivot selection.** Accumulate `list_max()` across the sorted cursors until the running
   sum exceeds `theta * threshold`. That cursor is the pivot; `pivot_doc` is its current
   doc. If the sum over all cursors never exceeds the threshold, no remaining document can
   enter the top-k — terminate.
4. **If `cursors[0].doc() != pivot_doc`**, some cursor is behind the pivot: `advance_to`
   it to `pivot_doc` and return to step 2.
5. **Block-max check.** Sum `block_max()` over cursors `0..=pivot`. If that sum
   `<= theta * threshold`, no document in the current block alignment can enter the top-k:
   skip. Advance past `min(block_last_doc())` across those cursors — one `advance_to` to
   `that_doc + 1` — and return to step 2. **This is where the work is saved:** an entire
   block of up to `BLOCK_SIZE` postings per cursor is bypassed without any of them being
   scored.
6. Otherwise fully evaluate `pivot_doc`: sum `score()` over every cursor currently at it,
   push into the heap, advance those cursors, return to step 2.

Correctness of the rank-safe case rests on `block_max()` being a true upper bound on any
member posting's contribution, and `list_max()` on any posting's contribution at all. Both
are computed as maxima over exactly the expression used to score, so both hold by
construction.

### Component 2 — the BM25 adapter

In `bm25_index.rs`, a `Bm25Cursor` implements `PostingCursor` over the existing
`Vec<(u32, u32, Vec<u32>)>` posting vector plus a block index for that term. No change to
how postings are stored.

**Impact.** For a posting `(doc_id, tf, _)`, with `idf` fixed per term and `avg_dl` fixed
per index, the impact must be **the identical expression** already at
`bm25_index.rs:244-249`, floor included — a bound computed from a paraphrase of the
scoring code is not a bound:

```
idf   = ln((n - df + 0.5) / (df + 0.5) + 1.0)
denom = tf + K1 * (1 - B + B * dl / max(avg_dl, 1.0))
impact = idf * tf * (K1 + 1) / max(denom, 1e-6)
```

Implementation note: both paths must call one shared scoring helper rather than each
writing out the formula, so the two can never drift.

`idf` is folded into the stored maxima, so `block_max` and `list_max` are directly
comparable to accumulated scores with no per-comparison multiply.

**Block index.**

```rust
/// Per-term block metadata: (last_doc_id_in_block, max_impact_in_block),
/// one entry per BLOCK_SIZE postings, in the same order as the postings.
struct BlockIndex {
    blocks: Vec<(u32, f32)>,
    list_max: f32,
}
```

Held in a `HashMap<String, BlockIndex>` on `Bm25Index`, marked `#[serde(skip)]`.

### Component 3 — when blocks exist, and when they do not

`Bm25Index` serves two very different roles, and the design splits on it:

- **Growing segments** mutate on every `insert`. Any block maximum would be invalidated
  immediately, and growing segments are small and in-memory by construction. They keep the
  existing brute-force path unchanged.
- **Sealed segments** are read-only. Blocks are built once, in a new
  `Bm25Index::build_blocks(&mut self)`, called from `Bm25Index::load` after
  deserialization and from the sealed writer after its final `insert`.

`search` therefore dispatches on whether the block index is populated: present → BMW,
absent → the existing accumulator. **This is deliberate and load-bearing for testing** —
both implementations live on the same struct over the same data, so a differential test
can call one against the other directly, with no fixture divergence.

### Why nothing is persisted

Block maxima are pure derived data: computable from `postings`, `doc_lengths` and
`doc_count`, all of which are already loaded. Building them is one pass over the postings.
`Bm25Index::load` already `serde_json`-parses the entire index into memory
(`bm25_index.rs:379`) — a JSON parse of every posting and every position list. The
derivation pass is small against that, and it costs nothing at query time.

Persisting would buy a marginal open-time saving and cost a format version, a migration
path for every existing sealed segment, and a permanent back-compat surface. Not worth it.

**Revisit if** BM25 postings ever move to mmap or a lazily-read on-disk format, at which
point loading postings solely to compute maxima would become the dominant cost.

### Component 4 — the approximate knob

`Bm25Query` (`rest/search.rs:138`) gains:

```rust
/// Pruning aggressiveness. Absent or 1.0 = rank-safe (results identical to
/// exhaustive scoring). Values above 1.0 raise the pruning threshold,
/// skipping more blocks at the cost of possibly missing true top-k results.
/// Values below 1.0 are rejected — they would only slow the query down.
#[serde(default)]
pub prune_theta: Option<f32>,
```

Per-request, matching the `slop` precedent on `MatchPhraseQuery` and the
`num_candidates -> ef_search` precedent at `rest/search.rs:967`. Per-request rather than
per-collection so the auto-tuner in `rest/tuning.rs` can sweep it.

Validation: `Some(t)` with `t < 1.0` or non-finite is a `400`. Absent behaves as `1.0`.

The value threads through to `Bm25Index::search_with_theta`; `search(query, k)` remains as
`search_with_theta(query, k, 1.0)` so no existing call site changes.

### Component 5 — the `search_phrase` fix

Replace the `self.search(phrase, usize::MAX)` call at `bm25_index.rs:360` with:

```rust
/// BM25 score for each doc in `doc_ids`, computed over the analyzed terms
/// of `query`. Returns scores in the same order as `doc_ids`.
fn score_docs(&self, query: &str, doc_ids: &[u32]) -> Vec<f32>;
```

`search_phrase` already knows exactly which docs matched positionally, so it scores those
and nothing else. Phrase results are unchanged; the corpus-sized intermediate disappears.

### Component 6 — deterministic ordering

Both `search` and `search_phrase` gain `doc_id` ascending as a tie-breaker after score
descending. This is a **deliberate behaviour change**, narrow in effect: it only orders
results that previously came back in arbitrary relative order. It is required for the
differential test to be meaningful, and it removes a source of nondeterministic API
responses.

## What does not change

Because the default is rank-safe, the top-`fetch_k` returned to `rest/search.rs` is the
same set with the same scores as today. Therefore:

- delete filtering, MVCC `read_ts` visibility and `evaluate_record` scalar filtering
  (`rest/search.rs:1065-1105`) are untouched;
- the 10× oversample in `bm25_fetch_k` keeps working with the same effectiveness;
- fusion (`veda-query/src/fusion.rs`) receives identical inputs, so RRF ranks and
  `weighted_sum` normalization are unaffected;
- recall and any benchmark measured against exact BM25 remain valid.

## Error handling

- `k == 0`, empty index, and query terms absent from the index all return an empty result,
  matching current behaviour.
- A term whose posting list is empty contributes no cursor.
- `build_blocks` on an empty index produces an empty block index; `search` then takes the
  brute-force path and returns empty.
- `prune_theta` validation is a request-level `400` at the REST boundary, not a panic in
  storage. `Bm25Index::search_with_theta` clamps a non-finite or sub-1.0 theta to `1.0`
  defensively rather than producing wrong results.

## Testing and verification

**Baseline first.** Before any behaviour change, capture current BM25 latency and postings
scored per query at realistic corpus sizes. Without this the feature cannot be shown to
work; "should be faster" is not a result.

1. **Differential correctness (the primary gate).** Over randomized corpora and randomized
   multi-term queries, assert `search_with_theta(q, k, 1.0)` on a block-built index returns
   exactly the same `(doc_id, score)` sequence as the brute-force path on the same data.
   Covers: single-term queries, many-term queries, terms absent from the index, `k`
   larger than the corpus, `k = 1`, duplicate query terms, and corpora smaller than one
   block.
2. **Pruning actually happens.** Instrument a counter of postings scored and blocks
   skipped; assert blocks skipped is greater than zero on a corpus large enough to contain
   multiple blocks. Without this the feature can be inert while every correctness test
   passes — a failure mode this project has hit four separate times.
3. **Upper bounds hold.** Property test: for every block, `max_impact` is `>=` the impact
   of every posting in that block; `list_max` is `>=` every `block_max`.
4. **`prune_theta`.** `theta = 1.0` equals brute force. `theta > 1.0` never returns more
   than `k`, never returns a doc that scores zero, and skips at least as many blocks as
   `theta = 1.0`.
5. **`search_phrase` unchanged.** All existing phrase and proximity tests
   (`bm25_index.rs:930-984`) pass against the `score_docs` implementation, plus an
   assertion that phrase scores equal the bag-of-words scores they equalled before.
6. **Growing-segment path unchanged.** A growing segment with no block index returns
   identical results to before.
7. **Measured improvement.** Re-run the baseline and report postings-scored reduction and
   latency at each corpus size. A result showing no improvement is reported as such, not
   buried.

Plus the standing gates: `cargo fmt --all --check`, `cargo clippy -p veda-storage
--all-targets -- -D warnings`, and the existing `veda-storage` and `veda-gateway` suites.

## Risks

| Risk | Mitigation |
|---|---|
| BMW is subtle; an off-by-one in pivot or block advancement silently drops results. | The differential test against brute force over randomized inputs is the gate, and both paths live on the same struct so they cannot diverge in fixture setup. |
| Pruning provides no measurable win at Veda's current corpus sizes. | Baseline is captured first. If the win is not there at realistic sizes, that is the finding, and it is reported rather than shipped as an unmeasured claim. |
| `build_blocks` at every segment open adds cold-start latency. | Measured as part of the baseline work. The pass is O(total postings) against a JSON parse of the same postings. |
| The `PostingCursor` trait fits BM25 but not sparse, making Cycle B rework it. | The trait is deliberately expressed in terms of doc ordering and impact bounds only — both of which sparse dot-product posting lists have. Cycle B's spec re-validates this before building on it. |

## Cycle B pointer

One finding here is likely to shrink Cycle B substantially: `sparse_storage.rs` also loads
its entire `Vec<SparseVec>` into memory at segment open. If the sparse inverted index can
likewise be derived in memory at open rather than persisted, Cycle B may need no on-disk
format change either — which would remove its migration story entirely. Cycle B's spec
should test that assumption against real sparse-vector sizes before assuming a new
storage format is required.
