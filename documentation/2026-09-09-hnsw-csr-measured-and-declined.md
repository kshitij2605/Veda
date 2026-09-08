# The HNSW CSR adjacency TODO: measured, and declined

**2026-09-09.** `hnsw/graph.rs` and `hnsw/mod.rs` carried a standing
`TODO(perf)`: replace the per-node `Vec<Vec<u32>>` edge lists with a CSR
layout (offsets + one flat neighbour array) so graph traversal is
cache-friendly. Half of it — the contiguous vector slab — already shipped.
This closes the other half as **not worth doing**, on measurement.

## What was measured

`cargo test --release -p veda-index --test ann_benchmark`, run **serially**
(`--test-threads=1 --exact`). Serially because the suite's tests run in
parallel by default and the tables interleave in the log: a first pass
attributed rows to the wrong cases, and the giveaway was a memory column that
could not belong to the case it appeared under. Any future run of this suite
that reports per-case numbers must serialise.

| Corpus (uniform) | Flat | HNSW | HNSW tuned |
|---|---|---|---|
| 10k × 128d | 2,068 QPS | 5,783 | 3,216 |
| 10k × 768d | 426 QPS | 1,339 | 964 |

## The argument

Flat is pure distance computation, so it is the control for how cost scales
with dimension. Going 128d → 768d, Flat drops 4.85× and HNSW drops 4.32×.
HNSW tracks the pure-distance index almost exactly, which is what you see when
distance dominates and traversal does not.

Fitting `time_per_query = fixed + k · dim` to the two HNSW points:

```
172.9 µs @128d, 746.8 µs @768d
  ->  k     ≈ 0.90 µs/dim
      fixed ≈ 58 µs
```

`fixed` is everything that is not distance: traversal, the candidate heap,
cache misses — the envelope CSR can attack, and it can only take part of it.

| dim | fixed as share of query time |
|---|---|
| 128 | ~34% |
| 768 | ~8% |
| **1024 (production)** | **~6%** |

Veda runs 1024-d embeddings (`EMBEDDING_DIMENSION=1024`; jina-v5-omni-small
and BAAI/bge-m3 are both 1024). So the *entire* non-distance envelope is ~6%
of query time there, and CSR would capture a fraction of that.

## Why that is a decline, not a deferral

The TODO's own note says "rewrite + on-disk format change". `HnswNode` derives
`Serialize`/`Deserialize` and is persisted, so a naive CSR breaks existing
indexes. That is avoidable — keep `Vec<Vec<u32>>` as the serialised form and
derive CSR into a `#[serde(skip)]` field after build and after load — so the
format objection is not the blocker. The blocker is the return.

This is graph traversal in a load-bearing index, where a subtle error surfaces
as degraded recall rather than a crash: the hardest failure class to notice,
and the one this repo has already been bitten by twice (`#15` Vamana pruning,
`#18` Vamana high-dimension connectivity). Paying that risk for ≤6% at the
dimension the product runs is a bad trade.

**Where the time should go instead:** shrinking the distance computation, not
the pointer chase. That path is already built — RaBitQ, RaBitQ-rerank, SQ4/SQ8
and `HnswRaBitQ` (graph + codes, no raw f32). The TODO pointed at the wrong
layer.

## What this does NOT say

The recall figures above are from the **uniform** corpus, which
`documentation/2026-08-18-corpus-shape-index-ranking.md` establishes
*understates HNSW badly* at 768d — 56.8% uniform against 99.1% clustered. The
59.7% here is that known artefact reproducing, not a finding. Nothing in this
note is a statement about HNSW recall; it is only about where query *time*
goes.

The measurement is at 10k vectors. Hop count grows with log N, so traversal
grows at larger N — but so does nothing else per hop, and distance still
dominates each one. If someone wants to reopen this, the case to make is at
128d or below, or at very large N with a measured hop-count profile, not on
cache-locality reasoning alone.
