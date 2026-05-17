# Deep RaBitQ → HNSW: design

Build the HNSW graph once over precise f32 vectors, then at search
time use 1-bit-per-dim quantized distance estimates so the working
set fits in cache. This is the "store HNSW edges + RaBitQ codes,
drop raw vectors" variant; complement to `RaBitQRerankIndex` which
keeps both raw and codes side-by-side.

## Status

**Not yet implemented**. The standalone `RaBitQIndex` (commit
`6c3f676`) and the candidate-generation-plus-rerank pattern
`RaBitQRerankIndex` (commit `73a3407`) are both in place. This doc
covers the next step: quantization deeply integrated into the HNSW
graph traversal itself.

## Why

| Approach | Memory per vec | Search compute | Recall | When to use |
|----------|----------------|----------------|--------|-------------|
| HNSW (raw)              | 4d B  | f32 dot     | Highest | < 1 M rows, recall-first |
| `RaBitQIndex` standalone | d/8 B | XOR+popcount | -3-5%  | ≥ 1 M rows, memory-tight |
| `RaBitQRerankIndex`     | 4d + d/8 B | XOR + f32 (top-K only) | ~HNSW | ≥ 1 M rows, mixed budget |
| **HNSW + RaBitQ deep** (this doc) | d/8 B + HNSW edges | XOR+popcount over graph traversal | ~HNSW | ≥ 10 M rows, memory + latency |

The proposition: get HNSW's logarithmic pruning *and* RaBitQ's 32×
memory + cache-friendly distance op. Not "instead of" the other
quantized modes — a different operating point.

## Build phase

1. Train RaBitQ centroid on the corpus (same as the existing
   `RaBitQuantizer::train`).
2. Build HNSW the way it builds today, BUT compute every distance via
   `RaBitQuantizer::estimate_distance` against pre-encoded codes
   rather than `compute_distance` against raw f32 slices. Edge
   selection uses the estimated distances directly — the resulting
   graph is calibrated against the same distance function search
   will use.
3. After build, drop the raw vectors. Persist only the codes (~d/8
   bytes per vector) plus the HNSW graph topology.

## Search phase

Drop-in replacement: every `compute_distance(query, vectors[id])` in
`crates/veda-index/src/hnsw/searcher.rs` becomes
`quantizer.estimate_distance(query, id)`. The query stays full-
precision (asymmetric distance).

## Why this is non-trivial

`HnswSearcher` currently takes `vectors: &[Vec<f32>]` as one
parameter. Changing that to a polymorphic distance source ripples
across:

- `search_layer`, `search_layer_filtered`, `search_layer_in_algorithm_filter`,
  `search_layer_post_filter`, `greedy_search` — all f32-slice
  hard-coded.
- `HnswIndex::search`, `search_filtered`, `insert` — same.
- Builder helpers — every neighbor-distance compute in
  `crates/veda-index/src/hnsw/builder.rs`.

A clean refactor introduces a `DistanceProvider` trait:

```rust
pub trait DistanceProvider {
    fn distance(&self, query: &[f32], id: u32) -> f32;
    fn len(&self) -> usize;
}

impl DistanceProvider for RawVectors { /* compute_distance */ }
impl DistanceProvider for RaBitQuantizer { /* estimate_distance */ }
```

Searcher takes `&dyn DistanceProvider` (or `&P: DistanceProvider`).
Same trait covers raw-f32 and quantized backends; the indexing-time
build path picks one.

## Migration

Optional via a new `IndexType::HnswRaBitQ` (or a `quantize: bool`
field on the existing HNSW config). Existing segments built without
quantization keep working unchanged.

Persisted format gains a flag in the segment HNSW header:
`is_quantized: bool`. When set, the segment dir also has
`{field}.rabitq` next to `{field}.hnsw`. Reader picks the right
distance provider at open time.

## Estimated effort

~2-3 days for the trait refactor + the deep wiring + tests. The
biggest risk is the build-time recall drop from training the graph
against quantized distances — Tian et al. (2024) show this is
manageable with a `refine_top_k` pass at the end of build, similar
to what `RaBitQRerankIndex` already does at search time. The proper
implementation also needs a benchmark against `RaBitQRerankIndex` on
the same corpus to confirm the recall delta is acceptable for the
memory savings.

## Why we deferred

The benchmark workload Veda has been tested against (rag_anything,
2.4 K rows) is well below the regime where this design starts to
pay off. Pure HNSW already runs at < 10 ms p50 with ~94% top-10
recall. The deep-quantized variant matters at ≥ 10 M rows where
HNSW's raw f32 vectors blow past L3 cache and become a bandwidth
bottleneck. Land this when a real ≥ 10 M-row workload is available
to validate the recall/latency trade.

## What lands here

This document is the design baseline. No code yet. The trait shape
above is the entry point for the implementation work when it's
scheduled.
