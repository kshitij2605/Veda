# Indexes

Veda ships seven index algorithms in one engine, so you can pick the right recall/latency/memory tradeoff per collection instead of working around a single index type.

## The seven indexes

| Index | Best for | Unique advantage |
|---|---|---|
| **HNSW** | General-purpose, low-latency search | Industry standard; tunable `M` / `ef` parameters |
| **DiskANN** | Billion-scale datasets that exceed RAM | Vamana graph on SSD, minimal memory footprint |
| **IVF-PQ** | Memory-constrained, high-throughput | Product quantization compresses vectors 8–32× |
| **Flat** | Small collections, 100% recall guarantee | Brute-force baseline, no build time |
| **RaBitQ** | Ultra-fast approximate screening | Binary quantization with Hamming distance |
| **Suffix array** | Exact text pattern matching | Substring + fuzzy matching; enables hybrid text + vector queries |
| **Scalar** (range / inverted / bitmap) | Metadata filtering | Pre-filter before ANN search |

## Quantization

Beyond IVF-PQ's product quantization, Veda supports **RaBitQ** (binary) and **SQ8** (scalar) quantization for memory-efficient vector compression with minimal recall loss.

## Distance computation

Distance kernels are SIMD-optimized — AVX2 fast paths with a scalar fallback — across six metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, and Jaccard. The hot loop runs in pure Rust with no GC pauses and no FFI boundary.

## MVCC-aware search

Every ANN index (HNSW, Flat, IVF-PQ, DiskANN) implements a native `search_filtered` that consults the snapshot-visibility bitmap *inside* the candidate walk — there is no oversample-and-post-filter step, so recall does not collapse when reading sparse historical snapshots (see [snapshot reads](/docs/search#snapshot-reads-time-travel)).

## Benchmarks

Measured on 10,000 vectors of dimension 768, single node, recall@10:

| Index | Recall@10 | Latency (p99) |
|---|---|---|
| HNSW (M=16, ef=200) | 95.5% | ~2ms |
| DiskANN (R=64, L=100) | 99.9% | ~5ms |
| IVF-PQ (nlist=256, m=48) | 89.2% | ~1ms |
| Flat (brute-force) | 100.0% | ~12ms |

Run the benchmark suite yourself:

```bash
cargo test --test ann_benchmark -- --ignored --nocapture
```
