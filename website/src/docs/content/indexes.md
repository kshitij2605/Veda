# Indexes

Veda ships nine index algorithms in one engine, so you can pick the right recall/latency/memory tradeoff per collection instead of working around a single index type.

## The nine indexes

| Index | Best for | Unique advantage |
|---|---|---|
| **HNSW** | General-purpose, low-latency search | Industry standard; tunable `M` / `ef` parameters |
| **DiskANN** | Billion-scale datasets that exceed RAM | Vamana graph on SSD, minimal memory footprint |
| **IVF-PQ** | Memory-constrained, high-throughput | Product quantization compresses vectors 8–32× |
| **Flat** | Small collections, 100% recall guarantee | Brute-force baseline, no build time |
| **RaBitQ** | Ultra-fast approximate screening | 1-bit binary quantization with Hamming distance, ~32× smaller |
| **Sq4** | Memory-tight collections that still need strong recall | 4-bit-per-dim scalar quantization — two dims per byte, 4× smaller than f32 |
| **Sq8** | Highest-recall quantized storage | 8-bit-per-dim scalar quantization — near-exact recall at half the memory |
| **Suffix array** | Exact text pattern matching | Substring + fuzzy matching; enables hybrid text + vector queries |
| **Scalar** (range / inverted / bitmap) | Metadata filtering | Pre-filter before ANN search |

## Choosing an index

Set the `index` property on a vector field at collection creation (default: `hnsw`):

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "fields": [
      {"name": "embedding", "type": "vector", "dimension": 1024,
       "metric": "cosine", "index": "sq8"}
    ]
  }'
```

Accepted values include `hnsw`, `diskann`, `ivf_pq`, `flat`, `rabitq`, `sq4`, and `sq8`.

## Scalar quantization (Sq4 / Sq8)

`Sq4` and `Sq8` quantize each dimension independently against per-dimension min/max calibration; `Sq4` packs two dimensions per byte. Measured recall@10:

| Index | Synthetic (d=1536) | Real BGE-M3 embeddings (n=13k, d=1024) | Bytes/vector (d=1536) |
|---|---|---|---|
| `Sq4` (4-bit/dim) | 0.879 | 1.000 | 768 |
| `Sq8` (8-bit/dim) | 0.995 | 1.000 | 1536 |

Rules of thumb: `Sq8` is the highest-recall quantized option — near-exact at half the memory of f32. `Sq4` halves memory again and, on real embedding distributions, matches it. Both are brute-force scans over compact codes (like RaBitQ), so they shine where memory dominates and collections are mid-sized; graph indexes (HNSW/DiskANN) win on latency at very large scale.

## Binary quantization (RaBitQ)

RaBitQ stores 1 bit per dimension (~32× smaller than f32) for ultra-fast Hamming screening, with rerank variants that re-score top candidates against precise vectors for higher recall.

## Distance computation

Distance kernels are **SimSIMD-backed and on by default**, covering **AVX-512, ARM NEON, and SVE** — measured up to ~9–11× faster than scalar for dot product at dimensions 768–3072 on an AVX-512 host. An AVX2/scalar path remains as the fallback (and via `--no-default-features`). Six metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard. The hot loop runs in pure Rust with no GC pauses and no FFI boundary.

## MVCC-aware search

Every ANN index implements a native `search_filtered` that consults the snapshot-visibility bitmap *inside* the candidate walk — there is no oversample-and-post-filter step, so recall does not collapse when reading sparse historical snapshots (see [snapshot reads](/docs/search#snapshot-reads-time-travel)).

## ANN benchmarks

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
