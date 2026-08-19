# Indexes

Veda ships **10 CPU vector index types** in one engine, so you can pick the right recall/latency/memory tradeoff per collection instead of working around a single index type. Scalar indexes for metadata filtering and the suffix array for exact text patterns are separate structures, covered further down.

## The 10 vector indexes

All 10 are reachable through the API and validated end-to-end.

| Index | API value | Best for |
|---|---|---|
| **Flat** | `flat` | Small collections; brute-force baseline, exact by construction, no build time |
| **HNSW** | `hnsw` | General-purpose, low-latency search; tunable `M` / `ef` |
| **DiskANN** | `diskann` | Datasets that exceed RAM; Vamana graph on SSD |
| **IVF-PQ** | `ivfpq` | Memory-constrained, high-throughput; PQ compresses 8–32× |
| **Sq8** | `sq8` | Quantized storage at half the memory of f32 |
| **Sq4** | `sq4` | Memory-tight collections; two dimensions per byte |
| **RaBitQ** | `rabitq` | 1-bit codes, ~32× smaller than f32, for fast screening |
| **RaBitQ + rerank** | `rabitq_rerank` | 1-bit screening, then re-scoring against precise vectors |
| **HNSW + RaBitQ** | `hnsw_rabitq` | Graph traversal over 1-bit codes |
| **RaBitQ ext rerank** | `rabitq_ext_rerank` | External rerank pass over quantized candidates |

A GPU **CAGRA** index is also available in builds compiled with NVIDIA cuVS — see [GPU index (CAGRA)](#gpu-index-cagra) below. It is not counted among the 10 CPU index types, and on a build without cuVS, creating a collection with `cagra` is rejected with a `400` rather than failing later at index-build time.

## Recall, measured on a corpus that looks like embeddings

A recall number without the corpus it was measured on is not a claim. Benchmarks routinely generate **uniform-random vectors**, where every point is roughly equidistant from every other in high dimension — so the resulting figures describe the corpus, not the index. Measured that way, HNSW at 768 dimensions scores 58.4%. Measured on a **clustered** corpus, the shape real embeddings actually have, the same index on the same data scores 99.6%.

So these are the numbers we publish, and the conditions they were taken under: **10,000 vectors, clustered corpus, cosine, recall@10 against exact search**.

| Index | 128d | 768d |
|---|---|---|
| **Flat** (exact by construction) | 100.0% | 100.0% |
| **HNSW** (`ef_search=128`) | 100.0% | 99.6% |
| **HNSW** (tuned, `ef_search=256`) | 100.0% | 99.9% |
| **IVF-PQ** (`nlist=64`) | 99.2% | 99.6% |

### Two indexes we are not quoting

**DiskANN** and **RaBitQ** are under investigation and we are not publishing a recall figure for either.

DiskANN is **dimension-dependent** right now: a graph-pruning fix brought it to 100.0% at 128 dimensions, but on the same clustered corpus at 768 dimensions it returns roughly 10%. Since most embedding models output 768 or more, that is the case that matters, and we would rather say so than publish the flattering half of the result.

RaBitQ's 1-bit encoding degrades on clustered data — its reconstruction error roughly triples against uniform-random — and recall follows. Its rerank variants exist precisely because 1-bit screening is meant to be re-scored against precise vectors, but we have not re-measured those either.

If recall matters to your choice of index — and it should — measure it on your own embeddings rather than trusting anyone's table, ours included. [RAG auto-tuning](/docs/auto-tuning) does this for you: it searches index configurations against a test set generated from your own corpus and reports recall per trial.

### Not vector indexes

| Structure | Purpose |
|---|---|
| **Suffix array** | Exact substring and fuzzy text matching; enables hybrid text + vector queries |
| **Scalar** (inverted / range / bitmap / hash) | Metadata filtering — pre-filter before ANN search |

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

Accepted values: `hnsw`, `diskann`, `ivfpq` (or `ivf_pq`), `flat`, `rabitq`, `rabitq_rerank`, `hnsw_rabitq`, `rabitq_ext_rerank`, `sq4`, and `sq8`.

## Scalar quantization (Sq4 / Sq8)

`Sq4` and `Sq8` quantize each dimension independently against per-dimension min/max calibration; `Sq4` packs two dimensions per byte. Measured recall@10, with both corpora named — the synthetic column is the unrepresentative one, and is shown only so the two can be compared:

| Index | Synthetic (d=1536) | Real BGE-M3 embeddings (n=13k, d=1024) | Bytes/vector (d=1536) |
|---|---|---|---|
| `Sq4` (4-bit/dim) | 0.879 | 1.000 | 768 |
| `Sq8` (8-bit/dim) | 0.995 | 1.000 | 1536 |

Rules of thumb: `Sq8` is the highest-recall quantized option — near-exact at half the memory of f32. `Sq4` halves memory again and, on real embedding distributions, matches it. Both are brute-force scans over compact codes (like RaBitQ), so they shine where memory dominates and collections are mid-sized; graph indexes (HNSW/DiskANN) win on latency at very large scale.

## GPU index (CAGRA)

Builds compiled with **NVIDIA cuVS** get a GPU CAGRA index. Its decisive advantage is **build time**.

Measured on an RTX 6000 Ada, clustered corpus, 128 dimensions, k=10, on an otherwise idle host:

| Corpus | Index | Build | Search | Recall@10 |
|---|---|---|---|---|
| 100k | CAGRA | **16.4s** | 1018µs | 1.000 |
| 100k | HNSW | 171.2s | **550µs** | 0.990 |
| 1M | CAGRA | **86.2s** | **878µs** | 0.930 |
| 1M | HNSW | 4184.4s | 4488µs | 0.913 |

At 1M vectors CAGRA builds the index in **86 seconds against HNSW's 70 minutes** — roughly 49× faster — at equal or better recall (0.930 vs 0.913). Its search latency is also nearly flat as the corpus grows (1018µs at 100k, 878µs at 1M) where HNSW's grows 8.2× over the same range.

**It is not a blanket win.** At 100k, HNSW is **1.85× faster per query** than CAGRA. The GPU only pays for its fixed overhead once the corpus is large enough; on this hardware the crossover sits somewhere between 100k and 1M. These latencies also come from a one-query-at-a-time interface, which is the worst case for a GPU — batched throughput was not measured, so do not read these as CAGRA's ceiling.

Requires a cuVS build. Without one, `cagra` is rejected at collection-create with a `400`.

## Binary quantization (RaBitQ)

RaBitQ stores 1 bit per dimension (~32× smaller than f32) for ultra-fast Hamming screening, with rerank variants that re-score top candidates against precise vectors for higher recall.

## Distance computation

Distance kernels are **SimSIMD-backed and on by default**, covering **AVX-512, ARM NEON, and SVE** — measured up to ~9–11× faster than scalar for dot product at dimensions 768–3072 on an AVX-512 host. An AVX2/scalar path remains as the fallback (and via `--no-default-features`). Six metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard. The hot loop runs in pure Rust with no GC pauses and no FFI boundary.

## MVCC-aware search

Every ANN index implements a native `search_filtered` that consults the snapshot-visibility bitmap *inside* the candidate walk — there is no oversample-and-post-filter step, so recall does not collapse when reading sparse historical snapshots (see [snapshot reads](/docs/search#snapshot-reads-time-travel)).

## ANN benchmarks

The bundled benchmark generates its corpus from uniform-random vectors, which — as described [above](#a-note-on-recall-figures) — is not representative of real embeddings, and its recall numbers should not be read as a ranking of these indexes. It is still useful for spotting a regression against itself, and for latency on your own hardware.

Run it yourself:

```bash
cargo test --test ann_benchmark -- --ignored --nocapture
```
