# Indexes

Veda ships **11 CPU vector index types** in one engine, so you can pick the right recall/latency/memory tradeoff per collection instead of working around a single index type. Scalar indexes for metadata filtering and the suffix array for exact text patterns are separate structures, covered further down.

## The 11 vector indexes

All 11 are reachable through the API and validated end-to-end.

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
| **SPANN** | `spann` | Indexes larger than the node that serves them; posting lists live in object storage |

A GPU **CAGRA** index is also available in builds compiled with NVIDIA cuVS — see [GPU index (CAGRA)](#gpu-index-cagra) below. It is not counted among the 11 CPU index types, and on a build without cuVS, creating a collection with `cagra` is rejected with a `400` rather than failing later at index-build time.

## Recall, measured on a corpus that looks like embeddings

A recall number without the corpus it was measured on is not a claim. Benchmarks routinely generate **uniform-random vectors**, where every point is roughly equidistant from every other in high dimension — so the resulting figures describe the corpus, not the index. Measured that way, HNSW at 768 dimensions scores 58.4%. Measured on a **clustered** corpus, the shape real embeddings actually have, the same index on the same data scores 99.6%.

So these are the numbers we publish, and the conditions they were taken under: **10,000 vectors, clustered corpus, cosine, recall@10 against exact search**.

| Index | 128d | 768d |
|---|---|---|
| **Flat** (exact by construction) | 100.0% | 100.0% |
| **HNSW** (`ef_search=128`) | 100.0% | 99.6% |
| **HNSW** (tuned, `ef_search=256`) | 100.0% | 99.9% |
| **IVF-PQ** (`nlist=64`) | 99.2% | 99.6% |
| **DiskANN** (Vamana, `r=64`) | 100.0% | 99.7% |
| **RaBitQ + rerank** (`rabitq_rerank`) | 100.0% | 100.0% |
| **RaBitQ** (bare 1-bit scan) | 2.4% | 1.2% |

### DiskANN, resolved

An earlier version of this page reported DiskANN at roughly 10% recall at 768 dimensions and declined to quote a figure. **That is fixed.** The cause was distance concentration: in high dimension the graph's pruning rule stops firing, so each node's edge budget fills with its nearest cluster-mates and the graph fragments — 999 of 10,000 nodes were reachable from the entry point. Reserving a few edges per node for long-range targets restores navigability.

DiskANN is included in the table above. Its remaining weakness is **build time**, not recall: at 768 dimensions the build is several times slower than HNSW for the same corpus. If build latency matters more than the memory profile, use HNSW.

### RaBitQ: use the rerank variants

`rabitq` on its own — the bare 1-bit scan with no re-scoring — scores about **2% recall@10 on a clustered corpus** at both 128 and 768 dimensions, and should not be selected for one.

This is an information limit rather than a tuning problem. One bit per dimension cannot order points *inside* a tight cluster: the distance estimate's error is a large fraction of the entire spread between a query's true neighbours. We tested the obvious fix — a k-means residual codebook, which is what RaBitQ is designed for — and it cut reconstruction error twentyfold while leaving recall unchanged.

What does work is giving the exact re-scoring pass enough candidates. `rabitq_rerank` reaches **1.000** on the same corpus. Use it, or `sq8`, and treat the bare scan as a screening primitive for uniform-ish data rather than a general-purpose index.

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

## SPANN — the index in object storage

Every other index here is *resident*: the node that answers a query holds the
whole structure. `spann` splits it in two.

- **Resident:** the centroids. Their size depends on the number of clusters,
  **not on the number of vectors**, so it does not grow as the collection does.
- **In object storage:** the posting lists, written as size-capped objects.
  A query finds its nearest centroids and fetches **only those lists**, by byte
  range.

Measured on a clustered corpus at 768 dimensions, recall@10 against exact
search:

| Vectors | Index size | Held on the node | Read per query | recall@10 |
|---|---|---|---|---|
| 10,000 | 30.8 MB | 0.31 MB (1.01%) | 2.96% | 0.950 |
| 50,000 | 153.8 MB | 0.69 MB (0.45%) | 3.83% | 0.912 |

The corpus grows fivefold and the resident *share* falls, because it tracks the
centroid count rather than the corpus.

`nprobe` is the dial: more probes read more of the index and return more of the
true neighbours. At 50,000 vectors, `nprobe=16` reaches recall 1.000 while still
reading 6.6% of the index per query.

**This is not the same thing as storing blobs on S3.** Veda has always kept
original files in MinIO or S3; here the *index itself* lives there, so a node
can serve a collection it could not hold.

### Building without holding the corpus

The default build receives the whole batch in memory, because that is the shape
the search executor hands every index type. A streaming build takes a
`VectorSource` instead and holds only a budgeted slice at a time, trading passes
over the source for memory. It produces a **byte-identical** index — same
centroids, same layout, same objects — when the budget admits the whole corpus.

20,000 vectors at 768 dimensions (61.5 MB of posting-list bytes):

| Budget | Groups | Passes | Peak held | vs corpus |
|---|---|---|---|---|
| 61.5 MB | 1 | 3 | 61.5 MB | 100% |
| 15.4 MB | 5 | 7 | 15.1 MB | 24.5% |
| 3.8 MB | 18 | 20 | 6.1 MB | 10.0% |
| 1.0 MB | 69 | 71 | 2.8 MB | **4.5%** |

The floor is the largest posting list, not the budget: a list is never split
across parts, so it is always buffered whole. `nlist` governs that floor, and a
budget below it buys extra passes for no saving.

**What it does not do yet:** this path is opt-in. `IndexType::Spann` still
builds through the in-memory route, so choosing `spann` in a collection lowers
what a node needs to *serve*, not yet what it needs to *build*. Closing that
gap needs a `VectorSource` backed by segment storage.

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
