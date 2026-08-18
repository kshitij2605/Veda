# Scale Benchmark Results

Measured numbers from `crates/veda-index/tests/scale_benchmark.rs`. The
test generates a deterministic random corpus at the configured `n`,
builds each index variant, runs 100 random queries, and reports build
time, p50/p95 query latency, and recall@10 against `FlatIndex` ground
truth.

Run with:

```
cargo test --release -p veda-index --test scale_benchmark -- --ignored --nocapture
```

> **The corpus below is uniform-random, and that decides most of what the
> table says.** In high dimension, uniformly random points are all nearly
> equidistant, so the true top-10 is close to arbitrary and *every* graph
> and quantization index scores near chance. Recall figures measured on
> this shape rank index types by how they behave on data no production
> corpus resembles. They are kept because the recorded results used this
> shape and changing it would make them incomparable — not because they
> are representative. See "Corpus shape decides recall" below before
> quoting any number from this section.

## 100K × 1024d (`scale_100k_dim_1024`, release build, single-host)

| Index            | Build (ms) | p50 (ms) | p95 (ms) | recall@10 |
|------------------|-----------:|---------:|---------:|----------:|
| flat (truth)     |        192 |    48.35 |    48.80 |     1.000 |
| hnsw             |    465,866 |     4.60 |     5.11 |     0.381 |
| rabitq           |        824 |   191.46 |   192.62 |     0.349 |
| rabitq+rerank    |      1,015 |   191.61 |   192.34 |     0.779 |
| hnsw+rabitq      |    461,697 |    13.57 |    15.46 |     0.275 |

### Observations

- **flat** is the latency floor for a 100K × 1024d corpus on this host
  (~48 ms per query). Useful as the recall and latency baseline.
- **hnsw** alone is ~10× faster than flat but recall@10 = 0.38 at the
  defaults. This was previously read as "the graph parameters need tuning
  for 1024d". That reading is wrong: the same HNSW implementation scores
  **1.000** on a clustered corpus at default parameters (measured below,
  and again on live data in the Elasticsearch comparison further down).
  0.38 is the corpus shape, not the parameters. The 466-second build cost
  is the real story here; that is the wall-clock motivating GPU CAGRA /
  cuVS for the deep-quantized HNSW path.
- **rabitq** (1-bit only) has the lowest build time at 0.8 s but recall
  collapses to 0.35. Also previously mis-read — as "expected for pure
  binary quantization at this dim". Binary quantization is not the whole
  cause; see the RaBitQ entry below, where the same index scores 0.032 on
  a *clustered* corpus, far worse than here. The defect is that this path
  quantizes against a single global centroid.
- **rabitq+rerank** climbs back to recall 0.78 by reranking the top-N
  binary hits with full-precision vectors. p95 latency stays at ~192 ms
  because rerank dominates; this is the "land where RaBitQ wins" once
  the corpus is large enough that flat-rerank's full pass costs more
  than the quantized prefilter + small rerank.
- **hnsw+rabitq** (quantized HNSW graph) finishes builds in roughly the
  same time as plain HNSW and queries in 14 ms p50, but recall@10 is
  0.27 — the same parameter-tuning gap as plain HNSW, compounded by the
  quantization loss. Calibrating `m` / `ef_construction` for 1024d is
  the next move before drawing a real conclusion on quantized-HNSW vs
  RaBitQRerank at this scale.

### Calibration vs literature claims

- The "RaBitQ wins above 1M rows" claim is consistent with what we see
  at 100K: rabitq+rerank already matches flat-class recall at a fraction
  of the build cost, but its 192 ms p95 is still slower than flat's 48
  ms — the crossover is corpus-size-driven. The 1M run will tell us
  where flat's linear scan stops being free and the quantized prefilter
  starts paying off.
- The "deep-quantized HNSW pays off at 10M" claim is **not** yet
  observable here — at 100K, untuned HNSW is already losing on recall.
  The 10M case is gated by index parameter tuning and GPU CAGRA wiring
  (see `documentation/gpu-indexing.md`).

## Corpus shape decides recall (2026-08-18)

Recall is a property of the corpus as much as of the index. Running the
same index types over a **clustered** corpus — the shape real embeddings
have, where intrinsic dimensionality sits far below the ambient dimension
— re-ranks them, in both directions. Measured with
`crates/veda-index/tests/ann_benchmark.rs` at 10,000 × 128d, Cosine,
10 clusters of ~1,000 points (`ann_benchmark_10k_128d_clustered` and
`ann_benchmark_10k_128d`):

| Index   | recall@10, uniform | recall@10, clustered |
|---------|-------------------:|---------------------:|
| Flat    |              1.000 |                1.000 |
| HNSW    |              0.892 |                1.000 |
| DiskANN |              0.965 |                0.990 |
| IVF-PQ  |              0.349 |                0.984 |
| RaBitQ  |              0.353 |                0.032 |

The RaBitQ row comes from the quantization entry points
(`ann_benchmark_quantization_clustered` and its uniform counterpart),
which run the same corpus size and dimension; every other row is from the
two entry points named above. HNSW is at defaults (`ef_search`=128).

Two findings came out of chasing the outliers in this table.

**DiskANN scored 0.202 on the clustered corpus before 2026-08-18** — the
worst result of any graph index, on the shape it should handle best. The
cause was in graph construction. `build_vamana` pruned over the `l_build`
closest candidates rather than over **V, the visited set**, which is what
Algorithm 3 of the DiskANN paper specifies. A node's closest candidates
are all its immediate neighbours, so pruning over them yields only short
edges; the visited set additionally holds the nodes the walk passed
through on its way in from the medoid, and those are where every
long-range edge comes from. The built graph had **106 cross-cluster edges
out of 618,617**, leaving one disconnected component per cluster, and
greedy search from a single global medoid could reach only 1,051 of
10,000 nodes. Out-degree was measured at 61.86 of `r`=64 — the graph was
not starved of edges, it was starved of *long* ones. Fixed; the guard is
`tests/diskann_connectivity.rs`, which asserts the structural property
rather than a recall number.

**RaBitQ scores 0.032 on the clustered corpus** — worse than on uniform
random, and worse than chance-adjacent. In the Cosine estimator,
`estimated_dot = v_norm * mean_abs * raw_dot` is divided by
`q_norm * v_norm`, so `v_norm` cancels and ranking depends only on
`raw_dot`, the sign-agreement count. On clustered data the sign pattern is
dominated by each cluster's offset from the single *global* centroid, so
every point in a cluster produces a near-identical `raw_dot` and
within-cluster ranking is arbitrary — which is exactly where the true
neighbours are. RaBitQ is designed to quantize **residuals from a nearest
IVF centroid**; using it standalone over globally-centred vectors is the
misuse. Adding a random orthogonal (Hadamard) rotation before binarization
was tried and **measured no better** (0.032 → 0.036): rotation is unitary,
so near-identical vectors stay near-identical and their sign bits still
agree. Not yet fixed; `rabitq+rerank` is unaffected, since reranking with
full-precision vectors repairs the ordering.

**Do not publish per-index recall without naming the corpus shape.** The
same index moves from 0.892 to 1.000, or from 0.353 to 0.032, on nothing
but the data.

## Next runs

- 1M × 1024d (`scale_1m_dim_1024`) — confirm the rabitq+rerank
  crossover.
- Re-run the 100K table on a **clustered** corpus. The uniform-random
  figures above cannot settle any index-quality question, and re-running
  them with tuned HNSW parameters would not change that — the previously
  planned tuning run has been superseded by this.
- Fix RaBitQ to quantize IVF residuals rather than a global centroid, and
  re-measure. Until then the standalone RaBitQ path should not be
  recommended for clustered (i.e. real) corpora.
- 10M × 768d on a GPU host once `gpu_cuvs` build path lands.

---

## Live REST benchmark vs Elasticsearch + the host-contention lesson (2026-06-22)

The table above is the **in-process** index micro-benchmark (`scale_benchmark.rs`). Separately,
the end-to-end **REST** path was benchmarked against a production Elasticsearch cluster on a real
30k × 1024d corpus. The full write-up is in `reports/2026-06-21-veda-vs-es-benchmark.md`; the
headline and the operational lesson belong here:

**Single-segment dense search, ann_search p50 — identical binary + identical data, two hosts:**

| ef | mactrn01 (quiet, 96-core) | gpu-host-a (shared, contended) | Elasticsearch (dedicated cluster) |
|---|---|---|---|
| 96 (default) | **0.53 ms** | 36.6 ms | 7 ms |
| 64 | 0.39 ms | 20 ms | — |
| 32 | 0.21 ms | 11 ms | — |

Recall@10 = 1.0 at ef=96 on both hosts (this corpus is weakly clustered, so the navigable graph
visits only ~1.5 % of nodes — confirmed by instrumentation — yet returns exact top-10).

**Takeaways:**
- On a fair (quiet/dedicated) host, Veda's single-segment dense search is **0.53 ms — roughly
  13× faster than Elasticsearch's 7 ms** at recall 1.0, and ~8× faster on ingest (1837 vs ~200 docs/s).
- gpu-host-a's 36 ms was **~99 % host contention**, not a Veda defect: that box is a heavily-shared
  multi-tenant node (ClickHouse, airflow, ML jobs saturating memory bandwidth). The same binary on
  mactrn01 was ~70× faster. The HNSW graph and AVX2+FMA distance path were healthy throughout.
- **Operational rule: never benchmark Veda on the shared `gpu-host-a` box.** Use a quiet/dedicated
  host (e.g. `mactrn01`) for any latency comparison, or the numbers are dominated by co-tenant
  memory-bandwidth contention.
- Process of elimination ruled out every code-level "fix" (contiguous slab → 0 ms; AVX2 distance
  already vectorized; heap churn negligible; record materialization O(1)). No index optimization is
  warranted for dense latency.
