# Corpus shape mis-ranks Veda's index types — measured

**Date:** 2026-08-18
**Origin:** the CAGRA repair (`documentation/2026-08-15-gpu-cuvs-cagra-design.md`), where
uniform-random vectors hid a recall collapse three separate times. That raised the question of
whether the other index types share the problem, since `tests/ann_benchmark.rs` used the same
uniform generator for every one of them.

**Answer: yes, and it mis-ranks in both directions.** Two index types look *better* on
uniform-random data than on a representative corpus, which is the dangerous direction — their
recorded numbers flatter them.

---

## Measured

`cargo test --release -p veda-index --test ann_benchmark -- --ignored --test-threads=1`,
Cosine, k=10, 100 queries, one host, nothing else running.

Both shapes use the same PRNG and seeds; only the shape differs. The clustered shape derives its
cluster count from the corpus size to hold ~1,000 points per cluster, so that top-10 is a ranking
problem *inside* a cluster rather than a cluster-identification problem — see "A trap worth
recording" below.

### recall@10, 10,000 x 128

| Index | uniform | clustered | verdict |
|---|---|---|---|
| Flat | 100.0% | 100.0% | exact either way, as expected |
| HNSW | 89.3% | **100.0%** | uniform **understates** |
| HNSW (ef=256) | 98.1% | **100.0%** | uniform understates |
| DiskANN | 97.2% | **20.3%** | uniform **overstates badly** |
| IVF-PQ | 34.4% | **98.0%** | uniform **understates badly** |

### recall@10, 50,000 x 128

| Index | uniform | clustered | verdict |
|---|---|---|---|
| Flat | 100.0% | 100.0% | — |
| HNSW | 56.8% | **99.1%** | uniform understates badly |
| HNSW (ef=256) | 77.9% | **100.0%** | uniform understates |
| DiskANN | 73.4% | **20.1%** | uniform overstates badly |
| IVF-PQ | 24.3% | **100.0%** | uniform understates badly |

### recall@10, quantizers, 10,000 x 128

| Quantizer | uniform | clustered | MAE uniform | MAE clustered |
|---|---|---|---|---|
| RaBitQ (1-bit) | 35.3% | **3.2%** | 0.0352 | 0.1030 |
| Scalar (SQ8) | 99.4% | 94.0% | 0.0003 | 0.0002 |

---

## What this changes

### 1. DiskANN needs its own investigation — probable defect, not tuning

97.2% -> 20.3% at 10k and 73.4% -> 20.1% at 50k. **R@100 is also ~20%**, and the figure barely
moves between two corpus sizes. A tuning problem would drift with scale and would improve at
k=100; a stable ~20% at both k values and both sizes looks structural — as though only a fraction
of the graph is ever reached.

This is not diagnosed here and should not be reported as a bug until it is. But DiskANN is
currently documented at 97.1% recall on the strength of the uniform measurement, and that number
does not describe its behaviour on a clustered corpus.

### 2. IVF-PQ's recorded weakness is an artifact

24.3–34.4% uniform against 98.0–100.0% clustered. IVF-PQ partitions by centroid, so a corpus with
no cluster structure is precisely its worst case and precisely what it was measured on. Nothing is
wrong with the index; the corpus was wrong.

### 3. HNSW is substantially understated

56.8% -> 99.1% at 50k. `documentation/scale-benchmark-results.md` concludes "at 100K, untuned HNSW
is already losing on recall" and recommends `rabitq+rerank` over it on that basis. That conclusion
rests on the uniform measurement and should be re-examined.

### 4. RaBitQ is the opposite case, and is probably correct behaviour

35.3% -> **3.2%**, with MAE tripling from 0.0352 to 0.1030. This is expected rather than a bug:
1-bit quantization keeps the sign per dimension, so points inside a tight cluster fall on the same
side in most dimensions and their codes become nearly identical — the quantizer cannot discriminate
*within* a cluster, which is exactly what top-10 requires here.

`scale-benchmark-results.md` says "recall collapses to 0.35 — expected for pure binary quantization
at this dim". Directionally right, but it **understates** the effect by an order of magnitude on
representative data, and it means the rerank variant is load-bearing rather than an optimisation.

---

## A trap worth recording

The first run of this comparison reported **100.0% for every index type** on the clustered corpus,
which looked like a clean result and was worthless. The generator used a fixed 1,000 centroids, so
at 10,000 points each cluster held 10 points — and with k=10 the answer set *is* one cluster. Every
index that found the right cluster scored perfectly.

That is the same failure as the uniform case wearing the opposite disguise: a corpus that cannot
discriminate between a good index and a bad one. The cluster count is now derived from the corpus
size to hold ~1,000 points per cluster, which is also the ratio the CAGRA benchmark happened to
use (1M points, 1,000 clusters) and the reason that benchmark was not saturated.

**Neither shape is "the" right answer.** Uniform-random is unrepresentative; a clustered corpus
with too few points per cluster is trivial; real embeddings are neither. What the comparison
supports is narrower and more useful than a new ranking: *any* recall number is meaningless
without the corpus it was measured on, and a single-shape benchmark cannot rank index types.

## What was changed

`crates/veda-index/tests/ann_benchmark.rs` gained a `Corpus` shape parameter threaded through
`run_benchmark` and `run_quantization_benchmark`, with four `_clustered` entry points beside the
existing ones. **The existing entry points keep `Corpus::Uniform`** so previously recorded numbers
stay comparable — the goal is to measure the delta, not to silently replace the baseline. The
`Uniform` variant's doc comment states that it is not representative and why it is retained.

## Not done

- No index code was changed. Nothing here is a fix.
- DiskANN's collapse is unexplained.
- Not measured on clustered data: `sq4`, `hnsw_rabitq`, `rabitq_ext_rerank`, `suffix_array`, and
  the 768-dimension cases. The 768d clustered entry point exists but was not run.
- `documentation/scale-benchmark-results.md` is **not** rewritten. Its conclusions about HNSW and
  RaBitQ are affected, and the numbers above are the input for revising it, but that is a separate
  change against a document with its own measurement provenance.
