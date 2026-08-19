# Recall re-measurement after the Vamana fix

**Measured:** 2026-08-19 · **Build:** `main` @ `7882b3c` (includes #15, "prune the Vamana graph over
the visited set") · **Harness:** the existing `ann_benchmark` clustered entry points, unmodified
**Raw output:** `reports/2026-08-19-recall-remeasure.txt`, `reports/2026-08-19-quant-remeasure.txt`

## Why

The 2026-08-18 brief (§9) pulled DiskANN and RaBitQ recall figures off the website as probable
defects, pending investigation. #15 fixed a real Vamana bug, so the question was whether the numbers
could go back.

**Partly. One index is fixed at low dimension only, and the other is unchanged.**

## Results — 10k vectors, clustered corpus, cosine, R@10

| Index | 128d | 768d |
|---|---:|---:|
| Flat (exact) | 100.0% | 100.0% |
| HNSW (ef=128) | 100.0% | 99.6% |
| HNSW (tuned, ef=256) | 100.0% | 99.9% |
| IVF-PQ (nlist=64) | 99.2% | 99.6% |
| **DiskANN** | **100.0%** | **10.0%** |

Quantizers, 10k × 128d:

| Quantizer | uniform | clustered | MAE (clustered) |
|---|---:|---:|---:|
| Scalar (SQ8) | 99.4% | 94.0% | 0.0002 |
| **RaBitQ** | 35.3% | **3.2%** | 0.1030 |

## What this says

**DiskANN is fixed at 128 dimensions and still broken at 768.** It went from the ~20% §9 recorded to
a clean 100.0% at 128d — #15 did real work. At 768d on the same clustered corpus it returns **10.0%**,
which is worse than the figure §9 retracted. Since production embeddings are 768–1024d, the fix does
not yet reach the case that matters, and **§9's hold on publishing a DiskANN recall figure should
stand**. The dimension split is new information: §9 did not separate 128d from 768d, so "DiskANN is
fixed" would be a fair reading of #15 alone and would be wrong.

**RaBitQ is unchanged at 3.2%** clustered, identical to §9. There is no RaBitQ fix on `main` — the
most recent commit touching `quantization/rabitq.rs` is unrelated CI work. Its MAE also rises 3× from
uniform to clustered (0.0352 → 0.1030), so the encoding itself degrades on clustered data rather than
just the ranking.

**HNSW, IVF-PQ, Flat and SQ8 are healthy on clustered data** and can be published with the corpus
stated. IVF-PQ in particular measures 99.2–99.6% clustered against 22–37% uniform, confirming §9's
point that the uniform corpus was slandering it.

**Uniform-random remains actively misleading**, and at 768d it misleads about HNSW too: HNSW scores
58.4% uniform against 99.6% clustered on the same size and dimension. Any recall figure without its
corpus shape is not a claim.

## Practical note

DiskANN build time at 10k × 768d was **457s clustered / 1302s uniform**, versus ~8s for HNSW on the
same data. Whatever is wrong with recall at high dimension, the build cost is also far out of line.

## Website

`/docs/indexes` now publishes the clustered figures for Flat, HNSW and IVF-PQ, each labelled with
corpus shape, size and dimension. DiskANN and RaBitQ remain unpublished, and the page says why —
including that DiskANN is dimension-dependent, so a reader running a 128d workload is not told the
index is broken for them.
