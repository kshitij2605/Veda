# RaBitQ on clustered corpora — what is actually wrong, and what fixed it

**Date:** 2026-08-20
**Status:** measured; the acted-on part shipped in `50cb93e`.

## The starting point

`IndexType::RaBitQ` scored **recall@10 = 3.2%** on a 10,000-vector clustered corpus
(10 clusters, sigma 0.10 — the shape `crates/veda-index/tests/ann_benchmark.rs` generates),
against 100% for Flat and 94% for SQ8. Reconstruction error was 0.035 on a uniform corpus and
0.103 on a clustered one, so the *encoding* degraded and not merely the ranking — which is what
ruled out a ranking-only repair.

## The hypothesis, and why it was wrong

RaBitQ is designed to quantize **IVF residuals**. The implementation centred every vector on one
global centroid, so on clustered data every point in a cluster sits on the same side of the
global mean in most dimensions and their sign patterns are nearly identical. Replacing that with
k-means centroids and per-vector residuals is the textbook fix.

It was built in full — k-means codebook, per-vector assignment, true `‖v‖` stored separately from
the residual norm, the estimator corrected to use the **raw** query (the old Cosine path centred
the query too, so the stored norm cancelled and ranking collapsed to sign agreement) — and
measured.

**It made recall worse.**

| | original | k-means residual |
|---|---|---|
| RaBitQ scan, 128d | 0.022 | 0.024 |
| RaBitQRerank, 128d | **0.250** | 0.172 |
| RaBitQExtRerank, 128d | **0.122** | 0.080 |
| RaBitQ scan, 768d | 0.026 | 0.012 |
| RaBitQRerank, 768d | **0.218** | 0.156 |
| RaBitQExtRerank, 768d | 0.098 | 0.096 |
| build time | 0.1–2.2 s | 1.7–5.5 s |

Five of six measurements down, at 10–20× the build cost — while **reconstruction error improved
20×** (0.0705 → 0.0033).

A sweep over centroid count made the reason unambiguous. At n=10,000, dim=128:

| k | recall@10 | mean abs estimate error |
|---|---|---|
| 1 | 0.010 | 0.0705 |
| 4 | 0.030 | 0.0036 |
| 16 | 0.027 | 0.0035 |
| 64 | 0.027 | 0.0034 |
| 256 | 0.027 | 0.0033 |
| 1024 | 0.030 | 0.0033 |

Recall is flat at ~3% across three orders of magnitude of `k` while the estimate error improves
20×. **The codebook was never the binding constraint, and reconstruction error was the wrong
proxy to optimise.**

## What the constraint actually is

One bit per dimension cannot order points *inside* a tight cluster. The estimator's error
(~0.0033 at 128d, ~0.0014 at 768d) is a large fraction of the entire intra-cluster distance
spread, which the DiskANN investigation measured at **[0.0084, 0.0120]** at 768d. The true top-10
of a query sit inside a band narrower than the estimator can resolve, so a pure scan selects
near-arbitrarily from within the right cluster. That is an information limit, not a tuning
problem.

## What did fix it

Rerank depth. Stage 2 re-scores exactly and cannot recover a neighbour stage 1 never returned, so
recall is governed by how deep the candidate pool is:

| multiplier | candidates | recall@10 (128d) | recall@10 (768d) |
|---|---|---|---|
| 10 (old default) | 100 | **0.250** | **0.218** |
| 25 | 250 | 0.448 | 0.418 |
| 50 | 500 | 0.724 | 0.690 |
| **100** | **1000** | **1.000** | **1.000** |
| 200 | 2000 | 1.000 | 1.000 |

`DEFAULT_RERANK_MULTIPLIER` is now 100. The cost does not grow with the corpus — the multiplier
scales with `k`, so it is a fixed ~1000 exact distance computations per `k=10` query at any
collection size.

Reproduce: `cargo test -p veda-index --release --test rabitq_rerank_sweep -- --ignored --nocapture`
and `--test rabitq_variants`.

## Two measurement errors made on the way here

Both were caught by cross-checking against an existing benchmark rather than by inspection, and
both are worth knowing because either one alone would have produced a confident wrong answer.

1. **The centroid sweep derived cluster centres from its per-call seed**, so the corpus and the
   query set had *different* centres and every query landed in the gaps between clusters. It
   reported 45% recall where the benchmark reported 2%. `ann_benchmark.rs::generate_dataset`
   documents this exact trap in a comment, and the benchmark was right.
2. **The depth sweep rebuilt the index inside its timing loop**, charging the first iteration for
   cache warm-up, which produced the nonsense that a *deeper* candidate pool was faster than a
   shallow one.

Latency is deliberately not quoted as a headline anywhere above: the exact-scan baseline moved
100× between two runs on this host. Only the recall figures are trustworthy, and those were
stable to three decimal places across both runs.

## Recommendation

- **Do not select `IndexType::RaBitQ` for a clustered corpus.** ~2% recall at 128d and 768d. Now
  documented on the enum variant itself.
- Use `RaBitQRerank` (1.000 at the new default) or `Sq8` (0.94).
- **Do not retry the k-means residual rewrite** on the strength of reconstruction error. It is a
  better encoding that produces a worse ranking, and this document exists so that experiment is
  not run a second time.
