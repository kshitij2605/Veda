# DiskANN 768d build cost — measured, and one optimisation rejected

**Date:** 2026-08-20. **Status:** open. No code shipped; this records what was tried so it is not
tried again.

## The cost, and where it is

The 10,000 × 768d Vamana build takes **382.8s** (measured this session on iitgpu11, release)
against ~29s for HNSW. Recall@10 is 0.994, so the graph is good — it is the build that is slow.

The reverse-edge pass dominates, and the shape of it is:

```
for each of n nodes p:
  for each of p's ~r new neighbours:
    if that neighbour's list exceeds r:
      robust_prune it: ~r selection rounds over an ~r+1 candidate list
```

At `n = 10,000`, `r = 64` that is roughly `n × r × r²/2` ≈ **1.3 × 10⁹ distance computations** at
768 dimensions, which is ~2 × 10¹² flops — consistent with the observed wall clock.

## What was tried and rejected

**Precomputing a pairwise distance matrix inside `robust_prune` for small candidate sets.**

The reasoning was that the selection loop asks for `dist(v, p*)` once per (selected `p*`,
surviving `v`) pair, and under the distance concentration documented in `builder.rs` almost
nothing is pruned, so `remaining` barely shrinks and the total should be ~`r × candidates`. A
matrix is `candidates²/2`, which looked ~1.8× cheaper for the `r + 1`-length lists the
reverse-edge pass produces.

**Measured, it was 35% slower**, at identical recall:

| | build | recall@10 |
|---|---|---|
| unchanged | **382.8s** | 0.994 |
| with the matrix | 516.8s | 0.994 |

The reasoning was wrong in two ways, both only visible from the measurement:

1. **The loop already shrinks.** The original code does `remaining.remove(0)` each round, so
   round *i* scans `candidates − i` elements, not `candidates`. The total is already
   ~`candidates²/2` — the same as the matrix. There was no saving to capture.
2. **The matrix computes pairs the loop never needs.** When pruning *does* remove candidates the
   loop skips those pairs entirely, while the matrix has already paid for all of them. Plus a
   ~17 KB allocation per prune call, roughly 600,000 times.

A differential test confirmed the two paths produce byte-identical output, so this was purely a
cost regression, not a behaviour change. Both were reverted.

## What is left

The cost is inherent to the algorithm at this `r`: `n × r³/2` distance computations. Three
levers, none free:

- **Reduce `r`.** Directly cuts the cube, and directly changes graph quality. Needs a recall
  sweep, not a guess.
- **Make the reverse-edge prune fire less often.** Already tried in a previous cycle as "degree
  slack" (letting lists run 25% over `r` before pruning): it helped 128d (24s → 13s) and made
  768d **worse** (435s → 630s), because longer lists inflate every visited set in turn. Recorded
  in `builder.rs`. Do not retry.
- **Parallelism.** The outer loop is sequential — each node reads the graph built so far — but a
  single node's reverse-edge updates touch distinct neighbour nodes and could run concurrently
  under per-node locking. This is the only untried lever with headroom, and it is a real change
  to `VamanaGraph`'s interior mutability, not a local optimisation.

## Note on the baseline

The 382.8s here does not match the ~509s recorded on 2026-08-19 for the same build. Neither run
had exclusive use of the host. Treat build-time figures from this machine as ±30% unless the two
arms were measured back to back, as the A/B above was.
