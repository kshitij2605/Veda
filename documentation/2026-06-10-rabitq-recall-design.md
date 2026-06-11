# Design: RaBitQ recall improvements + TurboQuant benchmark

**Date:** 2026-06-10
**Status:** Approved (brainstorming)
**Sub-project:** B of the turbovec-parity decomposition.
**Branch / workspace:** `feat/rabitq-recall` off `main`, in the worktree
`.claude/worktrees/rabitq-recall`.

## Context

A competitor audit (`turbovec`) plus a focused audit of Veda's RaBitQ family
(see the B-audit findings in the conversation history / commit log) established:

- **Length renormalization** (per-vector residual norm) is already correctly
  implemented in both Veda's 1-bit and multi-bit RaBitQ paths — NOT a turbovec
  differentiator.
- **Per-vector bias correction** of the inner-product estimate is the paper's
  unbiased RaBitQ estimator (divide the binary-code dot by the per-vector
  factor `⟨ō,o⟩`). Veda's **1-bit** path (`rabitq.rs`) substitutes a single
  **global** `mean_abs_component` constant — a mean-field approximation, not the
  per-vector unbiased estimator. Latent recall issue for the bare `RaBitQIndex`
  and for the coarse candidate-gen stage of the rerank indexes (muted in the
  rerank config by 10× oversampling).
- The "recommended default" `RaBitQExtRerankIndex` builds its 4-bit rerank
  quantizer with the **legacy** `ExtRaBitQuantizer::new` — i.e. **no Hadamard
  rotation, no Lloyd-Max codebook** — the lowest-recall multi-bit variant,
  despite the machinery existing and being serde-persisted.

Decisions taken: **benchmark TurboQuant before formally dropping it** (hard
numbers via an actual `turbovec` head-to-head), and **do both recall fixes**.

This sub-project closes both recall gaps and produces the head-to-head, all
measured benchmark-first.

## Goal

Improve Veda's RaBitQ recall by (a) wiring rotation + optimized codebook into
the default Extended rerank and (b) implementing the per-vector unbiased
estimator in the 1-bit path; then produce a measured turbovec head-to-head that
settles the TurboQuant drop.

## Pieces (sequenced benchmark-first)

### B1 — Benchmark harness + baseline (measurement only)

Extend `crates/veda-index/tests/scale_benchmark.rs` to report, per index,
**recall@10, p50/p95 search latency (→ QPS), build time, and resident index
memory** for the RaBitQ family — `RaBitQ`, `RaBitQRerank`, `RaBitQExtRerank` —
at **d=1536 and d=3072**, n≈100k, with recall measured against `FlatIndex`
exact top-10 on the same query set. Deterministic seeded corpus (reuse the
existing `vec_of`/`make_corpus` helpers). Gated `#[ignore]` like the existing
large cases. Captures the BASELINE (today's un-wired default) — no behavior
change. This harness is reused to measure B2 and B3 before/after.

### B2 — Fix-2: wire rotation + Lloyd-Max codebook into `RaBitQExtRerankIndex`

In `RaBitQExtRerankIndex::new` (`rabitq_ext_rerank.rs`), build the rerank
quantizer as `ExtRaBitQuantizer::new(metric).with_rotation(seed).with_optimized_codebook()`
instead of the bare `ExtRaBitQuantizer::new(metric)`. Use a fixed deterministic
`seed` constant so the rotation reproduces on reload.
- **Safety:** `HadamardRotation` + `LloydMaxCodebook` both derive
  `Serialize/Deserialize` and are fields of the persisted `ExtRaBitData`; the
  rotation is seed-derived and "reproduces on every load." A serde round-trip
  test (build → serialize → deserialize → identical search results) guards
  reload correctness.
- **Verification:** re-run B1 → before/after recall@10 for `RaBitQExtRerank`.
  Expected: a recall improvement (the rotation makes the per-dim distribution
  stationary, which is what the single shared codebook assumes).

### B3 — Fix-1: per-vector bias correction in 1-bit `rabitq.rs`

Replace the global `mean_abs_component` scoring with the RaBitQ paper's
per-vector unbiased estimator. Store a per-vector factor (e.g.
`dot_factor[i] = Σ|residual_i| / ‖residual‖` — the per-vector `⟨ō,o⟩` for
sign-based codes; +4 bytes/vector) and use it in `estimate_distance` in place
of the global constant.
- **The exact estimator formula is derived against the code's actual
  binary-code convention** (the `asymmetric_dot` definition and the ±1 vs
  ±1/√D scaling in `rabitq.rs`) during implementation — NOT assumed. A unit
  test asserts the estimator is unbiased on a small hand-checkable example.
- **Verification + guard:** the B1 recall benchmark for bare `RaBitQIndex`
  before/after. A correct unbiased estimator should raise (or at least not
  lower) recall@10. **If it cannot beat the current global approximation on the
  benchmark, keep the existing code and record the negative result** — do not
  ship a change that regresses recall.

### B4 — turbovec head-to-head (the TurboQuant gate evidence)

A Python harness (under `crates/veda-index/benches/` or a `scripts/` location)
that:
- `pip install turbovec` into a throwaway venv on the host.
- Generates the SAME deterministic vectors used in B1 (write the B1 corpus +
  queries to a file from the Rust bench, or regenerate with an identical
  algorithm in Python — whichever is simpler to keep them bit-identical;
  decided at implementation).
- Runs turbovec (2-bit and 4-bit) → recall@10 (vs the same exact ground truth)
  + QPS + index memory at d=1536/3072.
- Emits a comparison table: turbovec {2-bit,4-bit} vs Veda's POST-FIX
  `RaBitQExtRerank` (and `RaBitQ`).
- Produces the numbers for the formal TurboQuant drop decision, recorded in a
  short results note appended to this design doc.

## Execution

Like sub-project A: this design doc + a detailed plan in `feat/rabitq-recall`,
then subagent-driven TDD with two-stage review. B2 and B3 are the code changes
(TDD + reviews); B1 and B4 are benchmark harnesses validated by running them and
capturing real output.

## Out of scope

- Implementing TurboQuant as a new IndexType (the whole point of B4 is to decide
  whether it's even worth it; the audit says no).
- PyO3 in-process binding, framework adapters, single-file export (later
  sub-projects C/D and the deferred PyO3).
- Changing the multi-bit `estimate_distance` "dequantize-and-dot" scheme (it has
  no per-vector bias defect; only B2's rotation/codebook wiring touches that
  path, via construction not scoring).

## Risks

| Risk | Mitigation |
|---|---|
| `turbovec` PyPI package won't build/install on the host | Fall back to comparing against turbovec's PUBLISHED numbers, clearly labeled as not-same-hardware; state the limitation. |
| B3 estimator derived wrong vs the ±1 code convention → recall regresses | The B1 recall benchmark is the gate; keep existing code + record negative result if it doesn't beat the global approximation. |
| B2 rotation/codebook breaks on reload | Serde round-trip test (build → ser → de → identical results) before relying on it; rotation is seed-deterministic + persisted. |
| Cross-tool recall comparison is apples-to-oranges (different ground truth) | Use the SAME exact (Flat) top-10 ground truth on the SAME vectors for both tools. |
| Bench wall-clock (100k × d=3072, multiple indexes) is slow | `#[ignore]` + run in release; document run time. |

---

## Results (2026-06-11)

All measurements on iitgpu11 (x86_64, AVX-512), release build, deterministic
pseudo-random corpus (`vec_of`), recall@10 vs `FlatIndex` exact top-10.

### B1 — Baseline (d=1536, n=100k, uniform-random)

| Index | recall@10 | build | p50 |
|---|---|---|---|
| rabitq (1-bit) | 0.331 | 11s | 336ms |
| rabitq+rerank (1-bit + raw-f32 rerank) | 0.749 | 18s | 357ms |
| rabitq+extrerank (1-bit + 4-bit rerank) | 0.734 | 42s* | 374ms |

\* after the codebook-training subsampling fix (was ~836s before).

### B2 — Rotation + Lloyd-Max codebook in the default ExtRerank → REVERTED

Wiring `with_rotation().with_optimized_codebook()` into the default
`RaBitQExtRerankIndex` **regressed** recall on BOTH distributions:

| Data | extrerank no-rotation | extrerank +rotation |
|---|---|---|
| isotropic uniform (d=1536) | 0.734 | 0.721 |
| anisotropic low-rank (d=1536, rank=48) | 0.870 | 0.860 |

Both extrerank variants deliver the ~6.6× memory win (960 vs 6336 bytes/vec)
— but that comes from the 4-bit quantization, NOT rotation. Rotation did not
help even on anisotropic data (the distribution it targets). **Reverted** the
default to plain 4-bit; rotation/codebook kept as the opt-in
`new_with_rotation()` for future real-embedding evaluation. The
codebook-training subsampling perf fix (836s→42s) was kept.

### B3 — Per-vector unbiased estimator for 1-bit RaBitQ → REVERTED

Replacing the global `mean_abs_component` with the paper's per-vector
estimator (`raw_dot·‖c‖₂²/‖c‖₁`, exact when query==vector) did NOT improve
recall: neutral at d=1536 (0.331→0.333, within noise) and a ~5pp regression
at d=128 Euclidean (0.60→0.556). For high-d embeddings `‖c‖₁/‖c‖₂≈√(2D/π)` is
nearly constant across vectors, so the per-vector correction ≈ the global mean.
**Reverted**; the global approximation is retained.

### B4 — turbovec head-to-head (d=1536, n=100k, same corpus)

turbovec v0.8.0 (`pip install turbovec`), `TurboQuantIndex(dim, bit_width)`,
brute-force quantized scan:

| Method | recall@10 | build | p50 |
|---|---|---|---|
| **turbovec 4-bit** | **0.848** | 136s | 516ms |
| turbovec 2-bit | 0.511 | 138s | 294ms |
| Veda rabitq+rerank | 0.749 | 18s | 357ms |
| Veda rabitq+extrerank | 0.734 | 42s | 374ms |
| Veda rabitq (1-bit) | 0.331 | 11s | 336ms |

**turbovec's 4-bit scalar quantization beats Veda's best RaBitQ config on
recall (0.848 vs 0.749, +0.10)** on this benchmark. Caveats: turbovec is
brute-force-only (no ANN graph — Veda's HNSW/DiskANN sub-ms paths are a
separate axis turbovec lacks); turbovec builds 3–12× slower; synthetic
uniform data is a weak proxy; and the comparison isn't apples-to-apples
(turbovec pure-4-bit vs Veda 1-bit+rerank).

## Conclusions

1. **The two audit-hypothesized RaBitQ recall fixes do not help** under
   measurement — both reverted. Veda's existing RaBitQ family (global-mean
   1-bit estimator, no rotation) is near its practical optimum for these
   configs. turbovec's marketed "length-renormalization / per-vector bias
   correction" is NOT the source of its recall edge (Veda already has length
   renorm; the per-vector estimator didn't help).
2. **turbovec's real edge is its 4-bit scalar quantization recall/bit
   tradeoff** (0.848), which Veda's RaBitQ family does not match (≤0.749).
   This is a genuine, measured gap — NOT explained by the audit's
   "already-present" hypothesis.

## TurboQuant decision: RECONSIDERED → build a 4-bit scalar-quant index

Given B4's measured recall advantage, the decision flips from "drop" to
**pursue a TurboQuant-style low-bit scalar-quantization index** as its own
follow-up sub-project (brainstorm → design → plan → implement). Open questions
for that effort: validate the recall gap on REAL embeddings (not synthetic);
determine whether turbovec's edge is the TQ+ per-coordinate calibration vs
Veda's single shared codebook; and ensure any new index integrates with
Veda's ANN-graph path rather than competing only in the brute-force niche.

## Kept from this sub-project (production-relevant)

- Codebook-training subsampling (`rabitq_ext.rs`): 836s→42s build with no
  recall change — a real optimization for opt-in codebook users.
- `RaBitQExtRerankIndex::new_with_rotation()`: opt-in rotation/codebook for
  future real-embedding evaluation (serde-persisted, round-trip tested).
- Benchmark infrastructure: `run_rabitq_family`, `make_anisotropic_corpus`,
  `analytical_bytes_per_vec`, `run_anisotropic_ab` in `scale_benchmark.rs`;
  `scripts/bench/turbovec_compare.py`.
