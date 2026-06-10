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
