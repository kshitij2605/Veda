# turbovec Analysis — Changes & Outcomes

**Date:** 2026-06-11
**Trigger:** Competitive analysis of [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec), a Rust quantized vector index based on Google's TurboQuant, vs. Veda's vector-index layer.
**Status:** Complete. All resulting changes are merged to `main`.

This document summarizes what the turbovec analysis investigated, what shipped,
and — importantly — what was tried, measured, and deliberately reverted. The
honest negative results are as much a part of the outcome as the features.

---

## 1. What turbovec is, and the initial gap analysis

turbovec is a focused, in-process **quantized vector index library** (Rust core
+ Python bindings) implementing Google's TurboQuant scalar quantization with
random rotation. It is closer to a FAISS competitor than a Veda competitor — no
server, no replication, no multimodal pipeline; brute-force quantized scan only.

The initial audit identified candidate parity items (P1–P3). Rather than treat
them as one task, they were **decomposed into independent sub-projects** and
sequenced, gating the risky/uncertain ones behind cheap diagnostics. The items
that turned out to matter:

- **SIMD breadth** — turbovec ships NEON + AVX-512 kernels; Veda was AVX2-only.
- **Quantization recall** — turbovec markets "length-renormalization" and
  "per-vector bias correction"; the audit hypothesized matching these.
- **A 4-bit quantizer** — turbovec's headline is 16× compression at 2-bit / a
  competitive 4-bit mode.

---

## 2. SIMD acceleration (shipped)

**Finding:** Veda already had a `simd_simsimd.rs` module (SimSIMD-backed,
covering AVX-512 / ARM NEON / SVE) — but it was **dead code**, never dispatched
to; the hot path used AVX2-only. The build host even has AVX-512, so Veda was
leaving performance on the table.

**Change:** Per-function dispatch in `crates/veda-core/src/simd.rs` routes the
6 f32 distance ops + `i8_dot_product` to SimSIMD when the (now-default) `simsimd`
feature is on, with the AVX2/scalar path preserved as the `--no-default-features`
fallback. `manhattan_distance` stays on AVX2/scalar (SimSIMD has no f32 L1
kernel — routing it would infinite-loop). A zero-norm guard on the SimSIMD cosine
path preserves the documented `→ 1.0` contract on the rare branch only, keeping
the hot path to a single fused call.

**Result:** ~9–11× faster than the release-mode scalar baseline for dot product
at dims 768–3072 on the AVX-512 host. Verified by parity tests (SimSIMD ≈ scalar)
and a speedup benchmark. Merged to `main` independently.

---

## 3. RaBitQ recall investigation (two fixes tried, both reverted)

The audit hypothesized that turbovec's recall edge came from "length
renormalization" and "per-vector bias correction" missing in Veda's RaBitQ. A
benchmark-first sub-project tested both — **and both failed under measurement.**

### 3a. Rotation + Lloyd-Max codebook in the default ExtRerank → reverted

Veda's `RaBitQExtRerankIndex` shipped its 4-bit rerank quantizer *without* the
existing Hadamard rotation + Lloyd-Max codebook. Wiring them on **regressed**
recall on both isotropic (0.734→0.721) and anisotropic (0.870→0.860) synthetic
data — rotation did not help even on the anisotropic distribution it targets.
The ~6.6× memory win of the extended rerank comes from 4-bit quantization, not
rotation. **Reverted**; rotation/codebook kept as an opt-in `new_with_rotation()`
for future real-embedding evaluation.

### 3b. Per-vector unbiased estimator for 1-bit RaBitQ → reverted

Replacing the global `mean_abs_component` with the RaBitQ paper's per-vector
unbiased estimator (`raw_dot·‖c‖₂²/‖c‖₁`, exact when query==vector) did **not**
improve recall: neutral at d=1536 (0.331→0.333, noise) and a ~5pp regression at
d=128. For high-dim embeddings `‖c‖₁/‖c‖₂` is nearly constant across vectors, so
the per-vector correction ≈ the global mean. **Reverted**; the global
approximation retained.

### 3c. turbovec head-to-head — the real edge

A direct comparison (`pip install turbovec`, same vectors, recall@10 vs exact
top-10) revealed turbovec's *actual* advantage was neither marketed feature:

| Method | recall@10 (d=1536) | bytes/vec |
|---|---|---|
| turbovec 4-bit | 0.848 | ~768 |
| Veda rabitq+rerank | 0.749 | 6336 |
| Veda rabitq+extrerank | 0.734 | ~960 |
| Veda rabitq (1-bit) | 0.331 | ~192 |

turbovec simply uses **4-bit per-dimension scalar quantization**. That was the
gap — not calibration magic.

---

## 4. Scalar-quant diagnostic — collapsing the scope

Before committing to a research-grade TurboQuant reimplementation, a diagnostic
asked: is turbovec's edge *bits* or *calibration quality*? It benchmarked Veda's
existing per-dim 8-bit `ScalarQuantizer` (SQ8) and a 4-bit-per-dim prototype:

| Index | recall@10 (uniform) | recall@10 (aniso) | bytes/vec |
|---|---|---|---|
| Veda SQ8 (8-bit/dim, existing) | **0.995** | 0.996 | 1536 |
| 4-bit-per-dim prototype | 0.879 | 0.935 | 768 |
| turbovec 4-bit | 0.848 | — | ~768 |

**Conclusion:** turbovec's edge is purely "4 bits per dimension." Veda's existing
SQ8 already *dominates* turbovec on recall (0.995 vs 0.848) — it just wasn't
exposed as a standalone index and used 8 bits. A trivial 4-bit-per-dim quantizer
matches/beats turbovec at equal memory. The feared months-long reimplementation
collapsed into a focused index addition reusing existing calibration + packing.

---

## 5. Sq4 / Sq8 scalar-quant indexes (shipped)

**Change:**
- Generalized `crates/veda-index/src/quantization/scalar.rs::ScalarQuantizer` to
  a configurable bit width (4 or 8): per-dim min/max calibration (unchanged at
  8-bit), 4-bit codes packed 2 dims/byte, plus the index-supporting surface
  (`search`, `len`, `dimension`, `metric`, serde `save`/`load`).
- New `ScalarQuantIndex` (`crates/veda-index/src/quantization/scalar_index.rs`),
  a thin `VectorIndex` wrapper mirroring `RaBitQIndex`, with `sq4`/`sq8`
  constructors.
- New `IndexType::Sq4` and `IndexType::Sq8` variants; the REST collection-create
  parser accepts `"sq4"` / `"sq8"`.

**Result (recall@10):**

| Index | synthetic d=1536 | real BGE-M3 (n=13k, d=1024) | bytes/vec |
|---|---|---|---|
| `Sq4` (4-bit/dim) | 0.879 | 1.000 | 768 |
| `Sq8` (8-bit/dim) | 0.995 | 1.000 | 1536 |

`Sq4` **beats turbovec's 4-bit (0.848) at equal memory**, closing the only real
gap. `Sq8` is Veda's highest-recall quantized option, now reachable as an index.
Real-embedding validation extracted actual vectors from Veda's sealed-segment
`.vec` files and confirmed both at perfect recall on that corpus.

---

## 6. Net result on `main`

**Shipped:**
- SimSIMD-backed distance kernels (AVX-512/NEON/SVE), default-on.
- `IndexType::Sq4` (4-bit/dim) and `IndexType::Sq8` (8-bit/dim) standalone indexes.
- A codebook-training subsampling optimization (836s→42s build for opt-in
  codebook users) and the opt-in `RaBitQExtRerankIndex::new_with_rotation()`.
- Benchmark infrastructure: RaBitQ-family, anisotropic, scalar-quant diagnostic,
  turbovec head-to-head (`scripts/bench/turbovec_compare.py`), and real-embedding
  validation (`scripts/bench/real_embed_scalar_validation.py`).

**Deliberately NOT done (with evidence):**
- A from-scratch TurboQuant reimplementation — unnecessary; `Sq4`/`Sq8` close the
  gap by reusing existing code.
- The two marketed RaBitQ "improvements" — measured to not help, reverted.

**Documented follow-ups:**
- Graph-integrated scalar quant (an `Hnsw+Sq4` analog) for sub-ms latency at
  scale (`Sq4`/`Sq8` are brute-force scan, like turbovec).
- Runtime SIMD for the 4-bit scan; end-to-end gateway/schema + UX surfacing of
  the new index types.

**Design/plan/results docs** (on `main`, under `documentation/`):
- `2026-06-10-simd-acceleration-design.md` / `-plan.md`
- `2026-06-10-rabitq-recall-design.md` (incl. results) / `-plan.md`
- `2026-06-11-scalar-quant-index-design.md` (incl. results) / `-plan.md`

---

## 7. Process note

The defining feature of this run was **benchmark-gated, honest decision-making**:
two plausible audit hypotheses were implemented and then reverted because the
numbers said so; a cheap diagnostic turned a feared research-grade effort into a
small, low-risk addition; and a competitor's marketing claims were empirically
traced to their real (and modest) source. The negative results saved
substantially more effort than the features cost.
