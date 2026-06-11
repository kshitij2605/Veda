# Design: Scalar-quant indexes — Sq4 (4-bit/dim) + expose Sq8

**Date:** 2026-06-11
**Status:** Approved (brainstorming)
**Sub-project:** Follow-on to the turbovec-parity work; the "build a TurboQuant-style
index" decision, re-scoped by the scalar-quant diagnostic.
**Branch / workspace:** continues on `feat/rabitq-recall`, worktree
`.claude/worktrees/rabitq-recall` (where the diagnostic prototypes live).

## Context

The B4 turbovec head-to-head showed turbovec's 4-bit quantizer beating Veda's
RaBitQ family on recall (0.848 vs 0.749). A follow-up scalar-quant diagnostic
(`scale_benchmark.rs::run_scalar_diag`, commit e709655) then pinpointed WHY and
collapsed the scope:

| Index | recall@10 (uniform d=1536) | recall@10 (aniso) | bytes/vec |
|---|---|---|---|
| Veda SQ8 (8-bit/dim, existing `ScalarQuantizer`) | **0.995** | 0.996 | 1536 |
| 4-bit-per-dim prototype | 0.879 | 0.935 | 768 |
| turbovec 4-bit | 0.848 | — | ~768 |
| Veda rabitq+rerank | 0.749 | 0.985 | 6336 |

**Findings:** (1) turbovec's edge is purely "4 bits per dimension" — no special
calibration; (2) Veda's existing SQ8 already *dominates* turbovec on recall
(0.995 vs 0.848) but is NOT exposed as a standalone index and uses 8 bits;
(3) a trivial 4-bit-per-dim scalar quantizer matches/beats turbovec (0.879 vs
0.848) at the same ~768 B/vec. So the feared research-grade reimplementation is
actually: **add a 4-bit-per-dim scalar-quant index, and expose the existing SQ8.**

## Goal

Give Veda two first-class scalar-quant indexes: **`Sq4`** (4-bit/dim, ~768 B/vec
at d=1536, turbovec-competitive recall) and **`Sq8`** (8-bit/dim, existing
`ScalarQuantizer`, ~0.99 recall) — a clean recall/memory dial, reusing existing
code (per-dim calibration + 4-bit packing primitives).

## Design

### Quantizer: generalize `ScalarQuantizer` to a configurable bit width

`crates/veda-index/src/quantization/scalar.rs::ScalarQuantizer` already does
per-dimension min/max calibration + `encode`/`decode`/`estimate_distance` at
8-bit (uint8/dim). Generalize it to support **`bits: u8` ∈ {4, 8}**:
- 8-bit: unchanged (one byte/dim) — preserves the existing SQ8 behavior exactly.
- 4-bit: 16 levels per dim, packed 2 dims/byte using the existing
  `pack_bits`/`unpack_bits` primitives in `rabitq_ext.rs` (extract them to a
  shared location if needed, or reuse in place). Per-dim min/max calibration
  identical to 8-bit; only the level count + packing differ.
- `estimate_distance` decodes to the per-bin reconstruction (bin midpoint) and
  computes the metric — same structure as today, parameterized by bits.

Keep the change minimal and backward-compatible: existing SQ8 callers (if any)
get the same behavior via the default/8-bit path. Serde data carries `bits`.

### Indexes: two standalone `VectorIndex` impls + `IndexType` variants

Add `IndexType::Sq4` and `IndexType::Sq8` to the enum in
`crates/veda-core/src/types.rs` (with doc comments stating the recall/memory
tradeoff per the diagnostic). Add standalone `VectorIndex` wrappers (new file
`crates/veda-index/src/quantization/scalar_index.rs`, or alongside `scalar.rs`)
following the `RaBitQIndex` pattern (`rabitq_vector_index.rs`):
- `build(vectors)` → train the `ScalarQuantizer` (at the index's bit width) +
  encode/store all codes.
- `search(query, k)` → brute-force scan `estimate_distance` over all codes,
  return top-k. (Matches turbovec's model; Veda's HNSW/DiskANN remain the
  separate ANN-graph path. Graph-integrated SQ is out of scope for v1.)
- Implement the trait's persistence (save/load) consistent with the other
  quantizer indexes.

`Sq8Index` wraps the 8-bit config (the diagnostic's `Sq8BenchIndex`, productionized);
`Sq4Index` wraps the 4-bit config (the diagnostic's `Sq4PerDimBenchIndex`,
productionized + using the shared `ScalarQuantizer` rather than a throwaway).

### Verification

- **Unit tests:** encode→decode round-trip within per-bin error bounds (4 and
  8 bit); per-dim calibration correctness (a dim with a known range quantizes as
  expected); `estimate_distance` ordering sanity.
- **Recall benchmark:** promote the diagnostic into a kept bench asserting
  `Sq4` recall ≥ ~0.84 (turbovec-competitive) at 768 B/vec and `Sq8` ≥ ~0.99 on
  the d=1536 corpus.
- **Real-embedding validation (early plan step):** attempt to obtain real
  BGE-M3 embeddings (extract from a Veda collection's stored vectors, or
  re-embed the rag-anything corpus via the embedding service) and confirm
  `Sq4`/`Sq8` recall holds on real data. **Skip gracefully** (record the
  limitation) if embeddings can't be extracted in reasonable effort — the
  synthetic uniform+anisotropic results are consistent and the technique is
  well-understood.

## Out of scope (v1)

- Graph-integrated scalar quant (an HNSW+SQ4 analog to `HnswRaBitQ`) — later.
- turbovec's TQ+ Hadamard rotation — measured (sub-project B) to NOT help; the
  per-dim min/max calibration alone matches turbovec.
- Runtime SIMD for the 4-bit scan (could follow; the existing `i8_dot`/SimSIMD
  work is separate).
- Wiring `Sq4`/`Sq8` into the gateway/collection schema as user-selectable index
  types end-to-end (this design covers the `veda-index` layer; gateway exposure
  is a follow-up if desired).

## Risks

| Risk | Mitigation |
|---|---|
| Generalizing `ScalarQuantizer` breaks existing SQ8 behavior | 8-bit path unchanged + round-trip tests; existing callers use the default. |
| 4-bit packing bugs (2 dims/byte, odd dims) | Reuse the tested `pack_bits`/`unpack_bits`; round-trip unit tests incl. odd dim. |
| Real embeddings not extractable | Skip gracefully; rely on synthetic uniform+aniso (consistent, well-understood technique). |
| Brute-force scan too slow at scale | v1 matches turbovec's brute-force model; graph integration is the documented follow-up for latency. |
| `IndexType` additions ripple into match arms across crates | Grep all `match` on `IndexType`; add arms (most will route to the new index builders or error clearly). |

---

## Results (2026-06-11)

All on iitgpu11 (x86_64, AVX-512), recall@10 vs FlatIndex exact top-10.

### Synthetic (d=1536, n=50k, uniform), measured via the kept `sq_recall_targets_1536` bench

| Index | recall@10 | bytes/vec | vs turbovec 4-bit (0.848) |
|---|---|---|---|
| **Sq8** (8-bit/dim) | **0.995** | 1536 | +0.147 |
| **Sq4** (4-bit/dim) | **0.879** | 768 | +0.031 (at equal memory) |

Asserted thresholds (sq8 ≥ 0.97, sq4 ≥ 0.82) pass with margin.

### Real BGE-M3 embeddings (T1; n=13,046, d=1024, extracted from sealed-segment `.vec` files)

| Index | recall@10 |
|---|---|
| Sq8 (8-bit/dim) | 1.000 |
| Sq4 (4-bit/dim) | 1.000 |

Both perfect at this corpus size; the synthetic d=1536 figures (0.995 / 0.879) are
the conservative production estimate (larger corpora tighten the rank-10/11 margin).

## Outcome

Veda now has two first-class scalar-quant indexes in `veda-index`:
- **`IndexType::Sq4`** — 4-bit/dim, ~D/2 bytes/vec, recall 0.879 (synthetic d=1536) —
  **beats turbovec's 4-bit (0.848) at the same memory**, closing the only real gap
  the turbovec head-to-head (sub-project B / B4) surfaced.
- **`IndexType::Sq8`** — 8-bit/dim, ~D bytes/vec, recall 0.995 — Veda's
  highest-recall quantized option, now reachable as a standalone index (it was
  previously an internal-only quantizer).

Both wrap a generalized `ScalarQuantizer` (per-dim min/max calibration, 4/8-bit
packing, serde persistence) via a thin `ScalarQuantIndex` (`VectorIndex`), and are
accepted by the REST collection-create parser as `"sq4"` / `"sq8"`.

The turbovec investigation's full arc: the marketed "length-renorm / per-vector
bias correction" was NOT turbovec's edge (sub-project B reverted both as
non-improving); the real edge was simply "4-bit per-dimension scalar quant," which
Veda now matches (and exceeds at 8-bit). No research-grade TurboQuant
reimplementation was needed.

## Follow-ups (out of scope here)

- Graph-integrated scalar quant (an `Hnsw+Sq4` analog to `HnswRaBitQ`) for sub-ms
  latency at scale (Sq4/Sq8 are brute-force scan, like turbovec).
- Runtime SIMD for the 4-bit scan.
- End-to-end gateway/schema surfacing + docs for the new index types (the REST
  parser accepts them; user-facing docs/UX is a separate task).
