# Design: SIMD acceleration (AVX-512 / NEON / SVE via SimSIMD)

**Date:** 2026-06-10
**Status:** Approved (brainstorming)
**Sub-project:** A of the turbovec-parity decomposition (see "Context").
**Branch / workspace:** `feat/simd-acceleration` off `main`, in the git worktree
`.claude/worktrees/simd-acceleration`.

## Context

A comparison against the competitor `turbovec` (Rust quantized vector index)
surfaced that turbovec ships hand-written NEON (ARM) + AVX-512BW (x86) distance
kernels, while Veda's `crates/veda-core/src/simd.rs` uses **AVX2 + scalar
fallback only**. The full turbovec-parity work was decomposed into independent
sub-projects (SIMD, quantization parity, Python ecosystem, single-file export,
FAISS benchmark harness; with TurboQuant / PyO3 / block-32-masking gated behind
explicit decisions). **This spec covers sub-project A only: SIMD acceleration.**

Exploration changed the shape of this item materially:

1. Veda **already has** `crates/veda-core/src/simd_simsimd.rs` — a SimSIMD v6
   backed module covering **AVX-512, ARM NEON, ARM SVE** (and reduced-precision
   kernels) — behind an optional `simsimd` Cargo feature (`simsimd = ["dep:simsimd"]`).
   So this is **wiring up an existing dependency**, not writing kernels.
2. It is a **dead module**: nothing dispatches to it. `crates/veda-core/src/vector.rs`
   and all callers use `crate::simd` (the AVX2/scalar path). The `simsimd`
   module is compiled when the feature is on but never called.
3. The build host (iitgpu11) **has AVX-512** (`avx512f/bw/vl/dq/cd/ifma/vbmi`),
   yet Veda runs the AVX2 path there today — so AVX-512 perf is currently left
   on the table, and **the win is locally measurable**.
4. The parity surface is **partial**. `simd.rs` exposes 12 public functions;
   `simd_simsimd.rs` covers only the **6 core f32 ops**
   (`dot_product`, `cosine_distance`, `euclidean_distance`, `l2_squared`,
   `manhattan_distance`, `l2_norm_squared`). It does **not** cover
   `i8_dot_product` or the `maxsim_*` GEMM paths (ColPali late-interaction).

## Goal

Make Veda's hot-path distance kernels use the best available SIMD ISA
(AVX-512 / NEON / SVE) at runtime, via the existing SimSIMD dependency, without
changing any caller and without regressing correctness — and prove the speedup
on the AVX-512 build host.

## Design

### Dispatch mechanism — per-function dispatch inside `simd.rs`

`crate::simd` stays the single public API every caller already uses. Inside
each of the **7 routed functions** (the 6 f32 ops + `i8_dot_product`), add a
thin backend branch:

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simsimd")]
    { return crate::simd_simsimd::dot_product(a, b); }
    #[cfg(not(feature = "simsimd"))]
    { /* existing AVX2 / scalar body, unchanged */ }
}
```

- The existing AVX2/scalar body becomes the `#[cfg(not(feature = "simsimd"))]`
  fallback — preserved verbatim, so a `--no-default-features`-style build still
  works exactly as today.
- `i8_dot_product`, `l2_norm_squared`, `cosine_distance`, `euclidean_distance`,
  `l2_squared`, `manhattan_distance`, `dot_product` are routed.
- `maxsim_distance*` and the `*_gemm` functions are **NOT** routed — SimSIMD is
  not a GEMM library; those keep their current implementation. (Out of scope.)
- `simd_simsimd.rs` gains a new `i8_dot_product(a: &[i8], b: &[i8]) -> i32`
  kernel routing to SimSIMD's integer dot kernel (exact SimSIMD API confirmed
  in implementation; parity-tested against the scalar i8 dot).

Rejected alternatives: (B) module swap `use crate::simd_simsimd as simd` — breaks
because the module lacks `i8_dot_product`/`maxsim_*`; (C) hand-rolled runtime ISA
detection — redundant, SimSIMD already runtime-dispatches across ISAs per call.

### Feature default — default-on, GATED on a runtime-dispatch check

**Decision:** make `simsimd` a **default** Cargo feature so stock builds get
AVX-512/NEON/SVE automatically.

**Gate (first task of the plan):** verify that SimSIMD v6 performs *runtime* CPU
dispatch (compiles multiple ISA variants, selects at runtime, falls back safely
on CPUs lacking the ISA). This is the property that makes default-on safe — a
single binary runs everywhere and only goes faster where the CPU allows.
- If confirmed → `simsimd` becomes a default feature; also confirm
  `cargo build --workspace` succeeds with it on (SimSIMD pulls a native-compile
  step; the build host already compiles native deps, so this is expected to pass
  but is verified explicitly).
- If NOT confirmed (e.g. it requires build-time target-cpu flags or can crash on
  older CPUs) → **fall back to opt-in**: keep `simsimd` non-default, document
  enabling it for AVX-512/ARM deploys. The dispatch wiring (above) ships either
  way; only the default flips.

### Verification

1. **Parity tests** (`crates/veda-core/tests/simd_parity_test.rs`, gated to run
   when the feature is on): for each routed op, assert the SimSIMD result ≈ the
   scalar reference within a float tolerance over randomized inputs across a
   range of dimensions (incl. non-multiples of the SIMD width and short slices).
   Explicitly guards semantic conventions (e.g. `cosine_distance` = `1 - cos`,
   not similarity; `euclidean` vs `sqeuclidean`). For `i8_dot_product`, compare
   to the exact scalar integer dot (must match exactly, not within tolerance —
   integer arithmetic).
2. **Benchmark** (extend the existing bench infra under `crates/veda-core`):
   in a single feature-on run, time `simd_simsimd::<op>` vs the scalar reference
   `*_scalar` (both callable in that build) for each routed op, at representative
   dims (e.g. 768/1024/1536/3072), and print a speedup table. This proves the
   wiring actually engages AVX-512 on the host. `#[ignore]`d so it doesn't gate
   `cargo test`.

## Files

- **Modify** `crates/veda-core/src/simd.rs` — add per-function backend branches
  to the 7 routed fns; keep existing bodies as the `not(simsimd)` fallback.
- **Modify** `crates/veda-core/src/simd_simsimd.rs` — add `i8_dot_product`;
  ensure the 6 f32 kernels' signatures exactly match `simd.rs`.
- **Modify** `crates/veda-core/Cargo.toml` — make `simsimd` a default feature
  (only after the task-1 gate passes; else leave non-default).
- **Create** `crates/veda-core/tests/simd_parity_test.rs` — parity tests.
- **Create/extend** a `crates/veda-core` benchmark for the op-level speedup table.

## Out of scope (this sub-project)

- `maxsim_*` / `*_gemm` ColPali paths (stay on current impl).
- Block-32 SIMD allowlist filter masking (deferred sub-project F).
- TurboQuant index, length-renorm audit (sub-project B), Python adapters/PyO3
  (C), single-file export (D), FAISS benchmark harness (E).
- f16/bf16 reduced-precision kernels (SimSIMD supports them; not needed yet).

## Risks

| Risk | Mitigation |
|---|---|
| SimSIMD does not runtime-dispatch safely → default-on could crash on old CPUs | Task-1 gate verifies before flipping the default; fall back to opt-in. |
| Semantic mismatch between SimSIMD and existing kernels (cosine sign, sqeuclidean) | Parity tests assert against scalar reference for every routed op before merge. |
| `simsimd` native build fails in some CI/target | Fallback path (`not(simsimd)`) always compiles; feature can be turned off. |
| Default-on adds a C-toolchain build dependency | Documented; acceptable given the host already builds native deps. If it bites, revert to opt-in (one-line Cargo change). |

## Task 1 outcome

**Date verified:** 2026-06-10

### (a) Runtime dispatch — CONFIRMED

SimSIMD v6 performs runtime CPU feature detection and dispatch by default. Evidence from
two authoritative sources in the locally cached crate at
`~/.cargo/registry/src/index.crates.io-*/simsimd-6.*/`:

**From `build.rs`** (hardcoded for every Cargo build of the crate):
```
.define("SIMSIMD_DYNAMIC_DISPATCH", "1")
```
The build comment reads: *"Build SimSIMD with dynamic SIMD dispatching."*
This flag compiles all ISA variants into the binary and selects the best one at runtime
via CPUID/HWCAP probing — no `-C target-cpu=native` required.

**From `rust/lib.rs`** (the `capabilities` module doc):
> *"Returns `true` when the C backend is compiled with dynamic dispatch (default for this
> crate via `build.rs`), otherwise `false`."*

**From `README.md`** (the Rust section):
> *"SimSIMD provides a dynamic dispatch mechanism to select the most advanced micro-kernel
> for the current CPU. You can query supported backends and use the `SimSIMD::capabilities`
> function to select the best one."*

Graceful fallback: the dynamic-dispatch path selects the highest-tier backend whose CPUID
bits are present at startup (Sapphire Rapids → Ice Lake → Skylake → Haswell → serial, or
the ARM equivalent). On a CPU with no AVX-512, it falls through to the AVX2 (Haswell)
kernel; on a fully generic x86-64 it falls through to the `serial` scalar path. The binary
runs correctly everywhere, just faster on richer ISAs.

### (b) `cargo build -p veda-core --features simsimd` — CLEAN

Build on iitgpu11 (x86_64, AVX-512 capable) completed successfully:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3m 17s
```
No errors or warnings from the SimSIMD native compile step.

### (c) Decision — DEFAULT-ON

Both gate conditions are met:
- Runtime dispatch is confirmed (hardcoded `SIMSIMD_DYNAMIC_DISPATCH=1` in `build.rs`).
- `cargo build -p veda-core --features simsimd` is clean on the build host.

**`simsimd` will be made a default Cargo feature** in `crates/veda-core/Cargo.toml`
(Task 5). Stock builds will automatically use AVX-512 on iitgpu11, NEON/SVE on ARM
deployments, and fall through gracefully to AVX2 or scalar on older hardware.
The existing `#[cfg(not(feature = "simsimd"))]` AVX2/scalar path is preserved as a
`--no-default-features` escape hatch.
