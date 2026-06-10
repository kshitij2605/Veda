# SIMD Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route Veda's hot-path distance kernels to the existing-but-dead SimSIMD backend so they use AVX-512 / NEON / SVE at runtime, without changing any caller, and prove the speedup on the AVX-512 build host.

**Architecture:** Per-function dispatch inside `crates/veda-core/src/simd.rs` — each of 7 routed ops (6 f32 + `i8_dot_product`) gets a `#[cfg(feature = "simsimd")]` branch calling `crate::simd_simsimd`, with today's AVX2/scalar body kept as the `#[cfg(not(...))]` fallback. `maxsim_*`/GEMM paths are untouched. `simsimd` becomes a default Cargo feature, gated on confirming SimSIMD does safe runtime ISA dispatch.

**Tech Stack:** Rust 2024, the `simsimd` v6 crate (already an optional dep of `veda-core`), cargo features.

**Workspace:** All work happens in the git worktree on the remote:
`iitgpu11.hon.otsuka-shokai.co.jp:/home/mac/test/project-mdb/.claude/worktrees/simd-acceleration`, branch `feat/simd-acceleration` (off `main`). The local Windows path is a stale mirror — edit on the remote (scp down / Edit / scp up, or Write+scp for new files); run `cargo` over SSH from the worktree dir.

**Build/test prelude** (every cargo command assumes these are set):
```bash
export PROTOC=/home/mac/mitrix69/load/.venv/lib/python3.12/site-packages/torch/bin/protoc
export PATH=/home/mac/.cargo/bin:$PATH
cd /home/mac/test/project-mdb/.claude/worktrees/simd-acceleration
```

Design reference: `documentation/2026-06-10-simd-acceleration-design.md`.

---

## File Structure

- **Modify** `crates/veda-core/src/simd.rs` — add per-function backend branches to the 7 routed fns; keep existing bodies as the `not(simsimd)` fallback; suppress dead-code on the now-unused fallback helpers when the feature is on.
- **Modify** `crates/veda-core/src/simd_simsimd.rs` — add an `i8_dot_product` kernel; confirm the 6 f32 kernel signatures exactly match `simd.rs`.
- **Modify** `crates/veda-core/Cargo.toml` — make `simsimd` a default feature (only after the Task 1 gate passes).
- **Create** `crates/veda-core/tests/simd_parity_test.rs` — parity tests (feature-gated) covering all 7 routed ops, asserting both the direct `simd_simsimd::*` kernels and the dispatched `simd::*` public API match a scalar reference.
- **Create** `crates/veda-core/benches/simd_speedup.rs` (or a `#[ignore]` timing test) — op-level speedup table, run on the AVX-512 host.
- **Modify** `CHANGELOG.md` — record the change.

---

## Task 1: Gate — confirm SimSIMD runtime ISA dispatch, decide default

**Files:** none changed (research + decision). Records the outcome in the design doc.

This decides whether `simsimd` can safely be a **default** feature. The property required: SimSIMD v6 compiles multiple ISA variants and selects the best at runtime, falling back safely on CPUs lacking the ISA (so one binary runs everywhere and only goes faster where supported).

- [ ] **Step 1: Research SimSIMD's dispatch model**

Use WebFetch on the SimSIMD crate docs and README:
- `https://docs.rs/simsimd/latest/simsimd/`
- `https://github.com/ashvardanian/SimSIMD`

Answer concretely: Does SimSIMD do runtime CPU feature detection and dispatch (vs requiring build-time `-C target-cpu=native`)? Does it fall back gracefully on CPUs without AVX-512/NEON? Quote the relevant doc text.

- [ ] **Step 2: Confirm the workspace builds with the feature on**

```bash
cargo build -p veda-core --features simsimd 2>&1 | tail -5
```
Expected: clean build (SimSIMD's native compile step succeeds on this host). If it fails, capture the error.

- [ ] **Step 3: Record the decision**

Append a short "## Task 1 outcome" note to `documentation/2026-06-10-simd-acceleration-design.md` stating: (a) whether runtime dispatch is confirmed, (b) whether the feature build is clean, (c) the resulting decision — **default-on** (if both yes) or **opt-in fallback** (if either no). This decision governs Task 5.

- [ ] **Step 4: Commit**

```bash
git add documentation/2026-06-10-simd-acceleration-design.md
git commit -m "docs(simd): record SimSIMD runtime-dispatch gate outcome"
```

---

## Task 2: Add the `i8_dot_product` SimSIMD kernel (TDD)

**Files:**
- Test: `crates/veda-core/tests/simd_parity_test.rs` (create)
- Modify: `crates/veda-core/src/simd_simsimd.rs`

`simd_simsimd.rs` currently covers 6 f32 ops but not `i8_dot_product` (used by the quantised-centroid / PLAID path). Add it, TDD-first.

- [ ] **Step 1: Write the failing parity test**

Create `crates/veda-core/tests/simd_parity_test.rs` with ONLY the i8 test for now. The file is gated to the feature so it compiles to an empty test binary when the feature is off.

```rust
//! Parity tests: SimSIMD kernels (and the dispatched `simd` API) must match
//! a scalar reference. Only meaningful when the `simsimd` feature is on.
#![cfg(feature = "simsimd")]

use veda_core::simd_simsimd;

fn i8_dot_ref(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
}

fn seeded_i8(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as i32 % 256 - 128) as i8
        })
        .collect()
}

#[test]
fn i8_dot_product_matches_scalar_exactly() {
    for &dim in &[1usize, 7, 16, 31, 128, 768, 1024, 1536] {
        let a = seeded_i8(dim as u32, dim);
        let b = seeded_i8((dim as u32).wrapping_add(99), dim);
        assert_eq!(
            simd_simsimd::i8_dot_product(&a, &b),
            i8_dot_ref(&a, &b),
            "i8 dot mismatch at dim={dim}"
        );
    }
}
```

- [ ] **Step 2: Run the test to verify it fails (does not compile)**

```bash
cargo test -p veda-core --features simsimd --test simd_parity_test 2>&1 | tail -15
```
Expected: FAIL — `cannot find function i8_dot_product in module simd_simsimd`.

- [ ] **Step 3: Implement `i8_dot_product` in `simd_simsimd.rs`**

Append to `crates/veda-core/src/simd_simsimd.rs`:

```rust
/// i8 × i8 → i32 dot product via SimSIMD's integer spatial kernel.
/// SimSIMD returns the dot as f64; for realistic dims the i8×i8 sum is
/// well within f64's exact-integer range, so the cast is lossless.
pub fn i8_dot_product(a: &[i8], b: &[i8]) -> i32 {
    // NOTE: confirm the exact SimSIMD i8 API against the crate docs.
    // Expected form: `<i8 as SpatialSimilarity>::dot(a, b)` returning
    // `Option<f64>`. The parity test enforces exact correctness.
    i8::dot(a, b).unwrap_or(0.0) as i32
}
```

If `i8::dot` is not the correct path, consult the SimSIMD docs fetched in Task 1 for the i8 dot entry point (it may be `SpatialSimilarity::dot` with `i8` type params, or an `inner`/`dot` method on an integer trait). The exact-match parity test is the arbiter.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test -p veda-core --features simsimd --test simd_parity_test 2>&1 | tail -10
```
Expected: PASS — `i8_dot_product_matches_scalar_exactly`.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-core/src/simd_simsimd.rs crates/veda-core/tests/simd_parity_test.rs
git commit -m "feat(veda-core): SimSIMD i8 dot kernel + parity test"
```

---

## Task 3: Wire per-function dispatch in `simd.rs`

**Files:**
- Modify: `crates/veda-core/src/simd.rs`

Add a `#[cfg(feature = "simsimd")]` branch to each of the 7 routed public fns, returning the SimSIMD kernel; keep the existing body as the `#[cfg(not(feature = "simsimd"))]` fallback. `cosine_distance` and `euclidean_distance` route **directly** to SimSIMD's fused kernels (faster than composing from primitives).

- [ ] **Step 1: Route the three primitive f32 ops + i8**

For `dot_product`, replace the body so it reads:

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::dot_product(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { avx2::dot_product_avx2(a, b) };
            }
        }
        dot_product_scalar(a, b)
    }
}
```

Apply the identical pattern to `l2_squared` (fallback calls `avx2::l2_squared_avx2` / `l2_squared_scalar`), `manhattan_distance` (`avx2::manhattan_avx2` / `manhattan_scalar`), `l2_norm_squared` (`avx2::l2_norm_squared_avx2` / `l2_norm_squared_scalar`), and `i8_dot_product` (simsimd branch returns `crate::simd_simsimd::i8_dot_product(a, b)`; fallback calls `avx2::i8_dot_avx2` / `i8_dot_scalar`). Wrap each existing fallback body in the `#[cfg(not(feature = "simsimd"))]` block exactly as shown.

- [ ] **Step 2: Route `cosine_distance` and `euclidean_distance` directly**

Replace `cosine_distance`'s body:

```rust
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::cosine_distance(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        let dot = dot_product(a, b);
        let norm_a = l2_norm_squared(a).sqrt();
        let norm_b = l2_norm_squared(b).sqrt();
        let denom = norm_a * norm_b;
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }
}
```

Replace `euclidean_distance`'s body:

```rust
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::euclidean_distance(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        l2_squared(a, b).sqrt()
    }
}
```

- [ ] **Step 3: Build with the feature on; suppress dead-code on now-unused fallbacks**

```bash
cargo build -p veda-core --features simsimd 2>&1 | tail -20
```
With the feature on, the `avx2` module and the `*_scalar` helpers are unreferenced. To keep the build warning-clean, add `#[cfg_attr(feature = "simsimd", allow(dead_code))]` to: the `mod avx2 { ... }` declaration, and each scalar helper fn (`dot_product_scalar`, `l2_squared_scalar`, `manhattan_scalar`, `l2_norm_squared_scalar`, `i8_dot_scalar`). Re-run the build.
Expected: clean build, **zero new warnings**.

- [ ] **Step 4: Verify the non-feature build still compiles**

```bash
cargo build -p veda-core --no-default-features 2>&1 | tail -10
```
Expected: clean (the `not(simsimd)` fallback path is intact and self-sufficient). If `veda-core` has other default features required to build, use `--no-default-features --features <those>` minus `simsimd`; confirm the fallback compiles without `simsimd`.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-core/src/simd.rs
git commit -m "feat(veda-core): dispatch distance kernels to SimSIMD when enabled"
```

---

## Task 4: Parity tests for the 6 f32 ops + dispatched API

**Files:**
- Modify: `crates/veda-core/tests/simd_parity_test.rs`

Extend the parity test to cover the 6 f32 ops, asserting BOTH the direct `simd_simsimd::*` kernel AND the dispatched `simd::*` public API (which, with the feature on, routes to SimSIMD) match a scalar reference within tolerance. Testing `simd::*` verifies the Task 3 wiring end-to-end.

- [ ] **Step 1: Add the f32 parity tests**

Append to `crates/veda-core/tests/simd_parity_test.rs`:

```rust
use veda_core::simd;

fn seeded_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn l2sq_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn l1_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}
fn norm2_ref(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}
fn cosine_ref(a: &[f32], b: &[f32]) -> f32 {
    let d = dot_ref(a, b);
    let na = norm2_ref(a).sqrt();
    let nb = norm2_ref(b).sqrt();
    if na * nb == 0.0 { 1.0 } else { 1.0 - d / (na * nb) }
}

const DIMS: &[usize] = &[1, 7, 16, 31, 128, 768, 1024, 1536, 3072];
// Relative tolerance: SimSIMD may use fused/reordered ops vs naive scalar.
const TOL: f32 = 1e-3;

fn close(got: f32, want: f32, ctx: &str) {
    let denom = want.abs().max(1.0);
    assert!(
        ((got - want).abs() / denom) < TOL,
        "{ctx}: got {got}, want {want}"
    );
}

#[test]
fn f32_kernels_match_scalar_reference() {
    for &dim in DIMS {
        let a = seeded_f32(dim as u32, dim);
        let b = seeded_f32((dim as u32).wrapping_add(99), dim);

        // Direct SimSIMD kernels.
        close(simd_simsimd::dot_product(&a, &b), dot_ref(&a, &b), "ss dot");
        close(simd_simsimd::l2_squared(&a, &b), l2sq_ref(&a, &b), "ss l2sq");
        close(simd_simsimd::manhattan_distance(&a, &b), l1_ref(&a, &b), "ss l1");
        close(simd_simsimd::l2_norm_squared(&a), norm2_ref(&a), "ss norm2");
        close(simd_simsimd::cosine_distance(&a, &b), cosine_ref(&a, &b), "ss cos");
        close(simd_simsimd::euclidean_distance(&a, &b), l2sq_ref(&a, &b).sqrt(), "ss euc");

        // Dispatched public API (verifies Task 3 wiring).
        close(simd::dot_product(&a, &b), dot_ref(&a, &b), "api dot");
        close(simd::l2_squared(&a, &b), l2sq_ref(&a, &b), "api l2sq");
        close(simd::manhattan_distance(&a, &b), l1_ref(&a, &b), "api l1");
        close(simd::l2_norm_squared(&a), norm2_ref(&a), "api norm2");
        close(simd::cosine_distance(&a, &b), cosine_ref(&a, &b), "api cos");
        close(simd::euclidean_distance(&a, &b), l2sq_ref(&a, &b).sqrt(), "api euc");
    }
}
```

- [ ] **Step 2: Run the parity tests**

```bash
cargo test -p veda-core --features simsimd --test simd_parity_test 2>&1 | tail -15
```
Expected: PASS — both `i8_dot_product_matches_scalar_exactly` and `f32_kernels_match_scalar_reference`. If `cosine` fails, the SimSIMD `cosine` convention differs (similarity vs distance) — fix the routing in `simd_simsimd::cosine_distance` so it returns `1 - similarity`, then re-run.

- [ ] **Step 3: Commit**

```bash
git add crates/veda-core/tests/simd_parity_test.rs
git commit -m "test(veda-core): SimSIMD f32 parity incl. dispatched API"
```

---

## Task 5: Make `simsimd` a default feature (gated on Task 1)

**Files:**
- Modify: `crates/veda-core/Cargo.toml`

**Only do this task if Task 1 concluded default-on.** If Task 1 fell back to opt-in, SKIP this task and note in the commit log / CHANGELOG that the backend is opt-in via `--features simsimd`.

- [ ] **Step 1: Add `simsimd` to the default features**

In `crates/veda-core/Cargo.toml`, the `[features]` table currently has `simsimd = ["dep:simsimd"]` and (likely) a `default = [...]` line. Add `"simsimd"` to the `default` array. If no `default` key exists, add `default = ["simsimd"]`.

- [ ] **Step 2: Verify the workspace builds + tests green with simsimd on by default**

```bash
cargo build --workspace 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | grep -E "test result:|error" | tail -30
```
Expected: clean build; all lib tests pass (the parity test now runs by default since the feature is on by default — confirm `simd_parity_test` appears and passes).

- [ ] **Step 3: Commit**

```bash
git add crates/veda-core/Cargo.toml
git commit -m "feat(veda-core): enable SimSIMD backend by default"
```

---

## Task 6: Op-level speedup benchmark on the AVX-512 host

**Files:**
- Create: `crates/veda-core/tests/simd_speedup_bench.rs`

Prove the wiring engages AVX-512 by timing SimSIMD kernels vs the scalar reference at representative dims, on this host. Implemented as an `#[ignore]`d test (no extra bench harness/deps) so it doesn't gate `cargo test`.

- [ ] **Step 1: Write the benchmark**

Create `crates/veda-core/tests/simd_speedup_bench.rs`:

```rust
//! Op-level SIMD speedup, run on demand:
//!   cargo test -p veda-core --features simsimd --test simd_speedup_bench -- --ignored --nocapture
#![cfg(feature = "simsimd")]

use std::time::Instant;
use veda_core::simd_simsimd;

fn seeded(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345); ((s >> 16) as f32 / 32768.0) - 1.0 }).collect()
}
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

#[test]
#[ignore]
fn dot_product_speedup() {
    let iters = 200_000;
    for &dim in &[768usize, 1024, 1536, 3072] {
        let a = seeded(1, dim);
        let b = seeded(2, dim);

        let t = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters { acc += dot_scalar(&a, &b); }
        let scalar_ns = t.elapsed().as_nanos() as f64 / iters as f64;
        std::hint::black_box(acc);

        let t = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters { acc += simd_simsimd::dot_product(&a, &b); }
        let ss_ns = t.elapsed().as_nanos() as f64 / iters as f64;
        std::hint::black_box(acc);

        eprintln!("dim={dim:>5}  scalar={scalar_ns:>8.1}ns  simsimd={ss_ns:>8.1}ns  speedup={:.2}x", scalar_ns / ss_ns);
    }
}
```

- [ ] **Step 2: Run it and capture numbers**

```bash
cargo test -p veda-core --features simsimd --test simd_speedup_bench -- --ignored --nocapture 2>&1 | grep -E "dim=|speedup"
```
Expected: prints a speedup table; SimSIMD should be meaningfully faster than scalar at dim ≥ 768 on this AVX-512 host (a positive multiplier). Record the numbers for the CHANGELOG/commit message.

- [ ] **Step 3: Commit**

```bash
git add crates/veda-core/tests/simd_speedup_bench.rs
git commit -m "test(veda-core): op-level SIMD speedup benchmark"
```

---

## Task 7: CHANGELOG + final workspace verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a CHANGELOG entry**

Under the most recent version heading in `CHANGELOG.md`, add (in the appropriate `### Added` / `### Changed` block; create one if needed):

```markdown
- **SIMD acceleration via SimSIMD** — distance kernels (dot, cosine, euclidean, L2, L1, L2-norm, i8 dot) now dispatch to SimSIMD, using AVX-512 / ARM NEON / SVE at runtime instead of AVX2-only. Enabled by default; falls back to the AVX2/scalar path when built `--no-default-features`. Measured <RECORD speedup from Task 6> over scalar at dim≥768 on AVX-512.
```
Replace `<RECORD speedup from Task 6>` with the actual multiplier observed.

- [ ] **Step 2: Final full workspace test**

```bash
cargo test --workspace --lib 2>&1 | grep -E "test result:|error|FAILED" | tail -30
```
Expected: all green, 0 failures.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for SIMD acceleration"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** dispatch mechanism → Task 3; 6 f32 ops + i8 scope → Tasks 2-4; default-on gated → Tasks 1, 5; parity tests → Tasks 2, 4; benchmark on AVX-512 host → Task 6. `maxsim_*`/GEMM untouched (never modified in any task). All spec requirements mapped.
- **Type consistency:** `simd_simsimd::i8_dot_product(&[i8], &[i8]) -> i32` defined in Task 2, called from `simd::i8_dot_product` in Task 3 and tested in Tasks 2/4. The 6 f32 kernels keep their existing `&[f32] -> f32` signatures; `cosine_distance`/`euclidean_distance` route directly (not composed) per Task 3 step 2.
- **The one flagged uncertainty:** the exact SimSIMD i8 API (Task 2 step 3) — the exact-match parity test is the enforcement gate; confirm against the docs fetched in Task 1.
- **Cosine convention:** SimSIMD `cosine` returns a distance (`1 - similarity`) per the existing module; the parity test's `cosine_ref` asserts this. If SimSIMD returns similarity instead, fix `simd_simsimd::cosine_distance` to `1 - x`.
