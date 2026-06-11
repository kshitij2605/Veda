# RaBitQ Recall + TurboQuant Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two RaBitQ recall gaps (wire rotation + Lloyd-Max codebook into the default Extended rerank; replace the 1-bit global bias factor with the per-vector unbiased estimator) and produce a measured turbovec head-to-head that settles the TurboQuant drop.

**Architecture:** Benchmark-first. A bench harness (B1) measures the RaBitQ family before/after each fix; B2 and B3 are the two code changes (TDD + reviews); B4 is a Python turbovec head-to-head on the same vectors. All recall measured vs `FlatIndex` exact top-10.

**Tech Stack:** Rust 2024, `veda-index` crate, the `simsimd`-accelerated `veda-core` (now default), Python + the `turbovec` PyPI package for B4.

**Workspace:** git worktree on the remote: `gpu-host-a.internal.example:/home/mac/test/project-mdb/.claude/worktrees/rabitq-recall`, branch `feat/rabitq-recall` (off `main`). Edit on the remote (scp down / Edit / scp up, or Write+scp for new files); run `cargo` over SSH from the worktree dir.

**Build/test prelude** (every cargo command assumes these):
```bash
export PROTOC=/home/mac/mitrix69/load/.venv/lib/python3.12/site-packages/torch/bin/protoc
export PATH=/home/mac/.cargo/bin:$PATH
cd /home/mac/test/project-mdb/.claude/worktrees/rabitq-recall
```

Design reference: `documentation/2026-06-10-rabitq-recall-design.md`.

---

## File Structure

- **Modify** `crates/veda-index/tests/scale_benchmark.rs` — add a RaBitQ-family recall+latency+memory bench at d=1536/3072 (B1).
- **Modify** `crates/veda-index/src/quantization/rabitq_ext_rerank.rs` — wire rotation + codebook into the default rerank quantizer (B2).
- **Modify** `crates/veda-index/src/quantization/rabitq.rs` — per-vector unbiased estimator (B3).
- **Create** `scripts/bench/turbovec_compare.py` + a small Rust corpus-dump helper or shared generator — turbovec head-to-head (B4).
- **Modify** `documentation/2026-06-10-rabitq-recall-design.md` — append measured results (B1 baseline, B2/B3 deltas, B4 table, TurboQuant decision).

Each task leaves the workspace green and is committed independently.

---

## Task B1: RaBitQ-family recall/latency/memory benchmark

**Files:**
- Modify: `crates/veda-index/tests/scale_benchmark.rs`

Read the file first — it has `vec_of(seed, dim)`, `make_corpus(n, dim)`, `percentile`, `recall_at_k`, `run_scale`, `print_row`, and builds `FlatIndex`, `HnswIndex`, `RaBitQIndex`, `RaBitQRerankIndex`, `QuantizedHnswIndex`. You will add a focused RaBitQ-family bench that also reports recall (it already does), and ADD `RaBitQExtRerankIndex`, plus a resident-memory estimate per index.

- [ ] **Step 1: Add a RaBitQ-family scale function**

Append a new function and tests to `scale_benchmark.rs`. Reuse the existing helpers (`make_corpus`, `percentile`, `recall_at_k`, `print_row` pattern). Add imports at the top if missing: `use veda_index::quantization::rabitq_ext_rerank::RaBitQExtRerankIndex;` (confirm the exact module path via `cat -n` of `crates/veda-index/src/lib.rs` / `quantization/mod.rs`).

```rust
fn run_rabitq_family(label: &str, n: usize, dim: usize) {
    eprintln!();
    eprintln!("=== rabitq_family: {label}  n={n}  dim={dim} ===");
    let corpus = make_corpus(n, dim);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();

    let mut truth_idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth_idx.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES).map(|i| &corpus[(i * 7919) % n]).collect();
    let truths: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| truth_idx.search(q, K).unwrap().into_iter().map(|h| h.id).collect())
        .collect();

    // bytes/vector estimate for each index, printed alongside recall.
    let raw_bytes = (n * dim * 4) as f64 / (1024.0 * 1024.0); // f32 baseline, MB
    eprintln!("  raw f32 corpus: {raw_bytes:.1} MB");

    print_row("flat (truth)", n, 0.0, &truth_idx, &queries, &truths);

    let t0 = Instant::now();
    let mut rabitq = RaBitQIndex::new(DistanceMetric::Cosine);
    rabitq.build(&refs).unwrap();
    print_row("rabitq", n, t0.elapsed().as_secs_f64() * 1000.0, &rabitq, &queries, &truths);

    let t0 = Instant::now();
    let mut rerank = RaBitQRerankIndex::new(DistanceMetric::Cosine);
    rerank.build(&refs).unwrap();
    print_row("rabitq+rerank", n, t0.elapsed().as_secs_f64() * 1000.0, &rerank, &queries, &truths);

    let t0 = Instant::now();
    let mut ext = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
    ext.build(&refs).unwrap();
    print_row("rabitq+extrerank", n, t0.elapsed().as_secs_f64() * 1000.0, &ext, &queries, &truths);
}

#[test]
#[ignore]
fn rabitq_family_1536() {
    run_rabitq_family("d1536", 100_000, 1536);
}

#[test]
#[ignore]
fn rabitq_family_3072() {
    run_rabitq_family("d3072", 100_000, 3072);
}
```
If `RaBitQExtRerankIndex` does not implement `VectorIndex` with `new(metric)` + `build`/`search` exactly like the others (it does, per the source), adapt the call. If `print_row`'s generic bound doesn't accept `&RaBitQExtRerankIndex`, pass it the same way as `&rerank` (all implement `VectorIndex`).

- [ ] **Step 2: Build the test (compile only, fast)**

```bash
cargo test -p veda-index --test scale_benchmark --no-run 2>&1 | tail -15
```
Expected: compiles. Fix any import/path errors (e.g. the exact `RaBitQExtRerankIndex` module path).

- [ ] **Step 3: Run the baseline and capture numbers**

```bash
cargo test --release -p veda-index --test scale_benchmark -- --ignored --nocapture rabitq_family_1536 2>&1 | grep -E "rabitq_family|flat|rabitq|MB|recall"
```
(Optionally also `rabitq_family_3072` — it is slower; run if time permits.) Record the verbatim table: recall@10 + p50/p95 for `rabitq`, `rabitq+rerank`, `rabitq+extrerank` at d=1536. This is the BASELINE (pre-B2/B3).

- [ ] **Step 4: Commit**

```bash
git add crates/veda-index/tests/scale_benchmark.rs
git commit -m "test(veda-index): RaBitQ-family recall benchmark at d=1536/3072"
```

---

## Task B2: Wire rotation + Lloyd-Max codebook into RaBitQExtRerankIndex (TDD)

**Files:**
- Modify: `crates/veda-index/src/quantization/rabitq_ext_rerank.rs`

`RaBitQExtRerankIndex::new` currently builds `rerank: ExtRaBitQuantizer::new(metric)` (4-bit, no rotation, no codebook). Change it to enable both via the existing builders (`with_rotation(seed: u64)`, `with_optimized_codebook(self)`), with a fixed seed so reload is deterministic. Guard reload with a serde round-trip test.

- [ ] **Step 1: Write the failing serde round-trip test**

Read the file first to see existing tests + how the index serializes (look for `save`/`load`/`to_bytes`/`from_bytes` or serde on a `*Data` struct). The index persists via the `VectorIndex` trait's save/load (check `crate::traits::VectorIndex` for the exact method names — likely `save(&self, path)` / `load(path)` or `serialize`/`deserialize`). Add a test that builds the index, persists+reloads it, and asserts identical search results. Append to the file's `#[cfg(test)] mod tests`:

```rust
#[test]
fn extrerank_survives_roundtrip_with_rotation_and_codebook() {
    use veda_core::types::DistanceMetric;
    let dim = 64usize;
    let corpus: Vec<Vec<f32>> = (0..500u32)
        .map(|s| {
            let mut x = s.wrapping_mul(2654435761);
            (0..dim).map(|_| { x = x.wrapping_mul(1103515245).wrapping_add(12345); ((x >> 16) as f32 / 32768.0) - 1.0 }).collect()
        })
        .collect();
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    let mut idx = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
    idx.build(&refs).unwrap();
    let q = &corpus[7];
    let before = idx.search(q, 10).unwrap();

    // Persist + reload via the VectorIndex trait's save/load mechanism.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ext.idx");
    idx.save(&path).unwrap();
    let reloaded = RaBitQExtRerankIndex::load(&path).unwrap();
    let after = reloaded.search(q, 10).unwrap();

    let bi: Vec<u32> = before.iter().map(|r| r.id).collect();
    let ai: Vec<u32> = after.iter().map(|r| r.id).collect();
    assert_eq!(bi, ai, "rotation+codebook index must reload identically");
}
```
IMPORTANT: the `save`/`load` signatures above are a guess — replace them with the ACTUAL persistence API of `RaBitQExtRerankIndex` / the `VectorIndex` trait (read `crate::traits` and the existing index tests to find it; e.g. it may be `to_bytes()/from_bytes()`, or there may be no per-index save and persistence is via the quantizer's serde). If the index has no save/load, instead serialize the inner `rerank` quantizer (`ExtRaBitQuantizer` has `ExtRaBitData` serde) round-trip and assert its `estimate_distance` matches. Adapt the test to the real API; the GOAL is: prove a rotation+codebook-enabled quantizer reloads to identical results.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test -p veda-index --test <or lib> extrerank_survives_roundtrip 2>&1 | tail -15
```
(Determine whether the test is a lib unit test or needs a test file based on where you added it.) Expected: FAIL — currently the default has no rotation/codebook, so either the test asserts a property not yet enabled, or (if you wrote it to require rotation) it fails because rotation is absent. If it passes trivially because reload already works without rotation, that's fine — the real verification is Step 4's recall delta; in that case keep this as a regression guard.

- [ ] **Step 3: Enable rotation + codebook in `new`**

Add a seed constant near the top of `rabitq_ext_rerank.rs` (after `DEFAULT_RERANK_MULTIPLIER`):
```rust
/// Fixed seed for the Hadamard rotation so the index reloads deterministically.
const ROTATION_SEED: u64 = 0x5EED_1B17_4A11_0CD5;
```
Change the `rerank` field initializer in `RaBitQExtRerankIndex::new` from:
```rust
            rerank: ExtRaBitQuantizer::new(metric),
```
to:
```rust
            rerank: ExtRaBitQuantizer::new(metric)
                .with_rotation(ROTATION_SEED)
                .with_optimized_codebook(),
```

- [ ] **Step 4: Run the round-trip test + measure recall delta**

```bash
cargo test -p veda-index extrerank_survives_roundtrip 2>&1 | tail -10
```
Expected: PASS. Then re-run the B1 bench and compare `rabitq+extrerank` recall@10 to the B1 baseline:
```bash
cargo test --release -p veda-index --test scale_benchmark -- --ignored --nocapture rabitq_family_1536 2>&1 | grep -E "extrerank|recall"
```
Expected: recall@10 for `rabitq+extrerank` is >= the baseline (rotation+codebook should help or be neutral). Record before/after. If recall DROPS materially, investigate (wrong builder order / codebook not trained) before committing.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-index/src/quantization/rabitq_ext_rerank.rs
git commit -m "feat(veda-index): enable rotation + Lloyd-Max codebook in default ExtRerank"
```

---

## Task B3: Per-vector unbiased estimator in 1-bit RaBitQ (TDD)

**Files:**
- Modify: `crates/veda-index/src/quantization/rabitq.rs`

Replace the global `mean_abs_component` scoring with the RaBitQ per-vector unbiased estimator. Derivation (sign-based codes, `asymmetric_dot` returns `raw_dot = Σ q'ᵢ·signᵢ`):
the quantized unit vector is `ō = s/√D`; the estimator `⟨q',o⟩ ≈ ⟨q',ō⟩/⟨ō,o⟩` with `⟨ō,o⟩ = (1/√D)·Σ|cᵢ|/‖c‖₂` yields, after the √D cancels and scaling by `‖c‖₂`:

**`estimated_dot = raw_dot · ‖c‖₂² / ‖c‖₁`**, where `c` = centered residual, `‖c‖₂` = stored `norm`, `‖c‖₁` = `Σ|cᵢ|` (new per-vector field). This is exact when `q' == c` (verifiable). Store `l1_norms[i] = Σ|cᵢ|` per vector (+4 bytes/vector); remove the global `mean_abs_component`.

- [ ] **Step 1: Write the failing unbiasedness unit test**

Append to `rabitq.rs`'s `#[cfg(test)] mod tests` (read the file end to see the existing test module + helpers):

```rust
#[test]
fn estimator_is_exact_when_query_equals_vector() {
    // For q' == centered db vector c, the unbiased estimator must recover
    // ⟨c,c⟩ = ‖c‖² exactly (the global-mean approximation does not).
    let dim = 32usize;
    // Two vectors so the centroid isn't either of them; use a known one.
    let a: Vec<f32> = (0..dim).map(|i| ((i as f32) - 16.0) * 0.1).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((i as f32) - 8.0) * 0.05).collect();
    let mut q = RaBitQuantizer::new(DistanceMetric::DotProduct);
    q.train(&[a.as_slice(), b.as_slice()]);
    // The query equal to vector `a`; estimate_distance for DotProduct returns
    // -estimated_dot, and estimated_dot should ≈ ⟨a-centroid, a-centroid⟩.
    let centered_a_norm_sq: f32 = a.iter().zip(q.centroid_for_test())
        .map(|(x, c)| (x - c) * (x - c)).sum();
    let est_dot = -q.estimate_distance(&a, 0);
    let rel_err = (est_dot - centered_a_norm_sq).abs() / centered_a_norm_sq.max(1e-6);
    assert!(rel_err < 0.05, "unbiased estimator should ~recover ‖c‖²: est={est_dot}, want={centered_a_norm_sq}, rel_err={rel_err}");
}
```
This test needs read access to the centroid. If there's no accessor, add a `#[cfg(test)] pub(crate) fn centroid_for_test(&self) -> &[f32] { &self.centroid }` to the impl. (Adjust if a suitable accessor already exists.)

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test -p veda-index rabitq::tests::estimator_is_exact 2>&1 | tail -15
```
Expected: FAIL — the current global `mean_abs_component` estimator is biased per-vector, so `est_dot` won't recover `‖c‖²` within 5% in general (or `centroid_for_test` doesn't exist → compile error first; add it).

- [ ] **Step 3: Implement the per-vector estimator**

In `rabitq.rs`:
1. Struct + `RaBitQuantizerData`: replace `mean_abs_component: f32` with `l1_norms: Vec<f32>` (in BOTH the struct and the serde `RaBitQuantizerData`). Update `new()` to init `l1_norms: Vec::new()`.
2. In `train`, while iterating vectors, push `l1_norms.push(centered.iter().map(|x| x.abs()).sum())` alongside `norms.push(norm)`. Remove the `total_abs`/`total_components`/`mean_abs_component` computation block.
3. In `estimate_distance`, replace:
```rust
        let estimated_dot = v_norm * self.mean_abs_component * raw_dot;
```
with:
```rust
        let l1 = self.l1_norms[code_idx];
        let estimated_dot = if l1 > 1e-10 {
            raw_dot * (v_norm * v_norm) / l1
        } else {
            0.0
        };
```
4. Fix the serde struct conversions (the `From`/`into`/`save`/`load` paths that build `RaBitQuantizerData` from the quantizer and back) to carry `l1_norms` instead of `mean_abs_component`. Grep the file for `mean_abs_component` and replace every occurrence; ensure none remain.

- [ ] **Step 4: Run the unit test + full rabitq tests**

```bash
cargo test -p veda-index rabitq 2>&1 | grep -E "test result:|FAILED|estimator_is_exact" | tail -20
```
Expected: PASS, including `estimator_is_exact_when_query_equals_vector` and all pre-existing `rabitq` tests (any test that asserted old `mean_abs_component` behavior must be updated to the new estimator — fix those, don't delete coverage).

- [ ] **Step 5: Measure recall delta on bare RaBitQ**

```bash
cargo test --release -p veda-index --test scale_benchmark -- --ignored --nocapture rabitq_family_1536 2>&1 | grep -E "rabitq |recall"
```
Compare `rabitq` recall@10 to the B1 baseline. Expected: >= baseline (the unbiased estimator should improve or match candidate ordering). **If it regresses, revert the estimator change, keep the existing global approximation, and record the negative result in the design doc** — do not ship a recall regression.

- [ ] **Step 6: Commit**

```bash
git add crates/veda-index/src/quantization/rabitq.rs
git commit -m "feat(veda-index): per-vector unbiased estimator for 1-bit RaBitQ"
```

---

## Task B4: turbovec head-to-head

**Files:**
- Create: `scripts/bench/turbovec_compare.py`

A Python harness comparing turbovec vs Veda's post-fix RaBitQ on the SAME vectors + exact ground truth. Veda's numbers come from the B1 bench (post-B2/B3); turbovec's from this script. To keep vectors comparable, regenerate an identical deterministic corpus in Python (mirror `vec_of`: `s = seed*2654435761; loop s = s*1103515245+12345; val = ((s>>16) as f32/32768.0)-1.0`).

- [ ] **Step 1: Write the harness**

Create `scripts/bench/turbovec_compare.py`:

```python
"""turbovec vs Veda RaBitQ head-to-head. Run on the build host.
  python3 -m venv /tmp/tvenv && /tmp/tvenv/bin/pip install turbovec numpy
  /tmp/tvenv/bin/python scripts/bench/turbovec_compare.py
Mirrors veda-index/tests/scale_benchmark.rs::vec_of for identical vectors.
"""
import numpy as np, time

def vec_of(seed, dim):
    s = (seed * 2654435761) & 0xFFFFFFFF
    out = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = ((s >> 16) / 32768.0) - 1.0
    return out

def corpus(n, dim):
    return np.stack([vec_of(i, dim) for i in range(n)])

def exact_topk(mat, q, k):
    # cosine: normalize, dot, top-k
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    qn = q / (np.linalg.norm(q) + 1e-9)
    sims = mn @ qn
    return set(np.argsort(-sims)[:k].tolist())

def main():
    from turbovec import TurboQuantIndex
    N, K, NQ = 100_000, 10, 100
    for dim in (1536, 3072):
        mat = corpus(N, dim)
        queries = [mat[(i * 7919) % N] for i in range(NQ)]
        truths = [exact_topk(mat, q, K) for q in queries]
        for bits in (2, 4):
            idx = TurboQuantIndex(dim=dim, bit_width=bits)
            t = time.time(); idx.add(mat); build_ms = (time.time() - t) * 1000
            recalls, lat = [], []
            for q, truth in zip(queries, truths):
                t = time.time(); _, ind = idx.search(q, K); lat.append((time.time() - t) * 1e3)
                recalls.append(len(set(ind.tolist()) & truth) / K)
            lat.sort()
            print(f"turbovec  dim={dim:>5} bits={bits}  build={build_ms:8.1f}ms  "
                  f"p50={lat[len(lat)//2]:6.3f}ms  recall@10={sum(recalls)/len(recalls):.3f}")

if __name__ == "__main__":
    main()
```
Confirm the turbovec API (`TurboQuantIndex(dim, bit_width)`, `.add(np.ndarray)`, `.search(q, k) -> (scores, indices)`) against the installed package; adapt names if the package differs.

- [ ] **Step 2: Install turbovec + run**

```bash
ssh gpu-host-a.internal.example "python3 -m venv /tmp/tvenv && /tmp/tvenv/bin/pip install turbovec numpy 2>&1 | tail -3 && cd /home/mac/test/project-mdb/.claude/worktrees/rabitq-recall && /tmp/tvenv/bin/python scripts/bench/turbovec_compare.py 2>&1 | tail -20"
```
Expected: a table of turbovec recall@10/p50 at d=1536/3072 × {2,4}-bit. **If `pip install turbovec` fails** (native build issue on the host), record that it could not be installed and fall back to turbovec's PUBLISHED figures (R@1 leads FAISS IndexPQ by 0.4–3.4 pts at d=1536/3072; ~12–20% faster than FAISS FastScan), clearly labeled as not-same-hardware. Do not block the sub-project on the install.

- [ ] **Step 3: Commit the harness**

```bash
cd /home/mac/test/project-mdb/.claude/worktrees/rabitq-recall
git add scripts/bench/turbovec_compare.py
git commit -m "test(bench): turbovec vs Veda RaBitQ head-to-head harness"
```

---

## Task B5: Record results + TurboQuant decision

**Files:**
- Modify: `documentation/2026-06-10-rabitq-recall-design.md`

- [ ] **Step 1: Append a results section**

Append `## Results (2026-06-10)` to the design doc with: the B1 baseline table; the B2 recall delta (extrerank before/after rotation+codebook); the B3 recall delta (bare rabitq before/after unbiased estimator, OR the recorded negative result if reverted); the B4 turbovec-vs-Veda table (or the published-figures fallback + why); and a one-line **TurboQuant decision** (drop / reconsider) justified by the numbers.

- [ ] **Step 2: Final workspace verification**

```bash
cargo test -p veda-index 2>&1 | grep -E "test result:|FAILED|error" | tail -20
```
Expected: all green (the new unit/roundtrip tests pass; pre-existing veda-index tests still pass).

- [ ] **Step 3: Commit**

```bash
git add documentation/2026-06-10-rabitq-recall-design.md
git commit -m "docs: RaBitQ recall results + TurboQuant decision"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** B1 harness → Task B1; Fix-2 rotation/codebook → B2; Fix-1 unbiased estimator → B3; turbovec head-to-head → B4; results + decision → B5. All design pieces mapped.
- **The B3 estimator formula** `raw_dot · ‖c‖₂² / ‖c‖₁` is derived in the design + Task B3; the exactness-at-q==c unit test is the correctness guard, the B1 recall delta is the end-to-end guard (with an explicit revert-if-regress instruction).
- **Two API uncertainties flagged with concrete fallbacks:** (a) the index persistence API in B2 step 1 (find the real save/load; fall back to quantizer serde round-trip), (b) the turbovec Python API in B4 (confirm against the installed package; fall back to published figures if install fails). Neither blocks; both have explicit alternate paths.
- **Type consistency:** B3 replaces `mean_abs_component: f32` with `l1_norms: Vec<f32>` in BOTH the struct and `RaBitQuantizerData` and every conversion site (grep to confirm none remain). B1/B2/B4 add only tests/benches/scripts — no API changes to the indexes beyond B2's constructor body and B3's internals.
- **No recall regression ships:** both B2 and B3 have explicit "if recall drops, revert + record" gates measured against the B1 baseline.
