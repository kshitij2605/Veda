# Scalar-Quant Indexes (Sq4 + Sq8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a 4-bit-per-dimension scalar-quant index (`Sq4`, ~768 B/vec at d=1536, turbovec-competitive recall) and expose the existing 8-bit `ScalarQuantizer` as a standalone `Sq8` index (0.995 recall), giving a clean recall/memory dial.

**Architecture:** Generalize `ScalarQuantizer` to a configurable `bits ∈ {4,8}` and give it the index-supporting surface `RaBitQuantizer` already has (metric, search, len, save/load, dimension). One thin `ScalarQuantIndex` wrapper (mirroring `RaBitQIndex`) is exposed via two new `IndexType` variants `Sq4`/`Sq8`. Brute-force scan (matches turbovec; HNSW/DiskANN remain the separate ANN path).

**Tech Stack:** Rust 2024, `veda-index` + `veda-core` crates, `bincode`/`serde` for persistence.

**Workspace:** git worktree on the remote: `iitgpu11.hon.otsuka-shokai.co.jp:/home/mac/test/project-mdb/.claude/worktrees/rabitq-recall`, branch `feat/rabitq-recall` (continues the scalar-quant work; the diagnostic prototypes live here). Edit on the remote (scp down / Edit / scp up); run `cargo` over SSH from the worktree.

**Build/test prelude:**
```bash
export PROTOC=/home/mac/mitrix69/load/.venv/lib/python3.12/site-packages/torch/bin/protoc
export PATH=/home/mac/.cargo/bin:$PATH
cd /home/mac/test/project-mdb/.claude/worktrees/rabitq-recall
```

Design reference: `documentation/2026-06-11-scalar-quant-index-design.md`.

---

## File Structure

- **Modify** `crates/veda-index/src/quantization/scalar.rs` — generalize `ScalarQuantizer` to `bits ∈ {4,8}` + add metric/search/len/save/load/dimension + serde (T2).
- **Create** `crates/veda-index/src/quantization/scalar_index.rs` — `ScalarQuantIndex` (`VectorIndex` impl); constructors `sq4(metric)`/`sq8(metric)` (T3).
- **Modify** `crates/veda-index/src/quantization/mod.rs` — declare/export `scalar_index`.
- **Modify** `crates/veda-core/src/types.rs` — add `IndexType::Sq4`, `IndexType::Sq8` (T3).
- **Modify** the ≤3 `IndexType` match sites that need new arms (T3; grep to find them).
- **Modify** `crates/veda-index/tests/scale_benchmark.rs` — promote the diagnostic into a kept recall bench with target assertions (T4).
- **Modify** `documentation/2026-06-11-scalar-quant-index-design.md` — append results (T5).

Reuse: `compute_distance` (already used by ScalarQuantizer); `RaBitQuantizer`'s save/load shape (`bincode` of a `*Data` serde struct) as the template for ScalarQuantizer persistence.

---

## Task T1: Real-embedding validation (early, skip-gracefully)

**Files:** none committed unless embeddings are obtained (then a small script under `scripts/bench/`).

The synthetic uniform+anisotropic diagnostic already showed Sq4≈0.88/0.94, Sq8≈0.995. This task ATTEMPTS to confirm on real BGE-M3 embeddings; if extraction is impractical, record the limitation and proceed (the build does not depend on this result).

- [ ] **Step 1: Attempt to obtain real embeddings**

Look for real vectors on the host (in priority order):
- A Veda collection's stored vectors under `/home/mac/test/project-mdb/veda-data*/` or the live data dir (sealed segment `.vec` files) — check `ssh ... "find /home/mac/test -name '*.vec' -path '*rag_anything*' 2>/dev/null | head"` and whether a simple dump exists.
- Re-embedding the rag-anything corpus text via the embedding service (only if cheap/available — check `ssh ... "curl -s http://iitgpu03:8000/v1/models 2>/dev/null | head"`).
If neither is reachable in ~15 min of effort, STOP and record "real-embedding validation skipped: no readily-extractable embedding set; synthetic uniform+aniso results stand."

- [ ] **Step 2: If embeddings obtained, measure Sq4/Sq8 recall**

Write the embeddings to a file the Python diagnostic (`scripts/bench/turbovec_compare.py` pattern) or a small Rust bench can read; compute recall@10 of a 4-bit-per-dim and 8-bit-per-dim scalar quant vs exact cosine top-10 on that real set. Record the numbers.

- [ ] **Step 3: Record the outcome**

Note the real-embedding recall (or the skip) — it feeds the T5 results section. No production code is committed in this task. (If a throwaway extraction script was written, commit it under `scripts/bench/` with `test(bench): real-embedding scalar-quant validation`.)

---

## Task T2: Generalize ScalarQuantizer to bits ∈ {4,8} + index surface (TDD)

**Files:**
- Modify: `crates/veda-index/src/quantization/scalar.rs`

Read `scalar.rs` fully first. It currently stores `mins/maxs/codes/dimension` and has `new/train/encode/decode/estimate_distance` at 8-bit, with NO metric/search/save/load. Generalize + enrich it.

- [ ] **Step 1: Write failing round-trip + calibration unit tests**

Append a `#[cfg(test)] mod tests` (or extend it) to `scalar.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use veda_core::types::DistanceMetric;

    fn seeded(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..dim).map(|_| { s = s.wrapping_mul(1103515245).wrapping_add(12345); ((s>>16) as f32/32768.0)-1.0 }).collect()
    }

    #[test]
    fn encode_decode_roundtrip_8bit() {
        let v: Vec<Vec<f32>> = (0..50).map(|i| seeded(i, 17)).collect(); // odd dim
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 8);
        q.train(&refs);
        let code = q.encode(&v[3]);
        let dec = q.decode(&code);
        for (a, b) in v[3].iter().zip(&dec) {
            assert!((a-b).abs() < 0.02, "8-bit recon off: {a} vs {b}");
        }
    }

    #[test]
    fn encode_decode_roundtrip_4bit_odd_dim() {
        let v: Vec<Vec<f32>> = (0..50).map(|i| seeded(i, 17)).collect(); // odd dim → tests nibble padding
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let code = q.encode(&v[3]);
        assert_eq!(code.len(), 17usize.div_ceil(2), "4-bit packs 2 dims/byte");
        let dec = q.decode(&code);
        for (a, b) in v[3].iter().zip(&dec) {
            assert!((a-b).abs() < 0.15, "4-bit recon off: {a} vs {b}"); // coarser
        }
    }

    #[test]
    fn search_finds_exact_match() {
        let v: Vec<Vec<f32>> = (0..100).map(|i| seeded(i, 32)).collect();
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let hits = q.search(&v[42], 5, &std::collections::HashSet::new());
        assert!(hits.iter().any(|(id, _)| *id == 42));
    }

    #[test]
    fn save_load_roundtrip() {
        let v: Vec<Vec<f32>> = (0..40).map(|i| seeded(i, 24)).collect();
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let before = q.search(&v[5], 5, &std::collections::HashSet::new());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sq.bin");
        q.save(&p).unwrap();
        let q2 = ScalarQuantizer::load(&p).unwrap();
        let after = q2.search(&v[5], 5, &std::collections::HashSet::new());
        assert_eq!(before.iter().map(|x| x.0).collect::<Vec<_>>(),
                   after.iter().map(|x| x.0).collect::<Vec<_>>());
    }
}
```

- [ ] **Step 2: Run to verify failure**

`cargo test -p veda-index scalar:: 2>&1 | tail -20` → FAIL (no `with_bits`/`search`/`save`/`load`). Confirm RED.

- [ ] **Step 3: Implement the generalization**

Rewrite `scalar.rs`'s `ScalarQuantizer`:
- Fields: add `bits: u8`, `metric: DistanceMetric`. Keep `mins/maxs/codes/dimension`. `codes` stays `Vec<Vec<u8>>` (8-bit: 1 byte/dim; 4-bit: packed 2 dims/byte).
- `new()` keeps the 8-bit default for backward compat: `Self::with_bits(DistanceMetric::Cosine, 8)` — BUT note the old `new()` took no metric. Check callers: `grep -rn "ScalarQuantizer::new" crates/`. If existing callers exist, keep a `new()` that defaults metric to Cosine + bits 8, and ADD `with_bits(metric, bits)`. If no callers, replace `new()` with `with_bits` + a `new(metric)` convenience.
- `with_bits(metric: DistanceMetric, bits: u8) -> Self`: assert `bits == 4 || bits == 8`.
- `encode(&self, v) -> Vec<u8>`: per-dim normalize to `[0, levels]` where `levels = (1u32<<self.bits)-1` (15 or 255). For 8-bit: one byte/dim (as today). For 4-bit: quantize each dim to 0..=15, pack two dims per byte (`byte = lo | (hi << 4)`); output `dim.div_ceil(2)` bytes; if dim is odd the final byte's high nibble is 0.
- `decode(&self, code) -> Vec<f32>`: inverse — for 4-bit, unpack nibbles (low nibble = even dim, high nibble = odd dim), reconstruct `mins[i] + (q + 0.5)/(levels+1) * range` (bin midpoint) ... match the 8-bit reconstruction formula style (current 8-bit uses `mins[i] + (val/255)*range`; keep that exact form for 8-bit, and the analogous `mins[i] + (nibble as f32/15.0)*range` for 4-bit, to stay consistent with the diagnostic that measured 0.879).
- `estimate_distance(&self, query, code_idx, metric)`: keep, but use `self.metric` if you drop the param — simplest: keep the signature, ignore param or assert it matches `self.metric`. Prefer: change to `estimate_distance(&self, query, code_idx) -> f32` using `self.metric` (update the one call site in tests).
- ADD `search(&self, query, k, skip: &HashSet<u32>) -> Vec<(u32, f32)>`: scan all codes not in `skip`, compute `estimate_distance`, sort asc, truncate k. (Mirror `RaBitQuantizer::search`.)
- ADD `len(&self) -> usize` (= `codes.len()`), `is_empty`, `dimension(&self) -> u32`, `metric(&self) -> DistanceMetric`.
- ADD serde persistence: a `#[derive(Serialize,Deserialize)] struct ScalarQuantizerData { dimension, bits, metric, mins, maxs, codes }` + `save(&self, path) -> Result<()>` (bincode serialize, write file) and `load(path) -> Result<Self>` (read+deserialize) — mirror `RaBitQuantizer::save`/`load` exactly (read that for the precise `bincode` + `std::fs` + `VedaError` mapping).

- [ ] **Step 4: Run tests to verify pass**

`cargo test -p veda-index scalar:: 2>&1 | grep -E "test result:|FAILED" | tail` → PASS (4 new tests + any pre-existing). Fix the one `estimate_distance` call-site signature if you changed it.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-index/src/quantization/scalar.rs
git commit -m "feat(veda-index): ScalarQuantizer supports 4/8-bit + search/persistence"
```

---

## Task T3: ScalarQuantIndex wrapper + IndexType::Sq4/Sq8 (TDD)

**Files:**
- Create: `crates/veda-index/src/quantization/scalar_index.rs`
- Modify: `crates/veda-index/src/quantization/mod.rs`, `crates/veda-core/src/types.rs`, + ≤3 `IndexType` match sites.

- [ ] **Step 1: Write the failing index test**

Create `crates/veda-index/src/quantization/scalar_index.rs` with the impl (Step 3) AND this test module; write the test first mentally but you can author the file in one pass — run RED by referencing `ScalarQuantIndex` before the enum arms exist. Test:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VectorIndex;
    fn seeded(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..dim).map(|_| { s = s.wrapping_mul(1103515245).wrapping_add(12345); ((s>>16) as f32/32768.0)-1.0 }).collect()
    }
    #[test]
    fn sq4_build_search_delete_roundtrip() {
        for bits in [4u8, 8] {
            let mut idx = ScalarQuantIndex::with_bits(DistanceMetric::Cosine, bits);
            let v: Vec<Vec<f32>> = (0..100).map(|i| seeded(i, 48)).collect();
            let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
            idx.build(&refs).unwrap();
            let hits = idx.search(&v[10], 5).unwrap();
            assert!(hits.iter().any(|h| h.id == 10), "bits={bits} exact match in top-5");
            idx.delete(10).unwrap();
            let hits2 = idx.search(&v[10], 5).unwrap();
            assert!(hits2.iter().all(|h| h.id != 10), "bits={bits} deleted id absent");
            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("idx.bin");
            idx.save(&p).unwrap();
            let re = ScalarQuantIndex::load(&p, DistanceMetric::Cosine, 48).unwrap();
            assert_eq!(re.len(), idx.len() + 1); // reloaded has no soft-deletes
        }
    }
}
```

- [ ] **Step 2: Run RED**

`cargo test -p veda-index scalar_index 2>&1 | tail -15` → FAIL (type missing / not declared in mod).

- [ ] **Step 3: Implement the wrapper + enum + arms**

`scalar_index.rs` (mirror `rabitq_vector_index.rs` exactly — same `VectorIndex` methods, `deleted: HashSet`, `search_filtered` via skip-set, `insert`/`delete` soft-delete, `save`/`load` delegating to the quantizer):
```rust
use std::collections::HashSet;
use std::path::Path;
use roaring::RoaringBitmap;
use veda_core::error::Result;
use veda_core::types::DistanceMetric;
use crate::quantization::scalar::ScalarQuantizer;
use crate::traits::{SearchResult, VectorIndex};

pub struct ScalarQuantIndex { quantizer: ScalarQuantizer, deleted: HashSet<u32> }

impl ScalarQuantIndex {
    pub fn with_bits(metric: DistanceMetric, bits: u8) -> Self {
        Self { quantizer: ScalarQuantizer::with_bits(metric, bits), deleted: HashSet::new() }
    }
    pub fn sq4(metric: DistanceMetric) -> Self { Self::with_bits(metric, 4) }
    pub fn sq8(metric: DistanceMetric) -> Self { Self::with_bits(metric, 8) }
}

impl VectorIndex for ScalarQuantIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> { self.quantizer.train(vectors); self.deleted.clear(); Ok(()) }
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        Ok(self.quantizer.search(query, k, &self.deleted).into_iter().map(|(id, distance)| SearchResult { id, distance }).collect())
    }
    fn search_filtered(&self, query: &[f32], k: usize, filter: &RoaringBitmap) -> Result<Vec<SearchResult>> {
        let mut skip = self.deleted.clone();
        for id in 0..self.quantizer.len() as u32 { if !filter.contains(id) { skip.insert(id); } }
        Ok(self.quantizer.search(query, k, &skip).into_iter().map(|(id, distance)| SearchResult { id, distance }).collect())
    }
    fn insert(&mut self, id: u32, _v: &[f32]) -> Result<()> { self.deleted.remove(&id); Ok(()) }
    fn delete(&mut self, id: u32) -> Result<()> { self.deleted.insert(id); Ok(()) }
    fn len(&self) -> usize { self.quantizer.len().saturating_sub(self.deleted.len()) }
    fn save(&self, path: &Path) -> Result<()> { self.quantizer.save(path) }
    fn load(path: &Path, _metric: DistanceMetric, _dim: u32) -> Result<Self> where Self: Sized {
        Ok(Self { quantizer: ScalarQuantizer::load(path)?, deleted: HashSet::new() })
    }
    fn metric(&self) -> DistanceMetric { self.quantizer.metric() }
    fn dimension(&self) -> u32 { self.quantizer.dimension() }
}
```
In `mod.rs`: add `pub mod scalar_index;`.
In `crates/veda-core/src/types.rs`: add `Sq4,` and `Sq8,` variants to `enum IndexType` with doc comments (Sq4: "4-bit/dim scalar quant — ~D/2 bytes/vec, turbovec-competitive recall"; Sq8: "8-bit/dim scalar quant — D bytes/vec, ~0.99 recall, highest-recall quantized option").
Find the `IndexType` match sites: `grep -rn "match .*index_type\|IndexType::Flat =>" crates/*/src --include=*.rs` (≤3). Add `IndexType::Sq4 => ...` / `IndexType::Sq8 => ...` arms routing to `ScalarQuantIndex::sq4(metric)` / `sq8(metric)` (mirror how `IndexType::RaBitQ` builds `RaBitQIndex`). If a match site is a non-builder (e.g. a `name()`/display), add the appropriate arm there too.

- [ ] **Step 4: Run tests + full veda-index + workspace compile**

`cargo test -p veda-index scalar_index 2>&1 | tail` → PASS. Then `cargo build --workspace 2>&1 | tail -5` → clean (confirms the new enum variants' match arms are exhaustive everywhere).

- [ ] **Step 5: Commit**

```bash
git add crates/veda-index/src/quantization/scalar_index.rs crates/veda-index/src/quantization/mod.rs crates/veda-core/src/types.rs <the match-site files>
git commit -m "feat(veda-index): Sq4/Sq8 scalar-quant IndexType + ScalarQuantIndex"
```

---

## Task T4: Recall benchmark (promote the diagnostic)

**Files:**
- Modify: `crates/veda-index/tests/scale_benchmark.rs`

The diagnostic added `Sq8BenchIndex`/`Sq4PerDimBenchIndex` (throwaway). Replace their use with the real `ScalarQuantIndex` and add an asserting recall test.

- [ ] **Step 1: Add a kept recall test using the real indexes**

Add (reusing `make_corpus`, `print_row`, `QUERIES`, `K`):
```rust
#[test]
#[ignore]
fn sq_recall_targets_1536() {
    use veda_index::quantization::scalar_index::ScalarQuantIndex;
    let dim = 1536; let n = 50_000;
    let corpus = make_corpus(n, dim);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    let mut truth = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES).map(|i| &corpus[(i*7919)%n]).collect();
    let truths: Vec<std::collections::HashSet<u32>> = queries.iter()
        .map(|q| truth.search(q, K).unwrap().into_iter().map(|h| h.id).collect()).collect();
    let recall = |idx: &dyn veda_index::traits::VectorIndex| -> f64 {
        let mut s = 0.0;
        for (q, t) in queries.iter().zip(&truths) {
            let ids: Vec<u32> = idx.search(q, K).unwrap().iter().map(|h| h.id).collect();
            s += recall_at_k(&ids, t);
        }
        s / queries.len() as f64
    };
    let mut sq8 = ScalarQuantIndex::sq8(DistanceMetric::Cosine); sq8.build(&refs).unwrap();
    let mut sq4 = ScalarQuantIndex::sq4(DistanceMetric::Cosine); sq4.build(&refs).unwrap();
    let r8 = recall(&sq8); let r4 = recall(&sq4);
    eprintln!("sq8 recall@10={r8:.3} ({}B/vec)  sq4 recall@10={r4:.3} ({}B/vec)", dim, dim/2);
    assert!(r8 >= 0.97, "sq8 recall {r8} below 0.97");
    assert!(r4 >= 0.82, "sq4 recall {r4} below 0.82 (turbovec ~0.85)");
}
```
(`recall_at_k`'s exact signature: match the existing helper in the file. If it takes `&[u32], &HashSet<u32>`, adapt.)

- [ ] **Step 2: Run it**

`cargo test --release -p veda-index --test scale_benchmark -- --ignored --nocapture sq_recall_targets_1536 2>&1 | grep -E "sq8|sq4|recall|FAILED"` (slow; long timeout). Expected: sq8 ≥0.97, sq4 ≥0.82, test passes. Record the numbers.

- [ ] **Step 3: Commit**

```bash
git add crates/veda-index/tests/scale_benchmark.rs
git commit -m "test(veda-index): Sq4/Sq8 recall targets at d=1536"
```

---

## Task T5: Results + final verification

**Files:**
- Modify: `documentation/2026-06-11-scalar-quant-index-design.md`

- [ ] **Step 1: Append a results section**

`## Results (2026-06-11)`: the T4 measured Sq4/Sq8 recall + bytes/vec; the T1 real-embedding outcome (numbers or "skipped: <reason>"); confirmation that Sq4 matches turbovec (~0.85) at ~768 B/vec and Sq8 ≈0.99; note the new `IndexType::Sq4`/`Sq8` are available in `veda-index` (gateway/schema exposure remains a follow-up).

- [ ] **Step 2: Final workspace verification**

`cargo test -p veda-index -p veda-core 2>&1 | grep -E "test result:|FAILED|error" | tail -25` → all green. `cargo build --workspace 2>&1 | tail -3` → clean.

- [ ] **Step 3: Commit**

```bash
git add documentation/2026-06-11-scalar-quant-index-design.md
git commit -m "docs: Sq4/Sq8 scalar-quant results"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** generalize ScalarQuantizer → T2; Sq4/Sq8 indexes + IndexType → T3; recall verification → T4; real-embedding early validation → T1; results → T5. All design pieces mapped.
- **Type consistency:** `ScalarQuantizer::with_bits(metric, bits)` (T2) is used by `ScalarQuantIndex::with_bits/sq4/sq8` (T3) and the benches (T4). `ScalarQuantizer::search(query,k,&HashSet)->Vec<(u32,f32)>` and `save/load` (T2) are consumed by the wrapper (T3) exactly as `RaBitQuantizer`'s are by `RaBitQIndex`. `IndexType::Sq4`/`Sq8` (T3) — every `match IndexType` arm updated (T3 step 3 greps them; T3 step 4's `cargo build --workspace` enforces exhaustiveness).
- **Backward compat:** the 8-bit path must reproduce today's `ScalarQuantizer` behavior (the diagnostic's 0.995); keep the 8-bit encode/decode formula identical. Check `ScalarQuantizer::new` callers before changing its signature.
- **The 4-bit reconstruction formula** must match the diagnostic that measured 0.879 (per-dim min/max, 16 levels, `mins[i] + nibble/15 * range`) — otherwise recall may differ from the validated prototype.
- **Real-embedding (T1)** is skip-gracefully; do not block the build on it.
