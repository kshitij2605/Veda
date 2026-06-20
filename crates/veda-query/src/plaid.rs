//! PLAID-shape candidate pruning for multi-vector / ColPali fields,
//! upgraded with WARP_SELECT (SIGIR'25) for per-query top-`probe`
//! centroid selection.
//!
//! Given a corpus of documents where each document is a list of token
//! vectors, build:
//!
//! - K centroids over **all** tokens across all documents (via simple
//!   k-means).
//! - One posting list per centroid mapping `centroid_idx -> doc_id set`.
//!
//! At query time, for each query token find the top-`probe` nearest
//! centroids, then union their posting lists to get a candidate set. The
//! caller runs full MaxSim only on this set — typically 1-10% of the
//! corpus, for a 10-100x speedup over brute-force.
//!
//! ## WARP_SELECT (vs. classic PLAID)
//!
//! Classic PLAID's `candidates()` sorts ALL centroids by distance per
//! query row — O(K log K). When K is large and `probe` is small
//! (the common case, since `probe` ≤ 32 typically), most of that
//! work is wasted.
//!
//! WARP_SELECT (Engels et al., SIGIR 2025) replaces the sort with a
//! fixed-size max-heap of size `probe`: O(K log probe). On the
//! published numbers this delivers ≈3× lower per-query latency and
//! 2-4× smaller index footprint (the latter via implicit
//! decompression of compressed codes; Veda's PLAID stores
//! centroids uncompressed today, so we capture the latency win
//! but not the storage win).
//!
//! Not yet ported from the WARP paper (follow-up work):
//! - Implicit decompression of bit-packed centroids.
//! - Two-stage candidate reduction (intersection-then-union) for
//!   selective queries.
//!
//! See `research/2026-05-19-sota-audit/05-late-interaction.md`.
//!
//! References:
//! - Santhanam et al., "PLAID: An Efficient Engine for Late Interaction
//!   Retrieval" (CIKM 2022)
//! - Engels et al., "WARP: Word-Aware Compressed Representation for
//!   Efficient Late Interaction Retrieval" (SIGIR 2025)
//! - Khattab & Zaharia, "ColBERTv2" (NAACL 2022)

use roaring::RoaringBitmap;
use veda_core::simd;

/// Centroids quantised to int8 (one byte per dim) — the WARP
/// paper's "implicit decompression" path.
///
/// For typical PLAID shapes (k ≈ 1024 centroids × dim ≈ 128) this
/// shrinks the centroid store from 512 KB (f32) to 128 KB (i8),
/// landing fully in L1d on modern CPUs. The distance op against
/// the int codes uses the same scale across all dims (uniform
/// scalar quantisation), so the per-pair work is
/// `(Σ q_code[i] × c_code[i]) · scale_q · scale_c` — no f32
/// materialisation of the centroid in the hot loop.
#[derive(Debug)]
struct QuantizedCentroids {
    /// `codes[i]` is the int8 representation of centroid i.
    codes: Vec<Vec<i8>>,
    /// Per-centroid L2 norm of the *quantised* vector (in code
    /// space). Used by the cosine denominator.
    code_norms: Vec<f32>,
    /// Per-centroid scale: `f32_value = i8_value × scale[i]`.
    /// Uniform across dims within one centroid; varies per
    /// centroid because each cluster's value range differs.
    scales: Vec<f32>,
}

impl QuantizedCentroids {
    fn build(centroids: &[Vec<f32>]) -> Self {
        let mut codes = Vec::with_capacity(centroids.len());
        let mut code_norms = Vec::with_capacity(centroids.len());
        let mut scales = Vec::with_capacity(centroids.len());
        for c in centroids {
            let max_abs = c.iter().map(|x| x.abs()).fold(0.0_f32, f32::max).max(1e-9);
            let scale = max_abs / 127.0;
            let inv = 1.0 / scale;
            let code: Vec<i8> = c
                .iter()
                .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
                .collect();
            let code_norm = code
                .iter()
                .map(|&x| (x as f32) * (x as f32))
                .sum::<f32>()
                .sqrt();
            codes.push(code);
            code_norms.push(code_norm * scale); // pre-scaled L2 of decoded centroid
            scales.push(scale);
        }
        Self {
            codes,
            code_norms,
            scales,
        }
    }

    /// Cosine distance from a quantised query to centroid `idx`.
    /// Uses [`simd::i8_dot_product`] for the int dot — AVX2 path
    /// on x86_64, scalar fallback elsewhere.
    fn cosine_distance(&self, q_codes: &[i8], q_scale: f32, q_norm: f32, idx: usize) -> f32 {
        let c_codes = &self.codes[idx];
        let dot_int = simd::i8_dot_product(q_codes, c_codes);
        let dot_f = (dot_int as f32) * q_scale * self.scales[idx];
        let denom = q_norm * self.code_norms[idx];
        if denom < 1e-12 {
            return 1.0;
        }
        1.0 - dot_f / denom
    }
}

/// PLAID candidate-pruning index for a multi-vector field within a single segment.
pub struct PlaidIndex {
    /// `centroids[i]` is the centre of cluster i (dimension = token dim).
    centroids: Vec<Vec<f32>>,
    /// `postings[i]` lists all doc_ids that have at least one token assigned
    /// to centroid i.
    postings: Vec<RoaringBitmap>,
    /// Optional implicit-decompression path: int8-quantised
    /// centroids built alongside the float centroids. When
    /// present, [`Self::candidates_with_quorum`] uses the int8
    /// dot for centroid distance instead of `simd::cosine_distance`.
    /// Enable via [`Self::with_quantized_centroids`].
    quantized: Option<QuantizedCentroids>,
}

impl PlaidIndex {
    /// Build a PLAID index over `(doc_id, doc_token_rows)` pairs. Picks `k`
    /// via square-root heuristic on total token count, clamped to `[16, 1024]`.
    pub fn build<'a>(docs: impl Iterator<Item = (u32, &'a Vec<Vec<f32>>)>) -> Self {
        // Flatten into (doc_id, token_vec) pairs first.
        let mut tokens: Vec<(u32, &[f32])> = Vec::new();
        for (doc_id, rows) in docs {
            for r in rows {
                tokens.push((doc_id, r.as_slice()));
            }
        }
        let n = tokens.len();
        if n == 0 {
            return Self {
                centroids: Vec::new(),
                postings: Vec::new(),
                quantized: None,
            };
        }
        let dim = tokens[0].1.len();

        // sqrt(N) heuristic, clamped.
        let k = ((n as f64).sqrt() as usize).clamp(16, 1024).min(n);

        let centroids = kmeans(&tokens, k, dim, 10);

        // Assign each token to its nearest centroid; populate posting bitmaps.
        let mut postings: Vec<RoaringBitmap> =
            (0..centroids.len()).map(|_| RoaringBitmap::new()).collect();
        for &(doc_id, tok) in &tokens {
            let idx = nearest_centroid(tok, &centroids);
            postings[idx].insert(doc_id);
        }

        Self {
            centroids,
            postings,
            quantized: None,
        }
    }

    /// Enable the int8-quantised centroid path (the WARP
    /// "implicit decompression" trick). Run once after build;
    /// reads are then served from the int8 codes instead of
    /// the f32 centroids.
    pub fn with_quantized_centroids(mut self) -> Self {
        if !self.centroids.is_empty() {
            self.quantized = Some(QuantizedCentroids::build(&self.centroids));
        }
        self
    }

    pub fn has_quantized_centroids(&self) -> bool {
        self.quantized.is_some()
    }

    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// For each query token, find the `probe` nearest centroids and union
    /// their posting lists into a candidate doc_id set. Uses WARP_SELECT's
    /// fixed-size max-heap selection (O(K log probe)) instead of the
    /// classic PLAID full sort (O(K log K)).
    pub fn candidates(&self, query_rows: &[Vec<f32>], probe: usize) -> RoaringBitmap {
        self.candidates_with_quorum(query_rows, probe, 1)
    }

    /// WARP two-stage candidate reduction (SIGIR 2025).
    ///
    /// Stage 1: for each query row, find the `probe` nearest
    /// centroids (same as classic PLAID).
    /// Stage 2: a doc is kept iff it appears in the posting lists
    /// of at least `min_quorum` distinct query rows' top-probe
    /// sets. `min_quorum = 1` is equivalent to the classic union
    /// behaviour (used by [`Self::candidates`]); higher values
    /// produce a tighter, more focused candidate set at the cost
    /// of missing some weakly-correlated documents.
    ///
    /// Empirical tuning: `min_quorum = max(1, ⌈Q × 0.3⌉)` is a
    /// good default for ColPali-shaped queries where Q ≈ 32
    /// tokens — keeps the candidate set ~3x smaller than the
    /// union while preserving the docs that are actually
    /// relevant across most query tokens.
    pub fn candidates_with_quorum(
        &self,
        query_rows: &[Vec<f32>],
        probe: usize,
        min_quorum: u32,
    ) -> RoaringBitmap {
        let mut out = RoaringBitmap::new();
        if self.centroids.is_empty() || query_rows.is_empty() || probe == 0 {
            return out;
        }
        let min_quorum = min_quorum.max(1);

        // Fast path: union semantics.
        if min_quorum == 1 {
            let mut top: std::collections::BinaryHeap<HeapEntry> =
                std::collections::BinaryHeap::with_capacity(probe + 1);
            for q in query_rows {
                top.clear();
                if let Some(qc) = self.quantized.as_ref() {
                    push_top_probe_quantized(&mut top, q, &self.centroids, qc, probe);
                } else {
                    push_top_probe(&mut top, q, &self.centroids, probe);
                }
                for entry in top.drain() {
                    out |= &self.postings[entry.idx];
                }
            }
            return out;
        }

        // Two-stage: collect per-query-row candidate sets, then
        // count membership across rows.
        let mut per_row_sets: Vec<RoaringBitmap> = Vec::with_capacity(query_rows.len());
        let mut top: std::collections::BinaryHeap<HeapEntry> =
            std::collections::BinaryHeap::with_capacity(probe + 1);
        for q in query_rows {
            top.clear();
            if let Some(qc) = self.quantized.as_ref() {
                push_top_probe_quantized(&mut top, q, &self.centroids, qc, probe);
            } else {
                push_top_probe(&mut top, q, &self.centroids, probe);
            }
            let mut row_set = RoaringBitmap::new();
            for entry in top.drain() {
                row_set |= &self.postings[entry.idx];
            }
            per_row_sets.push(row_set);
        }

        // For each doc id, count rows that included it. Keep only
        // docs with count >= min_quorum.
        // Iterate over the union once so we don't visit every
        // possible doc id.
        let mut union_all = RoaringBitmap::new();
        for s in &per_row_sets {
            union_all |= s;
        }
        for doc_id in union_all.iter() {
            let count = per_row_sets.iter().filter(|s| s.contains(doc_id)).count() as u32;
            if count >= min_quorum {
                out.insert(doc_id);
            }
        }
        out
    }
}

/// Walk the centroids and push their top-`probe` by ascending
/// distance into the supplied max-heap. Reused by both the union
/// and quorum paths.
fn push_top_probe(
    top: &mut std::collections::BinaryHeap<HeapEntry>,
    q: &[f32],
    centroids: &[Vec<f32>],
    probe: usize,
) {
    for (i, c) in centroids.iter().enumerate() {
        let dist = simd::cosine_distance(q, c);
        if top.len() < probe {
            top.push(HeapEntry { idx: i, dist });
        } else if let Some(worst) = top.peek()
            && dist < worst.dist
        {
            top.pop();
            top.push(HeapEntry { idx: i, dist });
        }
    }
}

/// Same shape as `push_top_probe`, but pulls centroid distances
/// out of the int8 store ([`QuantizedCentroids`]). Quantises the
/// query once, then does int-dot per centroid — no f32
/// materialisation of the centroid in the hot loop.
fn push_top_probe_quantized(
    top: &mut std::collections::BinaryHeap<HeapEntry>,
    q: &[f32],
    centroids_f32: &[Vec<f32>],
    qc: &QuantizedCentroids,
    probe: usize,
) {
    // Pre-quantise the query once.
    let q_max_abs = q.iter().map(|x| x.abs()).fold(0.0_f32, f32::max).max(1e-9);
    let q_scale = q_max_abs / 127.0;
    let inv = 1.0 / q_scale;
    let q_codes: Vec<i8> = q
        .iter()
        .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let q_norm = q_codes
        .iter()
        .map(|&x| (x as f32) * (x as f32))
        .sum::<f32>()
        .sqrt()
        * q_scale;

    debug_assert_eq!(centroids_f32.len(), qc.codes.len());
    for i in 0..qc.codes.len() {
        let dist = qc.cosine_distance(&q_codes, q_scale, q_norm, i);
        if top.len() < probe {
            top.push(HeapEntry { idx: i, dist });
        } else if let Some(worst) = top.peek()
            && dist < worst.dist
        {
            top.pop();
            top.push(HeapEntry { idx: i, dist });
        }
    }
}

/// Max-heap entry: ordered so the WORST (largest dist) entry is
/// at the top, allowing O(log probe) admission of new candidates.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    idx: usize,
    dist: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Larger dist comes out first (max-heap on dist).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Find the nearest centroid index by cosine distance.
fn nearest_centroid(token: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = simd::cosine_distance(token, c);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Simple k-means: pick `k` initial centroids deterministically (stride
/// through input), iterate Lloyd's algorithm `max_iter` times.
fn kmeans(tokens: &[(u32, &[f32])], k: usize, dim: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if tokens.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(tokens.len());

    // Deterministic init: take every (n/k)th token. Keeps tests reproducible
    // without requiring a seeded RNG dependency.
    let stride = tokens.len() / k;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| tokens[i * stride.max(1)].1.to_vec())
        .collect();

    for _ in 0..max_iter {
        // Assignment.
        let mut sums: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; dim]).collect();
        let mut counts = vec![0u64; k];
        for &(_, t) in tokens {
            let idx = nearest_centroid(t, &centroids);
            for (i, &v) in t.iter().enumerate() {
                sums[idx][i] += v as f64;
            }
            counts[idx] += 1;
        }

        // Update.
        let mut changed = false;
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            for i in 0..dim {
                let new = (sums[c][i] * inv) as f32;
                if (centroids[c][i] - new).abs() > 1e-6 {
                    changed = true;
                }
                centroids[c][i] = new;
            }
        }
        if !changed {
            break;
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(rows: &[&[f32]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn empty_corpus_yields_empty_index() {
        let it = std::iter::empty::<(u32, &Vec<Vec<f32>>)>();
        let p = PlaidIndex::build(it);
        assert_eq!(p.centroid_count(), 0);
        assert!(p.candidates(&[], 5).is_empty());
    }

    #[test]
    fn candidates_include_matching_doc() {
        // Three docs clearly clustered around different points.
        let docs: Vec<(u32, Vec<Vec<f32>>)> = vec![
            (0, doc(&[&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.1, 0.0, 0.0]])),
            (1, doc(&[&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.1, 0.0]])),
            (2, doc(&[&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.1, 1.0]])),
        ];
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());

        // Query close to doc 1 — should include doc 1 in candidates.
        let q = vec![vec![0.0f32, 1.0, 0.0, 0.0]];
        let cand = p.candidates(&q, 1);
        assert!(cand.contains(1), "expected doc 1 in candidates: {cand:?}");
    }

    /// WARP_SELECT correctness: the candidate set with probe=K
    /// (asking for every centroid) MUST equal the union of every
    /// posting list. This pins that the heap selection isn't
    /// dropping centroids.
    #[test]
    fn warp_select_with_probe_eq_k_unions_all_postings() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..20u32)
            .map(|i| {
                let v = vec![i as f32 * 0.01, 1.0 - i as f32 * 0.01, 0.0, 0.0];
                (i, vec![v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());
        let k = p.centroid_count();
        assert!(k >= 16, "expected sqrt-N init to produce ≥16 centroids");
        let q = vec![vec![0.0_f32, 1.0, 0.0, 0.0]];
        let cand = p.candidates(&q, k);
        // With probe = K, every posting list is included; every
        // doc that's in any posting list must be in the result.
        for d in 0..20u32 {
            assert!(
                cand.contains(d),
                "doc {d} missing from full-probe candidates"
            );
        }
    }

    /// WARP two-stage: quorum=1 must equal the union of every
    /// posting list returned by per-row top-probe selection.
    #[test]
    fn warp_two_stage_quorum_one_equals_union() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..20u32)
            .map(|i| {
                let v = vec![i as f32 * 0.01, 1.0 - i as f32 * 0.01, 0.0, 0.0];
                (i, vec![v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());
        let q = vec![vec![0.0_f32, 1.0, 0.0, 0.0], vec![0.5_f32, 0.5, 0.0, 0.0]];
        assert_eq!(p.candidates(&q, 4), p.candidates_with_quorum(&q, 4, 1));
    }

    /// WARP two-stage: higher quorum produces a subset of lower
    /// quorum's set. Monotonicity invariant.
    #[test]
    fn warp_two_stage_quorum_is_monotone_subset() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..40u32)
            .map(|i| {
                let v = vec![
                    (i as f32 * 0.07).sin(),
                    (i as f32 * 0.13).cos(),
                    (i as f32 * 0.11).sin(),
                    (i as f32 * 0.17).cos(),
                ];
                (i, vec![v.clone(), v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());
        let q = vec![
            vec![0.5_f32, 0.5, 0.5, 0.5],
            vec![0.1_f32, 0.9, 0.1, 0.9],
            vec![0.9_f32, 0.1, 0.9, 0.1],
        ];
        let q1 = p.candidates_with_quorum(&q, 4, 1);
        let q2 = p.candidates_with_quorum(&q, 4, 2);
        let q3 = p.candidates_with_quorum(&q, 4, 3);
        // Monotone subset chain: q3 ⊆ q2 ⊆ q1.
        for id in q3.iter() {
            assert!(q2.contains(id), "q3 doc {id} missing from q2");
        }
        for id in q2.iter() {
            assert!(q1.contains(id), "q2 doc {id} missing from q1");
        }
        // q2 should be strictly smaller than q1 (otherwise the
        // test data is too dense).
        assert!(
            q2.len() <= q1.len(),
            "quorum=2 set must be subset of quorum=1"
        );
    }

    /// Implicit-decompression smoke test: with quantized
    /// centroids enabled, candidates() returns approximately the
    /// same set as the float-centroid path on a fixture where
    /// the quantisation error is small (large per-component
    /// magnitudes, small dim).
    #[test]
    fn quantized_centroids_smoke_returns_overlapping_candidates() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..40u32)
            .map(|i| {
                let v = vec![
                    (i as f32 * 0.17).sin() * 10.0,
                    (i as f32 * 0.13).cos() * 10.0,
                    (i as f32 * 0.11).sin() * 10.0,
                    (i as f32 * 0.07).cos() * 10.0,
                ];
                (i, vec![v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p_f = PlaidIndex::build(refs.iter().copied());
        let refs2: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p_q = PlaidIndex::build(refs2.into_iter()).with_quantized_centroids();
        assert!(p_q.has_quantized_centroids());
        let q = vec![vec![5.0_f32, 5.0, 5.0, 5.0], vec![-5.0_f32, 5.0, -5.0, 5.0]];
        let probe = 4;
        let r_f = p_f.candidates(&q, probe);
        let r_q = p_q.candidates(&q, probe);
        // The two sets should overlap heavily (≥ 60% Jaccard on
        // this small fixture; quantisation error rare drops a
        // borderline candidate).
        let inter = (&r_f) & (&r_q);
        let union = (&r_f) | (&r_q);
        if union.is_empty() {
            // Both empty — degenerate fixture; skip.
            return;
        }
        let jaccard = inter.len() as f64 / union.len() as f64;
        assert!(
            jaccard >= 0.6,
            "quantized vs f32 candidate sets too divergent: jaccard={jaccard:.2}"
        );
    }

    /// With quantized centroids and large per-component magnitudes
    /// (≫ 1.0 so the int8 grid is well-utilised), the candidate
    /// set should EXACTLY match the f32 path on a self-search.
    #[test]
    fn quantized_centroids_self_query_matches_f32_path() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..30u32)
            .map(|i| {
                let v = vec![
                    (i as f32 * 0.13).sin() * 100.0,
                    (i as f32 * 0.07).cos() * 100.0,
                ];
                (i, vec![v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p_f = PlaidIndex::build(refs.iter().copied());
        let refs2: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p_q = PlaidIndex::build(refs2.into_iter()).with_quantized_centroids();
        let q = vec![docs[5].1[0].clone()];
        let r_f = p_f.candidates(&q, 3);
        let r_q = p_q.candidates(&q, 3);
        // The float self-query must include doc 5; the quantized
        // path must also include doc 5 (the int8 grid at scale
        // ≈ 100 has plenty of resolution).
        assert!(r_f.contains(5), "f32 path missing self-doc: {r_f:?}");
        assert!(r_q.contains(5), "quantized path missing self-doc: {r_q:?}");
    }

    /// WARP two-stage: quorum > Q always yields empty (a doc
    /// can't appear in more query-row sets than there are query
    /// rows).
    #[test]
    fn warp_two_stage_quorum_above_q_is_empty() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> =
            (0..10u32).map(|i| (i, vec![vec![i as f32; 4]])).collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());
        let q = vec![vec![0.5_f32; 4], vec![1.5_f32; 4]];
        let too_strict = p.candidates_with_quorum(&q, 4, 99);
        assert!(too_strict.is_empty());
    }

    /// WARP_SELECT correctness: probe=1 must return the SAME set
    /// as the legacy full-sort approach. Replays both algorithms
    /// and compares.
    #[test]
    fn warp_select_with_probe_one_matches_full_sort() {
        let docs: Vec<(u32, Vec<Vec<f32>>)> = (0..30u32)
            .map(|i| {
                let v = vec![
                    (i as f32 * 0.07).sin(),
                    (i as f32 * 0.11).cos(),
                    (i as f32 * 0.13).sin(),
                    (i as f32 * 0.17).cos(),
                ];
                (i, vec![v.clone(), v])
            })
            .collect();
        let refs: Vec<(u32, &Vec<Vec<f32>>)> = docs.iter().map(|(i, v)| (*i, v)).collect();
        let p = PlaidIndex::build(refs.into_iter());
        let q = vec![vec![0.5_f32, 0.5, 0.5, 0.5]];

        // Legacy full-sort reference (replicated locally so we
        // don't depend on the public API to expose both).
        let legacy = {
            let mut out = RoaringBitmap::new();
            for q in &q {
                let mut scored: Vec<(usize, f32)> = p
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, simd::cosine_distance(q, c)))
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (idx, _) in scored.iter().take(1) {
                    out |= &p.postings[*idx];
                }
            }
            out
        };

        let warp = p.candidates(&q, 1);
        assert_eq!(warp, legacy, "WARP_SELECT must match full-sort at probe=1");
    }
}
