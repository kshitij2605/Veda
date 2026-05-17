//! PLAID (Performance-optimized Late Interaction Driver) style candidate
//! pruning for multi-vector / ColPali fields.
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
//! References:
//! - Santhanam et al., "PLAID: An Efficient Engine for Late Interaction
//!   Retrieval" (CIKM 2022)
//! - Khattab & Zaharia, "ColBERTv2" (NAACL 2022)

use roaring::RoaringBitmap;
use veda_core::simd;

/// PLAID candidate-pruning index for a multi-vector field within a single segment.
pub struct PlaidIndex {
    /// `centroids[i]` is the centre of cluster i (dimension = token dim).
    centroids: Vec<Vec<f32>>,
    /// `postings[i]` lists all doc_ids that have at least one token assigned
    /// to centroid i.
    postings: Vec<RoaringBitmap>,
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

        Self { centroids, postings }
    }

    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// For each query token, find the `probe` nearest centroids and union
    /// their posting lists into a candidate doc_id set.
    pub fn candidates(&self, query_rows: &[Vec<f32>], probe: usize) -> RoaringBitmap {
        let mut out = RoaringBitmap::new();
        if self.centroids.is_empty() || query_rows.is_empty() {
            return out;
        }
        for q in query_rows {
            let mut scored: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, simd::cosine_distance(q, c)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, _) in scored.iter().take(probe) {
                out |= &self.postings[*idx];
            }
        }
        out
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
}
