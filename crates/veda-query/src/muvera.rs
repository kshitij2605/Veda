//! MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encoding).
//!
//! Collapses a multi-vector (ColBERT / ColPali style) representation into a
//! single fixed-dimension vector that approximates MaxSim scoring under
//! standard cosine similarity. Used as a fast recall stage upstream of
//! exact MaxSim rerank — 70-80% memory reduction and 3-8× faster initial
//! retrieval in published benchmarks (Google PLAID / Weaviate / Qdrant /
//! Milvus implementations 2025-2026).
//!
//! ## Algorithm
//!
//! 1. **Setup**: choose `k_sim` random unit-norm anchor vectors in the
//!    same dimension as the document tokens. Anchors are deterministically
//!    derived from a seed so query encoding and document encoding produce
//!    compatible FDEs.
//!
//! 2. **Encode** a document/query of `n` token vectors:
//!    - For each token, find the nearest anchor (max cosine).
//!    - Add the token's projection onto that anchor to the FDE slot for
//!      that anchor.
//!    - Result is a `k_sim`-dimensional vector.
//!
//! 3. **Score** two FDEs with plain inner product or cosine. The expected
//!    inner product approximates the MaxSim sum.
//!
//! ## Defaults
//!
//! `k_sim = 128` works well for token dim ~128-1024 (ColBERT, ColPali). For
//! very high-dimensional tokens (e.g. ColQwen 2048-d) raise to 256.
//!
//! ## References
//!
//! - Dhulipala et al., "MUVERA: Multi-Vector Retrieval via Fixed Dimensional
//!   Encodings" (Google 2024) — <https://arxiv.org/abs/2405.19504>
//! - Weaviate's implementation guide — <https://weaviate.io/blog/muvera>

use serde::{Deserialize, Serialize};

/// Default number of anchor partitions. Higher = more expressive FDE but
/// larger memory; 128 is the sweet spot for typical ColBERT/ColPali dims.
pub const DEFAULT_K_SIM: usize = 128;

/// MUVERA encoder: deterministic anchors + token-to-FDE projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuveraEncoder {
    /// `k_sim` anchor vectors, each `dim` floats, L2-normalised.
    anchors: Vec<Vec<f32>>,
    pub dim: usize,
    pub k_sim: usize,
}

impl MuveraEncoder {
    /// Build an encoder for `dim`-dimensional tokens with `k_sim` anchors.
    /// `seed` controls the deterministic anchor generation — the same seed
    /// must be used when encoding queries as when encoding documents.
    pub fn new(dim: usize, k_sim: usize, seed: u64) -> Self {
        let mut anchors = Vec::with_capacity(k_sim);
        let mut state = if seed == 0 { 0xC0FFEE_u64 } else { seed };
        for _ in 0..k_sim {
            let mut v = Vec::with_capacity(dim);
            let mut norm_sq = 0.0f32;
            for _ in 0..dim {
                // xorshift64*
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1.0, 1.0).
                let f = (state as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                v.push(f);
                norm_sq += f * f;
            }
            let norm = norm_sq.sqrt().max(1e-10);
            for x in &mut v {
                *x /= norm;
            }
            anchors.push(v);
        }
        Self {
            anchors,
            dim,
            k_sim,
        }
    }

    /// Encode a multi-vector (list of `dim`-dimensional tokens) into a
    /// single `k_sim`-dimensional FDE. Empty input yields the zero vector.
    pub fn encode(&self, tokens: &[Vec<f32>]) -> Vec<f32> {
        let mut fde = vec![0.0f32; self.k_sim];
        if tokens.is_empty() {
            return fde;
        }
        for tok in tokens {
            if tok.len() != self.dim {
                continue;
            }
            // Find argmax anchor.
            let mut best_idx = 0usize;
            let mut best_dot = f32::MIN;
            for (i, a) in self.anchors.iter().enumerate() {
                let d: f32 = tok.iter().zip(a).map(|(t, a)| t * a).sum();
                if d > best_dot {
                    best_dot = d;
                    best_idx = i;
                }
            }
            // Accumulate the projection onto that anchor.
            fde[best_idx] += best_dot;
        }
        fde
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn deterministic_anchors_for_same_seed() {
        let a = MuveraEncoder::new(8, 16, 42);
        let b = MuveraEncoder::new(8, 16, 42);
        assert_eq!(a.anchors, b.anchors);
    }

    #[test]
    fn encode_empty_yields_zero() {
        let enc = MuveraEncoder::new(4, 16, 1);
        let fde = enc.encode(&[]);
        assert_eq!(fde.len(), 16);
        assert_eq!(norm(&fde), 0.0);
    }

    #[test]
    fn similar_multi_vectors_have_similar_fdes() {
        // Two documents with very similar token distributions should
        // produce nearby FDEs.
        let enc = MuveraEncoder::new(4, 32, 7);
        let doc_a = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let doc_b = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.95, 0.05, 0.0, 0.0],
        ];
        let unrelated = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let fa = enc.encode(&doc_a);
        let fb = enc.encode(&doc_b);
        let fu = enc.encode(&unrelated);
        let cos = |x: &[f32], y: &[f32]| -> f32 {
            let dot: f32 = x.iter().zip(y).map(|(a, b)| a * b).sum();
            let denom = norm(x) * norm(y);
            if denom < 1e-10 { 0.0 } else { dot / denom }
        };
        let sim_ab = cos(&fa, &fb);
        let sim_au = cos(&fa, &fu);
        assert!(sim_ab > sim_au, "similar docs should be closer: ab={sim_ab} au={sim_au}");
    }
}
