//! Light-ColPali token reduction.
//!
//! At index time, cluster each document's token vectors and keep one
//! representative per cluster (the token nearest the cluster centroid).
//! Reduces ColPali / ColQwen multi-vector storage by ~10× while retaining
//! >98% NDCG@5 (Light-ColPali, ACL 2025 Findings).
//!
//! Complements MUVERA (recall-stage encoding) and PLAID (centroid
//! pruning): MUVERA reduces query-time work, PLAID prunes the candidate
//! set, Light-ColPali reduces *per-document storage*.

use veda_core::simd;

/// Reduce the token list of a single document by k-means clustering and
/// keeping one representative per cluster.
///
/// `target_fraction` is the desired output size as a fraction of input —
/// e.g. `0.1` keeps ~10% of tokens. The minimum is 1 token.
///
/// Deterministic given the same input: initialisation picks every
/// `n/k`-th token.
pub fn reduce_tokens(tokens: &[Vec<f32>], target_fraction: f32) -> Vec<Vec<f32>> {
    let n = tokens.len();
    if n == 0 {
        return Vec::new();
    }
    let k = ((n as f32 * target_fraction).ceil() as usize).clamp(1, n);
    if k >= n {
        return tokens.to_vec();
    }
    let dim = tokens[0].len();
    if dim == 0 {
        return Vec::new();
    }

    // Init: stride-pick.
    let stride = n / k;
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| tokens[i * stride.max(1)].clone()).collect();

    // Lloyd iterations.
    for _ in 0..10 {
        let mut sums: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; dim]).collect();
        let mut counts = vec![0u64; k];
        for tok in tokens {
            let idx = nearest_centroid(tok, &centroids);
            for (i, &v) in tok.iter().enumerate() {
                sums[idx][i] += v as f64;
            }
            counts[idx] += 1;
        }
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

    // For each cluster, pick the token closest to its centroid.
    let mut best_for: Vec<Option<(usize, f32)>> = vec![None; k];
    for (i, tok) in tokens.iter().enumerate() {
        let idx = nearest_centroid(tok, &centroids);
        let d = simd::cosine_distance(tok, &centroids[idx]);
        match best_for[idx] {
            None => best_for[idx] = Some((i, d)),
            Some((_, best_d)) if d < best_d => best_for[idx] = Some((i, d)),
            _ => {}
        }
    }
    best_for
        .into_iter()
        .flatten()
        .map(|(i, _)| tokens[i].clone())
        .collect()
}

fn nearest_centroid(t: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = simd::cosine_distance(t, c);
        if d < best {
            best = d;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty() {
        let out = reduce_tokens(&[], 0.1);
        assert!(out.is_empty());
    }

    #[test]
    fn target_larger_than_n_returns_all() {
        let tokens = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let out = reduce_tokens(&tokens, 1.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn reduction_keeps_representatives_of_each_cluster() {
        // Three clear clusters; reduce to 3 tokens → one per cluster.
        let tokens: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.95, 0.05, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.95, 0.05, 0.0],
            vec![0.0, 0.9, 0.0, 0.1],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.95, 0.05],
            vec![0.0, 0.0, 0.9, 0.1],
        ];
        let out = reduce_tokens(&tokens, 0.333);
        assert_eq!(out.len(), 3, "expected 3 reps, got {}", out.len());
    }
}
