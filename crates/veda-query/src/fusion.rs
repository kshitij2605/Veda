use std::collections::HashMap;

use uuid::Uuid;

use crate::search::ScoredRecord;

/// Fusion strategy for combining multi-vector search results.
///
/// **Choosing**: RRF (k=60) is the unsupervised default every
/// major vector DB ships. Convex combination beats RRF on every
/// BEIR slice when you have ≥50 labelled queries to tune weights
/// (Bruch et al., TOIS 2023). DBSF is the cheaper label-free
/// middle ground — uses each leg's distribution to normalise
/// rather than rank. See
/// `research/2026-05-19-sota-audit/06-sparse-fusion-rerank.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))
    Rrf { k: u32 },
    /// Weighted sum of min-max-normalized similarities (1 -
    /// normalized_distance). Weights are taken from the
    /// per-field `weight` field and applied as-is (no
    /// normalisation of the weight vector).
    WeightedSum,
    /// Convex combination: same shape as `WeightedSum`, but the
    /// weights are auto-normalised to sum to 1.0. The label-tuned
    /// variant that beats RRF on supervised retrieval benchmarks.
    /// If all weights are zero the strategy falls back to
    /// uniform 1/N per field.
    ConvexCombination,
    /// Distribution-Based Score Fusion (Qdrant 1.11+,
    /// Mar 2024). Normalises each leg's distances using the
    /// `[μ - 3σ, μ + 3σ]` range, converts to similarity, then
    /// sums. Robust to legs producing differently-scaled scores
    /// (e.g. cosine vs Euclidean vs BM25). Label-free.
    Dbsf,
    /// Only return results appearing in ALL field results, scored via RRF.
    Intersection,
}

/// A query over multiple vector fields.
#[derive(Debug, Clone)]
pub struct MultiVectorQuery {
    pub queries: Vec<FieldQuery>,
    pub fusion: FusionStrategy,
    pub limit: usize,
}

/// A single field's query within a multi-vector query.
#[derive(Debug, Clone)]
pub struct FieldQuery {
    pub field: String,
    pub vector: Vec<f32>,
    pub weight: f32, // used by WeightedSum
}

/// Fused result with combined score.
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub id: Uuid,
    pub segment_id: Uuid,
    pub doc_id: u32,
    pub score: f32,                       // fused score (higher = better)
    pub field_scores: Vec<(String, f32)>, // per-field distances for debugging
}

/// Accumulator for per-document data during fusion.
struct DocAccumulator {
    segment_id: Uuid,
    doc_id: u32,
    /// Per-field: (field_index, 1-based rank, raw distance)
    field_entries: Vec<(usize, usize, f32)>,
}

/// Fuse results from multiple vector field searches.
///
/// `field_results` contains `(field_name, results)` per field.
/// `weights` contains per-field weights (used by `WeightedSum`).
/// Returns at most `limit` fused results sorted by score descending.
pub fn fuse_results(
    field_results: &[(String, Vec<ScoredRecord>)],
    strategy: FusionStrategy,
    weights: &[f32],
    limit: usize,
) -> Vec<FusedResult> {
    if field_results.is_empty() {
        return Vec::new();
    }

    // Build a per-document accumulator keyed by record UUID.
    let mut doc_scores: HashMap<Uuid, DocAccumulator> = HashMap::new();

    for (field_idx, (_field_name, results)) in field_results.iter().enumerate() {
        for (rank, result) in results.iter().enumerate() {
            let entry = doc_scores.entry(result.id).or_insert_with(|| DocAccumulator {
                segment_id: result.segment_id,
                doc_id: result.doc_id,
                field_entries: Vec::new(),
            });
            entry.field_entries.push((field_idx, rank + 1, result.distance));
        }
    }

    match strategy {
        FusionStrategy::Rrf { k } => fuse_rrf(&doc_scores, field_results, k, limit),
        FusionStrategy::WeightedSum => {
            fuse_weighted_sum(&doc_scores, field_results, weights, limit)
        }
        FusionStrategy::ConvexCombination => {
            // Normalise weights to sum to 1 then reuse the
            // weighted-sum kernel.
            let n = field_results.len();
            let normalised: Vec<f32> = normalise_weights(weights, n);
            fuse_weighted_sum(&doc_scores, field_results, &normalised, limit)
        }
        FusionStrategy::Dbsf => fuse_dbsf(&doc_scores, field_results, limit),
        FusionStrategy::Intersection => {
            fuse_intersection(&doc_scores, field_results, limit)
        }
    }
}

/// Auto-normalise a weight vector so it sums to 1. Missing
/// entries are filled with 0; all-zero input falls back to
/// uniform 1/n.
fn normalise_weights(weights: &[f32], n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let mut out: Vec<f32> = (0..n)
        .map(|i| weights.get(i).copied().unwrap_or(0.0).max(0.0))
        .collect();
    let sum: f32 = out.iter().sum();
    if sum > f32::EPSILON {
        for w in &mut out {
            *w /= sum;
        }
    } else {
        let uniform = 1.0 / n as f32;
        for w in &mut out {
            *w = uniform;
        }
    }
    out
}

/// Reciprocal Rank Fusion.
///
/// For each document `d` appearing in any result list:
///   score(d) = sum over all fields i: 1.0 / (k + rank_i(d))
/// where rank_i(d) is the 1-based rank of `d` in field i's results.
/// If `d` does not appear in field i, that field is skipped.
fn fuse_rrf(
    doc_scores: &HashMap<Uuid, DocAccumulator>,
    field_results: &[(String, Vec<ScoredRecord>)],
    k: u32,
    limit: usize,
) -> Vec<FusedResult> {
    let mut results: Vec<FusedResult> = doc_scores
        .iter()
        .map(|(id, acc)| {
            let score: f32 = acc
                .field_entries
                .iter()
                .map(|&(_field_idx, rank, _dist)| 1.0 / (k as f32 + rank as f32))
                .sum();

            let field_scores = acc
                .field_entries
                .iter()
                .map(|&(field_idx, _rank, dist)| (field_results[field_idx].0.clone(), dist))
                .collect();

            FusedResult {
                id: *id,
                segment_id: acc.segment_id,
                doc_id: acc.doc_id,
                score,
                field_scores,
            }
        })
        .collect();

    // Sort by score descending (higher = better). Break ties by id for determinism.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Weighted sum of normalized scores.
///
/// 1. Normalize each field's distances to [0, 1].
/// 2. Convert to similarity: sim = 1.0 - norm_dist.
/// 3. score(d) = sum(weight_i * sim_i(d)).
fn fuse_weighted_sum(
    doc_scores: &HashMap<Uuid, DocAccumulator>,
    field_results: &[(String, Vec<ScoredRecord>)],
    weights: &[f32],
    limit: usize,
) -> Vec<FusedResult> {
    let num_fields = field_results.len();

    // Compute min/max distance per field for normalization.
    let mut min_dists = vec![f32::MAX; num_fields];
    let mut max_dists = vec![f32::MIN; num_fields];

    // Compute min/max from all documents' entries.
    for acc in doc_scores.values() {
        for &(field_idx, _rank, dist) in &acc.field_entries {
            if dist < min_dists[field_idx] {
                min_dists[field_idx] = dist;
            }
            if dist > max_dists[field_idx] {
                max_dists[field_idx] = dist;
            }
        }
    }

    let mut results: Vec<FusedResult> = doc_scores
        .iter()
        .map(|(id, acc)| {
            let mut score: f32 = 0.0;

            for &(field_idx, _rank, dist) in &acc.field_entries {
                let range = max_dists[field_idx] - min_dists[field_idx];
                let norm_dist = if range > f32::EPSILON {
                    (dist - min_dists[field_idx]) / range
                } else {
                    0.0 // all distances are the same -> normalized to 0 -> similarity = 1
                };
                let similarity = 1.0 - norm_dist;
                let w = if field_idx < weights.len() {
                    weights[field_idx]
                } else {
                    1.0
                };
                score += w * similarity;
            }

            let field_scores = acc
                .field_entries
                .iter()
                .map(|&(field_idx, _rank, dist)| (field_results[field_idx].0.clone(), dist))
                .collect();

            FusedResult {
                id: *id,
                segment_id: acc.segment_id,
                doc_id: acc.doc_id,
                score,
                field_scores,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Intersection fusion: only return results appearing in ALL fields, scored via RRF.
fn fuse_intersection(
    doc_scores: &HashMap<Uuid, DocAccumulator>,
    field_results: &[(String, Vec<ScoredRecord>)],
    limit: usize,
) -> Vec<FusedResult> {
    let num_fields = field_results.len();
    let default_k: u32 = 60;

    let mut results: Vec<FusedResult> = doc_scores
        .iter()
        .filter(|(_id, acc)| {
            // Check that this document appears in every field.
            let mut seen_fields = vec![false; num_fields];
            for &(field_idx, _rank, _dist) in &acc.field_entries {
                seen_fields[field_idx] = true;
            }
            seen_fields.iter().all(|&seen| seen)
        })
        .map(|(id, acc)| {
            // Score using RRF with default k.
            let score: f32 = acc
                .field_entries
                .iter()
                .map(|&(_field_idx, rank, _dist)| 1.0 / (default_k as f32 + rank as f32))
                .sum();

            let field_scores = acc
                .field_entries
                .iter()
                .map(|&(field_idx, _rank, dist)| (field_results[field_idx].0.clone(), dist))
                .collect();

            FusedResult {
                id: *id,
                segment_id: acc.segment_id,
                doc_id: acc.doc_id,
                score,
                field_scores,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Distribution-Based Score Fusion (Qdrant 1.11+, Mar 2024).
///
/// For each field, compute mean μ and stddev σ of the per-result
/// distances. Map each distance into `[0, 1]` using the
/// `[μ - 3σ, μ + 3σ]` range (clamped), invert to similarity, then
/// sum the per-field similarities for each document. Missing
/// fields contribute zero.
fn fuse_dbsf(
    doc_scores: &HashMap<Uuid, DocAccumulator>,
    field_results: &[(String, Vec<ScoredRecord>)],
    limit: usize,
) -> Vec<FusedResult> {
    let num_fields = field_results.len();

    // Per-field μ and σ over the distances actually returned for
    // that field. Use the field's own results vec rather than
    // walking doc_scores so we don't count cross-field membership.
    let mut means = vec![0.0_f32; num_fields];
    let mut stddevs = vec![0.0_f32; num_fields];
    for (i, (_name, results)) in field_results.iter().enumerate() {
        if results.is_empty() {
            continue;
        }
        let n = results.len() as f32;
        let sum: f32 = results.iter().map(|r| r.distance).sum();
        let mean = sum / n;
        let var: f32 = results
            .iter()
            .map(|r| (r.distance - mean).powi(2))
            .sum::<f32>()
            / n;
        means[i] = mean;
        stddevs[i] = var.sqrt();
    }

    let mut results: Vec<FusedResult> = doc_scores
        .iter()
        .map(|(id, acc)| {
            let mut score = 0.0_f32;
            for &(field_idx, _rank, dist) in &acc.field_entries {
                let mu = means[field_idx];
                let sigma = stddevs[field_idx];
                let lo = mu - 3.0 * sigma;
                let hi = mu + 3.0 * sigma;
                let range = hi - lo;
                let norm_dist = if range > f32::EPSILON {
                    ((dist - lo) / range).clamp(0.0, 1.0)
                } else {
                    // Degenerate distribution (all distances
                    // equal) — every result is equally good.
                    0.0
                };
                score += 1.0 - norm_dist;
            }
            let field_scores = acc
                .field_entries
                .iter()
                .map(|&(field_idx, _rank, dist)| (field_results[field_idx].0.clone(), dist))
                .collect();
            FusedResult {
                id: *id,
                segment_id: acc.segment_id,
                doc_id: acc.doc_id,
                score,
                field_scores,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scored(id: Uuid, distance: f32) -> ScoredRecord {
        ScoredRecord {
            id,
            segment_id: Uuid::new_v4(),
            doc_id: 0,
            distance,
        }
    }

    #[test]
    fn test_fuse_empty_input() {
        let results = fuse_results(&[], FusionStrategy::Rrf { k: 60 }, &[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuse_single_field_rrf() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let field_results = vec![(
            "embedding".to_string(),
            vec![
                make_scored(id1, 0.1),
                make_scored(id2, 0.5),
                make_scored(id3, 0.9),
            ],
        )];

        let results = fuse_results(&field_results, FusionStrategy::Rrf { k: 60 }, &[], 10);
        assert_eq!(results.len(), 3);
        // First result should be id1 (rank 1 -> highest RRF score).
        assert_eq!(results[0].id, id1);
        assert_eq!(results[1].id, id2);
        assert_eq!(results[2].id, id3);
    }

    /// Two-field fusion fixture: id_top is rank-1 in field A and
    /// rank-2 in field B; id_mid is rank-2/rank-1; id_low is in
    /// neither's top-2.
    fn two_field_fixture() -> (Uuid, Uuid, Uuid, Vec<(String, Vec<ScoredRecord>)>) {
        let id_top = Uuid::new_v4();
        let id_mid = Uuid::new_v4();
        let id_low = Uuid::new_v4();
        let field_a = (
            "dense".to_string(),
            vec![
                make_scored(id_top, 0.10),
                make_scored(id_mid, 0.40),
                make_scored(id_low, 0.80),
            ],
        );
        let field_b = (
            "sparse".to_string(),
            vec![
                make_scored(id_mid, 0.05),
                make_scored(id_top, 0.20),
                make_scored(id_low, 0.95),
            ],
        );
        (id_top, id_mid, id_low, vec![field_a, field_b])
    }

    #[test]
    fn convex_combination_normalises_weights() {
        let (id_top, id_mid, id_low, fixture) = two_field_fixture();
        let results_un = fuse_results(
            &fixture,
            FusionStrategy::ConvexCombination,
            &[3.0, 1.0],
            10,
        );
        let results_norm = fuse_results(
            &fixture,
            FusionStrategy::ConvexCombination,
            &[0.75, 0.25],
            10,
        );
        // Scores from `[3, 1]` and `[0.75, 0.25]` must be identical
        // because CC normalises the weight vector internally.
        assert_eq!(results_un.len(), results_norm.len());
        for (a, b) in results_un.iter().zip(results_norm.iter()) {
            assert_eq!(a.id, b.id);
            assert!(
                (a.score - b.score).abs() < 1e-5,
                "CC must be invariant to weight scale: {} vs {}",
                a.score,
                b.score
            );
        }
        // Sanity: id_top is best on both fields, id_low worst on both.
        assert_eq!(results_un[0].id, id_top);
        // id_mid is middle on field A, top on field B. id_low is bottom on both.
        assert_eq!(results_un[1].id, id_mid);
        assert_eq!(results_un[2].id, id_low);
    }

    #[test]
    fn convex_combination_zero_weights_fall_back_uniform() {
        let (id_top, _, id_low, fixture) = two_field_fixture();
        let r = fuse_results(
            &fixture,
            FusionStrategy::ConvexCombination,
            &[0.0, 0.0],
            10,
        );
        assert_eq!(r.len(), 3);
        // Equivalent to (0.5, 0.5) — id_top dominates.
        assert_eq!(r[0].id, id_top);
        assert_eq!(r.last().unwrap().id, id_low);
    }

    #[test]
    fn dbsf_handles_differently_scaled_legs() {
        // Construct a fixture where field B uses a wildly
        // different distance scale (e.g. BM25 vs cosine).
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();
        let field_dense = (
            "dense".to_string(),
            vec![
                make_scored(id_a, 0.02),
                make_scored(id_b, 0.04),
                make_scored(id_c, 0.08),
            ],
        );
        let field_bm25 = (
            "bm25".to_string(),
            vec![
                make_scored(id_c, 20.0),
                make_scored(id_a, 30.0),
                make_scored(id_b, 40.0),
            ],
        );
        let fixture = vec![field_dense, field_bm25];
        let r = fuse_results(&fixture, FusionStrategy::Dbsf, &[], 10);
        // All three appear; sort order is implementation-defined
        // but every score must be finite + bounded.
        assert_eq!(r.len(), 3);
        for hit in &r {
            assert!(hit.score.is_finite());
            // Each leg contributes a 0..1 similarity, so the
            // total over 2 legs must be in 0..2.
            assert!((0.0..=2.0).contains(&hit.score));
        }
        // The top hit by DBSF on this fixture is the one with the
        // best joint (μ-3σ, μ+3σ)-normalised position. Verify
        // ordering is stable (no ties producing UB).
        for w in r.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn dbsf_equal_distances_collapse_to_zero_diff() {
        // If σ = 0 the normaliser falls back to "all equal";
        // every doc gets the same per-leg sim contribution.
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let field = (
            "f".to_string(),
            vec![make_scored(id_a, 0.5), make_scored(id_b, 0.5)],
        );
        let r = fuse_results(&[field], FusionStrategy::Dbsf, &[], 10);
        assert_eq!(r.len(), 2);
        assert!(
            (r[0].score - r[1].score).abs() < 1e-5,
            "degenerate σ=0 must yield equal scores"
        );
    }
}
