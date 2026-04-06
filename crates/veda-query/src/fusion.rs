use std::collections::HashMap;

use uuid::Uuid;

use crate::search::ScoredRecord;

/// Fusion strategy for combining multi-vector search results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))
    Rrf { k: u32 },
    /// Weighted sum of normalized scores.
    WeightedSum,
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
        FusionStrategy::Intersection => {
            fuse_intersection(&doc_scores, field_results, limit)
        }
    }
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
}
