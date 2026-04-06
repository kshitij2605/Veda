use std::collections::HashSet;

use uuid::Uuid;
use veda_query::fusion::{fuse_results, FusedResult, FusionStrategy};
use veda_query::search::ScoredRecord;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Create a `ScoredRecord` with a specific UUID, segment, doc_id, and distance.
fn make_record(id: Uuid, segment_id: Uuid, doc_id: u32, distance: f32) -> ScoredRecord {
    ScoredRecord {
        id,
        segment_id,
        doc_id,
        distance,
    }
}

/// Shorthand: create a record with a fresh segment_id.
fn record(id: Uuid, distance: f32) -> ScoredRecord {
    make_record(id, Uuid::new_v4(), 0, distance)
}

/// Assert that results are sorted by score descending.
fn assert_sorted_descending(results: &[FusedResult]) {
    for window in results.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "Results not sorted descending: {} < {}",
            window[0].score,
            window[1].score,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RRF Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_rrf_three_fields_overlap() {
    // 3 fields, each returning 20 results. Some documents overlap across fields.
    let all_ids: Vec<Uuid> = (0..50).map(|_| Uuid::new_v4()).collect();

    // Field A: ids 0..20
    let field_a: Vec<ScoredRecord> = (0..20)
        .map(|i| record(all_ids[i], i as f32 * 0.1))
        .collect();

    // Field B: ids 10..30 (overlap with A on ids 10..20)
    let field_b: Vec<ScoredRecord> = (10..30)
        .map(|i| record(all_ids[i], (i - 10) as f32 * 0.15))
        .collect();

    // Field C: ids 15..35 (overlap with A on 15..20, B on 15..30)
    let field_c: Vec<ScoredRecord> = (15..35)
        .map(|i| record(all_ids[i], (i - 15) as f32 * 0.12))
        .collect();

    let field_results = vec![
        ("text_embed".to_string(), field_a),
        ("image_embed".to_string(), field_b),
        ("audio_embed".to_string(), field_c),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    // Verify basic properties.
    assert_eq!(results.len(), 10);
    assert_sorted_descending(&results);

    // Documents appearing in multiple fields should rank higher.
    // ids 15..20 appear in all 3 fields; they should dominate the top results.
    let top_ids: HashSet<Uuid> = results.iter().map(|r| r.id).collect();
    let triple_overlap: HashSet<Uuid> = (15..20).map(|i| all_ids[i]).collect();

    // At least some of the triple-overlap docs should be in top 10.
    let overlap_in_top = top_ids.intersection(&triple_overlap).count();
    assert!(
        overlap_in_top >= 3,
        "Expected at least 3 triple-overlap docs in top 10, got {}",
        overlap_in_top,
    );

    // The top result should have the highest RRF score.
    assert!(results[0].score >= results[1].score);
}

#[test]
fn test_rrf_limit_works() {
    let ids: Vec<Uuid> = (0..20).map(|_| Uuid::new_v4()).collect();

    let field_a: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.1))
        .collect();

    let field_results = vec![("embed".to_string(), field_a)];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        5,
    );

    assert_eq!(results.len(), 5);
    assert_sorted_descending(&results);
}

#[test]
fn test_rrf_score_computation() {
    // Verify exact RRF score computation with k=60.
    let id_a = Uuid::new_v4();
    let id_b = Uuid::new_v4();

    // id_a appears at rank 1 in field 0, rank 2 in field 1.
    // id_b appears at rank 2 in field 0, rank 1 in field 1.
    let field_0 = vec![record(id_a, 0.1), record(id_b, 0.2)];
    let field_1 = vec![record(id_b, 0.1), record(id_a, 0.2)];

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    // Both should have the same score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    let expected = 1.0_f32 / 61.0 + 1.0 / 62.0;
    assert_eq!(results.len(), 2);
    assert!((results[0].score - expected).abs() < 1e-6);
    assert!((results[1].score - expected).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// WeightedSum Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_weighted_sum_two_fields() {
    // 2 fields with different weights (0.7, 0.3).
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    // Field 0 (weight 0.7): id1 is best (distance 0.1), id2 middle (0.5), id3 worst (1.0)
    let field_0 = vec![
        record(id1, 0.1),
        record(id2, 0.5),
        record(id3, 1.0),
    ];

    // Field 1 (weight 0.3): id3 is best (0.1), id2 middle (0.5), id1 worst (1.0)
    let field_1 = vec![
        record(id3, 0.1),
        record(id2, 0.5),
        record(id1, 1.0),
    ];

    let field_results = vec![
        ("text".to_string(), field_0),
        ("image".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::WeightedSum,
        &[0.7, 0.3],
        10,
    );

    assert_eq!(results.len(), 3);
    assert_sorted_descending(&results);

    // All scores should be positive.
    for r in &results {
        assert!(r.score > 0.0, "Score should be positive, got {}", r.score);
    }

    // id1 should rank higher because the higher-weighted field (0.7) has it
    // at distance 0.1 (best similarity).
    // id1 score: 0.7 * (1.0 - 0.0) + 0.3 * (1.0 - 1.0) = 0.7
    // id3 score: 0.7 * (1.0 - 1.0) + 0.3 * (1.0 - 0.0) = 0.3
    // id2 score: 0.7 * (1.0 - (0.5-0.1)/(1.0-0.1)) + 0.3 * (1.0 - (0.5-0.1)/(1.0-0.1))
    //          = 0.7 * (1.0 - 4/9) + 0.3 * (1.0 - 4/9)
    //          = (0.7 + 0.3) * (5/9) = 5/9 ~ 0.5556
    assert_eq!(results[0].id, id1, "id1 should be ranked first");
    assert!((results[0].score - 0.7).abs() < 1e-5, "Expected ~0.7, got {}", results[0].score);
}

#[test]
fn test_weighted_sum_all_scores_positive() {
    let ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

    let field_0: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.5))
        .collect();

    let field_1: Vec<ScoredRecord> = ids
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.3))
        .collect();

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::WeightedSum,
        &[0.6, 0.4],
        10,
    );

    assert_eq!(results.len(), 10);
    assert_sorted_descending(&results);

    for r in &results {
        assert!(r.score >= 0.0, "Score should be non-negative, got {}", r.score);
    }
}

#[test]
fn test_weighted_sum_higher_weight_dominates() {
    // A document that is best in the higher-weighted field but worst in the
    // lower-weighted field should still rank above a doc that is best in the
    // lower-weighted field but worst in the higher-weighted field.
    let id_text_best = Uuid::new_v4();
    let id_image_best = Uuid::new_v4();

    // Field 0 (weight 0.9): text_best at dist 0.0, image_best at dist 1.0
    let field_0 = vec![
        record(id_text_best, 0.0),
        record(id_image_best, 1.0),
    ];

    // Field 1 (weight 0.1): image_best at dist 0.0, text_best at dist 1.0
    let field_1 = vec![
        record(id_image_best, 0.0),
        record(id_text_best, 1.0),
    ];

    let field_results = vec![
        ("text".to_string(), field_0),
        ("image".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::WeightedSum,
        &[0.9, 0.1],
        10,
    );

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].id, id_text_best,
        "Text-best doc should rank first with 0.9 weight on text field"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Intersection Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_intersection_basic() {
    // 2 fields, 10 results each. Only 3 documents in common.
    let common_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
    let field_0_only: Vec<Uuid> = (0..7).map(|_| Uuid::new_v4()).collect();
    let field_1_only: Vec<Uuid> = (0..7).map(|_| Uuid::new_v4()).collect();

    let mut field_0 = Vec::new();
    for (i, &id) in common_ids.iter().enumerate() {
        field_0.push(record(id, i as f32 * 0.1));
    }
    for (i, &id) in field_0_only.iter().enumerate() {
        field_0.push(record(id, (i + 3) as f32 * 0.1));
    }

    let mut field_1 = Vec::new();
    for (i, &id) in common_ids.iter().enumerate() {
        field_1.push(record(id, i as f32 * 0.2));
    }
    for (i, &id) in field_1_only.iter().enumerate() {
        field_1.push(record(id, (i + 3) as f32 * 0.2));
    }

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Intersection,
        &[],
        10,
    );

    // Only 3 results should be returned (the intersection set).
    assert_eq!(results.len(), 3);

    let result_ids: HashSet<Uuid> = results.iter().map(|r| r.id).collect();
    let common_set: HashSet<Uuid> = common_ids.iter().copied().collect();
    assert_eq!(result_ids, common_set);

    assert_sorted_descending(&results);
}

#[test]
fn test_intersection_all_disjoint() {
    // Two fields with completely disjoint results.
    let field_0: Vec<ScoredRecord> = (0..5)
        .map(|i| record(Uuid::new_v4(), i as f32 * 0.1))
        .collect();

    let field_1: Vec<ScoredRecord> = (0..5)
        .map(|i| record(Uuid::new_v4(), i as f32 * 0.2))
        .collect();

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Intersection,
        &[],
        10,
    );

    assert!(results.is_empty(), "Disjoint results should yield empty intersection");
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_single_field_passthrough() {
    // Single field -> pass-through (RRF with one field = just rank order).
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

    let field: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.5))
        .collect();

    let field_results = vec![("embed".to_string(), field)];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    assert_eq!(results.len(), 5);
    assert_sorted_descending(&results);

    // Order should match original rank order: id[0] first (rank 1 = highest RRF).
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.id, ids[i], "Position {} mismatch", i);
    }
}

#[test]
fn test_empty_results_from_one_field() {
    // Empty results from one field -> RRF still works (skips missing field).
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

    let field_0: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.1))
        .collect();

    let field_1: Vec<ScoredRecord> = Vec::new(); // empty

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    // Should still return results from field 0.
    assert_eq!(results.len(), 5);
    assert_sorted_descending(&results);
}

#[test]
fn test_duplicate_uuid_across_fields() {
    // Same document searched in two embedding spaces (same UUID in both fields).
    let shared_id = Uuid::new_v4();

    let field_0 = vec![record(shared_id, 0.3)];
    let field_1 = vec![record(shared_id, 0.7)];

    let field_results = vec![
        ("text".to_string(), field_0),
        ("image".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    // Should produce exactly one fused result.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, shared_id);

    // It should have 2 field scores.
    assert_eq!(results[0].field_scores.len(), 2);

    // RRF score: 1/(60+1) + 1/(60+1) = 2/61
    let expected = 2.0_f32 / 61.0;
    assert!((results[0].score - expected).abs() < 1e-6);
}

#[test]
fn test_empty_input() {
    let results = fuse_results(&[], FusionStrategy::Rrf { k: 60 }, &[], 10);
    assert!(results.is_empty());
}

#[test]
fn test_limit_zero() {
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    let field: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32))
        .collect();

    let field_results = vec![("f0".to_string(), field)];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        0,
    );

    assert!(results.is_empty());
}

#[test]
fn test_field_scores_populated() {
    let id = Uuid::new_v4();

    let field_0 = vec![record(id, 0.25)];
    let field_1 = vec![record(id, 0.75)];

    let field_results = vec![
        ("text_embed".to_string(), field_0),
        ("image_embed".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        10,
    );

    assert_eq!(results.len(), 1);
    let field_scores = &results[0].field_scores;
    assert_eq!(field_scores.len(), 2);

    // Check that field names and distances are recorded correctly.
    let text_score = field_scores.iter().find(|(name, _)| name == "text_embed");
    let image_score = field_scores.iter().find(|(name, _)| name == "image_embed");

    assert!(text_score.is_some());
    assert!(image_score.is_some());
    assert!((text_score.unwrap().1 - 0.25).abs() < 1e-6);
    assert!((image_score.unwrap().1 - 0.75).abs() < 1e-6);
}

#[test]
fn test_intersection_with_empty_field() {
    // Intersection where one field is empty should return nothing.
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

    let field_0: Vec<ScoredRecord> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| record(id, i as f32 * 0.1))
        .collect();

    let field_1: Vec<ScoredRecord> = Vec::new();

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Intersection,
        &[],
        10,
    );

    assert!(
        results.is_empty(),
        "Intersection with empty field should be empty"
    );
}

#[test]
fn test_weighted_sum_single_element_fields() {
    // When each field has a single result with the same distance,
    // normalization should still work (range = 0, similarity = 1).
    let id = Uuid::new_v4();

    let field_0 = vec![record(id, 0.5)];
    let field_1 = vec![record(id, 0.5)];

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::WeightedSum,
        &[0.5, 0.5],
        10,
    );

    assert_eq!(results.len(), 1);
    // With range=0, norm_dist=0, sim=1. Score = 0.5*1 + 0.5*1 = 1.0
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "Expected score ~1.0, got {}",
        results[0].score,
    );
}

#[test]
fn test_rrf_documents_in_more_fields_rank_higher() {
    // A document appearing in 3 fields should rank higher than one in 1 field,
    // even if the single-field doc has rank 1 in that field.
    let id_triple = Uuid::new_v4();
    let id_single = Uuid::new_v4();

    // id_triple at rank 3 in each of 3 fields.
    // id_single at rank 1 in field 0 only.
    let field_0 = vec![
        record(id_single, 0.01),
        record(Uuid::new_v4(), 0.02),
        record(id_triple, 0.03),
    ];
    let field_1 = vec![
        record(Uuid::new_v4(), 0.01),
        record(Uuid::new_v4(), 0.02),
        record(id_triple, 0.03),
    ];
    let field_2 = vec![
        record(Uuid::new_v4(), 0.01),
        record(Uuid::new_v4(), 0.02),
        record(id_triple, 0.03),
    ];

    let field_results = vec![
        ("f0".to_string(), field_0),
        ("f1".to_string(), field_1),
        ("f2".to_string(), field_2),
    ];

    let results = fuse_results(
        &field_results,
        FusionStrategy::Rrf { k: 60 },
        &[],
        20,
    );

    // id_triple: 3 * 1/(60+3) = 3/63 ~ 0.0476
    // id_single: 1/(60+1) = 1/61 ~ 0.0164
    let triple_result = results.iter().find(|r| r.id == id_triple).unwrap();
    let single_result = results.iter().find(|r| r.id == id_single).unwrap();

    assert!(
        triple_result.score > single_result.score,
        "Triple-field doc ({}) should outrank single-field doc ({})",
        triple_result.score,
        single_result.score,
    );
}
