use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::NamedTempFile;
use veda_core::types::DistanceMetric;
use veda_index::distance::compute_distance;
use veda_index::flat::FlatIndex;
use veda_index::traits::VectorIndex;

/// Generate `count` random f32 vectors of the given dimension.
fn random_vectors(rng: &mut StdRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Brute-force ground-truth: compute all distances and return the `k` smallest.
fn ground_truth_search(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
    deleted: &HashSet<u32>,
) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| !deleted.contains(&(*i as u32)))
        .map(|(i, v)| (i as u32, compute_distance(query, v, metric)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_index_search_returns_empty() {
    let idx = FlatIndex::new(DistanceMetric::Cosine, 4);
    let results = idx.search(&[0.1, 0.2, 0.3, 0.4], 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_cosine_1000_vectors_top10() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;
    let vecs = random_vectors(&mut rng, 1000, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    idx.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 10).unwrap();
    let truth = ground_truth_search(&vecs, &query, 10, DistanceMetric::Cosine, &HashSet::new());

    assert_eq!(results.len(), 10);
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0, "id mismatch at ground truth comparison");
        assert!(
            (r.distance - t.1).abs() < 1e-5,
            "distance mismatch: {} vs {}",
            r.distance,
            t.1
        );
    }
}

#[test]
fn test_euclidean_metric() {
    let mut rng = StdRng::seed_from_u64(123);
    let dim = 64;
    let vecs = random_vectors(&mut rng, 500, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 5).unwrap();
    let truth = ground_truth_search(&vecs, &query, 5, DistanceMetric::Euclidean, &HashSet::new());

    assert_eq!(results.len(), 5);
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0);
    }
}

#[test]
fn test_dot_product_metric() {
    let mut rng = StdRng::seed_from_u64(7);
    let dim = 32;
    let vecs = random_vectors(&mut rng, 200, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::DotProduct, dim as u32);
    idx.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 5).unwrap();
    let truth = ground_truth_search(
        &vecs,
        &query,
        5,
        DistanceMetric::DotProduct,
        &HashSet::new(),
    );

    assert_eq!(results.len(), 5);
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0);
    }
    // For dot product, lower (more negative) distance = higher raw dot product = more similar
    assert!(results[0].distance <= results[1].distance);
}

#[test]
fn test_manhattan_metric() {
    let mut rng = StdRng::seed_from_u64(99);
    let dim = 16;
    let vecs = random_vectors(&mut rng, 300, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Manhattan, dim as u32);
    idx.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 5).unwrap();
    let truth = ground_truth_search(&vecs, &query, 5, DistanceMetric::Manhattan, &HashSet::new());

    assert_eq!(results.len(), 5);
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0);
    }
}

#[test]
fn test_build_vs_incremental_insert_same_results() {
    let mut rng = StdRng::seed_from_u64(55);
    let dim = 32;
    let vecs = random_vectors(&mut rng, 100, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    // Build path
    let mut idx_build = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    idx_build.build(&refs).unwrap();

    // Incremental insert path
    let mut idx_insert = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    for (i, v) in vecs.iter().enumerate() {
        idx_insert.insert(i as u32, v).unwrap();
    }

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results_build = idx_build.search(&query, 10).unwrap();
    let results_insert = idx_insert.search(&query, 10).unwrap();

    assert_eq!(results_build.len(), results_insert.len());
    for (a, b) in results_build.iter().zip(results_insert.iter()) {
        assert_eq!(a.id, b.id);
        assert!((a.distance - b.distance).abs() < 1e-6);
    }
}

#[test]
fn test_delete_vectors_excluded_from_results() {
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 16;
    let vecs = random_vectors(&mut rng, 50, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).unwrap();

    // Delete a few vectors
    let deleted_ids: HashSet<u32> = vec![0, 10, 20, 30, 40].into_iter().collect();
    for &id in &deleted_ids {
        idx.delete(id).unwrap();
    }

    assert_eq!(idx.len(), 45);

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 50).unwrap();

    // No deleted id should appear in results
    for r in &results {
        assert!(
            !deleted_ids.contains(&r.id),
            "Deleted id {} found in results",
            r.id
        );
    }
    assert_eq!(results.len(), 45);

    // Verify against ground truth
    let truth = ground_truth_search(&vecs, &query, 45, DistanceMetric::Euclidean, &deleted_ids);
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0);
    }
}

#[test]
fn test_save_load_roundtrip() {
    let mut rng = StdRng::seed_from_u64(88);
    let dim = 64;
    let vecs = random_vectors(&mut rng, 200, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    idx.build(&refs).unwrap();
    idx.delete(5).unwrap();
    idx.delete(100).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results_before = idx.search(&query, 10).unwrap();

    // Save and load
    let tmp = NamedTempFile::new().unwrap();
    idx.save(tmp.path()).unwrap();

    let loaded = FlatIndex::load(tmp.path(), DistanceMetric::Cosine, dim as u32).unwrap();
    let results_after = loaded.search(&query, 10).unwrap();

    assert_eq!(results_before.len(), results_after.len());
    for (a, b) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(a.id, b.id);
        assert!((a.distance - b.distance).abs() < 1e-6);
    }
    assert_eq!(loaded.len(), idx.len());
    assert_eq!(loaded.metric(), DistanceMetric::Cosine);
    assert_eq!(loaded.dimension(), dim as u32);
}

#[test]
fn test_k_greater_than_n_returns_all() {
    let mut rng = StdRng::seed_from_u64(33);
    let dim = 8;
    let vecs = random_vectors(&mut rng, 5, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).unwrap();

    // Delete one so there are only 4 non-deleted vectors
    idx.delete(2).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 100).unwrap();
    assert_eq!(results.len(), 4);
}

#[test]
fn test_dimension_validation_on_insert() {
    let mut idx = FlatIndex::new(DistanceMetric::Cosine, 4);
    let result = idx.insert(0, &[1.0, 2.0, 3.0]); // wrong dim (3 instead of 4)
    assert!(result.is_err());
}

#[test]
fn test_dimension_validation_on_search() {
    let mut idx = FlatIndex::new(DistanceMetric::Cosine, 4);
    idx.insert(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = idx.search(&[1.0, 2.0], 1); // wrong dim
    assert!(result.is_err());
}

#[test]
fn test_dimension_validation_on_build() {
    let mut idx = FlatIndex::new(DistanceMetric::Cosine, 4);
    let bad_vec: Vec<f32> = vec![1.0, 2.0]; // wrong dim
    let refs: Vec<&[f32]> = vec![bad_vec.as_slice()];
    let result = idx.build(&refs);
    assert!(result.is_err());
}

#[test]
fn test_realistic_rag_scenario_10000_dim768() {
    let mut rng = StdRng::seed_from_u64(2024);
    let dim = 768;
    let n = 10_000;
    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    idx.build(&refs).unwrap();
    assert_eq!(idx.len(), n);

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Results should be sorted by ascending distance
    for window in results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }

    // Verify these are actually the 10 nearest neighbors
    let truth = ground_truth_search(&vecs, &query, 10, DistanceMetric::Cosine, &HashSet::new());
    for (r, t) in results.iter().zip(truth.iter()) {
        assert_eq!(r.id, t.0);
    }
}

#[test]
fn test_results_sorted_ascending() {
    let mut rng = StdRng::seed_from_u64(11);
    let dim = 16;
    let vecs = random_vectors(&mut rng, 100, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut idx = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = idx.search(&query, 20).unwrap();

    for window in results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted: {} > {}",
            window[0].distance,
            window[1].distance
        );
    }
}

#[test]
fn test_len_and_is_empty() {
    let mut idx = FlatIndex::new(DistanceMetric::Cosine, 2);
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);

    idx.insert(0, &[1.0, 2.0]).unwrap();
    assert!(!idx.is_empty());
    assert_eq!(idx.len(), 1);

    idx.insert(1, &[3.0, 4.0]).unwrap();
    assert_eq!(idx.len(), 2);

    idx.delete(0).unwrap();
    assert_eq!(idx.len(), 1);
}

#[test]
fn test_metric_and_dimension_accessors() {
    let idx = FlatIndex::new(DistanceMetric::Manhattan, 256);
    assert_eq!(idx.metric(), DistanceMetric::Manhattan);
    assert_eq!(idx.dimension(), 256);
}
