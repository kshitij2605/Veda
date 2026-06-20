use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::NamedTempFile;
use veda_core::types::DistanceMetric;
use veda_index::diskann::DiskAnnIndex;
use veda_index::flat::FlatIndex;
use veda_index::traits::VectorIndex;

/// Generate `count` random f32 vectors of the given dimension.
fn random_vectors(rng: &mut StdRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Compute recall@k: the fraction of ground-truth top-k results that appear
/// in the approximate result set, averaged over all queries.
fn compute_recall(approx_results: &[Vec<u32>], flat_results: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(approx_results.len(), flat_results.len());
    let mut total_recall = 0.0;
    for (approx, flat) in approx_results.iter().zip(flat_results.iter()) {
        let approx_set: HashSet<u32> = approx.iter().take(k).copied().collect();
        let flat_set: HashSet<u32> = flat.iter().take(k).copied().collect();
        let intersection = approx_set.intersection(&flat_set).count();
        total_recall += intersection as f64 / k as f64;
    }
    total_recall / approx_results.len() as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Test 1: Recall@10 on 5000 vectors, dim 64, Cosine
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_recall_5k_cosine() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 64;
    let n = 5_000;
    let k = 10;
    let num_queries = 100;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann = DiskAnnIndex::with_params(
        DistanceMetric::Cosine,
        dim as u32,
        64,  // r
        100, // l_build
        1.2, // alpha
        100, // l_search
    );
    diskann.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut diskann_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let dr = diskann.search(query, k).unwrap();
        let fr = flat.search(query, k).unwrap();
        diskann_results.push(dr.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(fr.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&diskann_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (Cosine, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.85,
        "Recall@{k} = {recall:.4} is below 85% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: Different metrics — Euclidean
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_recall_euclidean() {
    let mut rng = StdRng::seed_from_u64(7);
    let dim = 64;
    let n = 5_000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann =
        DiskAnnIndex::with_params(DistanceMetric::Euclidean, dim as u32, 64, 100, 1.2, 100);
    diskann.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut diskann_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let dr = diskann.search(query, k).unwrap();
        let fr = flat.search(query, k).unwrap();
        diskann_results.push(dr.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(fr.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&diskann_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (Euclidean, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.85,
        "Recall@{k} = {recall:.4} is below 85% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2b: Different metrics — DotProduct
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_recall_dot_product() {
    let mut rng = StdRng::seed_from_u64(99);
    let dim = 64;
    let n = 5_000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann =
        DiskAnnIndex::with_params(DistanceMetric::DotProduct, dim as u32, 64, 100, 1.2, 100);
    diskann.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::DotProduct, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut diskann_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let dr = diskann.search(query, k).unwrap();
        let fr = flat.search(query, k).unwrap();
        diskann_results.push(dr.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(fr.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&diskann_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (DotProduct, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.85,
        "Recall@{k} = {recall:.4} is below 85% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: Delete — deleted vectors excluded from results
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_delete_excludes_from_results() {
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 32;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann = DiskAnnIndex::new(DistanceMetric::Euclidean, dim as u32);
    diskann.build(&refs).unwrap();

    // Delete 10% of vectors.
    let num_deleted = n / 10;
    let deleted_ids: HashSet<u32> = (0..num_deleted).map(|i| (i * 10) as u32).collect();
    for &id in &deleted_ids {
        diskann.delete(id).unwrap();
    }

    assert_eq!(diskann.len(), n - deleted_ids.len());

    // Search and verify no deleted IDs appear.
    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = diskann.search(query, 20).unwrap();
        for r in &results {
            assert!(
                !deleted_ids.contains(&r.id),
                "Deleted ID {} appeared in search results",
                r.id
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: Save/load round-trip
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_save_load_roundtrip() {
    let mut rng = StdRng::seed_from_u64(88);
    let dim = 64;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann = DiskAnnIndex::new(DistanceMetric::Cosine, dim as u32);
    diskann.build(&refs).unwrap();
    diskann.delete(5).unwrap();
    diskann.delete(100).unwrap();

    let queries = random_vectors(&mut rng, 10, dim);
    let results_before: Vec<Vec<_>> = queries
        .iter()
        .map(|q| diskann.search(q, 10).unwrap())
        .collect();

    let tmp = NamedTempFile::new().unwrap();
    diskann.save(tmp.path()).unwrap();

    let loaded = DiskAnnIndex::load(tmp.path(), DistanceMetric::Cosine, dim as u32).unwrap();

    let results_after: Vec<Vec<_>> = queries
        .iter()
        .map(|q| loaded.search(q, 10).unwrap())
        .collect();

    for (before, after) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a.id, b.id, "ID mismatch after load");
            assert!(
                (a.distance - b.distance).abs() < 1e-6,
                "Distance mismatch after load"
            );
        }
    }

    assert_eq!(loaded.len(), diskann.len());
    assert_eq!(loaded.metric(), DistanceMetric::Cosine);
    assert_eq!(loaded.dimension(), dim as u32);
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: Parameter tuning — higher L_search → higher recall
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_l_search_tuning() {
    let mut rng = StdRng::seed_from_u64(33);
    let dim = 64;
    let n = 5_000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann =
        DiskAnnIndex::with_params(DistanceMetric::Cosine, dim as u32, 64, 100, 1.2, 100);
    diskann.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);

    let flat_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| flat.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();

    // Test with low L_search.
    diskann.set_l_search(16);
    let low_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| diskann.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();
    let recall_low = compute_recall(&low_results, &flat_results, k);

    // Test with high L_search.
    diskann.set_l_search(256);
    let high_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| diskann.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();
    let recall_high = compute_recall(&high_results, &flat_results, k);

    eprintln!("l_search=16: recall={recall_low:.4}, l_search=256: recall={recall_high:.4}");
    assert!(
        recall_high >= recall_low,
        "Higher l_search should have equal or better recall: {recall_high:.4} < {recall_low:.4}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 6: Edge cases
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_index() {
    let diskann = DiskAnnIndex::new(DistanceMetric::Cosine, 4);
    assert!(diskann.is_empty());
    assert_eq!(diskann.len(), 0);

    let results = diskann.search(&[1.0, 2.0, 3.0, 4.0], 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_single_vector() {
    let mut diskann = DiskAnnIndex::new(DistanceMetric::Euclidean, 3);
    diskann.build(&[&[1.0, 2.0, 3.0]]).unwrap();

    assert_eq!(diskann.len(), 1);

    let results = diskann.search(&[1.0, 2.0, 3.0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!(results[0].distance < 1e-6);
}

#[test]
fn test_k_equals_1() {
    let mut rng = StdRng::seed_from_u64(11);
    let dim = 16;
    let n = 100;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann = DiskAnnIndex::new(DistanceMetric::Cosine, dim as u32);
    diskann.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let dr = diskann.search(&query, 1).unwrap();
    let fr = flat.search(&query, 1).unwrap();

    assert_eq!(dr.len(), 1);
    assert_eq!(dr[0].id, fr[0].id);
}

#[test]
fn test_k_zero_returns_empty() {
    let mut diskann = DiskAnnIndex::new(DistanceMetric::Cosine, 3);
    diskann.build(&[&[1.0, 2.0, 3.0]]).unwrap();
    let results = diskann.search(&[1.0, 2.0, 3.0], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_insert_returns_error() {
    let mut diskann = DiskAnnIndex::new(DistanceMetric::Cosine, 3);
    let result = diskann.insert(0, &[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_dimension_validation() {
    let mut diskann = DiskAnnIndex::new(DistanceMetric::Cosine, 4);
    assert!(diskann.build(&[&[1.0, 2.0, 3.0]]).is_err()); // wrong dim
    assert!(diskann.search(&[1.0, 2.0], 1).is_err()); // wrong dim
}

#[test]
fn test_results_sorted_ascending() {
    let mut rng = StdRng::seed_from_u64(44);
    let dim = 32;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut diskann = DiskAnnIndex::new(DistanceMetric::Euclidean, dim as u32);
    diskann.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = diskann.search(&query, 20).unwrap();

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
fn test_metric_and_dimension_accessors() {
    let idx = DiskAnnIndex::new(DistanceMetric::Manhattan, 256);
    assert_eq!(idx.metric(), DistanceMetric::Manhattan);
    assert_eq!(idx.dimension(), 256);
}
