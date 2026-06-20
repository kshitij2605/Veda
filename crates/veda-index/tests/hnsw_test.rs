use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::NamedTempFile;
use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::traits::VectorIndex;

/// Generate `count` random f32 vectors of the given dimension.
fn random_vectors(rng: &mut StdRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Compute recall@k: the fraction of ground-truth top-k results that appear
/// in the approximate result set, averaged over all queries.
fn compute_recall(hnsw_results: &[Vec<u32>], flat_results: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(hnsw_results.len(), flat_results.len());
    let mut total_recall = 0.0;
    for (hnsw_res, flat_res) in hnsw_results.iter().zip(flat_results.iter()) {
        let hnsw_set: HashSet<u32> = hnsw_res.iter().take(k).copied().collect();
        let flat_set: HashSet<u32> = flat_res.iter().take(k).copied().collect();
        let intersection = hnsw_set.intersection(&flat_set).count();
        total_recall += intersection as f64 / k as f64;
    }
    total_recall / hnsw_results.len() as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Test 1: Recall@10 on 10K vectors, dim 128, Cosine — MOST IMPORTANT
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_recall_10k_cosine_dim128() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;
    let n = 10_000;
    let k = 10;
    let num_queries = 100;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    // Build HNSW index.
    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();

    // Build FlatIndex as ground truth.
    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    // Generate queries and compare.
    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut hnsw_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let hnsw_res = hnsw.search(query, k).unwrap();
        let flat_res = flat.search(query, k).unwrap();
        hnsw_results.push(hnsw_res.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(flat_res.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&hnsw_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (Cosine, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.90,
        "Recall@{k} = {recall:.4} is below 90% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: Small correctness test — all top-1 match FlatIndex
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_small_correctness_top1() {
    let mut rng = StdRng::seed_from_u64(123);
    let dim = 32;
    let n = 100;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    // Use each vector as a query — top-1 should be itself.
    for (i, vec) in vecs.iter().enumerate() {
        let hnsw_res = hnsw.search(vec, 1).unwrap();
        let flat_res = flat.search(vec, 1).unwrap();
        assert_eq!(
            hnsw_res[0].id, flat_res[0].id,
            "Top-1 mismatch for vector {i}: HNSW={}, Flat={}",
            hnsw_res[0].id, flat_res[0].id
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: Different metrics — Euclidean
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

    let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, dim as u32);
    hnsw.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut hnsw_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let hnsw_res = hnsw.search(query, k).unwrap();
        let flat_res = flat.search(query, k).unwrap();
        hnsw_results.push(hnsw_res.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(flat_res.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&hnsw_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (Euclidean, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.90,
        "Recall@{k} = {recall:.4} is below 90% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3b: Different metrics — DotProduct
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

    let mut hnsw = HnswIndex::new(DistanceMetric::DotProduct, dim as u32);
    hnsw.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::DotProduct, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut hnsw_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let hnsw_res = hnsw.search(query, k).unwrap();
        let flat_res = flat.search(query, k).unwrap();
        hnsw_results.push(hnsw_res.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(flat_res.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&hnsw_results, &flat_results, k);
    eprintln!("Recall@{k} on {n} vectors (DotProduct, dim {dim}): {recall:.4}");
    assert!(
        recall > 0.90,
        "Recall@{k} = {recall:.4} is below 90% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: Incremental insert — insert vectors one by one
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_incremental_insert_recall() {
    let mut rng = StdRng::seed_from_u64(55);
    let dim = 64;
    let n = 2_000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);

    // Build HNSW incrementally.
    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    for (i, v) in vecs.iter().enumerate() {
        hnsw.insert(i as u32, v).unwrap();
    }

    // Build FlatIndex for ground truth.
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut hnsw_results = Vec::new();
    let mut flat_results = Vec::new();

    for query in &queries {
        let hnsw_res = hnsw.search(query, k).unwrap();
        let flat_res = flat.search(query, k).unwrap();
        hnsw_results.push(hnsw_res.iter().map(|r| r.id).collect::<Vec<_>>());
        flat_results.push(flat_res.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let recall = compute_recall(&hnsw_results, &flat_results, k);
    eprintln!("Recall@{k} (incremental insert, {n} vectors): {recall:.4}");
    assert!(
        recall > 0.90,
        "Recall@{k} = {recall:.4} is below 90% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: Delete — deleted vectors don't appear in results
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_delete_excludes_from_results() {
    let mut rng = StdRng::seed_from_u64(77);
    let dim = 32;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, dim as u32);
    hnsw.build(&refs).unwrap();

    // Delete 10% of vectors.
    let num_deleted = n / 10;
    let deleted_ids: HashSet<u32> = (0..num_deleted).map(|i| (i * 10) as u32).collect();
    for &id in &deleted_ids {
        hnsw.delete(id).unwrap();
    }

    assert_eq!(hnsw.len(), n - deleted_ids.len());

    // Search and verify no deleted IDs appear.
    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = hnsw.search(query, 20).unwrap();
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
// Test 6: Save/load round-trip
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_save_load_roundtrip() {
    let mut rng = StdRng::seed_from_u64(88);
    let dim = 64;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();
    hnsw.delete(5).unwrap();
    hnsw.delete(100).unwrap();

    let queries = random_vectors(&mut rng, 10, dim);
    let results_before: Vec<Vec<_>> = queries
        .iter()
        .map(|q| hnsw.search(q, 10).unwrap())
        .collect();

    // Save and load.
    let tmp = NamedTempFile::new().unwrap();
    hnsw.save(tmp.path()).unwrap();

    let loaded = HnswIndex::load(tmp.path(), DistanceMetric::Cosine, dim as u32).unwrap();

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

    assert_eq!(loaded.len(), hnsw.len());
    assert_eq!(loaded.metric(), DistanceMetric::Cosine);
    assert_eq!(loaded.dimension(), dim as u32);
}

// ────────────────────────────────────────────────────────────────────────────
// Test 7: ef_search tuning — higher ef → better recall
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ef_search_tuning() {
    let mut rng = StdRng::seed_from_u64(33);
    let dim = 64;
    let n = 5_000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let queries = random_vectors(&mut rng, num_queries, dim);

    // Compute ground truth once.
    let flat_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| flat.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();

    // Test with low ef_search.
    hnsw.set_ef_search(16);
    let low_ef_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| hnsw.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();
    let recall_low = compute_recall(&low_ef_results, &flat_results, k);

    // Test with high ef_search.
    hnsw.set_ef_search(256);
    let high_ef_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| hnsw.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect();
    let recall_high = compute_recall(&high_ef_results, &flat_results, k);

    eprintln!("ef_search=16: recall={recall_low:.4}, ef_search=256: recall={recall_high:.4}");
    assert!(
        recall_high >= recall_low,
        "Higher ef_search should have equal or better recall: {recall_high:.4} < {recall_low:.4}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 8: Edge cases
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_index() {
    let hnsw = HnswIndex::new(DistanceMetric::Cosine, 4);
    assert!(hnsw.is_empty());
    assert_eq!(hnsw.len(), 0);

    let results = hnsw.search(&[1.0, 2.0, 3.0, 4.0], 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_single_vector() {
    let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, 3);
    hnsw.insert(0, &[1.0, 2.0, 3.0]).unwrap();

    assert_eq!(hnsw.len(), 1);

    let results = hnsw.search(&[1.0, 2.0, 3.0], 1).unwrap();
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

    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let hnsw_res = hnsw.search(&query, 1).unwrap();
    let flat_res = flat.search(&query, 1).unwrap();

    assert_eq!(hnsw_res.len(), 1);
    assert_eq!(hnsw_res[0].id, flat_res[0].id);
}

#[test]
fn test_k_zero_returns_empty() {
    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, 3);
    hnsw.insert(0, &[1.0, 2.0, 3.0]).unwrap();
    let results = hnsw.search(&[1.0, 2.0, 3.0], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_dimension_validation() {
    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, 4);
    assert!(hnsw.insert(0, &[1.0, 2.0, 3.0]).is_err()); // wrong dim
    assert!(hnsw.search(&[1.0, 2.0], 1).is_err()); // wrong dim
}

#[test]
fn test_results_sorted_ascending() {
    let mut rng = StdRng::seed_from_u64(44);
    let dim = 32;
    let n = 500;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, dim as u32);
    hnsw.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = hnsw.search(&query, 20).unwrap();

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
    let idx = HnswIndex::new(DistanceMetric::Manhattan, 256);
    assert_eq!(idx.metric(), DistanceMetric::Manhattan);
    assert_eq!(idx.dimension(), 256);
}
