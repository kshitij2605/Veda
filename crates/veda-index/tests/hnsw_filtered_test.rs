use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;
use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::traits::{SearchResult, VectorIndex};

/// Generate `count` random f32 vectors of the given dimension.
fn random_vectors(rng: &mut StdRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Build an HNSW index from a set of vectors.
fn build_hnsw(vecs: &[Vec<f32>], dim: usize, metric: DistanceMetric) -> HnswIndex {
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mut hnsw = HnswIndex::new(metric, dim as u32);
    hnsw.build(&refs).unwrap();
    hnsw
}

/// Compute recall: fraction of ground-truth results found in the approximate set.
fn recall(approx: &[SearchResult], truth: &[SearchResult], k: usize) -> f64 {
    let approx_ids: HashSet<u32> = approx.iter().take(k).map(|r| r.id).collect();
    let truth_ids: HashSet<u32> = truth.iter().take(k).map(|r| r.id).collect();
    if truth_ids.is_empty() {
        return 1.0;
    }
    approx_ids.intersection(&truth_ids).count() as f64 / truth_ids.len() as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Test 1: Basic filtered search — 50% filter, all results must be even IDs
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_basic_filtered_search() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 64;
    let n = 5000;
    let k = 10;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Cosine);

    // Filter: even-numbered IDs only (50%)
    let mut filter = RoaringBitmap::new();
    for id in (0..n as u32).step_by(2) {
        filter.insert(id);
    }

    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = hnsw.search_filtered(query, k, &filter).unwrap();
        assert!(results.len() <= k);
        for r in &results {
            assert!(
                r.id % 2 == 0,
                "Non-even ID {} found in filtered results",
                r.id
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: Filtered recall — compare against filtered FlatIndex, recall > 85%
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_filtered_recall() {
    let mut rng = StdRng::seed_from_u64(100);
    let dim = 64;
    let n = 3000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Euclidean);

    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    // Filter: 30% of IDs
    let mut filter = RoaringBitmap::new();
    for id in (0..n as u32).step_by(3) {
        filter.insert(id);
    }

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut total_recall = 0.0;

    for query in &queries {
        let hnsw_res = hnsw.search_filtered(query, k, &filter).unwrap();
        let flat_res = flat.search_filtered(query, k, &filter).unwrap();
        total_recall += recall(&hnsw_res, &flat_res, k);
    }

    let avg_recall = total_recall / num_queries as f64;
    eprintln!("Filtered recall@{k} (30% filter, {n} vectors): {avg_recall:.4}");
    assert!(
        avg_recall > 0.85,
        "Filtered recall@{k} = {avg_recall:.4} is below 85% threshold"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: High selectivity (>50%) — 80% of vectors in filter
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_high_selectivity_filter() {
    let mut rng = StdRng::seed_from_u64(200);
    let dim = 64;
    let n = 3000;
    let k = 10;
    let num_queries = 50;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Cosine);

    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    flat.build(&refs).unwrap();

    // Filter: 80% of IDs (0, 1, 2, 3 out of every 5)
    let mut filter = RoaringBitmap::new();
    for id in 0..n as u32 {
        if id % 5 != 4 {
            filter.insert(id);
        }
    }
    assert!(filter.len() as f64 / n as f64 > 0.5);

    let queries = random_vectors(&mut rng, num_queries, dim);
    let mut total_recall = 0.0;

    for query in &queries {
        let hnsw_res = hnsw.search_filtered(query, k, &filter).unwrap();
        let flat_res = flat.search_filtered(query, k, &filter).unwrap();

        // All results must be in the filter
        for r in &hnsw_res {
            assert!(filter.contains(r.id), "ID {} not in filter", r.id);
        }

        total_recall += recall(&hnsw_res, &flat_res, k);
    }

    let avg_recall = total_recall / num_queries as f64;
    eprintln!("High selectivity recall@{k} (80% filter): {avg_recall:.4}");
    assert!(
        avg_recall > 0.85,
        "High selectivity recall = {avg_recall:.4} below 85%"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: Medium selectivity (1-50%) — 10% of vectors in filter
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_medium_selectivity_filter() {
    let mut rng = StdRng::seed_from_u64(300);
    let dim = 64;
    let n = 3000;
    let k = 10;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Euclidean);

    // Filter: 10% of IDs
    let mut filter = RoaringBitmap::new();
    for id in (0..n as u32).step_by(10) {
        filter.insert(id);
    }
    let selectivity = filter.len() as f64 / n as f64;
    assert!(selectivity > 0.01 && selectivity <= 0.5);

    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = hnsw.search_filtered(query, k, &filter).unwrap();
        for r in &results {
            assert!(filter.contains(r.id), "ID {} not in filter", r.id);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: Low selectivity (<1%) — only 20 vectors out of 5000
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_low_selectivity_filter() {
    let mut rng = StdRng::seed_from_u64(400);
    let dim = 64;
    let n = 5000;
    let k = 10;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Euclidean);

    // Filter: only 20 specific IDs (0.4% selectivity)
    let mut filter = RoaringBitmap::new();
    let chosen: Vec<u32> = (0..20).map(|i| (i * 250) as u32).collect();
    for &id in &chosen {
        filter.insert(id);
    }
    let selectivity = filter.len() as f64 / n as f64;
    assert!(selectivity < 0.01, "selectivity = {selectivity}");

    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = hnsw.search_filtered(query, k, &filter).unwrap();
        assert!(
            results.len() <= k,
            "Got more than k results: {}",
            results.len()
        );
        for r in &results {
            assert!(
                filter.contains(r.id),
                "ID {} not in the low-selectivity filter set",
                r.id
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 6: Empty filter — should return empty results
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_filter() {
    let mut rng = StdRng::seed_from_u64(500);
    let dim = 32;
    let n = 500;
    let k = 10;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Cosine);

    let filter = RoaringBitmap::new();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = hnsw.search_filtered(&query, k, &filter).unwrap();
    assert!(
        results.is_empty(),
        "Expected empty results for empty filter"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 7: Filter with all IDs — should behave like unfiltered search
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_filter_all_ids() {
    let mut rng = StdRng::seed_from_u64(600);
    let dim = 64;
    let n = 2000;
    let k = 10;
    let num_queries = 30;

    let vecs = random_vectors(&mut rng, n, dim);
    let hnsw = build_hnsw(&vecs, dim, DistanceMetric::Cosine);

    // Filter contains ALL IDs
    let mut filter = RoaringBitmap::new();
    for id in 0..n as u32 {
        filter.insert(id);
    }

    let queries = random_vectors(&mut rng, num_queries, dim);
    for query in &queries {
        let unfiltered = hnsw.search(query, k).unwrap();
        let filtered = hnsw.search_filtered(query, k, &filter).unwrap();

        // With all IDs in the filter, the filtered search should find the
        // same (or very similar) results as the unfiltered search.
        let r = recall(&filtered, &unfiltered, k);
        assert!(
            r > 0.8,
            "All-IDs filter should match unfiltered: recall={r:.4}"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 8: Deleted + filtered — vectors both deleted and in filter must NOT appear
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_deleted_and_filtered() {
    let mut rng = StdRng::seed_from_u64(700);
    let dim = 32;
    let n = 2000;
    let k = 10;

    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mut hnsw = HnswIndex::new(DistanceMetric::Euclidean, dim as u32);
    hnsw.build(&refs).unwrap();

    // Delete every 10th vector
    let mut deleted_ids: HashSet<u32> = HashSet::new();
    for id in (0..n as u32).step_by(10) {
        hnsw.delete(id).unwrap();
        deleted_ids.insert(id);
    }

    // Filter includes some deleted and some non-deleted vectors
    let mut filter = RoaringBitmap::new();
    for id in (0..n as u32).step_by(2) {
        filter.insert(id); // Includes IDs 0, 10, 20, ... which are deleted
    }

    let queries = random_vectors(&mut rng, 20, dim);
    for query in &queries {
        let results = hnsw.search_filtered(query, k, &filter).unwrap();
        for r in &results {
            assert!(filter.contains(r.id), "ID {} not in filter", r.id);
            assert!(
                !deleted_ids.contains(&r.id),
                "Deleted ID {} appeared in filtered results",
                r.id
            );
        }
    }
}
