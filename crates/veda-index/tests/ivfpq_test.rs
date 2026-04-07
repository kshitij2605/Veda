use std::collections::HashSet;

use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::ivfpq::IvfPqIndex;
use veda_index::traits::{SearchResult, VectorIndex};

/// Deterministic PRNG (LCG) for reproducible test data.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate a float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Use bits 32..63 for good randomness from LCG high bits.
        ((self.state >> 32) as u32 as f32) / (u32::MAX as f32)
    }

    /// Generate a float in [-1, 1).
    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Generate clustered vectors: `num_clusters` cluster centers with `per_cluster`
/// points each, perturbed by Gaussian-like noise with given `spread`.
/// This produces data with structure that IVF-PQ can exploit.
fn generate_clustered_vectors(
    num_clusters: usize,
    per_cluster: usize,
    dim: usize,
    spread: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut vecs = Vec::with_capacity(num_clusters * per_cluster);

    for _ in 0..num_clusters {
        // Random cluster center.
        let center: Vec<f32> = (0..dim).map(|_| rng.next_f32_signed() * 5.0).collect();

        for _ in 0..per_cluster {
            let v: Vec<f32> = center
                .iter()
                .map(|&c| c + rng.next_f32_signed() * spread)
                .collect();
            vecs.push(v);
        }
    }

    vecs
}

/// Generate uniform random vectors.
fn generate_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut vecs = Vec::with_capacity(count);
    for _ in 0..count {
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32_signed()).collect();
        vecs.push(v);
    }
    vecs
}

/// Compute recall@k: fraction of ground-truth top-k IDs found in approximate results.
fn recall_at_k(approx: &[SearchResult], ground_truth: &[SearchResult]) -> f64 {
    let gt_ids: HashSet<u32> = ground_truth.iter().map(|r| r.id).collect();
    let found = approx.iter().filter(|r| gt_ids.contains(&r.id)).count();
    found as f64 / ground_truth.len() as f64
}

#[test]
fn test_recall_5000_vectors() {
    let dim = 64;
    let k = 10;
    // Generate clustered data: 50 clusters x 100 vectors = 5000 vectors.
    let vectors = generate_clustered_vectors(50, 100, dim, 1.0, 42);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    // Build ground-truth with FlatIndex.
    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    // Build IVF-PQ index.
    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        50,  // nlist
        20,  // nprobe
        8,   // m_pq
        8,   // nbits
    );
    ivfpq.build(&refs).unwrap();

    // Query with several vectors and measure average recall.
    let queries = generate_vectors(50, dim, 999);
    let mut total_recall = 0.0;
    for q in &queries {
        let gt = flat.search(q, k).unwrap();
        let approx = ivfpq.search(q, k).unwrap();
        total_recall += recall_at_k(&approx, &gt);
    }
    let avg_recall = total_recall / queries.len() as f64;

    assert!(
        avg_recall > 0.70,
        "Recall@{k} = {avg_recall:.3}, expected > 0.70"
    );
}

#[test]
fn test_higher_nprobe_higher_recall() {
    let dim = 64;
    let k = 10;
    let vectors = generate_clustered_vectors(20, 100, dim, 1.0, 123);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        20, // nlist
        1,  // nprobe (low)
        8,  // m_pq
        8,  // nbits
    );
    ivfpq.build(&refs).unwrap();

    let queries = generate_vectors(20, dim, 777);

    // Measure recall with nprobe=1.
    let mut recall_low = 0.0;
    for q in &queries {
        let gt = flat.search(q, k).unwrap();
        let approx = ivfpq.search(q, k).unwrap();
        recall_low += recall_at_k(&approx, &gt);
    }
    recall_low /= queries.len() as f64;

    // Increase nprobe.
    ivfpq.set_nprobe(10);
    let mut recall_high = 0.0;
    for q in &queries {
        let gt = flat.search(q, k).unwrap();
        let approx = ivfpq.search(q, k).unwrap();
        recall_high += recall_at_k(&approx, &gt);
    }
    recall_high /= queries.len() as f64;

    assert!(
        recall_high >= recall_low,
        "Higher nprobe should yield >= recall: nprobe=1 recall={recall_low:.3}, nprobe=10 recall={recall_high:.3}"
    );
}

#[test]
fn test_delete_excludes_vectors() {
    let dim = 32;
    let n = 100;
    let vectors = generate_vectors(n, dim, 55);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        8, 8, 4, 8,
    );
    ivfpq.build(&refs).unwrap();

    // Delete some vectors.
    let deleted_ids: Vec<u32> = vec![0, 5, 10, 50, 99];
    for &id in &deleted_ids {
        ivfpq.delete(id).unwrap();
    }

    // Search and verify none of the deleted IDs appear.
    let query = &vectors[0];
    let results = ivfpq.search(query, 20).unwrap();
    for r in &results {
        assert!(
            !deleted_ids.contains(&r.id),
            "Deleted vector {} should not appear in results",
            r.id
        );
    }
}

#[test]
fn test_save_and_load() {
    let dim = 32;
    let n = 200;
    let vectors = generate_vectors(n, dim, 88);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        16, 4, 4, 8,
    );
    ivfpq.build(&refs).unwrap();

    let query = generate_vectors(1, dim, 111)[0].clone();
    let results_before = ivfpq.search(&query, 5).unwrap();

    // Save and load.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ivfpq.json");
    ivfpq.save(&path).unwrap();

    let loaded = IvfPqIndex::load(&path, DistanceMetric::Euclidean, dim as u32).unwrap();
    let results_after = loaded.search(&query, 5).unwrap();

    assert_eq!(results_before.len(), results_after.len());
    for (a, b) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(a.id, b.id);
        assert!((a.distance - b.distance).abs() < 1e-6);
    }
}

#[test]
fn test_insert_after_build() {
    let dim = 32;
    let n = 100;
    let vectors = generate_vectors(n, dim, 66);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        8, 8, 4, 8,
    );
    ivfpq.build(&refs).unwrap();

    // Insert a new vector.
    let new_vec = vec![0.0f32; dim];
    let new_id = n as u32;
    ivfpq.insert(new_id, &new_vec).unwrap();

    // Search with the new vector as query — it should find itself (distance ~0).
    let results = ivfpq.search(&new_vec, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, new_id);
}

#[test]
fn test_empty_index() {
    let ivfpq = IvfPqIndex::new(DistanceMetric::Euclidean, 32);
    let results = ivfpq.search(&[0.0; 32], 10).unwrap();
    assert!(results.is_empty());
    assert!(ivfpq.is_empty());
}

#[test]
fn test_search_k_equals_1() {
    let dim = 16;
    let vectors = generate_vectors(50, dim, 77);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        4, 4, 4, 8,
    );
    ivfpq.build(&refs).unwrap();

    let results = ivfpq.search(&vectors[0], 1).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_single_vector() {
    let dim = 16;
    let vec = vec![1.0f32; dim];
    let refs: Vec<&[f32]> = vec![vec.as_slice()];

    let mut ivfpq = IvfPqIndex::with_params(
        DistanceMetric::Euclidean,
        dim as u32,
        1, 1, 4, 8,
    );
    ivfpq.build(&refs).unwrap();

    let results = ivfpq.search(&vec, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}
