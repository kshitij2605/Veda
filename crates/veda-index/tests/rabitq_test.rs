use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::NamedTempFile;
use veda_core::types::DistanceMetric;
use veda_index::distance::compute_distance;
use veda_index::flat::FlatIndex;
use veda_index::quantization::rabitq::RaBitQuantizer;
use veda_index::quantization::scalar::ScalarQuantizer;
use veda_index::traits::VectorIndex;

fn random_vectors(rng: &mut StdRng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// RaBitQ tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rabitq_basic_encode() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;
    let vecs = random_vectors(&mut rng, 1000, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut quantizer = RaBitQuantizer::new(DistanceMetric::Cosine);
    quantizer.train(&refs);

    assert_eq!(quantizer.len(), 1000);

    // Each binary code should have ceil(128/64) = 2 u64 words
    let code = quantizer.encode(&vecs[0]);
    assert_eq!(code.len(), 2);
}

#[test]
fn test_rabitq_distance_estimation_quality_cosine() {
    let mut rng = StdRng::seed_from_u64(123);
    let dim = 128;
    let n = 1000;
    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut quantizer = RaBitQuantizer::new(DistanceMetric::Cosine);
    quantizer.train(&refs);

    // Compare estimated distances vs true distances for random query-vector pairs
    let num_queries = 100;
    let mut total_abs_error = 0.0f64;
    let mut count = 0;

    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        for (idx, vec) in vecs.iter().enumerate().take(10) {
            let true_dist = compute_distance(&query, vec, DistanceMetric::Cosine);
            let est_dist = quantizer.estimate_distance(&query, idx);
            total_abs_error += (true_dist - est_dist).abs() as f64;
            count += 1;
        }
    }

    let mean_abs_error = total_abs_error / count as f64;
    assert!(
        mean_abs_error < 0.3,
        "Mean absolute error {mean_abs_error:.4} exceeds threshold 0.3"
    );
}

#[test]
fn test_rabitq_distance_positive_correlation() {
    let mut rng = StdRng::seed_from_u64(456);
    let dim = 128;
    let n = 500;
    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut quantizer = RaBitQuantizer::new(DistanceMetric::Cosine);
    quantizer.train(&refs);

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let mut true_dists: Vec<f32> = Vec::new();
    let mut est_dists: Vec<f32> = Vec::new();
    for (i, vec) in vecs.iter().enumerate() {
        true_dists.push(compute_distance(&query, vec, DistanceMetric::Cosine));
        est_dists.push(quantizer.estimate_distance(&query, i));
    }

    // Compute Spearman rank correlation (simplified: use Pearson on ranks)
    let true_ranks = ranks(&true_dists);
    let est_ranks = ranks(&est_dists);

    let n_f = n as f64;
    let mean_t: f64 = true_ranks.iter().sum::<f64>() / n_f;
    let mean_e: f64 = est_ranks.iter().sum::<f64>() / n_f;

    let mut cov = 0.0f64;
    let mut var_t = 0.0f64;
    let mut var_e = 0.0f64;
    for i in 0..n {
        let dt = true_ranks[i] - mean_t;
        let de = est_ranks[i] - mean_e;
        cov += dt * de;
        var_t += dt * dt;
        var_e += de * de;
    }

    let correlation = cov / (var_t.sqrt() * var_e.sqrt());
    assert!(
        correlation > 0.3,
        "Distance correlation {correlation:.4} should be positive (>0.3)"
    );
}

fn ranks(values: &[f32]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut result = vec![0.0f64; values.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        result[idx] = rank as f64;
    }
    result
}

#[test]
fn test_rabitq_search_recall() {
    let mut rng = StdRng::seed_from_u64(789);
    let dim = 128;
    let n = 2000;
    let k = 10;

    // Generate vectors with varying norms and structure.
    // Use Gaussian components (more realistic for embeddings than uniform).
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let scale = rng.gen_range(0.5..2.0);
            (0..dim)
                .map(|_| {
                    // Box-Muller for Gaussian
                    let u1: f32 = rng.gen_range(0.0001f32..1.0);
                    let u2: f32 = rng.gen_range(0.0f32..std::f32::consts::TAU);
                    scale * (-2.0 * u1.ln()).sqrt() * u2.cos()
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    // Build FlatIndex for ground truth
    let mut flat = FlatIndex::new(DistanceMetric::Euclidean, dim as u32);
    flat.build(&refs).unwrap();

    // Build RaBitQ
    let mut quantizer = RaBitQuantizer::new(DistanceMetric::Euclidean);
    quantizer.train(&refs);

    let num_queries = 50;
    let mut total_recall = 0.0f64;

    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let truth = flat.search(&query, k).unwrap();
        let truth_ids: HashSet<u32> = truth.iter().map(|r| r.id).collect();

        let approx = quantizer.search(&query, k, &HashSet::new());
        let approx_ids: HashSet<u32> = approx.iter().map(|(id, _)| *id).collect();

        let hits = truth_ids.intersection(&approx_ids).count();
        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall > 0.60,
        "Recall@{k} = {avg_recall:.4} should be > 0.60"
    );
}

#[test]
fn test_rabitq_save_load_roundtrip() {
    let mut rng = StdRng::seed_from_u64(101);
    let dim = 64;
    let vecs = random_vectors(&mut rng, 200, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut quantizer = RaBitQuantizer::new(DistanceMetric::Cosine);
    quantizer.train(&refs);

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results_before = quantizer.search(&query, 10, &HashSet::new());

    let tmp = NamedTempFile::new().unwrap();
    quantizer.save(tmp.path()).unwrap();

    let loaded = RaBitQuantizer::load(tmp.path()).unwrap();
    assert_eq!(loaded.len(), quantizer.len());

    let results_after = loaded.search(&query, 10, &HashSet::new());
    assert_eq!(results_before.len(), results_after.len());
    for (a, b) in results_before.iter().zip(results_after.iter()) {
        assert_eq!(a.0, b.0, "id mismatch after load");
        assert!(
            (a.1 - b.1).abs() < 1e-6,
            "distance mismatch after load: {} vs {}",
            a.1,
            b.1
        );
    }
}

#[test]
fn test_packed_binary_hamming() {
    // Manually verify XOR + popcount for known bit patterns
    let a: Vec<u64> = vec![0xFF00_FF00_FF00_FF00, 0x0000_0000_0000_0000];
    let b: Vec<u64> = vec![0x00FF_00FF_00FF_00FF, 0x0000_0000_0000_0000];

    // XOR should give all 1s in first word
    let xor = a[0] ^ b[0];
    assert_eq!(xor, 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(xor.count_ones(), 64);

    // Hamming distance between a and b
    let hamming: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    assert_eq!(hamming, 64);

    // Same codes should have hamming distance 0
    let hamming_same: u32 = a
        .iter()
        .zip(a.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    assert_eq!(hamming_same, 0);
}

// ────────────────────────────────────────────────────────────────────────────
// Scalar quantization tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scalar_quantization_reconstruction_error() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 64;
    let n = 1000;
    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut sq = ScalarQuantizer::new();
    sq.train(&refs);

    // Check reconstruction error per dimension
    let mut total_error = 0.0f64;

    for v in &vecs {
        let code = sq.encode(v);
        let reconstructed = sq.decode(&code);
        for (&orig, &recon) in v.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs() as f64;
            total_error += error;
            // Range is max - min for this dimension; for relative error we
            // compare against the data range per dimension
        }
    }

    // Average absolute error per element
    let avg_error = total_error / (n * dim) as f64;
    // For uniform [-1, 1], range = 2, quantization step = 2/255 ~ 0.0078
    // Average error should be roughly step/2 ~ 0.004
    // 1% of range (2.0) = 0.02 -- our average should be well under that
    let range = 2.0; // approximate range for uniform [-1, 1]
    let relative_error = avg_error / range;
    assert!(
        relative_error < 0.01,
        "Average relative reconstruction error {relative_error:.6} exceeds 1% threshold"
    );
}

#[test]
fn test_scalar_quantization_encode_decode_roundtrip() {
    let mut sq = ScalarQuantizer::new();
    let vectors: Vec<Vec<f32>> = vec![
        vec![0.0, 0.5, 1.0, -1.0],
        vec![-0.5, 0.25, 0.75, -0.75],
    ];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    sq.train(&refs);

    // Encode and decode the first vector
    let code = sq.encode(&vectors[0]);
    assert_eq!(code.len(), 4);

    let decoded = sq.decode(&code);
    assert_eq!(decoded.len(), 4);

    // Each dimension should be close to original
    for (orig, recon) in vectors[0].iter().zip(decoded.iter()) {
        assert!(
            (orig - recon).abs() < 0.02,
            "Reconstruction error too large: {orig} vs {recon}"
        );
    }
}

#[test]
fn test_scalar_quantization_distance_estimation() {
    let mut rng = StdRng::seed_from_u64(99);
    let dim = 64;
    let n = 500;
    let vecs = random_vectors(&mut rng, n, dim);
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let mut sq = ScalarQuantizer::new();
    sq.train(&refs);

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // SQ8 distance estimates should be very close to true distances
    for (i, vec) in vecs.iter().enumerate().take(20) {
        let true_dist = compute_distance(&query, vec, DistanceMetric::Euclidean);
        let est_dist = sq.estimate_distance(&query, i, DistanceMetric::Euclidean);
        let rel_error = if true_dist > 0.01 {
            (true_dist - est_dist).abs() / true_dist
        } else {
            (true_dist - est_dist).abs()
        };
        assert!(
            rel_error < 0.05,
            "SQ8 distance estimation error too large at idx {i}: true={true_dist:.4} est={est_dist:.4} rel_error={rel_error:.4}"
        );
    }
}
