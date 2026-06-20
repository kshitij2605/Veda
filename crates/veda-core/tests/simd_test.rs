use veda_core::simd;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Scalar reference implementations for comparison against SIMD dispatch.
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn scalar_manhattan(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn scalar_l2_norm_squared(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

fn scalar_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = scalar_l2_norm_squared(a).sqrt();
    let norm_b = scalar_l2_norm_squared(b).sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    scalar_l2_squared(a, b).sqrt()
}

/// Assert that `actual` is close to `expected` within a relative tolerance.
/// Uses the formula: |actual - expected| <= tol * max(|actual|, |expected|, epsilon)
fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
    let diff = (actual - expected).abs();
    let max_abs = expected.abs().max(actual.abs()).max(1e-10);
    assert!(
        diff <= tol * max_abs + 1e-6,
        "{label}: expected {expected}, got {actual} (diff {diff})"
    );
}

/// Generate a deterministic pseudo-random vector of length `dim`.
fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        // Simple LCG for reproducibility
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Map to [-1.0, 1.0]
        let val = ((state >> 16) as f32 / 32768.0) - 1.0;
        v.push(val);
    }
    v
}

// ── SIMD vs scalar consistency across various dimensions ────────────────────

const DIMENSIONS: &[usize] = &[3, 8, 16, 128, 256, 512, 768, 1024, 1536];

#[test]
fn test_dot_product_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_dot_product(&a, &b);
        let actual = simd::dot_product(&a, &b);
        assert_close(actual, expected, 1e-4, &format!("dot_product dim={dim}"));
    }
}

#[test]
fn test_l2_squared_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_l2_squared(&a, &b);
        let actual = simd::l2_squared(&a, &b);
        assert_close(actual, expected, 1e-4, &format!("l2_squared dim={dim}"));
    }
}

#[test]
fn test_manhattan_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_manhattan(&a, &b);
        let actual = simd::manhattan_distance(&a, &b);
        assert_close(actual, expected, 1e-4, &format!("manhattan dim={dim}"));
    }
}

#[test]
fn test_l2_norm_squared_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let expected = scalar_l2_norm_squared(&a);
        let actual = simd::l2_norm_squared(&a);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("l2_norm_squared dim={dim}"),
        );
    }
}

#[test]
fn test_cosine_distance_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_cosine_distance(&a, &b);
        let actual = simd::cosine_distance(&a, &b);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("cosine_distance dim={dim}"),
        );
    }
}

#[test]
fn test_euclidean_distance_matches_scalar_all_dimensions() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_euclidean_distance(&a, &b);
        let actual = simd::euclidean_distance(&a, &b);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("euclidean_distance dim={dim}"),
        );
    }
}

// ── Edge cases: zero vectors ────────────────────────────────────────────────

#[test]
fn test_zero_vectors_dot_product() {
    let z = vec![0.0_f32; 128];
    let a = make_vector(128, 42);
    assert_close(simd::dot_product(&z, &z), 0.0, 1e-5, "zero dot zero");
    assert_close(simd::dot_product(&z, &a), 0.0, 1e-5, "zero dot nonzero");
}

#[test]
fn test_zero_vectors_euclidean() {
    let z = vec![0.0_f32; 128];
    assert_close(
        simd::euclidean_distance(&z, &z),
        0.0,
        1e-5,
        "euclidean(zero, zero)",
    );
}

#[test]
fn test_zero_vectors_manhattan() {
    let z = vec![0.0_f32; 128];
    assert_close(
        simd::manhattan_distance(&z, &z),
        0.0,
        1e-5,
        "manhattan(zero, zero)",
    );
}

#[test]
fn test_zero_vector_cosine_returns_one() {
    let z = vec![0.0_f32; 128];
    let a = make_vector(128, 42);
    // Cosine distance with a zero vector should return 1.0 by convention
    assert_close(
        simd::cosine_distance(&z, &a),
        1.0,
        1e-5,
        "cosine(zero, nonzero)",
    );
    assert_close(
        simd::cosine_distance(&z, &z),
        1.0,
        1e-5,
        "cosine(zero, zero)",
    );
}

#[test]
fn test_zero_vector_l2_norm() {
    let z = vec![0.0_f32; 128];
    assert_close(simd::l2_norm_squared(&z), 0.0, 1e-5, "norm(zero)");
}

// ── Edge cases: identical vectors ───────────────────────────────────────────

#[test]
fn test_identical_vectors_euclidean() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 7);
        assert_close(
            simd::euclidean_distance(&a, &a),
            0.0,
            1e-5,
            &format!("euclidean(a, a) dim={dim}"),
        );
    }
}

#[test]
fn test_identical_vectors_manhattan() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 7);
        assert_close(
            simd::manhattan_distance(&a, &a),
            0.0,
            1e-5,
            &format!("manhattan(a, a) dim={dim}"),
        );
    }
}

#[test]
fn test_identical_vectors_cosine() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 7);
        assert_close(
            simd::cosine_distance(&a, &a),
            0.0,
            1e-4,
            &format!("cosine(a, a) dim={dim}"),
        );
    }
}

// ── Edge cases: opposite vectors ────────────────────────────────────────────

#[test]
fn test_opposite_vectors_cosine() {
    for &dim in DIMENSIONS {
        let a = make_vector(dim, 7);
        let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();
        assert_close(
            simd::cosine_distance(&a, &neg_a),
            2.0,
            1e-4,
            &format!("cosine(a, -a) dim={dim}"),
        );
    }
}

#[test]
fn test_opposite_vectors_dot_product() {
    let a = make_vector(128, 7);
    let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();
    let dot_aa = simd::dot_product(&a, &a);
    let dot_a_neg = simd::dot_product(&a, &neg_a);
    assert_close(dot_a_neg, -dot_aa, 1e-5, "dot(a, -a) == -dot(a, a)");
}

// ── Edge cases: unit vectors / orthogonal ───────────────────────────────────

#[test]
fn test_orthogonal_unit_vectors_cosine_distance_one() {
    // Standard basis vectors in R^128: e_0 and e_1
    let mut e0 = vec![0.0_f32; 128];
    let mut e1 = vec![0.0_f32; 128];
    e0[0] = 1.0;
    e1[1] = 1.0;
    assert_close(simd::cosine_distance(&e0, &e1), 1.0, 1e-5, "cosine(e0, e1)");
}

#[test]
fn test_unit_vectors_norm() {
    let mut e = vec![0.0_f32; 768];
    e[42] = 1.0;
    assert_close(simd::l2_norm_squared(&e), 1.0, 1e-5, "norm(unit vector)");
}

#[test]
fn test_orthogonal_unit_vectors_dot_product_zero() {
    let mut e0 = vec![0.0_f32; 256];
    let mut e1 = vec![0.0_f32; 256];
    e0[0] = 1.0;
    e1[1] = 1.0;
    assert_close(simd::dot_product(&e0, &e1), 0.0, 1e-5, "dot(e0, e1)");
}

// ── Remainder handling (non-multiple-of-8 dimensions) ───────────────────────

#[test]
fn test_remainder_handling_dot_product() {
    // Dimensions that are not multiples of 8
    for dim in [1, 2, 3, 5, 7, 9, 10, 13, 15, 17, 31, 33, 63, 65, 127, 129] {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_dot_product(&a, &b);
        let actual = simd::dot_product(&a, &b);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("dot_product remainder dim={dim}"),
        );
    }
}

#[test]
fn test_remainder_handling_l2_squared() {
    for dim in [1, 2, 3, 5, 7, 9, 10, 13, 15, 17, 31, 33, 63, 65, 127, 129] {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_l2_squared(&a, &b);
        let actual = simd::l2_squared(&a, &b);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("l2_squared remainder dim={dim}"),
        );
    }
}

#[test]
fn test_remainder_handling_manhattan() {
    for dim in [1, 2, 3, 5, 7, 9, 10, 13, 15, 17, 31, 33, 63, 65, 127, 129] {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);
        let expected = scalar_manhattan(&a, &b);
        let actual = simd::manhattan_distance(&a, &b);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("manhattan remainder dim={dim}"),
        );
    }
}

#[test]
fn test_remainder_handling_l2_norm_squared() {
    for dim in [1, 2, 3, 5, 7, 9, 10, 13, 15, 17, 31, 33, 63, 65, 127, 129] {
        let a = make_vector(dim, 42);
        let expected = scalar_l2_norm_squared(&a);
        let actual = simd::l2_norm_squared(&a);
        assert_close(
            actual,
            expected,
            1e-4,
            &format!("l2_norm_squared remainder dim={dim}"),
        );
    }
}

// ── Known-value tests ───────────────────────────────────────────────────────

#[test]
fn test_known_dot_product() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
    assert_close(simd::dot_product(&a, &b), 120.0, 1e-5, "known dot 8-elem");
}

#[test]
fn test_known_euclidean() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    assert_close(
        simd::euclidean_distance(&a, &b),
        5.0,
        1e-5,
        "known euclidean",
    );
}

#[test]
fn test_known_manhattan() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 3.0];
    // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    assert_close(
        simd::manhattan_distance(&a, &b),
        7.0,
        1e-5,
        "known manhattan",
    );
}

#[test]
fn test_known_cosine_identical_direction() {
    let a = [1.0, 0.0];
    let b = [2.0, 0.0]; // same direction, different magnitude
    assert_close(
        simd::cosine_distance(&a, &b),
        0.0,
        1e-5,
        "cosine identical direction",
    );
}

#[test]
fn test_known_cosine_opposite() {
    let a = [1.0, 0.0];
    let b = [-1.0, 0.0];
    assert_close(simd::cosine_distance(&a, &b), 2.0, 1e-5, "cosine opposite");
}

#[test]
fn test_known_cosine_orthogonal() {
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    assert_close(
        simd::cosine_distance(&a, &b),
        1.0,
        1e-5,
        "cosine orthogonal",
    );
}

#[test]
fn test_l2_norm_squared_known() {
    let a = [3.0, 4.0];
    assert_close(simd::l2_norm_squared(&a), 25.0, 1e-5, "norm squared 3-4");
}

// ── DenseVector integration (ensure vector.rs delegates correctly) ──────────

use veda_core::vector::DenseVector;

#[test]
fn test_dense_vector_uses_simd_cosine() {
    let a = DenseVector::new(make_vector(768, 1));
    let b = DenseVector::new(make_vector(768, 2));
    let via_vector = a.cosine_distance(&b);
    let via_simd = simd::cosine_distance(a.as_slice(), b.as_slice());
    assert!(
        (via_vector - via_simd).abs() < 1e-7,
        "DenseVector::cosine_distance should delegate to simd module"
    );
}

#[test]
fn test_dense_vector_uses_simd_euclidean() {
    let a = DenseVector::new(make_vector(768, 1));
    let b = DenseVector::new(make_vector(768, 2));
    let via_vector = a.euclidean_distance(&b);
    let via_simd = simd::euclidean_distance(a.as_slice(), b.as_slice());
    assert!(
        (via_vector - via_simd).abs() < 1e-7,
        "DenseVector::euclidean_distance should delegate to simd module"
    );
}

#[test]
fn test_dense_vector_uses_simd_dot_product() {
    let a = DenseVector::new(make_vector(768, 1));
    let b = DenseVector::new(make_vector(768, 2));
    let via_vector = a.dot_product(&b);
    let via_simd = simd::dot_product(a.as_slice(), b.as_slice());
    assert!(
        (via_vector - via_simd).abs() < 1e-7,
        "DenseVector::dot_product should delegate to simd module"
    );
}

#[test]
fn test_dense_vector_uses_simd_manhattan() {
    let a = DenseVector::new(make_vector(768, 1));
    let b = DenseVector::new(make_vector(768, 2));
    let via_vector = a.manhattan_distance(&b);
    let via_simd = simd::manhattan_distance(a.as_slice(), b.as_slice());
    assert!(
        (via_vector - via_simd).abs() < 1e-7,
        "DenseVector::manhattan_distance should delegate to simd module"
    );
}

// ── Large embedding dimensions commonly used in multimodal models ───────────

#[test]
fn test_large_dimension_768_text_embedding() {
    let a = make_vector(768, 100);
    let b = make_vector(768, 200);
    let expected = scalar_cosine_distance(&a, &b);
    let actual = simd::cosine_distance(&a, &b);
    assert_close(actual, expected, 1e-4, "cosine 768-dim");
}

#[test]
fn test_large_dimension_1536_openai_embedding() {
    let a = make_vector(1536, 100);
    let b = make_vector(1536, 200);
    let expected = scalar_cosine_distance(&a, &b);
    let actual = simd::cosine_distance(&a, &b);
    assert_close(actual, expected, 1e-4, "cosine 1536-dim");
}

#[test]
fn test_large_dimension_512_image_embedding() {
    let a = make_vector(512, 100);
    let b = make_vector(512, 200);
    let expected = scalar_euclidean_distance(&a, &b);
    let actual = simd::euclidean_distance(&a, &b);
    assert_close(actual, expected, 1e-4, "euclidean 512-dim");
}
