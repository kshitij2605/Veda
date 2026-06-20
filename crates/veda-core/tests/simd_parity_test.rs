//! Parity tests: SimSIMD kernels (and the dispatched `simd` API) must match
//! a scalar reference. Only meaningful when the `simsimd` feature is on.
#![cfg(feature = "simsimd")]

use veda_core::simd_simsimd;

fn i8_dot_ref(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

fn seeded_i8(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as i32 % 256 - 128) as i8
        })
        .collect()
}

#[test]
fn i8_dot_product_matches_scalar_exactly() {
    for &dim in &[1usize, 7, 16, 31, 128, 768, 1024, 1536] {
        let a = seeded_i8(dim as u32, dim);
        let b = seeded_i8((dim as u32).wrapping_add(99), dim);
        assert_eq!(
            simd_simsimd::i8_dot_product(&a, &b),
            i8_dot_ref(&a, &b),
            "i8 dot mismatch at dim={dim}"
        );
    }
}

use veda_core::simd;

fn seeded_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn l2sq_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn l1_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}
fn norm2_ref(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}
fn cosine_ref(a: &[f32], b: &[f32]) -> f32 {
    let d = dot_ref(a, b);
    let na = norm2_ref(a).sqrt();
    let nb = norm2_ref(b).sqrt();
    if na * nb == 0.0 {
        1.0
    } else {
        1.0 - d / (na * nb)
    }
}

const DIMS: &[usize] = &[1, 7, 16, 31, 128, 768, 1024, 1536, 3072];
const TOL: f32 = 1e-3;

fn close(got: f32, want: f32, ctx: &str) {
    let denom = want.abs().max(1.0);
    assert!(
        ((got - want).abs() / denom) < TOL,
        "{ctx}: got {got}, want {want}"
    );
}

#[test]
fn f32_kernels_match_scalar_reference() {
    for &dim in DIMS {
        let a = seeded_f32(dim as u32, dim);
        let b = seeded_f32((dim as u32).wrapping_add(99), dim);

        // Direct SimSIMD kernels.
        close(simd_simsimd::dot_product(&a, &b), dot_ref(&a, &b), "ss dot");
        close(
            simd_simsimd::l2_squared(&a, &b),
            l2sq_ref(&a, &b),
            "ss l2sq",
        );
        close(
            simd_simsimd::manhattan_distance(&a, &b),
            l1_ref(&a, &b),
            "ss l1",
        );
        close(simd_simsimd::l2_norm_squared(&a), norm2_ref(&a), "ss norm2");
        close(
            simd_simsimd::cosine_distance(&a, &b),
            cosine_ref(&a, &b),
            "ss cos",
        );
        close(
            simd_simsimd::euclidean_distance(&a, &b),
            l2sq_ref(&a, &b).sqrt(),
            "ss euc",
        );

        // Dispatched public API (verifies the wiring).
        close(simd::dot_product(&a, &b), dot_ref(&a, &b), "api dot");
        close(simd::l2_squared(&a, &b), l2sq_ref(&a, &b), "api l2sq");
        close(simd::manhattan_distance(&a, &b), l1_ref(&a, &b), "api l1");
        close(simd::l2_norm_squared(&a), norm2_ref(&a), "api norm2");
        close(simd::cosine_distance(&a, &b), cosine_ref(&a, &b), "api cos");
        close(
            simd::euclidean_distance(&a, &b),
            l2sq_ref(&a, &b).sqrt(),
            "api euc",
        );
    }
}

#[test]
fn cosine_zero_norm_returns_one() {
    // The zero-norm guard: either operand having zero norm must yield the
    // max-distance convention (1.0), on both the direct kernel and the
    // dispatched API. (The non-zero parity test never hits this branch.)
    let zero = vec![0.0f32; 128];
    let b = seeded_f32(7, 128);
    assert_eq!(
        simd_simsimd::cosine_distance(&zero, &b),
        1.0,
        "ss zero-norm"
    );
    assert_eq!(
        simd_simsimd::cosine_distance(&b, &zero),
        1.0,
        "ss zero-norm (swapped)"
    );
    assert_eq!(
        simd_simsimd::cosine_distance(&zero, &zero),
        1.0,
        "ss both-zero"
    );
    assert_eq!(simd::cosine_distance(&zero, &b), 1.0, "api zero-norm");
}
