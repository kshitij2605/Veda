//! Optional SimSIMD-backed distance kernels.
//!
//! Compiled only when the `simsimd` feature is enabled. Veda's primary path
//! (`crate::simd`) ships AVX2+FMA on x86_64 and a scalar fallback; SimSIMD
//! adds support for AVX-512, ARM NEON, ARM SVE, and other ISAs through a
//! single dependency. The module exposes the same function names as the
//! primary `simd` module, so callers can switch backends with a feature flag.
//!
//! ## When to enable
//!
//! - Deploying on ARM64 (Apple Silicon, AWS Graviton, Ampere Altra).
//! - Deploying on x86_64 systems with AVX-512.
//! - Wanting reduced-precision (f16/bf16/i8) distance kernels.
//!
//! On stock x86_64 with AVX2, the primary path is already optimal — no
//! reason to enable this feature.

#![cfg(feature = "simsimd")]

use simsimd::SpatialSimilarity;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    SpatialSimilarity::dot(a, b).unwrap_or(0.0) as f32
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Guard: SimSIMD returns 0.0 (not an error) for zero-norm inputs, but
    // our documented convention is 1.0 (maximum distance). Check cheaply.
    let norm_a: f32 = a.iter().map(|x| x * x).sum();
    let norm_b: f32 = b.iter().map(|x| x * x).sum();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    SpatialSimilarity::cosine(a, b).unwrap_or(1.0) as f32
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    SpatialSimilarity::sqeuclidean(a, b)
        .unwrap_or(0.0)
        .sqrt() as f32
}

pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    SpatialSimilarity::sqeuclidean(a, b).unwrap_or(0.0) as f32
}

pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    // SimSIMD has no L1 fast-path for f32 — fall through to the scalar
    // implementation in the primary `simd` module.
    crate::simd::manhattan_distance(a, b)
}

pub fn l2_norm_squared(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

pub fn i8_dot_product(a: &[i8], b: &[i8]) -> i32 {
    // i8×i8 sums fit exactly in f64 for all practical vector dims, so the
    // cast back to i32 is lossless.
    <i8 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as i32
}
