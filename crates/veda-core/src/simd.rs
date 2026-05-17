//! SIMD-optimized distance functions for dense vectors.
//!
//! On x86_64 with AVX2+FMA support, operations process 8 f32 values at a time.
//! A scalar fallback is used on other architectures or when AVX2 is not available.

// ── Scalar fallback implementations ─────────────────────────────────────────

/// Scalar dot product of two f32 slices.
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar L2 squared distance between two f32 slices.
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Scalar Manhattan (L1) distance between two f32 slices.
fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Scalar squared L2 norm of an f32 slice.
fn l2_norm_squared_scalar(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

// ── AVX2 + FMA implementations ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontally sum all 8 lanes of a `__m256` into a single `f32`.
    ///
    /// Strategy: extract high/low 128-bit halves, add them, then use
    /// `movehdup` + `movehl` to reduce the 4-lane `__m128` to a scalar.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX is available.
    #[inline(always)]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        unsafe {
            // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
            let high = _mm256_extractf128_ps(v, 1); // [a4 a5 a6 a7]
            let low = _mm256_castps256_ps128(v); // [a0 a1 a2 a3]
            let sum128 = _mm_add_ps(low, high); // [a0+a4, a1+a5, a2+a6, a3+a7]
            let shuf = _mm_movehdup_ps(sum128); // [a1+a5, a1+a5, a3+a7, a3+a7]
            let sums = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5, ?, a2+a3+a6+a7, ?]
            let high128 = _mm_movehl_ps(sums, sums); // [a2+a3+a6+a7, ?, ?, ?]
            let result = _mm_add_ss(sums, high128);
            _mm_cvtss_f32(result)
        }
    }

    /// Dot product using AVX2 + FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 8;
            let remainder = n % 8;

            let mut acc = _mm256_setzero_ps();

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            let mut sum = hsum_avx(acc);

            // Handle remainder elements
            let tail_start = chunks * 8;
            for i in 0..remainder {
                sum += a[tail_start + i] * b[tail_start + i];
            }

            sum
        }
    }

    /// Squared L2 distance using AVX2 + FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 8;
            let remainder = n % 8;

            let mut acc = _mm256_setzero_ps();

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                let diff = _mm256_sub_ps(va, vb);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }

            let mut sum = hsum_avx(acc);

            let tail_start = chunks * 8;
            for i in 0..remainder {
                let d = a[tail_start + i] - b[tail_start + i];
                sum += d * d;
            }

            sum
        }
    }

    /// Manhattan (L1) distance using AVX2.
    ///
    /// Absolute value is computed via `andnot` with the sign-bit mask, which
    /// clears the sign bit of each f32 lane.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn manhattan_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 8;
            let remainder = n % 8;

            let mut acc = _mm256_setzero_ps();

            // Sign-bit mask: 0x80000000 in every lane
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000));

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                let diff = _mm256_sub_ps(va, vb);
                // abs(diff) = andnot(sign_mask, diff) — clears sign bit
                let abs_diff = _mm256_andnot_ps(sign_mask, diff);
                acc = _mm256_add_ps(acc, abs_diff);
            }

            let mut sum = hsum_avx(acc);

            let tail_start = chunks * 8;
            for i in 0..remainder {
                sum += (a[tail_start + i] - b[tail_start + i]).abs();
            }

            sum
        }
    }

    /// Squared L2 norm using AVX2 + FMA.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn l2_norm_squared_avx2(a: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 8;
            let remainder = n % 8;

            let mut acc = _mm256_setzero_ps();

            let a_ptr = a.as_ptr();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                acc = _mm256_fmadd_ps(va, va, acc);
            }

            let mut sum = hsum_avx(acc);

            let tail_start = chunks * 8;
            for i in 0..remainder {
                sum += a[tail_start + i] * a[tail_start + i];
            }

            sum
        }
    }
}

// ── Public dispatch functions ───────────────────────────────────────────────

/// Compute the dot product of two f32 slices.
///
/// Automatically dispatches to AVX2+FMA on supported x86_64 CPUs,
/// falling back to scalar otherwise.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// Compute the squared L2 distance between two f32 slices.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::l2_squared_avx2(a, b) };
        }
    }
    l2_squared_scalar(a, b)
}

/// Compute the Manhattan (L1) distance between two f32 slices.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::manhattan_avx2(a, b) };
        }
    }
    manhattan_scalar(a, b)
}

/// Compute the squared L2 norm of an f32 slice.
pub fn l2_norm_squared(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::l2_norm_squared_avx2(a) };
        }
    }
    l2_norm_squared_scalar(a)
}

/// Cosine distance: `1.0 - cosine_similarity(a, b)`.
///
/// Returns 0.0 for identical directions, 1.0 for orthogonal, 2.0 for opposite.
/// If either vector has zero norm, returns 1.0 by convention.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");

    let dot = dot_product(a, b);
    let norm_a = l2_norm_squared(a).sqrt();
    let norm_b = l2_norm_squared(b).sqrt();
    let denom = norm_a * norm_b;

    if denom == 0.0 {
        return 1.0;
    }

    1.0 - (dot / denom)
}

/// Euclidean (L2) distance: `sqrt(l2_squared(a, b))`.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// MaxSim distance (ColBERT/ColPali late-interaction): for each query row
/// take the maximum similarity (= -distance) over all document rows, then
/// sum. Returned as a distance — lower = more similar.
pub fn maxsim_distance(
    query_rows: &[Vec<f32>],
    doc_rows: &[Vec<f32>],
    per_row_distance: fn(&[f32], &[f32]) -> f32,
) -> f32 {
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let mut sum_max_sim = 0.0f32;
    for q in query_rows {
        let mut min_dist = f32::MAX;
        for d in doc_rows {
            let dist = per_row_distance(q, d);
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_max_sim += -min_dist;
    }
    -sum_max_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two f32 values are close within a relative tolerance.
    fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
        let diff = (actual - expected).abs();
        let max_abs = expected.abs().max(actual.abs()).max(1e-10);
        assert!(
            diff <= tol * max_abs + 1e-7,
            "{label}: expected {expected}, got {actual} (diff {diff})"
        );
    }

    #[test]
    fn test_dot_product_scalar_matches_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_product_scalar(&a, &b);
        assert_close(result, 32.0, 1e-5, "dot_product_scalar");
    }

    #[test]
    fn test_simd_dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_product(&a, &b);
        assert_close(result, expected, 1e-5, "simd dot_product 8-elem");
    }

    #[test]
    fn test_simd_dot_product_remainder() {
        // 11 elements — 8 via SIMD + 3 remainder
        let a: Vec<f32> = (0..11).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..11).map(|i| (10 - i) as f32 * 0.1).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_product(&a, &b);
        assert_close(result, expected, 1e-5, "simd dot_product 11-elem");
    }

    #[test]
    fn test_euclidean_distance_known() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let result = euclidean_distance(&a, &b);
        assert_close(result, 5.0, 1e-5, "euclidean_distance");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let result = cosine_distance(&a, &b);
        assert_close(result, 1.0, 1e-5, "cosine_distance orthogonal");
    }

    #[test]
    fn test_manhattan_distance_known() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let result = manhattan_distance(&a, &b);
        assert_close(result, 7.0, 1e-5, "manhattan_distance");
    }
}
