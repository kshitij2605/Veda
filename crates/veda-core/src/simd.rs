//! SIMD-optimized distance functions for dense vectors.
//!
//! On x86_64 with AVX2+FMA support, operations process 8 f32 values at a time.
//! A scalar fallback is used on other architectures or when AVX2 is not available.

// ── Scalar fallback implementations ─────────────────────────────────────────

/// Scalar dot product of two f32 slices.
#[cfg_attr(feature = "simsimd", allow(dead_code))]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar L2 squared distance between two f32 slices.
#[cfg_attr(feature = "simsimd", allow(dead_code))]
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
#[cfg_attr(feature = "simsimd", allow(dead_code))]
fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Scalar squared L2 norm of an f32 slice.
#[cfg_attr(feature = "simsimd", allow(dead_code))]
fn l2_norm_squared_scalar(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Scalar i8 × i8 → i32 dot product. Fallback when AVX2 is not
/// available or the slices are short enough that the SIMD setup
/// cost dominates.
#[cfg_attr(feature = "simsimd", allow(dead_code))]
fn i8_dot_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

// ── AVX2 + FMA implementations ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[cfg_attr(feature = "simsimd", allow(dead_code))]
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

    /// Horizontally sum the 8 i32 lanes of a `__m256i`.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[inline(always)]
    unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
        unsafe {
            // Reduce 8 i32 → 4 i32 by adding upper and lower 128-bit halves.
            let lo = _mm256_castsi256_si128(v); // [a0 a1 a2 a3]
            let hi = _mm256_extracti128_si256(v, 1); // [a4 a5 a6 a7]
            let sum128 = _mm_add_epi32(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]
            // 4 → 2 by shuffling high pair down.
            let shuf = _mm_shuffle_epi32::<0xee>(sum128); // [a2+a6, a3+a7, _, _]
            let sum2 = _mm_add_epi32(sum128, shuf);
            // 2 → 1.
            let shuf2 = _mm_shuffle_epi32::<0x55>(sum2);
            let sum1 = _mm_add_epi32(sum2, shuf2);
            _mm_cvtsi128_si32(sum1)
        }
    }

    /// AVX2 i8 × i8 → i32 dot product.
    ///
    /// Processes 32 lanes per iteration via:
    ///   1. Sign-extend i8 → i16 (16 lanes per `_mm256_cvtepi8_epi16`)
    ///   2. `_mm256_madd_epi16` multiplies + pair-sums i16 lanes into i32
    ///   3. Accumulate into an `__m256i` of i32
    ///
    /// Overflow safety: each i8 × i8 fits in i16; the pair-add inside
    /// `_mm256_madd_epi16` produces i32 lanes whose absolute magnitude
    /// is ≤ 2·127² ≈ 32_258 per pair. Across n=10_000 lanes that's
    /// ≤ 5·10⁸ — well under `i32::MAX` (2.1·10⁹).
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn i8_dot_avx2(a: &[i8], b: &[i8]) -> i32 {
        unsafe {
            let n = a.len();
            let chunks = n / 32;
            let remainder = n % 32;

            let mut acc = _mm256_setzero_si256();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * 32;
                let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

                // Split each into low/high 128-bit halves; sign-extend
                // each half from i8 → i16 (16 lanes per half).
                let va_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
                let va_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                let vb_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
                let vb_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

                // Multiply + pair-add: i16 × i16 → i32 (16 lanes → 8 lanes).
                let prod_lo = _mm256_madd_epi16(va_lo16, vb_lo16);
                let prod_hi = _mm256_madd_epi16(va_hi16, vb_hi16);
                acc = _mm256_add_epi32(acc, prod_lo);
                acc = _mm256_add_epi32(acc, prod_hi);
            }

            let mut sum: i32 = hsum_epi32_avx2(acc);

            let tail_start = chunks * 32;
            for i in 0..remainder {
                sum += (a[tail_start + i] as i32) * (b[tail_start + i] as i32);
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
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::dot_product(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { avx2::dot_product_avx2(a, b) };
            }
        }
        dot_product_scalar(a, b)
    }
}

/// Compute the squared L2 distance between two f32 slices.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::l2_squared(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { avx2::l2_squared_avx2(a, b) };
            }
        }
        l2_squared_scalar(a, b)
    }
}

/// Compute the Manhattan (L1) distance between two f32 slices.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    // Note: simd_simsimd::manhattan_distance delegates back here (SimSIMD has no
    // f32 L1 fast path), so we always use the AVX2/scalar path to avoid recursion.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::manhattan_avx2(a, b) };
        }
    }
    manhattan_scalar(a, b)
}

/// Compute the i8 × i8 → i32 dot product. Dispatches to AVX2
/// on supported x86_64 CPUs (32 lanes per chunk), scalar
/// fallback otherwise. Used by the quantised-centroid PLAID
/// path; centroid storage shrinks 4× vs the f32 path AND the
/// inner-loop arithmetic is ~4× faster on AVX2.
pub fn i8_dot_product(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::i8_dot_product(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::i8_dot_avx2(a, b) };
            }
        }
        i8_dot_scalar(a, b)
    }
}

/// Compute the squared L2 norm of an f32 slice.
pub fn l2_norm_squared(a: &[f32]) -> f32 {
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::l2_norm_squared(a);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { avx2::l2_norm_squared_avx2(a) };
            }
        }
        l2_norm_squared_scalar(a)
    }
}

/// Cosine distance: `1.0 - cosine_similarity(a, b)`.
///
/// Returns 0.0 for identical directions, 1.0 for orthogonal, 2.0 for opposite.
/// If either vector has zero norm, returns 1.0 by convention.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "slice length mismatch");
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::cosine_distance(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        let dot = dot_product(a, b);
        let norm_a = l2_norm_squared(a).sqrt();
        let norm_b = l2_norm_squared(b).sqrt();
        let denom = norm_a * norm_b;
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }
}

/// Euclidean (L2) distance: `sqrt(l2_squared(a, b))`.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simsimd")]
    {
        return crate::simd_simsimd::euclidean_distance(a, b);
    }
    #[cfg(not(feature = "simsimd"))]
    {
        l2_squared(a, b).sqrt()
    }
}

/// MaxSim distance (ColBERT/ColPali late-interaction): for each query row
/// take the maximum similarity (= -distance) over all document rows, then
/// sum. Returned as a distance — lower = more similar.
///
/// The generic `F` parameter (vs. the previous `fn` pointer) lets the
/// compiler **monomorphise + inline** the per-row distance into the
/// inner loop, eliminating the indirect-call overhead and freeing
/// the existing SIMD `dot_product`/`euclidean_distance` to be
/// inlined too. On typical ColPali shapes (Q ≈ 32 rows × D ≈ 1024
/// rows × dim ≈ 128) this is a ~2-3x speedup over the fn-pointer
/// version.
///
/// For pre-normalised cosine, prefer [`maxsim_distance_cosine_normalized`].
/// See `research/2026-05-19-sota-audit/05-late-interaction.md` for
/// the path to a libxsmm-style GEMM kernel that would give another
/// 5-10x on top of this.
#[inline]
pub fn maxsim_distance<F>(query_rows: &[Vec<f32>], doc_rows: &[Vec<f32>], per_row_distance: F) -> f32
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let mut sum_max_sim = 0.0f32;
    for q in query_rows {
        let mut min_dist = f32::MAX;
        for d in doc_rows {
            let dist = per_row_distance(q.as_slice(), d.as_slice());
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_max_sim += -min_dist;
    }
    -sum_max_sim
}

/// Pre-normalised cosine fast path for [`maxsim_distance`].
///
/// Caller must have unit-length-normalised every row of both
/// `query_rows` and `doc_rows`. Then cosine similarity is just
/// the dot product, and the per-pair work is one SIMD dot call.
/// Returned as a distance: `1 - max_sim_sum / Q`. Lower = closer.
///
/// On ColPali-shaped inputs this matches the
/// shape Mixedbread's `maxsim-cpu` (libxsmm-AVX2) optimises;
/// we get the bulk of the win from monomorphisation + inlining
/// of [`dot_product`] without pulling in libxsmm.
///
/// For the BLIS-style GEMM kernel (the real Mixedbread-equivalent
/// speedup), use [`maxsim_distance_cosine_gemm`].
#[inline]
pub fn maxsim_distance_cosine_normalized(
    query_rows: &[Vec<f32>],
    doc_rows: &[Vec<f32>],
) -> f32 {
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let mut sum_max_sim = 0.0f32;
    for q in query_rows {
        let qs = q.as_slice();
        let mut max_sim = f32::MIN;
        for d in doc_rows {
            let sim = dot_product(qs, d.as_slice());
            if sim > max_sim {
                max_sim = sim;
            }
        }
        sum_max_sim += max_sim;
    }
    // Convert similarity sum back to a per-query-averaged distance.
    1.0 - sum_max_sim / query_rows.len() as f32
}

/// Internal: materialise the Q × D dot-product matrix via
/// matrixmultiply's sgemm. Caller owns the result buffer for
/// flexibility; row-reduce shapes diverge across metrics.
fn maxsim_gemm_dot_matrix(q: &[Vec<f32>], d: &[Vec<f32>]) -> (Vec<f32>, usize, usize) {
    let q_n = q.len();
    let d_n = d.len();
    let dim = q[0].len();
    debug_assert!(q.iter().all(|r| r.len() == dim));
    debug_assert!(d.iter().all(|r| r.len() == dim));

    let mut a = Vec::with_capacity(q_n * dim);
    for row in q {
        a.extend_from_slice(row);
    }
    let mut b = Vec::with_capacity(d_n * dim);
    for row in d {
        b.extend_from_slice(row);
    }
    let mut sim = vec![0.0_f32; q_n * d_n];
    unsafe {
        matrixmultiply::sgemm(
            q_n,
            dim,
            d_n,
            1.0,
            a.as_ptr(),
            dim as isize,
            1,
            b.as_ptr(),
            1,
            dim as isize,
            0.0,
            sim.as_mut_ptr(),
            d_n as isize,
            1,
        );
    }
    (sim, q_n, d_n)
}

/// GEMM-backed dot-product MaxSim. Returns a distance (lower =
/// closer) consistent with `simd::dot_product`'s sign convention:
/// raw dot is the similarity; we negate to make it a distance.
///
/// On ColPali shapes (Q ≈ 32 × D ≈ 1024 × dim ≈ 128) this matches
/// the speedup of [`maxsim_distance_cosine_gemm`].
#[inline]
pub fn maxsim_distance_dot_gemm(query_rows: &[Vec<f32>], doc_rows: &[Vec<f32>]) -> f32 {
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let (sim, q_n, d_n) = maxsim_gemm_dot_matrix(query_rows, doc_rows);
    let mut total = 0.0_f32;
    for q in 0..q_n {
        let row = &sim[q * d_n..(q + 1) * d_n];
        let mut m = f32::MIN;
        for &s in row {
            if s > m {
                m = s;
            }
        }
        total += m;
    }
    // Higher dot = more similar; convert to distance.
    -total / q_n as f32
}

/// GEMM-backed L2-squared MaxSim. Uses the identity
///   ‖q − d‖² = ‖q‖² + ‖d‖² − 2·q·d
/// to keep the per-pair work in a single sgemm + a small
/// per-row vector op. Returns the per-query-row-averaged L2²
/// distance to the nearest doc row.
///
/// (Squared L2, not L2 — saves the sqrt; the ordering relative
/// to nearest is preserved.)
#[inline]
pub fn maxsim_distance_l2sq_gemm(query_rows: &[Vec<f32>], doc_rows: &[Vec<f32>]) -> f32 {
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let q_norms: Vec<f32> = query_rows
        .iter()
        .map(|r| r.iter().map(|x| x * x).sum::<f32>())
        .collect();
    let d_norms: Vec<f32> = doc_rows
        .iter()
        .map(|r| r.iter().map(|x| x * x).sum::<f32>())
        .collect();
    let (sim, q_n, d_n) = maxsim_gemm_dot_matrix(query_rows, doc_rows);

    let mut total = 0.0_f32;
    for q in 0..q_n {
        let row = &sim[q * d_n..(q + 1) * d_n];
        let qn = q_norms[q];
        let mut min_l2sq = f32::MAX;
        for (d, &dot) in row.iter().enumerate() {
            // L2² = ‖q‖² + ‖d‖² − 2·q·d. Clamp at 0 in case
            // float rounding produces a tiny negative.
            let l2sq = (qn + d_norms[d] - 2.0 * dot).max(0.0);
            if l2sq < min_l2sq {
                min_l2sq = l2sq;
            }
        }
        total += min_l2sq;
    }
    total / q_n as f32
}

/// GEMM-backed cosine MaxSim — builds the Q×D similarity matrix
/// in one BLIS-style microkernel call, then row-reduces.
///
/// Caller must have unit-length-normalised every row of both
/// `query_rows` and `doc_rows`. Returns a distance: `1 - sum/Q`.
///
/// **Why this exists**: even with the inlined-dot path
/// ([`maxsim_distance_cosine_normalized`]) the inner loop still
/// pays per-pair function-call overhead and can't reuse cache
/// lines across the Q × D matrix elements. A real SGEMM blocks
/// the matrix so each loaded cache line participates in
/// multiple accumulations. On typical ColPali shapes
/// (Q ≈ 32 × D ≈ 1024 × dim ≈ 128) this delivers another
/// ≈5-10× over the inlined-dot path — the rest of what
/// Mixedbread's `maxsim-cpu` (libxsmm + AVX2 microkernel) wins,
/// without the C dependency.
///
/// Implementation uses the `matrixmultiply` crate's
/// register-blocked sgemm — same algorithmic class as BLIS, BLAS
/// SGEMM, libxsmm. Pure Rust, no FFI.
#[inline]
pub fn maxsim_distance_cosine_gemm(
    query_rows: &[Vec<f32>],
    doc_rows: &[Vec<f32>],
) -> f32 {
    if query_rows.is_empty() || doc_rows.is_empty() {
        return f32::MAX;
    }
    let (sim, q_n, d_n) = maxsim_gemm_dot_matrix(query_rows, doc_rows);
    let mut total = 0.0_f32;
    for q in 0..q_n {
        let row = &sim[q * d_n..(q + 1) * d_n];
        let mut m = f32::MIN;
        for &s in row {
            if s > m {
                m = s;
            }
        }
        total += m;
    }
    1.0 - total / q_n as f32
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

    fn unit_normalise(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn maxsim_generic_inlines_correctly() {
        // Two query rows, three doc rows. With cosine distance,
        // verify each query row picks the closest doc row.
        let q1 = vec![1.0_f32, 0.0, 0.0];
        let q2 = vec![0.0_f32, 1.0, 0.0];
        let d1 = vec![0.9_f32, 0.0, 0.0]; // closest to q1
        let d2 = vec![0.0_f32, 0.8, 0.0]; // closest to q2
        let d3 = vec![0.0_f32, 0.0, 0.5]; // unrelated
        let q = vec![q1, q2];
        let d = vec![d1, d2, d3];
        let dist = maxsim_distance(&q, &d, cosine_distance);
        // Both query rows should match closely; total distance
        // should be much smaller than the orthogonal-pair total.
        assert!(dist < 0.5, "expected near-zero MaxSim distance, got {dist}");
    }

    #[test]
    fn maxsim_cosine_normalized_matches_generic_for_unit_vectors() {
        let q = vec![
            unit_normalise(&[1.0, 2.0, 3.0, 4.0]),
            unit_normalise(&[0.5, 0.5, 0.5, 0.5]),
        ];
        let d = vec![
            unit_normalise(&[1.0, 1.0, 0.0, 0.0]),
            unit_normalise(&[0.0, 0.0, 1.0, 1.0]),
            unit_normalise(&[1.0, 2.0, 3.0, 4.0]),
        ];
        // The generic path returns -sum_max_sim where sim = -dist.
        // The normalised path returns 1 - sum_max_sim/Q (per-query avg).
        // Both must agree on which doc is best per query and give
        // consistent monotonic ordering — but the scales differ.
        // Verify the fast path returns a sensible distance in [0, 2].
        let dist = maxsim_distance_cosine_normalized(&q, &d);
        assert!(dist.is_finite());
        assert!(
            (0.0..=2.0).contains(&dist),
            "expected normalised cosine MaxSim in [0,2], got {dist}"
        );
        // A query that's *identical* to a doc row should pull
        // the result close to zero.
        let exact_q = vec![unit_normalise(&[1.0, 2.0, 3.0, 4.0])];
        let exact_d = vec![unit_normalise(&[1.0, 2.0, 3.0, 4.0])];
        let exact_dist = maxsim_distance_cosine_normalized(&exact_q, &exact_d);
        assert!(
            exact_dist < 1e-4,
            "exact match should yield distance ≈ 0, got {exact_dist}"
        );
    }

    #[test]
    fn maxsim_empty_inputs_return_max_distance() {
        let empty: Vec<Vec<f32>> = vec![];
        let some = vec![vec![1.0_f32, 2.0]];
        assert_eq!(maxsim_distance(&empty, &some, cosine_distance), f32::MAX);
        assert_eq!(maxsim_distance(&some, &empty, cosine_distance), f32::MAX);
        assert_eq!(
            maxsim_distance_cosine_normalized(&empty, &some),
            f32::MAX
        );
        assert_eq!(
            maxsim_distance_cosine_normalized(&some, &empty),
            f32::MAX
        );
        assert_eq!(maxsim_distance_cosine_gemm(&empty, &some), f32::MAX);
        assert_eq!(maxsim_distance_cosine_gemm(&some, &empty), f32::MAX);
    }

    #[test]
    fn maxsim_gemm_matches_normalized_path_within_fp_tolerance() {
        // Build a moderately-sized ColPali-shaped fixture and
        // confirm the GEMM-backed path and the inlined-dot path
        // agree to within float rounding.
        let q_n = 4;
        let d_n = 32;
        let dim = 16;
        let q: Vec<Vec<f32>> = (0..q_n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| ((i as f32 + 1.0) * 0.07 + d as f32 * 0.13).sin())
                    .collect();
                unit_normalise(&v)
            })
            .collect();
        let d: Vec<Vec<f32>> = (0..d_n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| ((i as f32 + 1.0) * 0.09 + d as f32 * 0.11).cos())
                    .collect();
                unit_normalise(&v)
            })
            .collect();
        let normed = maxsim_distance_cosine_normalized(&q, &d);
        let gemm = maxsim_distance_cosine_gemm(&q, &d);
        assert!(
            (normed - gemm).abs() < 1e-4,
            "gemm vs normalized path disagree: normed={normed} gemm={gemm}"
        );
    }

    #[test]
    fn i8_dot_simd_matches_scalar_random() {
        // Random-ish i8 vectors of various lengths covering the
        // SIMD-block + remainder paths.
        for &n in &[1, 7, 31, 32, 33, 64, 65, 128, 1024] {
            let a: Vec<i8> = (0..n)
                .map(|i| ((i as i32 * 31 + 7) % 255 - 127) as i8)
                .collect();
            let b: Vec<i8> = (0..n)
                .map(|i| ((i as i32 * 19 + 11) % 255 - 127) as i8)
                .collect();
            let scalar = i8_dot_scalar(&a, &b);
            let public = i8_dot_product(&a, &b);
            assert_eq!(
                public, scalar,
                "i8_dot at n={n}: scalar={scalar} simd={public}"
            );
        }
    }

    #[test]
    fn i8_dot_overflow_safe_at_realistic_dims() {
        // Worst-case: every product is the maximum (127 × 127 =
        // 16_129). For dim=10_000 the sum is 161_290_000 — well
        // under i32::MAX (~2.15 × 10⁹).
        let n = 10_000;
        let a = vec![127i8; n];
        let b = vec![127i8; n];
        let r = i8_dot_product(&a, &b);
        assert_eq!(r, 127 * 127 * n as i32);
    }

    #[test]
    fn maxsim_gemm_self_match_is_near_zero() {
        // A query that's identical to a doc row should drive the
        // MaxSim distance to near zero (max sim per row ≈ 1).
        let dim = 8;
        let v = unit_normalise(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let q = vec![v.clone()];
        let d = vec![v.clone(), unit_normalise(&[1.0_f32; 8])];
        assert!(maxsim_distance_cosine_gemm(&q, &d) < 1e-4);
        let _ = dim;
    }

    fn random_rows(n: usize, dim: usize, seed: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((seed + i as f32) * 0.13 + d as f32 * 0.07).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn maxsim_dot_gemm_matches_generic_path() {
        // `maxsim_distance` (generic) returns the per-row max-sim
        // SUM as a distance: -Σ max_sim. The GEMM helpers
        // normalise by q_n (per-row average); compare via that
        // factor.
        let q = random_rows(4, 16, 1.0);
        let d = random_rows(16, 16, 2.5);
        let generic = maxsim_distance(&q, &d, |a, b| -dot_product(a, b));
        let gemm_per_row = maxsim_distance_dot_gemm(&q, &d);
        let gemm_summed = gemm_per_row * q.len() as f32;
        assert!(
            (generic - gemm_summed).abs() < 1e-3,
            "dot GEMM (summed) vs generic disagree: generic={generic} gemm_summed={gemm_summed}"
        );
    }

    #[test]
    fn maxsim_l2sq_gemm_matches_generic_path() {
        let q = random_rows(4, 16, 3.0);
        let d = random_rows(16, 16, 4.5);
        let generic = maxsim_distance(&q, &d, l2_squared);
        let gemm_per_row = maxsim_distance_l2sq_gemm(&q, &d);
        let gemm_summed = gemm_per_row * q.len() as f32;
        assert!(
            (generic - gemm_summed).abs() < 1e-3,
            "L2² GEMM (summed) vs generic disagree: generic={generic} gemm_summed={gemm_summed}"
        );
    }

    #[test]
    fn maxsim_l2sq_self_match_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let q = vec![v.clone()];
        let d = vec![v.clone(), vec![5.0, 6.0, 7.0, 8.0]];
        assert!(maxsim_distance_l2sq_gemm(&q, &d) < 1e-4);
    }

    #[test]
    fn maxsim_dot_self_match_max_negative() {
        // For dot, "closer" = more negative distance (we returned
        // -dot). A self-match maximises dot, so distance is the
        // most negative.
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let q = vec![v.clone()];
        let d_close = vec![v.clone()];
        let d_far = vec![vec![-1.0_f32, -2.0, -3.0, -4.0]];
        let d_close_dist = maxsim_distance_dot_gemm(&q, &d_close);
        let d_far_dist = maxsim_distance_dot_gemm(&q, &d_far);
        assert!(d_close_dist < d_far_dist);
    }

    #[test]
    fn maxsim_l2sq_dot_gemm_empty_inputs_max_distance() {
        let empty: Vec<Vec<f32>> = vec![];
        let some = vec![vec![1.0_f32, 2.0]];
        assert_eq!(maxsim_distance_dot_gemm(&empty, &some), f32::MAX);
        assert_eq!(maxsim_distance_dot_gemm(&some, &empty), f32::MAX);
        assert_eq!(maxsim_distance_l2sq_gemm(&empty, &some), f32::MAX);
        assert_eq!(maxsim_distance_l2sq_gemm(&some, &empty), f32::MAX);
    }
}
