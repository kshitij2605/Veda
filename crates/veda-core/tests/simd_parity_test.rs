//! Parity tests: SimSIMD kernels (and the dispatched `simd` API) must match
//! a scalar reference. Only meaningful when the `simsimd` feature is on.
#![cfg(feature = "simsimd")]

use veda_core::simd_simsimd;

fn i8_dot_ref(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
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
