//! Op-level SIMD speedup, run on demand:
//!   cargo test -p veda-core --test simd_speedup_bench -- --ignored --nocapture
#![cfg(feature = "simsimd")]

use std::time::Instant;
use veda_core::simd_simsimd;

fn seeded(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345); ((s >> 16) as f32 / 32768.0) - 1.0 }).collect()
}
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

#[test]
#[ignore]
fn dot_product_speedup() {
    let iters = 200_000;
    for &dim in &[768usize, 1024, 1536, 3072] {
        let a = seeded(1, dim);
        let b = seeded(2, dim);

        let t = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters { acc += dot_scalar(&a, &b); }
        let scalar_ns = t.elapsed().as_nanos() as f64 / iters as f64;
        std::hint::black_box(acc);

        let t = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters { acc += simd_simsimd::dot_product(&a, &b); }
        let ss_ns = t.elapsed().as_nanos() as f64 / iters as f64;
        std::hint::black_box(acc);

        eprintln!("dim={dim:>5}  scalar={scalar_ns:>8.1}ns  simsimd={ss_ns:>8.1}ns  speedup={:.2}x", scalar_ns / ss_ns);
    }
}
