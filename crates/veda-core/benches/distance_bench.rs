//! Simple benchmark comparing SIMD-dispatched distance functions.
//!
//! Run with: `cargo bench -p veda-core`

use std::hint::black_box;
use std::time::Instant;

use veda_core::simd;

/// Generate a deterministic pseudo-random vector.
fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let val = ((state >> 16) as f32 / 32768.0) - 1.0;
        v.push(val);
    }
    v
}

fn bench_fn(name: &str, iterations: u32, f: impl Fn()) {
    // Warm-up
    for _ in 0..100 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!("{name:>30}: {per_iter_ns:>8.1} ns/iter ({iterations} iterations)");
}

fn main() {
    let dimensions = [128, 256, 512, 768, 1024, 1536];
    let iterations = 100_000;

    println!("Distance function benchmarks (SIMD-dispatched)");
    println!("===============================================\n");

    for &dim in &dimensions {
        let a = make_vector(dim, 42);
        let b = make_vector(dim, 99);

        println!("--- Dimension: {dim} ---");

        bench_fn(&format!("dot_product({dim})"), iterations, || {
            black_box(simd::dot_product(black_box(&a), black_box(&b)));
        });

        bench_fn(&format!("l2_squared({dim})"), iterations, || {
            black_box(simd::l2_squared(black_box(&a), black_box(&b)));
        });

        bench_fn(&format!("euclidean_distance({dim})"), iterations, || {
            black_box(simd::euclidean_distance(black_box(&a), black_box(&b)));
        });

        bench_fn(&format!("cosine_distance({dim})"), iterations, || {
            black_box(simd::cosine_distance(black_box(&a), black_box(&b)));
        });

        bench_fn(&format!("manhattan_distance({dim})"), iterations, || {
            black_box(simd::manhattan_distance(black_box(&a), black_box(&b)));
        });

        bench_fn(&format!("l2_norm_squared({dim})"), iterations, || {
            black_box(simd::l2_norm_squared(black_box(&a)));
        });

        println!();
    }
}
