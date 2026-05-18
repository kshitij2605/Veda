//! Scale benchmark for VectorIndex variants.
//!
//! Calibrates the literature claims ("RaBitQ wins above 1M rows",
//! "deep-quantized HNSW pays off at 10M") against the actual code in
//! this tree. Each test:
//! 1. Generates `n` deterministic 1024-d random vectors with a seeded
//!    RNG (no allocations beyond the corpus).
//! 2. Builds Flat (ground truth), HNSW, RaBitQ, and RaBitQRerank.
//! 3. Runs `Q` queries, measures p50 + p95 wall-clock latency.
//! 4. Computes top-10 recall against Flat's exact answers.
//! 5. Prints one table line per index variant.
//!
//! The 10k case runs by default. Bigger cases are `#[ignore]`d so they
//! don't gate `cargo test --workspace`; run them with
//! `cargo test --test scale_benchmark -- --ignored --nocapture`.

use std::collections::HashSet;
use std::time::Instant;

use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::hnsw::quantized::QuantizedHnswIndex;
use veda_index::quantization::rabitq_rerank::RaBitQRerankIndex;
use veda_index::quantization::rabitq_vector_index::RaBitQIndex;
use veda_index::traits::VectorIndex;

const QUERIES: usize = 100;
const K: usize = 10;

/// Cheap deterministic pseudo-random vector. Same algorithm as the
/// unit-test helper in the quantization tests so behaviour is
/// reproducible across files.
fn vec_of(seed: u32, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..dim)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn make_corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n as u32).map(|i| vec_of(i, dim)).collect()
}

fn percentile(mut sorted: Vec<f64>, p: f64) -> f64 {
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn recall_at_k(index_ids: &[u32], truth_ids: &HashSet<u32>) -> f64 {
    let hit = index_ids.iter().take(K).filter(|id| truth_ids.contains(id)).count();
    hit as f64 / K as f64
}

fn run_scale(label: &str, n: usize, dim: usize) {
    eprintln!();
    eprintln!("=== scale_benchmark: {label}  n={n}  dim={dim} ===");
    let corpus = make_corpus(n, dim);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();

    // Ground truth (Flat exact distance).
    let t0 = Instant::now();
    let mut truth_idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth_idx.build(&refs).unwrap();
    let truth_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Pick a few corpus members as queries (so we know the expected answer).
    let queries: Vec<&Vec<f32>> = (0..QUERIES)
        .map(|i| {
            // Stride across the corpus to vary the regions hit.
            let q_idx = (i * 7919) % n;
            &corpus[q_idx]
        })
        .collect();
    let truths: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| {
            let hits = truth_idx.search(q, K).unwrap();
            hits.into_iter().map(|h| h.id).collect()
        })
        .collect();

    // Print the truth row first (latency = 100% recall by construction).
    print_row("flat (truth)", n, truth_build_ms, &truth_idx, &queries, &truths);

    // HNSW.
    let t0 = Instant::now();
    let mut hnsw = HnswIndex::new(DistanceMetric::Cosine, dim as u32);
    hnsw.build(&refs).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    print_row("hnsw", n, build_ms, &hnsw, &queries, &truths);

    // RaBitQ (1-bit quantized brute-force).
    let t0 = Instant::now();
    let mut rabitq = RaBitQIndex::new(DistanceMetric::Cosine);
    rabitq.build(&refs).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    print_row("rabitq", n, build_ms, &rabitq, &queries, &truths);

    // RaBitQ + raw rerank (two-stage).
    let t0 = Instant::now();
    let mut rerank = RaBitQRerankIndex::new(DistanceMetric::Cosine);
    rerank.build(&refs).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    print_row("rabitq+rerank", n, build_ms, &rerank, &queries, &truths);

    // Deep-quantized HNSW: graph + RaBitQ codes, no raw vectors.
    let t0 = Instant::now();
    let mut qhnsw = QuantizedHnswIndex::new(DistanceMetric::Cosine, dim as u32);
    qhnsw.build(&refs).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    print_row("hnsw+rabitq", n, build_ms, &qhnsw, &queries, &truths);
}

fn print_row<I: VectorIndex + ?Sized>(
    label: &str,
    _n: usize,
    build_ms: f64,
    idx: &I,
    queries: &[&Vec<f32>],
    truths: &[HashSet<u32>],
) {
    let mut latencies_ms = Vec::with_capacity(queries.len());
    let mut recalls = Vec::with_capacity(queries.len());
    for (q, truth) in queries.iter().zip(truths) {
        let t0 = Instant::now();
        let hits = idx.search(q, K).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(dt);
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        recalls.push(recall_at_k(&ids, truth));
    }
    let p50 = percentile(latencies_ms.clone(), 0.5);
    let p95 = percentile(latencies_ms.clone(), 0.95);
    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    eprintln!(
        "  {:<14}  build={:>9.1}ms  p50={:>7.3}ms  p95={:>7.3}ms  recall@10={:.3}",
        label, build_ms, p50, p95, mean_recall
    );
}

#[test]
fn scale_10k_dim_128() {
    // Default smoke — runs under `cargo test --workspace`. Small dim
    // keeps the wall-clock under a few seconds.
    run_scale("smoke", 10_000, 128);
}

#[test]
#[ignore]
fn scale_100k_dim_1024() {
    run_scale("medium", 100_000, 1024);
}

#[test]
#[ignore]
fn scale_1m_dim_768() {
    // 1M rows, dim 768 — close to a real BGE-M3 production shard.
    // Runs ~10-15 min on a desktop; gated behind --ignored.
    run_scale("large", 1_000_000, 768);
}
