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
use std::path::Path;
use std::time::Instant;

use veda_core::error::Result;
use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::hnsw::quantized::QuantizedHnswIndex;
use veda_index::quantization::rabitq_ext_rerank::RaBitQExtRerankIndex;
use veda_index::quantization::rabitq_rerank::RaBitQRerankIndex;
use veda_index::quantization::rabitq_vector_index::RaBitQIndex;
use veda_index::quantization::scalar::ScalarQuantizer;
use veda_index::traits::{SearchResult, VectorIndex};

// ---------------------------------------------------------------------------
// Diagnostic-only VectorIndex wrappers (bench-only, not production code)
// ---------------------------------------------------------------------------

/// SQ8: wraps Veda's existing ScalarQuantizer (8-bit per-dim).
/// build() trains + encodes; search() scans estimate_distance over all, top-k.
struct Sq8BenchIndex {
    quantizer: ScalarQuantizer,
    metric: DistanceMetric,
    n: usize,
    dim: u32,
}

impl Sq8BenchIndex {
    fn new(metric: DistanceMetric) -> Self {
        Self {
            quantizer: ScalarQuantizer::new(),
            metric,
            n: 0,
            dim: 0,
        }
    }
}

impl VectorIndex for Sq8BenchIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.n = vectors.len();
        self.dim = vectors.first().map(|v| v.len() as u32).unwrap_or(0);
        self.quantizer.train(vectors);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut scores: Vec<(u32, f32)> = (0..self.n)
            .map(|i| {
                (
                    i as u32,
                    self.quantizer.estimate_distance(query, i, self.metric),
                )
            })
            .collect();
        scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        unimplemented!()
    }
    fn delete(&mut self, _id: u32) -> Result<()> {
        unimplemented!()
    }
    fn len(&self) -> usize {
        self.n
    }
    fn save(&self, _path: &Path) -> Result<()> {
        unimplemented!()
    }
    fn load(_path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self> {
        unimplemented!()
    }
    fn metric(&self) -> DistanceMetric {
        self.metric
    }
    fn dimension(&self) -> u32 {
        self.dim
    }
}

/// SQ4 per-dim: throwaway 4-bit per-dimension scalar quantizer prototype.
/// For each dimension d, per-dim min/max over training set → 16 uniform levels
/// in [min_d, max_d]. Codes stored packed 2 dims/byte (lo nibble = even dim,
/// hi nibble = odd dim). search() reconstructs to bin midpoint and computes
/// cosine distance over the reconstructed vector.
struct Sq4PerDimBenchIndex {
    mins: Vec<f32>,
    maxs: Vec<f32>,
    /// Packed 4-bit codes: each entry has `ceil(dim/2)` bytes.
    codes: Vec<Vec<u8>>,
    metric: DistanceMetric,
    n: usize,
    dim: u32,
}

impl Sq4PerDimBenchIndex {
    fn new(metric: DistanceMetric) -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            codes: Vec::new(),
            metric,
            n: 0,
            dim: 0,
        }
    }

    fn encode_vec(&self, v: &[f32]) -> Vec<u8> {
        let dim = v.len();
        let packed_len = dim.div_ceil(2);
        let mut out = vec![0u8; packed_len];
        for (d, &val) in v.iter().enumerate() {
            let range = self.maxs[d] - self.mins[d];
            let nibble = if range < 1e-10 {
                7u8
            } else {
                let normalized = (val - self.mins[d]) / range;
                (normalized.clamp(0.0, 1.0) * 15.0).round() as u8
            };
            let byte_idx = d / 2;
            if d % 2 == 0 {
                out[byte_idx] |= nibble; // low nibble
            } else {
                out[byte_idx] |= nibble << 4; // high nibble
            }
        }
        out
    }

    fn decode_vec(&self, code: &[u8]) -> Vec<f32> {
        let dim = self.dim as usize;
        let mut out = Vec::with_capacity(dim);
        for d in 0..dim {
            let byte_idx = d / 2;
            let nibble = if d % 2 == 0 {
                code[byte_idx] & 0x0F
            } else {
                (code[byte_idx] >> 4) & 0x0F
            };
            let range = self.maxs[d] - self.mins[d];
            // decode to bin midpoint: (nibble + 0.5) / 16
            let val = self.mins[d] + ((nibble as f32 + 0.5) / 16.0) * range;
            out.push(val);
        }
        out
    }
}

impl VectorIndex for Sq4PerDimBenchIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        let dim = vectors[0].len();
        self.dim = dim as u32;
        self.n = vectors.len();

        // Per-dimension min/max
        self.mins = vec![f32::MAX; dim];
        self.maxs = vec![f32::MIN; dim];
        for v in vectors {
            for (d, &val) in v.iter().enumerate() {
                if val < self.mins[d] {
                    self.mins[d] = val;
                }
                if val > self.maxs[d] {
                    self.maxs[d] = val;
                }
            }
        }

        // Encode all vectors
        self.codes = vectors.iter().map(|v| self.encode_vec(v)).collect();
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut scores: Vec<(u32, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let reconstructed = self.decode_vec(code);
                let dist =
                    veda_index::distance::compute_distance(query, &reconstructed, self.metric);
                (i as u32, dist)
            })
            .collect();
        scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        unimplemented!()
    }
    fn delete(&mut self, _id: u32) -> Result<()> {
        unimplemented!()
    }
    fn len(&self) -> usize {
        self.n
    }
    fn save(&self, _path: &Path) -> Result<()> {
        unimplemented!()
    }
    fn load(_path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self> {
        unimplemented!()
    }
    fn metric(&self) -> DistanceMetric {
        self.metric
    }
    fn dimension(&self) -> u32 {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Scalar diagnostic runner
// ---------------------------------------------------------------------------

fn run_scalar_diag(label: &str, corpus: Vec<Vec<f32>>, dim: usize) {
    eprintln!(
        "\n=== scalar_diag: {label} n={} dim={dim} ===",
        corpus.len()
    );
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    let mut truth = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES)
        .map(|i| &corpus[(i * 7919) % corpus.len()])
        .collect();
    let truths: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| {
            truth
                .search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    print_row("flat(truth)", corpus.len(), 0.0, &truth, &queries, &truths);

    // SQ8 (8-bit per-dim): bytes/vec = dim
    let mut sq8 = Sq8BenchIndex::new(DistanceMetric::Cosine);
    sq8.build(&refs).unwrap();
    print_row("sq8(8bit/dim)", corpus.len(), 0.0, &sq8, &queries, &truths);
    eprintln!("    bytes/vec = {dim}");

    // SQ4 per-dim (4-bit per-dim): bytes/vec = ceil(dim/2)
    let mut sq4 = Sq4PerDimBenchIndex::new(DistanceMetric::Cosine);
    sq4.build(&refs).unwrap();
    print_row(
        "sq4-perdim(4b/dim)",
        corpus.len(),
        0.0,
        &sq4,
        &queries,
        &truths,
    );
    eprintln!("    bytes/vec = {}", dim.div_ceil(2));
}

#[test]
#[ignore]
fn scalar_diag_uniform_1536() {
    run_scalar_diag("uniform", make_corpus(50_000, 1536), 1536);
}

#[test]
#[ignore]
fn scalar_diag_aniso_1536() {
    run_scalar_diag("aniso", make_anisotropic_corpus(50_000, 1536, 48), 1536);
}

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

/// Anisotropic corpus: each vector is a random combination of `rank`
/// fixed basis directions plus small isotropic noise, then L2-normalized.
/// Variance concentrates in `rank` dims — mimicking real embedding
/// distributions (unlike `make_corpus`'s isotropic uniform vectors).
fn make_anisotropic_corpus(n: usize, dim: usize, rank: usize) -> Vec<Vec<f32>> {
    // Fixed random basis (rank x dim), seeded.
    let basis: Vec<Vec<f32>> = (0..rank)
        .map(|r| vec_of(1_000_000 + r as u32, dim))
        .collect();
    (0..n)
        .map(|i| {
            // Per-vector coefficients over the basis (heavy-tailed: first
            // dirs dominate) + small noise.
            let coeffs = vec_of(i as u32, rank);
            let noise = vec_of(2_000_000 + i as u32, dim);
            let mut v = vec![0.0f32; dim];
            for (r, b) in basis.iter().enumerate() {
                // decay weight so early basis dirs carry more variance
                let w = coeffs[r] / ((r + 1) as f32);
                for d in 0..dim {
                    v[d] += w * b[d];
                }
            }
            for d in 0..dim {
                v[d] += 0.05 * noise[d];
            }
            // L2 normalize (cosine space)
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

fn percentile(mut sorted: Vec<f64>, p: f64) -> f64 {
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn recall_at_k(index_ids: &[u32], truth_ids: &HashSet<u32>) -> f64 {
    let hit = index_ids
        .iter()
        .take(K)
        .filter(|id| truth_ids.contains(id))
        .count();
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
    print_row(
        "flat (truth)",
        n,
        truth_build_ms,
        &truth_idx,
        &queries,
        &truths,
    );

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

fn run_rabitq_family(label: &str, n: usize, dim: usize) {
    eprintln!();
    eprintln!("=== rabitq_family: {label}  n={n}  dim={dim} ===");
    let corpus = make_corpus(n, dim);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();

    let mut truth_idx = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth_idx.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES).map(|i| &corpus[(i * 7919) % n]).collect();
    let truths: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| {
            truth_idx
                .search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    let raw_mb = (n * dim * 4) as f64 / (1024.0 * 1024.0);
    eprintln!("  raw f32 corpus: {raw_mb:.1} MB");

    print_row("flat (truth)", n, 0.0, &truth_idx, &queries, &truths);

    let t0 = Instant::now();
    let mut rabitq = RaBitQIndex::new(DistanceMetric::Cosine);
    rabitq.build(&refs).unwrap();
    print_row(
        "rabitq",
        n,
        t0.elapsed().as_secs_f64() * 1000.0,
        &rabitq,
        &queries,
        &truths,
    );

    let t0 = Instant::now();
    let mut rerank = RaBitQRerankIndex::new(DistanceMetric::Cosine);
    rerank.build(&refs).unwrap();
    print_row(
        "rabitq+rerank",
        n,
        t0.elapsed().as_secs_f64() * 1000.0,
        &rerank,
        &queries,
        &truths,
    );

    let t0 = Instant::now();
    let mut ext = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
    ext.build(&refs).unwrap();
    print_row(
        "rabitq+extrerank",
        n,
        t0.elapsed().as_secs_f64() * 1000.0,
        &ext,
        &queries,
        &truths,
    );
}

#[test]
#[ignore]
fn rabitq_family_1536() {
    run_rabitq_family("d1536", 100_000, 1536);
}

#[test]
#[ignore]
fn rabitq_family_3072() {
    run_rabitq_family("d3072", 100_000, 3072);
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

fn analytical_bytes_per_vec(label: &str, dim: usize) -> f64 {
    // 1-bit coarse = dim/8; raw-f32 rerank = dim*4; 4-bit ext rerank = dim*0.5
    match label {
        "rabitq" => (dim as f64) / 8.0 + 4.0, // codes + norm
        "rabitq+rerank" => (dim as f64) / 8.0 + (dim as f64) * 4.0, // + raw f32
        l if l.starts_with("extrerank") => (dim as f64) / 8.0 + (dim as f64) * 0.5, // + 4-bit
        _ => 0.0,
    }
}

fn run_anisotropic_ab(n: usize, dim: usize, rank: usize) {
    eprintln!();
    eprintln!("=== anisotropic A/B: n={n} dim={dim} rank={rank} ===");
    let corpus = make_anisotropic_corpus(n, dim, rank);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    let mut truth = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES).map(|i| &corpus[(i * 7919) % n]).collect();
    let truths: Vec<std::collections::HashSet<u32>> = queries
        .iter()
        .map(|q| {
            truth
                .search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    let mut rerank = RaBitQRerankIndex::new(DistanceMetric::Cosine);
    rerank.build(&refs).unwrap();
    print_row("rabitq+rerank", n, 0.0, &rerank, &queries, &truths);
    eprintln!(
        "    bytes/vec ≈ {:.0}",
        analytical_bytes_per_vec("rabitq+rerank", dim)
    );

    let mut ext_default = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
    ext_default.build(&refs).unwrap();
    print_row("extrerank-default", n, 0.0, &ext_default, &queries, &truths);
    eprintln!(
        "    bytes/vec ≈ {:.0}",
        analytical_bytes_per_vec("extrerank", dim)
    );

    let mut ext_rot = RaBitQExtRerankIndex::new_with_rotation(DistanceMetric::Cosine);
    ext_rot.build(&refs).unwrap();
    print_row("extrerank-rotation", n, 0.0, &ext_rot, &queries, &truths);
    eprintln!(
        "    bytes/vec ≈ {:.0}",
        analytical_bytes_per_vec("extrerank", dim)
    );
}

#[test]
#[ignore]
fn anisotropic_ab_1536() {
    run_anisotropic_ab(50_000, 1536, 48);
}

#[test]
#[ignore]
fn sq_recall_targets_1536() {
    use veda_index::quantization::scalar_index::ScalarQuantIndex;
    let dim = 1536usize;
    let n = 50_000usize;
    let corpus = make_corpus(n, dim);
    let refs: Vec<&[f32]> = corpus.iter().map(|v| v.as_slice()).collect();
    let mut truth = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
    truth.build(&refs).unwrap();
    let queries: Vec<&Vec<f32>> = (0..QUERIES).map(|i| &corpus[(i * 7919) % n]).collect();
    let truths: Vec<std::collections::HashSet<u32>> = queries
        .iter()
        .map(|q| {
            truth
                .search(q, K)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();

    // Helper: compute mean recall@K over all queries for a given index.
    fn measure_recall(
        idx: &dyn veda_index::traits::VectorIndex,
        queries: &[&Vec<f32>],
        truths: &[std::collections::HashSet<u32>],
        name: &str,
    ) -> f64 {
        let mut sum = 0.0f64;
        for (q, t) in queries.iter().zip(truths) {
            let ids: Vec<u32> = idx.search(q, K).unwrap().iter().map(|h| h.id).collect();
            sum += recall_at_k(&ids, t);
        }
        let r = sum / queries.len() as f64;
        eprintln!("{name} recall@10={r:.3}");
        r
    }

    let mut sq8 = ScalarQuantIndex::sq8(DistanceMetric::Cosine);
    sq8.build(&refs).unwrap();
    let mut sq4 = ScalarQuantIndex::sq4(DistanceMetric::Cosine);
    sq4.build(&refs).unwrap();

    eprintln!("sq8 = {} B/vec, sq4 = {} B/vec", dim, dim / 2);

    let r8 = measure_recall(&sq8, &queries, &truths, "sq8(8bit)");
    let r4 = measure_recall(&sq4, &queries, &truths, "sq4(4bit)");

    assert!(r8 >= 0.97, "sq8 recall {r8:.3} below 0.97");
    assert!(r4 >= 0.82, "sq4 recall {r4:.3} below 0.82 (turbovec ~0.85)");
}
