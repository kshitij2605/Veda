//! ANN Benchmark Suite for Veda Index Types
//!
//! Measures recall and performance of all index types against FlatIndex ground truth.
//! Run with: cargo test -p veda-index --test ann_benchmark -- --ignored --nocapture

use std::collections::HashSet;
use std::time::Instant;

use veda_core::types::DistanceMetric;
use veda_index::diskann::DiskAnnIndex;
use veda_index::distance::compute_distance;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::ivfpq::IvfPqIndex;
use veda_index::quantization::rabitq::RaBitQuantizer;
use veda_index::quantization::scalar::ScalarQuantizer;
use veda_index::traits::{SearchResult, VectorIndex};

// ---------------------------------------------------------------------------
// Deterministic PRNG (LCG) for reproducible benchmarks
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 32) as u32 as f32) / (u32::MAX as f32)
    }

    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

fn generate_random_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32_signed()).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Ground truth and recall
// ---------------------------------------------------------------------------

fn compute_ground_truth(flat: &FlatIndex, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| flat.search(q, k).unwrap().iter().map(|r| r.id).collect())
        .collect()
}

fn compute_recall(predicted: &[Vec<SearchResult>], ground_truth: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(predicted.len(), ground_truth.len());
    if predicted.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for (pred, gt) in predicted.iter().zip(ground_truth.iter()) {
        let gt_set: HashSet<u32> = gt.iter().take(k).copied().collect();
        let found = pred
            .iter()
            .take(k)
            .filter(|r| gt_set.contains(&r.id))
            .count();
        total += found as f64 / gt_set.len().max(1) as f64;
    }
    total / predicted.len() as f64
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

struct BenchmarkResult {
    index_type: String,
    build_time_ms: f64,
    queries_per_second: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    memory_estimate_mb: f64,
}

// ---------------------------------------------------------------------------
// Table printing
// ---------------------------------------------------------------------------

fn print_header(dataset_size: usize, dimension: usize, metric: &str) {
    println!();
    println!("\u{2554}{:\u{2550}<78}\u{2557}", "");
    println!("\u{2551} {:^76} \u{2551}", "Veda ANN Benchmark Results");
    println!(
        "\u{2551} {:^76} \u{2551}",
        format!(
            "Dataset: {} vectors, dim={}, {}",
            dataset_size, dimension, metric
        )
    );
    println!(
        "\u{2560}{:\u{2550}<15}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<17}\u{2563}",
        "", "", "", "", "", ""
    );
    println!(
        "\u{2551} {:<13} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>15} \u{2551}",
        "Index", "Build(ms)", "QPS", "R@10", "R@100", "Memory(MB)"
    );
    println!(
        "\u{2560}{:\u{2550}<15}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<17}\u{2563}",
        "", "", "", "", "", ""
    );
}

fn print_row(r: &BenchmarkResult) {
    println!(
        "\u{2551} {:<13} \u{2551} {:>8.1} \u{2551} {:>8.0} \u{2551} {:>7.1}% \u{2551} {:>7.1}% \u{2551} {:>15.1} \u{2551}",
        r.index_type,
        r.build_time_ms,
        r.queries_per_second,
        r.recall_at_10 * 100.0,
        r.recall_at_100 * 100.0,
        r.memory_estimate_mb,
    );
}

fn print_footer() {
    println!(
        "\u{255A}{:\u{2550}<15}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<17}\u{255D}",
        "", "", "", "", "", ""
    );
}

// ---------------------------------------------------------------------------
// Index benchmark runners
// ---------------------------------------------------------------------------

fn benchmark_flat(
    dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: u32,
    metric: DistanceMetric,
) -> (FlatIndex, BenchmarkResult) {
    let refs: Vec<&[f32]> = dataset.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let mut flat = FlatIndex::new(metric, dim);
    flat.build(&refs).unwrap();
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let mut all_results = Vec::with_capacity(queries.len());
    for q in queries {
        all_results.push(flat.search(q, 100).unwrap());
    }
    let query_time = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / query_time;

    let memory_mb = (dataset.len() * dim as usize * 4) as f64 / (1024.0 * 1024.0);

    let result = BenchmarkResult {
        index_type: "Flat".to_string(),
        build_time_ms,
        queries_per_second: qps,
        recall_at_10: 1.0,
        recall_at_100: 1.0,
        memory_estimate_mb: memory_mb,
    };

    (flat, result)
}

// Benchmark harness helper: the parameter list mirrors the knobs under test;
// a params struct would obscure the call sites without reuse.
#[allow(clippy::too_many_arguments)]
fn benchmark_hnsw(
    dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: u32,
    metric: DistanceMetric,
    gt_10: &[Vec<u32>],
    gt_100: &[Vec<u32>],
    ef_search: usize,
    label: &str,
) -> BenchmarkResult {
    let refs: Vec<&[f32]> = dataset.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let mut hnsw = HnswIndex::with_params(metric, dim, 16, 200, ef_search);
    hnsw.build(&refs).unwrap();
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let results_100: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| hnsw.search(q, 100).unwrap())
        .collect();
    let query_time = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / query_time;

    let results_10: Vec<Vec<SearchResult>> = results_100
        .iter()
        .map(|r| r.iter().take(10).cloned().collect())
        .collect();

    let recall_10 = compute_recall(&results_10, gt_10, 10);
    let recall_100 = compute_recall(&results_100, gt_100, 100);

    // HNSW memory: vectors + graph overhead (~M*2 neighbors per node * 4 bytes per id * levels)
    let vector_bytes = dataset.len() * dim as usize * 4;
    let graph_overhead = dataset.len() * 16 * 2 * 4; // rough: M * 2 * sizeof(u32) per node
    let memory_mb = (vector_bytes + graph_overhead) as f64 / (1024.0 * 1024.0);

    BenchmarkResult {
        index_type: label.to_string(),
        build_time_ms,
        queries_per_second: qps,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        memory_estimate_mb: memory_mb,
    }
}

fn benchmark_diskann(
    dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: u32,
    metric: DistanceMetric,
    gt_10: &[Vec<u32>],
    gt_100: &[Vec<u32>],
) -> BenchmarkResult {
    let refs: Vec<&[f32]> = dataset.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let mut diskann = DiskAnnIndex::with_params(metric, dim, 64, 100, 1.2, 100);
    diskann.build(&refs).unwrap();
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let results_100: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| diskann.search(q, 100).unwrap())
        .collect();
    let query_time = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / query_time;

    let results_10: Vec<Vec<SearchResult>> = results_100
        .iter()
        .map(|r| r.iter().take(10).cloned().collect())
        .collect();

    let recall_10 = compute_recall(&results_10, gt_10, 10);
    let recall_100 = compute_recall(&results_100, gt_100, 100);

    // DiskANN memory: vectors + graph (R neighbors per node)
    let vector_bytes = dataset.len() * dim as usize * 4;
    let graph_overhead = dataset.len() * 64 * 4; // R=64 neighbors * sizeof(u32)
    let memory_mb = (vector_bytes + graph_overhead) as f64 / (1024.0 * 1024.0);

    BenchmarkResult {
        index_type: "DiskANN".to_string(),
        build_time_ms,
        queries_per_second: qps,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        memory_estimate_mb: memory_mb,
    }
}

fn benchmark_ivfpq(
    dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: u32,
    metric: DistanceMetric,
    gt_10: &[Vec<u32>],
    gt_100: &[Vec<u32>],
    nlist: usize,
) -> BenchmarkResult {
    let refs: Vec<&[f32]> = dataset.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let mut ivfpq = IvfPqIndex::with_params(metric, dim, nlist, 8, 8, 8);
    ivfpq.build(&refs).unwrap();
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    let results_100: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| ivfpq.search(q, 100).unwrap())
        .collect();
    let query_time = start.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / query_time;

    let results_10: Vec<Vec<SearchResult>> = results_100
        .iter()
        .map(|r| r.iter().take(10).cloned().collect())
        .collect();

    let recall_10 = compute_recall(&results_10, gt_10, 10);
    let recall_100 = compute_recall(&results_100, gt_100, 100);

    // IVF-PQ memory: PQ codes (n * m_pq bytes) + centroids + overhead
    let pq_bytes = dataset.len() * 8; // m_pq=8 sub-quantizers, 1 byte each
    let centroid_bytes = nlist * dim as usize * 4;
    let pq_centroid_bytes = 8 * 256 * (dim as usize / 8) * 4; // m_pq * 256 codes * sub_dim * f32
    let memory_mb = (pq_bytes + centroid_bytes + pq_centroid_bytes) as f64 / (1024.0 * 1024.0);

    BenchmarkResult {
        index_type: "IVF-PQ".to_string(),
        build_time_ms,
        queries_per_second: qps,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        memory_estimate_mb: memory_mb,
    }
}

// ---------------------------------------------------------------------------
// Core benchmark runner
// ---------------------------------------------------------------------------

fn run_benchmark(dataset_size: usize, dimension: usize) {
    let metric = DistanceMetric::Cosine;
    let num_queries = 100;
    let dim = dimension as u32;

    println!(
        "\n  Generating dataset: {} vectors x {} dimensions...",
        dataset_size, dimension
    );
    let dataset = generate_random_dataset(dataset_size, dimension, 42);
    let queries = generate_random_dataset(num_queries, dimension, 12345);

    // 1. Flat index (ground truth)
    println!("  Building FlatIndex (ground truth)...");
    let (flat, flat_result) = benchmark_flat(&dataset, &queries, dim, metric);

    let gt_10 = compute_ground_truth(&flat, &queries, 10);
    let gt_100 = compute_ground_truth(&flat, &queries, 100);

    // 2. HNSW (default ef_search=128)
    println!("  Building HnswIndex (ef_search=128)...");
    let hnsw_result = benchmark_hnsw(
        &dataset, &queries, dim, metric, &gt_10, &gt_100, 128, "HNSW",
    );

    // 3. HNSW (tuned ef_search=256)
    println!("  Building HnswIndex (ef_search=256)...");
    let hnsw_tuned_result = benchmark_hnsw(
        &dataset,
        &queries,
        dim,
        metric,
        &gt_10,
        &gt_100,
        256,
        "HNSW (tuned)",
    );

    // 4. DiskANN
    println!("  Building DiskAnnIndex...");
    let diskann_result = benchmark_diskann(&dataset, &queries, dim, metric, &gt_10, &gt_100);

    // 5. IVF-PQ (nlist scaled to dataset size)
    let nlist = if dataset_size <= 10_000 { 64 } else { 128 };
    println!("  Building IvfPqIndex (nlist={})...", nlist);
    let ivfpq_result = benchmark_ivfpq(&dataset, &queries, dim, metric, &gt_10, &gt_100, nlist);

    // Print results table
    print_header(dataset_size, dimension, "Cosine");
    print_row(&flat_result);
    print_row(&hnsw_result);
    print_row(&hnsw_tuned_result);
    print_row(&diskann_result);
    print_row(&ivfpq_result);
    print_footer();
}

// ---------------------------------------------------------------------------
// Quantization benchmarks
// ---------------------------------------------------------------------------

fn run_quantization_benchmark() {
    let dataset_size = 10_000;
    let dimension = 128;
    let num_queries = 100;
    let metric = DistanceMetric::Cosine;

    println!(
        "\n  Generating quantization benchmark dataset: {} vectors x {} dimensions...",
        dataset_size, dimension
    );
    let dataset = generate_random_dataset(dataset_size, dimension, 42);
    let queries = generate_random_dataset(num_queries, dimension, 12345);
    let refs: Vec<&[f32]> = dataset.iter().map(|v| v.as_slice()).collect();

    // Ground truth via exact distances
    let mut flat = FlatIndex::new(metric, dimension as u32);
    flat.build(&refs).unwrap();
    let gt_10 = compute_ground_truth(&flat, &queries, 10);

    // ── RaBitQ ──
    println!("  Training RaBitQuantizer...");
    let start = Instant::now();
    let mut rabitq = RaBitQuantizer::new(metric);
    rabitq.train(&refs);
    let rabitq_train_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Encoding speed (re-encode all vectors)
    let start = Instant::now();
    for v in &dataset {
        let _ = rabitq.encode(v);
    }
    let rabitq_encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Search speed and recall
    let skip = HashSet::new();
    let start = Instant::now();
    let rabitq_results: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| rabitq.search(q, 10, &skip))
        .collect();
    let rabitq_search_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rabitq_qps = num_queries as f64 / (rabitq_search_ms / 1000.0);

    // Recall for RaBitQ
    let rabitq_predicted: Vec<Vec<SearchResult>> = rabitq_results
        .iter()
        .map(|r| {
            r.iter()
                .map(|&(id, distance)| SearchResult { id, distance })
                .collect()
        })
        .collect();
    let rabitq_recall = compute_recall(&rabitq_predicted, &gt_10, 10);

    // Memory: 1 bit per dimension + norm (f32) per vector + centroid
    let rabitq_bits = dataset_size * dimension; // 1 bit per dim per vector
    let rabitq_norms = dataset_size * 4; // f32 per vector
    let rabitq_centroid = dimension * 4; // f32 per dim
    let rabitq_mem_mb =
        (rabitq_bits / 8 + rabitq_norms + rabitq_centroid) as f64 / (1024.0 * 1024.0);

    // ── ScalarQuantizer ──
    println!("  Training ScalarQuantizer...");
    let start = Instant::now();
    let mut sq = ScalarQuantizer::new();
    sq.train(&refs);
    let sq_train_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Encoding speed
    let start = Instant::now();
    for v in &dataset {
        let _ = sq.encode(v);
    }
    let sq_encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Distance estimation speed and recall
    let start = Instant::now();
    let sq_results: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<SearchResult> = (0..dataset_size)
                .map(|i| SearchResult {
                    id: i as u32,
                    distance: sq.estimate_distance(q, i, metric),
                })
                .collect();
            dists.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            dists.truncate(10);
            dists
        })
        .collect();
    let sq_search_ms = start.elapsed().as_secs_f64() * 1000.0;
    let sq_qps = num_queries as f64 / (sq_search_ms / 1000.0);

    let sq_recall = compute_recall(&sq_results, &gt_10, 10);

    // SQ memory: 1 byte per dimension per vector + mins/maxs
    let sq_bytes = dataset_size * dimension; // 1 byte per dim
    let sq_params = dimension * 4 * 2; // mins + maxs
    let sq_mem_mb = (sq_bytes + sq_params) as f64 / (1024.0 * 1024.0);

    // Compute distance estimation accuracy (mean absolute error vs exact)
    let mut rabitq_total_err = 0.0f64;
    let mut sq_total_err = 0.0f64;
    let sample_count = num_queries * 100; // check first 100 vectors per query

    for q in &queries {
        for (i, vec) in dataset.iter().enumerate().take(100) {
            let exact = compute_distance(q, vec, metric);
            let rabitq_est = rabitq.estimate_distance(q, i);
            let sq_est = sq.estimate_distance(q, i, metric);
            rabitq_total_err += (exact - rabitq_est).abs() as f64;
            sq_total_err += (exact - sq_est).abs() as f64;
        }
    }
    let rabitq_mae = rabitq_total_err / sample_count as f64;
    let sq_mae = sq_total_err / sample_count as f64;

    // Print quantization results
    println!();
    println!("\u{2554}{:\u{2550}<78}\u{2557}", "");
    println!(
        "\u{2551} {:^76} \u{2551}",
        "Veda Quantization Benchmark Results"
    );
    println!(
        "\u{2551} {:^76} \u{2551}",
        format!(
            "Dataset: {} vectors, dim={}, Cosine",
            dataset_size, dimension
        )
    );
    println!(
        "\u{2560}{:\u{2550}<13}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<9}\u{2563}",
        "", "", "", "", "", "", ""
    );
    println!(
        "\u{2551} {:<11} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>8} \u{2551} {:>7} \u{2551}",
        "Quantizer", "Train(ms)", "Enc(ms)", "QPS", "R@10", "MAE", "Mem(MB)"
    );
    println!(
        "\u{2560}{:\u{2550}<13}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<10}\u{256C}{:\u{2550}<9}\u{2563}",
        "", "", "", "", "", "", ""
    );
    println!(
        "\u{2551} {:<11} \u{2551} {:>8.1} \u{2551} {:>8.1} \u{2551} {:>8.0} \u{2551} {:>7.1}% \u{2551} {:>8.4} \u{2551} {:>7.2} \u{2551}",
        "RaBitQ",
        rabitq_train_ms,
        rabitq_encode_ms,
        rabitq_qps,
        rabitq_recall * 100.0,
        rabitq_mae,
        rabitq_mem_mb,
    );
    println!(
        "\u{2551} {:<11} \u{2551} {:>8.1} \u{2551} {:>8.1} \u{2551} {:>8.0} \u{2551} {:>7.1}% \u{2551} {:>8.4} \u{2551} {:>7.2} \u{2551}",
        "Scalar(SQ8)",
        sq_train_ms,
        sq_encode_ms,
        sq_qps,
        sq_recall * 100.0,
        sq_mae,
        sq_mem_mb,
    );
    println!(
        "\u{255A}{:\u{2550}<13}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<10}\u{2569}{:\u{2550}<9}\u{255D}",
        "", "", "", "", "", "", ""
    );
}

// ---------------------------------------------------------------------------
// Test entry points (all #[ignore] so they don't run in normal cargo test)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn ann_benchmark_10k_128d() {
    println!("\n{}", "=".repeat(70));
    println!("  ANN Benchmark: 10,000 vectors, 128 dimensions, Cosine");
    println!("{}", "=".repeat(70));
    run_benchmark(10_000, 128);
}

#[test]
#[ignore]
fn ann_benchmark_10k_768d() {
    println!("\n{}", "=".repeat(70));
    println!("  ANN Benchmark: 10,000 vectors, 768 dimensions, Cosine");
    println!("{}", "=".repeat(70));
    run_benchmark(10_000, 768);
}

#[test]
#[ignore]
fn ann_benchmark_50k_128d() {
    println!("\n{}", "=".repeat(70));
    println!("  ANN Benchmark: 50,000 vectors, 128 dimensions, Cosine");
    println!("{}", "=".repeat(70));
    run_benchmark(50_000, 128);
}

#[test]
#[ignore]
fn ann_benchmark_50k_768d() {
    println!("\n{}", "=".repeat(70));
    println!("  ANN Benchmark: 50,000 vectors, 768 dimensions, Cosine");
    println!("{}", "=".repeat(70));
    run_benchmark(50_000, 768);
}

#[test]
#[ignore]
fn ann_benchmark_quantization() {
    println!("\n{}", "=".repeat(70));
    println!("  Quantization Benchmark: 10,000 vectors, 128 dimensions, Cosine");
    println!("{}", "=".repeat(70));
    run_quantization_benchmark();
}
