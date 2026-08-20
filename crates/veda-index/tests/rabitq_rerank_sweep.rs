//! How many candidates does RaBitQ's rerank stage actually need?
//!
//! On a clustered corpus the 1-bit coarse pass ranks near-arbitrarily *within*
//! the correct cluster: its distance-estimate error is a large fraction of the
//! whole intra-cluster spread, which the DiskANN work measured at
//! [0.0084, 0.0120] at 768d. Improving the codebook does not fix that — a
//! k-means-residual rewrite cut reconstruction error 20x and made recall
//! *worse*.
//!
//! What governs rerank recall instead is how deep the candidate pool is, since
//! stage 2 re-scores exactly and cannot recover a neighbour stage 1 never
//! returned. `DEFAULT_RERANK_MULTIPLIER = 10` means k=10 inspects 100 of 10,000
//! rows — 1% of the corpus. This measures recall and cost against that depth.
//!
//! cargo test -p veda-index --release --test rabitq_rerank_sweep -- --ignored --nocapture

use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::quantization::rabitq_rerank::RaBitQRerankIndex;
use veda_index::traits::VectorIndex;

const CENTROID_SEED: u64 = 0xC0FF_EE01;
const CLUSTER_SIGMA: f32 = 0.10;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn clustered(n: usize, dim: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut crng = Lcg(CENTROID_SEED);
    let centres: Vec<Vec<f32>> = (0..k)
        .map(|_| (0..dim).map(|_| crng.f()).collect())
        .collect();
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let idx = if n >= k { i % k } else { i * k / n.max(1) };
            let c = &centres[idx];
            (0..dim).map(|d| c[d] + rng.f() * CLUSTER_SIGMA).collect()
        })
        .collect()
}

#[test]
#[ignore]
fn sweep_rerank_depth() {
    for &dim in &[128usize, 768] {
        let n = 10_000;
        let data = clustered(n, dim, 10, 42);
        let queries = clustered(50, dim, 10, 12345);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut flat = FlatIndex::new(DistanceMetric::Cosine, dim as u32);
        flat.build(&refs).unwrap();
        let truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| {
                flat.search(q, 10)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        // Exact scan, for the latency the rerank path is trying to beat.
        let t = std::time::Instant::now();
        for q in &queries {
            std::hint::black_box(flat.search(q, 10).unwrap());
        }
        let flat_us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;

        // Build ONCE and vary only the multiplier. Rebuilding inside the loop
        // charged the first iteration for cache warm-up and produced the
        // nonsense that a deeper candidate pool was *faster* than a shallow
        // one.
        let mut idx = RaBitQRerankIndex::new(DistanceMetric::Cosine);
        idx.build(&refs).unwrap();
        for q in queries.iter().take(10) {
            std::hint::black_box(idx.search(q, 10).unwrap()); // warm up
        }

        println!(
            "\n=== n={n} dim={dim} clustered — Flat is {flat_us:.0} us/query at recall 1.000 ==="
        );
        println!("  mult  candidates  recall@10  us/query  vs Flat");
        for &mult in &[10usize, 25, 50, 100, 200, 500] {
            idx.set_rerank_multiplier(mult);

            let mut sum = 0.0;
            for (q, tr) in queries.iter().zip(&truth) {
                let got = idx.search(q, 10).unwrap();
                sum += got.iter().filter(|r| tr.contains(&r.id)).count() as f64 / 10.0;
            }
            let t = std::time::Instant::now();
            for q in &queries {
                std::hint::black_box(idx.search(q, 10).unwrap());
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
            println!(
                "{mult:6}  {:10}     {:.3}  {us:8.0}   {:.1}x",
                (10 * mult).min(n),
                sum / queries.len() as f64,
                us / flat_us
            );
        }
    }
}
