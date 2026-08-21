//! Does residual coding help the RaBitQ variants that actually rerank?
//!
//! The centroid sweep showed reconstruction error improving 20x while recall
//! stayed at ~3% for every centroid count. That says the 1-bit code cannot rank
//! within a tight cluster no matter how good the codebook is: the estimator's
//! error (~0.0033 at 128d, ~0.0014 at 768d) is a large fraction of the entire
//! intra-cluster distance spread, which the DiskANN work measured at
//! [0.0084, 0.0120] at 768d. A pure scan is therefore picking near-arbitrarily
//! from inside the right cluster.
//!
//! Reranking is the documented answer — `RaBitQRerank` and `RaBitQExtRerank`
//! exist precisely because the coarse pass is meant to produce *candidates*,
//! not an ordering. This measures whether they clear the bar on the same corpus.
//!
//! cargo test -p veda-index --release --test rabitq_variants -- --ignored --nocapture

use veda_core::types::DistanceMetric;
use veda_index::flat::FlatIndex;
use veda_index::quantization::rabitq_ext_rerank::RaBitQExtRerankIndex;
use veda_index::quantization::rabitq_rerank::RaBitQRerankIndex;
use veda_index::quantization::rabitq_vector_index::RaBitQIndex;
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

fn recall(idx: &dyn VectorIndex, truth: &[Vec<u32>], queries: &[Vec<f32>]) -> f64 {
    let mut sum = 0.0;
    for (q, t) in queries.iter().zip(truth) {
        let got = idx.search(q, 10).unwrap();
        let hit = got.iter().filter(|r| t.contains(&r.id)).count();
        sum += hit as f64 / 10.0;
    }
    sum / queries.len() as f64
}

#[test]
#[ignore]
fn variants_on_a_clustered_corpus() {
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

        println!("\n=== n={n} dim={dim} clustered, recall@10 ===");

        let t = std::time::Instant::now();
        let mut plain = RaBitQIndex::new(DistanceMetric::Cosine);
        plain.build(&refs).unwrap();
        println!(
            "RaBitQ (scan, no rerank)  {:.3}   build {:.1}s",
            recall(&plain, &truth, &queries),
            t.elapsed().as_secs_f64()
        );

        let t = std::time::Instant::now();
        let mut rr = RaBitQRerankIndex::new(DistanceMetric::Cosine);
        rr.build(&refs).unwrap();
        println!(
            "RaBitQRerank              {:.3}   build {:.1}s",
            recall(&rr, &truth, &queries),
            t.elapsed().as_secs_f64()
        );

        let t = std::time::Instant::now();
        let mut ext = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
        ext.build(&refs).unwrap();
        println!(
            "RaBitQExtRerank           {:.3}   build {:.1}s",
            recall(&ext, &truth, &queries),
            t.elapsed().as_secs_f64()
        );
    }
}
