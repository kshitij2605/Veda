//! SPANN streaming build at production scale.
//!
//! `spann_streaming_build_test` proves the *routing* — that `build_indexes`
//! takes the streaming path — on a 4,000-vector corpus. That is enough to
//! catch a regression and not enough to put a number in front of anyone.
//!
//! This measures the thing the marketing brief gates on: whether peak build
//! memory tracks the **budget** or the **corpus**. It runs three sizes at
//! 1024d — the dimension this product actually runs — and prints peak against
//! corpus for each. If peak stays roughly flat while the corpus triples, the
//! claim holds; if it grows with the corpus, it does not, however green the
//! 4k test is.
//!
//! `#[ignore]` because it materialises up to ~400 MB of vectors and takes
//! minutes. Run with:
//!   cargo test --release -p veda-query --test spann_scale_measurement \
//!     -- --ignored --nocapture

use std::sync::Arc;

use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, IndexType};
use veda_core::vector::DenseVector;
use veda_query::local_shard::LocalShardSearcher;
use veda_storage::segment_manager::SegmentManager;

/// Production dimension: `EMBEDDING_DIMENSION=1024`, and both jina-v5-omni
/// and BAAI/bge-m3 are 1024.
const DIM: u32 = 1024;
/// Deliberately small relative to every corpus below, so "peak follows the
/// budget" and "peak follows the corpus" give visibly different answers.
const BUDGET_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn schema() -> CollectionSchema {
    CollectionSchema::builder("spann_scale")
        .add_field(
            FieldDefinition::vector("embedding", DIM, DistanceMetric::Cosine)
                .with_index_type(IndexType::Spann),
        )
        .build()
        .unwrap()
}

fn measure(n: usize) -> (usize, usize, u32, u32) {
    let blob_dir = tempfile::tempdir().unwrap();
    let blob = Arc::new(veda_storage::blob_store::LocalBlobStore::new(blob_dir.path()).unwrap());
    veda_index::spann::register_store(Arc::new(veda_index::spann::BlobRangeStore(blob)));

    let tmp = tempfile::tempdir().unwrap();
    // One sealed segment, so peak is comparable against the whole corpus.
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 1usize << 40).unwrap();

    let mut rng = Lcg(0xC0FFEE);
    let centres: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..DIM as usize).map(|_| rng.next_f32()).collect())
        .collect();
    for i in 0..n {
        let c = &centres[i % centres.len()];
        let v: Vec<f32> = (0..DIM as usize)
            .map(|d| c[d] + rng.next_f32() * 0.10)
            .collect();
        let mut rec = Record::new();
        rec.set("embedding", FieldValue::Vector(DenseVector::new(v)));
        sm.insert(rec).unwrap();
    }
    sm.seal_growing().unwrap();
    let seg_id = sm.sealed_segments()[0].metadata.id;

    let mut searcher = LocalShardSearcher::new(sm);
    searcher.build_indexes().unwrap();
    let stats = searcher
        .spann_build_stats(&seg_id, "embedding")
        .expect("SPANN must build through the streaming path");

    let corpus = n * DIM as usize * std::mem::size_of::<f32>();
    (
        corpus,
        stats.peak_buffered_bytes,
        stats.passes,
        stats.groups,
    )
}

#[test]
#[ignore]
fn spann_peak_tracks_the_budget_not_the_corpus() {
    // SAFETY: single test in its own binary.
    unsafe {
        std::env::set_var("VEDA_SPANN_BUILD_BUDGET_BYTES", BUDGET_BYTES.to_string());
    }

    println!();
    println!(
        "SPANN streaming build, dim={DIM}, budget={} MiB",
        BUDGET_BYTES >> 20
    );
    println!(
        "{:>9} {:>14} {:>14} {:>9} {:>7} {:>7}",
        "vectors", "corpus(MiB)", "peak(MiB)", "peak/corpus", "passes", "groups"
    );

    let mut peaks = Vec::new();
    for n in [25_000usize, 50_000, 100_000] {
        let (corpus, peak, passes, groups) = measure(n);
        println!(
            "{:>9} {:>14.1} {:>14.1} {:>9.1}% {:>7} {:>7}",
            n,
            corpus as f64 / (1 << 20) as f64,
            peak as f64 / (1 << 20) as f64,
            100.0 * peak as f64 / corpus as f64,
            passes,
            groups
        );
        peaks.push((n, corpus, peak));
    }

    // The claim under test. Peak must not scale with the corpus: quadrupling
    // the corpus must not quadruple peak. Stated as a bound rather than a
    // ratio because the floor is the largest posting list, which `nlist`
    // governs — a budget below that adds passes without lowering peak.
    let (_, small_corpus, small_peak) = peaks[0];
    let (_, big_corpus, big_peak) = peaks[peaks.len() - 1];
    let corpus_growth = big_corpus as f64 / small_corpus as f64;
    let peak_growth = big_peak as f64 / small_peak as f64;
    println!("\ncorpus grew {corpus_growth:.1}x, peak grew {peak_growth:.1}x");
    assert!(
        peak_growth < corpus_growth / 2.0,
        "peak grew {peak_growth:.1}x against a {corpus_growth:.1}x corpus — that is \
         corpus-proportional, so the streaming build is not bounding memory"
    );
    // And the absolute claim: at the largest size, peak is a small fraction.
    assert!(
        big_peak * 4 < big_corpus,
        "peak {big_peak} should be far below the {big_corpus}-byte corpus"
    );
}
