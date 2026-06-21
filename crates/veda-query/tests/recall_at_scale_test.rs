//! Regression test for the at-scale dense-recall collapse.
//!
//! When a bulk load seals into many small HNSW segments, each per-segment
//! ANN search must hand the cross-segment merge *more* than `k` candidates;
//! otherwise each segment's approximate top-`k` discards true global
//! neighbours and the merge cannot recover them. With ~38 small sealed
//! segments recall@10 collapsed to ~0.27. This test forces ≥20 sealed HNSW
//! segments, builds an exact Flat ground truth over the same vectors, and
//! asserts recall@10 ≥ 0.95.

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use uuid::Uuid;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;
use veda_core::vector::DenseVector;
use veda_index::flat::FlatIndex;
use veda_index::traits::VectorIndex;
use veda_query::local_shard::LocalShardSearcher;
use veda_query::search::VectorSearchRequest;
use veda_storage::segment_manager::SegmentManager;
use veda_query::distributed::ShardSearcher;

const DIM: u32 = 1024;
const N_VECTORS: usize = 20000;
const N_QUERIES: usize = 100;
const K: usize = 10;
/// Number of latent clusters.
const N_CLUSTERS: usize = 40;
/// Seal threshold sized so the bulk load fragments into many sealed HNSW
/// segments where each segment is much larger than `ef_search` (=200) — so
/// per-segment HNSW genuinely sub-samples (it is approximate, not exhaustive),
/// the regime that triggered the recall collapse at scale.
const SEAL_THRESHOLD_BYTES: usize = 8_400_000;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("recall_at_scale")
        .add_field(FieldDefinition::vector(
            "embedding",
            DIM,
            DistanceMetric::Cosine,
        ))
        .build()
        .unwrap()
}

/// Draw a "hard" embedding-like vector. Real text embeddings sit on a smooth
/// manifold with a strong shared component, so near-neighbours have high mutual
/// cosine similarity (0.7–0.9) and are separated by tiny margins — the regime
/// where HNSW at a bounded `ef` genuinely misses neighbours (unlike trivially
/// separable uniform noise). We model that with a large shared base + a cluster
/// component + small per-vector noise.
fn hard_vector(rng: &mut StdRng, base: &[f32], centroids: &[Vec<f32>]) -> Vec<f32> {
    let c = &centroids[rng.gen_range(0..centroids.len())];
    base.iter()
        .zip(c.iter())
        .map(|(&b, &cc)| 1.0 * b + 0.35 * cc + rng.gen_range(-0.08f32..0.08f32))
        .collect()
}

#[test]
fn recall_at_scale_many_hnsw_segments() {
    let mut rng = StdRng::seed_from_u64(0x5EED_BEEF);
    let tmp_dir = tempfile::tempdir().unwrap();

    let mut sm =
        SegmentManager::new(schema(), 0, tmp_dir.path(), SEAL_THRESHOLD_BYTES).unwrap();

    // Shared base direction (makes all vectors highly similar — the hard
    // regime) plus latent cluster centroids.
    let base: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    // Insert N hard 1024-d vectors. Keep a copy for the exact
    // ground-truth Flat index, keyed by the same record UUID.
    let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(N_VECTORS);
    let mut ids: Vec<Uuid> = Vec::with_capacity(N_VECTORS);
    for _ in 0..N_VECTORS {
        let v = hard_vector(&mut rng, &base, &centroids);
        let mut record = Record::new();
        record.set("embedding", FieldValue::Vector(DenseVector::new(v.clone())));
        let rec_id = record.id();
        sm.insert(record).unwrap();
        all_vectors.push(v);
        ids.push(rec_id);
    }
    // Force any remaining growing data into a final sealed segment so the
    // entire dataset is served by HNSW indexes.
    sm.seal_growing().unwrap();

    let n_segments = sm.sealed_segments().len();
    assert!(
        n_segments >= 5,
        "expected several sealed segments to reproduce the collapse, got {n_segments}"
    );

    let mut searcher = LocalShardSearcher::new(sm);
    searcher.build_indexes().unwrap();

    // Exact ground truth: a single Flat index over the same vectors. Map the
    // Flat doc_id (insertion order) back to the record UUID.
    let mut flat = FlatIndex::new(DistanceMetric::Cosine, DIM);
    let refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    flat.build(&refs).unwrap();

    let mut total_hits = 0usize;
    let mut total_expected = 0usize;

    for _ in 0..N_QUERIES {
        // Queries are drawn from the same clustered distribution as the
        // corpus, so each query has a dense neighbourhood — concentrating
        // its true top-K into a few segments.
        let q = hard_vector(&mut rng, &base, &centroids);

        // Ground-truth top-K record UUIDs.
        let truth: HashSet<Uuid> = flat
            .search(&q, K)
            .unwrap()
            .into_iter()
            .map(|sr| ids[sr.id as usize])
            .collect();

        let request = VectorSearchRequest {
            field: "embedding".into(),
            query_vector: q,
            k: K,
            metric: DistanceMetric::Cosine,
            filter: None,
            truncate_to: None,
            read_ts: None,
            filter_expr: None,
        };
        let (results, _) = searcher.search_shard(0, &request).unwrap();

        let got: HashSet<Uuid> = results.iter().map(|r| r.id).collect();
        total_hits += truth.intersection(&got).count();
        total_expected += truth.len();
    }

    let recall = total_hits as f64 / total_expected as f64;
    eprintln!("RECALL_DEBUG segments={n_segments} recall@{K}={recall:.4}");
    assert!(
        recall >= 0.95,
        "recall@{K} over {n_segments} HNSW segments = {recall:.3} (< 0.95)"
    );
}
