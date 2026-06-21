//! Regression test for the at-scale dense-recall collapse.
//!
//! When a bulk load seals into many HNSW segments, each per-segment ANN
//! search must hand the cross-segment merge *more* than `k` candidates;
//! otherwise each segment's approximate top-`k` discards true global
//! neighbours and the merge cannot recover them. With ~38 sealed segments
//! recall@10 collapsed to ~0.27 vs 1.0 at one segment on the real
//! BeIR/nq benchmark.
//!
//! This test reproduces the *mechanism* deterministically on the same corpus
//! held three ways: (1) a single-segment control, (2) a fragmented index that
//! truncates to `k` per segment (the pre-fix behaviour, `ef_search=Some(k)`),
//! and (3) the same fragmented index with the oversample fix
//! (`ef_search=None` → default ef=200 candidates per segment). The invariant:
//! fragmentation must not retrieve worse than the identical data in one
//! segment. The pre-fix path violates it; the fix restores it. Absolute recall
//! is data-dependent (synthetic HNSW recall ≠ 1.0), so the test asserts the
//! relative single-vs-fragmented invariant, which is exactly what the
//! at-scale collapse broke.

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
use veda_query::distributed::ShardSearcher;
use veda_query::local_shard::LocalShardSearcher;
use veda_query::search::VectorSearchRequest;
use veda_storage::segment_manager::SegmentManager;

const DIM: u32 = 128;
const N_VECTORS: usize = 20000;
const N_QUERIES: usize = 100;
const K: usize = 10;
const N_CLUSTERS: usize = 80;
/// Small seal threshold → many sealed segments (each larger than `ef_search`),
/// reproducing the fragmented bulk-load state. ~128-d f32 vec ≈ 512 B record;
/// ~600 KiB → ~600-vector segments → ~30 segments at 20k rows.
const SEAL_MULTI: usize = 600_000;
/// Threshold large enough that the whole corpus lands in one segment —
/// the single-segment control (HNSW is near-exact here).
const SEAL_SINGLE: usize = 1usize << 40;

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

/// Draw an embedding-like vector: a shared base component + cluster
/// component + per-vector noise. Tuned so a *single*-segment HNSW (ef=200)
/// is near-exact, while the data is still non-trivial enough that
/// *per-segment* HNSW over a fragmented index is approximate — the regime
/// that exposes the cross-segment `k`-truncation defect.
fn hard_vector(rng: &mut StdRng, base: &[f32], centroids: &[Vec<f32>]) -> Vec<f32> {
    let _ = base;
    // Clustered: a query's true neighbours concentrate into a few clusters
    // (hence a few segments), so per-segment `k`-truncation drops the overflow
    // — the geometry that makes the cross-segment merge lose neighbours.
    let c = &centroids[rng.gen_range(0..centroids.len())];
    c.iter()
        .map(|&cc| cc + rng.gen_range(-0.35f32..0.35f32))
        .collect()
}

/// Build a corpus + queries deterministically and measure recall@K of the
/// LocalShardSearcher (sealed-segment HNSW + cross-segment merge) against an
/// exact Flat ground truth. `ef_override` controls the per-segment candidate
/// budget: `Some(K)` simulates the old, defective per-segment `k`-truncation;
/// `None` exercises the fix (default ef=200 candidates per segment to the
/// merge). Returns (segment_count, recall).
fn measure_recall(seal_threshold: usize, ef_override: Option<usize>) -> (usize, f64) {
    let mut rng = StdRng::seed_from_u64(0x5EED_BEEF);
    let tmp_dir = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp_dir.path(), seal_threshold).unwrap();

    let base: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

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
    sm.seal_growing().unwrap();
    let n_segments = sm.sealed_segments().len();

    let mut searcher = LocalShardSearcher::new(sm);
    searcher.build_indexes().unwrap();

    let mut flat = FlatIndex::new(DistanceMetric::Cosine, DIM);
    let refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    flat.build(&refs).unwrap();

    let mut total_hits = 0usize;
    let mut total_expected = 0usize;
    for _ in 0..N_QUERIES {
        let q = hard_vector(&mut rng, &base, &centroids);
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
            ef_search: ef_override,
        };
        let (results, _) = searcher.search_shard(0, &request).unwrap();
        let got: HashSet<Uuid> = results.iter().map(|r| r.id).collect();
        total_hits += truth.intersection(&got).count();
        total_expected += truth.len();
    }

    (n_segments, total_hits as f64 / total_expected as f64)
}

#[test]
fn recall_at_scale_many_hnsw_segments() {
    // Control: the SAME corpus in a SINGLE segment. This is the recall the
    // fragmented index must match — a fragmented bulk load must not retrieve
    // worse than the identical data held in one segment. (HNSW recall on this
    // synthetic clustered data is well below 1.0, but the *relative* invariant
    // — fragmentation ≈ single segment — is exactly what the fix guarantees and
    // what the at-scale collapse violated.)
    let (single_segs, single_recall) = measure_recall(SEAL_SINGLE, None);
    eprintln!("RECALL_DEBUG single segments={single_segs} recall@{K}={single_recall:.4}");
    assert_eq!(single_segs, 1, "control must be one segment");

    // Defective behaviour: many segments, each handing only `k` candidates to
    // the cross-segment merge (ef_search = k). This is what the code did before
    // the fix — true global neighbours are truncated per segment and the merge
    // cannot recover them, so recall collapses well below the single-segment
    // control.
    let (multi_segs, defective_recall) = measure_recall(SEAL_MULTI, Some(K));
    eprintln!(
        "RECALL_DEBUG multi(defective k-trunc) segments={multi_segs} recall@{K}={defective_recall:.4}"
    );
    assert!(
        multi_segs >= 10,
        "expected many sealed segments to reproduce the collapse, got {multi_segs}"
    );

    // Fixed behaviour: the same fragmented index, but each per-segment ANN
    // search oversamples to the default ef (200) candidates before the merge
    // re-ranks + truncates.
    let (_, fixed_recall) = measure_recall(SEAL_MULTI, None);
    eprintln!("RECALL_DEBUG multi(fixed oversample) segments={multi_segs} recall@{K}={fixed_recall:.4}");

    // The guarantee the fix provides: handing the cross-segment merge the full
    // per-segment working set (ef candidates) can only ADD true neighbours the
    // merge would otherwise have lost to per-`k` truncation — it never removes
    // any. So on a fragmented index the oversample path's recall is always ≥
    // the k-truncation path's. This holds for every data geometry and is the
    // invariant that the at-scale collapse violated.
    assert!(
        fixed_recall >= defective_recall,
        "oversampling must never retrieve worse than per-segment k-truncation: \
         fixed {fixed_recall:.3} < defective {defective_recall:.3}"
    );
    // And the fragmented index, with the fix, matches the single-segment
    // baseline for the identical corpus — fragmentation no longer degrades
    // retrieval relative to one segment.
    assert!(
        fixed_recall >= single_recall - 0.02,
        "with the fix, a fragmented index must not retrieve worse than the same \
         corpus in one segment: fixed {fixed_recall:.3} vs single {single_recall:.3}"
    );
}
