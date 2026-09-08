//! SPANN builds through the streaming path at INGEST, not only at serving.
//!
//! `SpannIndex::build_into_streaming` shipped in PR #20 but was unreachable
//! from a real collection: `build_indexes` materialised every segment into a
//! `Vec<Vec<f32>>` before handing it to `build_vector_index`, because that is
//! the shape every other index type needs. So choosing `spann` lowered what a
//! node needed to *serve* a collection and not what it needed to *build* one.
//!
//! **No assertion about the result can catch that regression.** The in-memory
//! and streaming builds produce the same index — byte-identical when the
//! budget admits the whole corpus — so recall, neighbours and every query
//! answer are identical either way. A recall test passes on both paths. The
//! only evidence that the cheap path was taken is `StreamingBuildStats`,
//! which is why `LocalShardSearcher` records it rather than only logging it.

use std::sync::Arc;

use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, IndexType};
use veda_core::vector::DenseVector;
use veda_query::distributed::ShardSearcher;
use veda_query::local_shard::LocalShardSearcher;
use veda_query::search::VectorSearchRequest;
use veda_storage::segment_manager::SegmentManager;

const DIM: u32 = 64;
const N: usize = 4_000;
/// Deliberately far below the corpus (1,024,000 B) so a build that ignored
/// the budget and buffered everything is unmistakable in the stats.
const BUDGET_BYTES: usize = 64 * 1024;

fn corpus_bytes() -> usize {
    N * DIM as usize * std::mem::size_of::<f32>()
}

fn schema() -> CollectionSchema {
    CollectionSchema::builder("spann_stream")
        .add_field(
            FieldDefinition::vector("embedding", DIM, DistanceMetric::Cosine)
                .with_index_type(IndexType::Spann),
        )
        .build()
        .unwrap()
}

/// Deterministic clustered corpus — SPANN is an IVF, so clustered data is
/// the shape its posting lists are actually meant for.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn spann_builds_from_the_segment_without_materialising_it() {
    // SAFETY: single test in its own test binary; nothing else in this
    // process reads or writes the environment concurrently.
    unsafe {
        std::env::set_var("VEDA_SPANN_BUILD_BUDGET_BYTES", BUDGET_BYTES.to_string());
    }

    // SPANN posting lists live in the object store; a local blob store is the
    // same code path the single-node deployment uses.
    let blob_dir = tempfile::tempdir().unwrap();
    let blob = Arc::new(veda_storage::blob_store::LocalBlobStore::new(blob_dir.path()).unwrap());
    veda_index::spann::register_store(Arc::new(veda_index::spann::BlobRangeStore(blob)));

    let tmp = tempfile::tempdir().unwrap();
    // Threshold high enough that the whole corpus lands in ONE sealed
    // segment, so `peak_buffered_bytes` is comparable against the full
    // corpus rather than against a fraction of it.
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 1usize << 40).unwrap();

    let mut rng = Lcg(0xC0FFEE);
    let centres: Vec<Vec<f32>> = (0..16)
        .map(|_| (0..DIM as usize).map(|_| rng.next_f32()).collect())
        .collect();
    let mut first_vector: Option<Vec<f32>> = None;
    for i in 0..N {
        let c = &centres[i % centres.len()];
        let v: Vec<f32> = (0..DIM as usize)
            .map(|d| c[d] + rng.next_f32() * 0.10)
            .collect();
        if first_vector.is_none() {
            first_vector = Some(v.clone());
        }
        let mut rec = Record::new();
        rec.set("embedding", FieldValue::Vector(DenseVector::new(v)));
        sm.insert(rec).unwrap();
    }
    sm.seal_growing().unwrap();
    let sealed: Vec<uuid::Uuid> = sm.sealed_segments().iter().map(|h| h.metadata.id).collect();
    assert_eq!(sealed.len(), 1, "test assumes a single sealed segment");
    let seg_id = sealed[0];

    let mut searcher = LocalShardSearcher::new(sm);
    searcher.build_indexes().unwrap();

    // 1. The streaming path was taken at all. Before this change SPANN went
    //    through `build_vector_index`, which records nothing, so this is
    //    `None` — the routing assertion.
    let stats = searcher
        .spann_build_stats(&seg_id, "embedding")
        .expect("SPANN field did not build through the streaming path");

    // 2. It genuinely streamed. The floor is the largest posting list (a list
    //    is never split across parts), so this is not asserted against the
    //    budget itself — only that peak is a small fraction of the corpus,
    //    which the in-memory build could never satisfy.
    assert!(
        stats.peak_buffered_bytes * 4 < corpus_bytes(),
        "peak {} should be far below the {}-byte corpus; a build that \
         buffered everything would return the same index and the same recall",
        stats.peak_buffered_bytes,
        corpus_bytes()
    );
    assert!(stats.groups >= 1, "a streaming build processes >=1 group");

    // 3. And the index still answers. Correctness cannot detect the
    //    regression above, but a broken build would fail here, so the
    //    streaming route must still be exercised for retrieval.
    let request = VectorSearchRequest {
        field: "embedding".to_owned(),
        query_vector: first_vector.unwrap(),
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };
    let (results, _) = searcher.search_shard(0, &request).unwrap();
    assert!(
        !results.is_empty(),
        "the streaming-built SPANN index returned nothing"
    );
}
