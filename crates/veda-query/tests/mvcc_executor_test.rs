//! Phase F: precise per-doc visibility inside the search
//! executor. With `read_ts` set on the request, the
//! LocalShardSearcher filters growing-segment hits through
//! `visible_doc_ids_at` so future / superseded versions never
//! surface from the search walk itself (not just the gateway
//! post-filter).

use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;
use veda_core::vector::DenseVector;
use veda_query::distributed::ShardSearcher;
use veda_query::local_shard::LocalShardSearcher;
use veda_query::search::VectorSearchRequest;
use veda_storage::segment_manager::SegmentManager;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("mvcc_exec")
        .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
        .shard_count(1)
        .build()
        .unwrap()
}

fn record(values: [f32; 2]) -> Record {
    let mut r = Record::new();
    r.set(
        "v",
        FieldValue::Vector(DenseVector::new(vec![values[0], values[1]])),
    );
    r
}

#[test]
fn search_at_past_ts_filters_future_versions() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();

    // Three distinct records committed at different ts.
    sm.insert_mvcc(100, record([1.0, 0.0])).unwrap();
    sm.insert_mvcc(200, record([0.9, 0.1])).unwrap();
    sm.insert_mvcc(300, record([0.8, 0.2])).unwrap();

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: Some(150),
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    // Only the record committed at ts=100 is visible at read_ts=150.
    assert_eq!(
        hits.len(),
        1,
        "expected 1 visible hit at read_ts=150, got {hits:?}"
    );
}

#[test]
fn search_at_far_future_ts_sees_every_visible_record() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    sm.insert_mvcc(100, record([1.0, 0.0])).unwrap();
    sm.insert_mvcc(200, record([0.9, 0.1])).unwrap();
    sm.insert_mvcc(300, record([0.8, 0.2])).unwrap();

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: Some(u128::MAX),
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn search_supersession_drops_older_version_at_high_read_ts() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();

    // Same record id with two versions; vectors picked so v1's
    // distance to the query is BETTER (would beat v2 if it
    // weren't superseded).
    let r1 = record([1.0, 0.0]);
    let rid = r1.id();
    sm.insert_mvcc(100, r1).unwrap();
    let mut r2 = Record::with_id(rid);
    r2.set("v", FieldValue::Vector(DenseVector::new(vec![0.5, 0.5])));
    sm.insert_mvcc(200, r2).unwrap();

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: Some(250),
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    // v1 has been superseded; only one hit, and it's v2.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, rid);
    // Distance reflects v2's vector (0.5, 0.5) not v1's (1.0, 0.0).
    // For cosine: query [1,0] vs [0.5,0.5] normalised = ~0.293.
    // For cosine: query [1,0] vs [1.0,0] = 0. So v1 would give dist=0
    // but the supersession means we get v2's distance instead.
    assert!(
        hits[0].distance > 0.1,
        "expected v2's distance, got {}",
        hits[0].distance
    );
}

#[test]
fn search_without_read_ts_returns_every_doc() {
    // read_ts = None: legacy / non-MVCC semantics. The visibility
    // filter is skipped entirely, so even though MVCC stamps
    // exist, every doc surfaces.
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    sm.insert_mvcc(100, record([1.0, 0.0])).unwrap();
    sm.insert_mvcc(200, record([0.9, 0.1])).unwrap();

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn sealed_segment_carries_commit_ts_through_seal_reopen() {
    // Insert with MVCC, seal, then search via the LocalShardSearcher
    // against the sealed segment. The visibility filter must work
    // identically to the growing-segment case.
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    sm.insert_mvcc(100, record([1.0, 0.0])).unwrap();
    sm.insert_mvcc(200, record([0.9, 0.1])).unwrap();
    sm.insert_mvcc(300, record([0.8, 0.2])).unwrap();
    // Force a seal: drains the growing segment to disk and starts
    // a fresh empty growing.
    sm.seal_growing().expect("seal").expect("non-empty");
    // The growing segment is now empty; all 3 docs live in a
    // sealed segment that should carry commit_ts through.
    assert_eq!(sm.growing_segment().count(), 0);
    assert_eq!(sm.sealed_segments().len(), 1);

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: Some(150),
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "sealed-segment visibility at read_ts=150 should keep only the doc committed at ts=100; got {:?}",
        hits
    );
}

#[test]
fn sealed_segment_without_commit_ts_treats_every_doc_visible() {
    // Legacy (non-MVCC) inserts → seal → search with read_ts set.
    // The sealed segment has no commit_ts sidecar; every doc must
    // surface regardless of read_ts (always-visible semantics).
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.insert(record([1.0, 0.0])).unwrap();
    sm.insert(record([0.9, 0.1])).unwrap();
    sm.seal_growing().expect("seal").expect("non-empty");
    assert_eq!(sm.sealed_segments().len(), 1);

    let searcher = LocalShardSearcher::new(sm);
    let query = VectorSearchRequest {
        field: "v".into(),
        query_vector: vec![1.0, 0.0],
        k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        truncate_to: None,
        read_ts: Some(0), // would drop every MVCC doc; legacy stays
        filter_expr: None,
        ef_search: None,
    };
    let (hits, _t) = searcher.search_shard(0, &query).unwrap();
    assert_eq!(hits.len(), 2, "legacy sealed segment should pass through");
}
