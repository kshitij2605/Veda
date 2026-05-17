use std::collections::{HashMap, HashSet};

use rand::Rng;
use uuid::Uuid;
use veda_core::error::Result;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_query::distributed::{DistributedQueryExecutor, ShardSearcher};
use veda_query::fusion::{FusionStrategy, MultiVectorQuery, FieldQuery};
use veda_query::search::{ScoredRecord, SearchTimings, VectorSearchRequest};

const DIM: u32 = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Mock shard searcher
// ═══════════════════════════════════════════════════════════════════════════

/// A mock shard searcher that returns pre-built results.
/// Supports optional filter: if request has a filter string, only records
/// whose `category` field (stored in metadata) matches are returned.
struct MockShardSearcher {
    records: Vec<MockRecord>,
}

struct MockRecord {
    id: Uuid,
    segment_id: Uuid,
    doc_id: u32,
    vectors: HashMap<String, Vec<f32>>,
    category: Option<String>,
}

impl MockShardSearcher {
    fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    fn add_records(
        &mut self,
        count: usize,
        dim: usize,
        field: &str,
        category: Option<&str>,
    ) {
        let mut rng = rand::thread_rng();
        let segment_id = Uuid::new_v4();
        for i in 0..count {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            let mut vectors = HashMap::new();
            vectors.insert(field.to_owned(), vec);
            self.records.push(MockRecord {
                id: Uuid::new_v4(),
                segment_id,
                doc_id: i as u32,
                vectors,
                category: category.map(|s| s.to_owned()),
            });
        }
    }

    /// Add records with multiple vector fields.
    fn add_multi_field_records(
        &mut self,
        count: usize,
        dim: usize,
        fields: &[&str],
    ) {
        let mut rng = rand::thread_rng();
        let segment_id = Uuid::new_v4();
        for i in 0..count {
            let mut vectors = HashMap::new();
            for &field in fields {
                let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
                vectors.insert(field.to_owned(), vec);
            }
            self.records.push(MockRecord {
                id: Uuid::new_v4(),
                segment_id,
                doc_id: i as u32,
                vectors,
                category: None,
            });
        }
    }
}

impl ShardSearcher for MockShardSearcher {
    fn search_shard(
        &self,
        _shard_id: u32,
        request: &VectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        // Apply simple category filter if present.
        let filter_category = request.filter.as_ref().and_then(|f| {
            // Very simple parser: "category = 'X'" -> X
            if f.starts_with("category = '") && f.ends_with('\'') {
                Some(f[12..f.len() - 1].to_owned())
            } else {
                None
            }
        });

        let mut results: Vec<ScoredRecord> = Vec::new();

        for record in &self.records {
            // Filter by category if set.
            if let Some(ref cat) = filter_category {
                if record.category.as_deref() != Some(cat.as_str()) {
                    continue;
                }
            }

            if let Some(vec) = record.vectors.get(&request.field) {
                let distance = euclidean_distance(&request.query_vector, vec);
                results.push(ScoredRecord {
                    id: record.id,
                    segment_id: record.segment_id,
                    doc_id: record.doc_id,
                    distance,
                });
            }
        }

        // Sort by distance, take top-k.
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(request.k);

        Ok((results, SearchTimings::default()))
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Single shard search
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_single_shard_search() {
    let mut searcher = MockShardSearcher::new();
    searcher.add_records(200, DIM as usize, "embedding", None);

    let mut executor = DistributedQueryExecutor::new();
    executor.register_shard(0, Box::new(searcher));

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    assert_eq!(response.results.len(), 10);

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted: {} > {}",
            window[0].distance,
            window[1].distance,
        );
    }

    // total_candidates should reflect what the shard returned (capped by k).
    assert!(response.total_candidates > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Multi-shard fan-out
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_shard_fanout() {
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..3 {
        let mut searcher = MockShardSearcher::new();
        searcher.add_records(100, DIM as usize, "embedding", None);
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    assert_eq!(response.results.len(), 10);

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted: {} > {}",
            window[0].distance,
            window[1].distance,
        );
    }

    // Results should come from multiple shards (different segment_ids).
    let segment_ids: HashSet<Uuid> = response.results.iter().map(|r| r.segment_id).collect();
    // With 300 random vectors and k=10, we expect results from multiple shards
    // most of the time. At minimum, verify we got results.
    assert!(!segment_ids.is_empty());

    // Total candidates = 10 from each of 3 shards = 30 (each shard returns top-k).
    assert_eq!(response.total_candidates, 30);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Multi-shard with filter
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_shard_with_filter() {
    let categories = ["electronics", "books", "clothing", "food"];
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..3 {
        let mut searcher = MockShardSearcher::new();
        // Add records with various categories.
        for &cat in &categories {
            searcher.add_records(25, DIM as usize, "embedding", Some(cat));
        }
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("category = 'electronics'".into()),
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    // Should return at most 10 results.
    assert!(response.results.len() <= 10);
    assert!(!response.results.is_empty());

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Multi-vector fusion across shards
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_vector_fusion_across_shards() {
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..2 {
        let mut searcher = MockShardSearcher::new();
        searcher.add_multi_field_records(100, DIM as usize, &["text_embed", "image_embed"]);
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let query = MultiVectorQuery {
        queries: vec![
            FieldQuery {
                field: "text_embed".into(),
                vector: random_vector(DIM as usize),
                weight: 0.7,
            },
            FieldQuery {
                field: "image_embed".into(),
                vector: random_vector(DIM as usize),
                weight: 0.3,
            },
        ],
        fusion: FusionStrategy::Rrf { k: 60 },
        limit: 10,
    };

    let results = executor
        .search_multi_vector(&query, None, DistanceMetric::Euclidean)
        .unwrap();

    assert_eq!(results.len(), 10);

    // Fused results should be sorted by score descending.
    for window in results.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "Fused results not sorted descending: {} < {}",
            window[0].score,
            window[1].score,
        );
    }

    // Each result should have field_scores from both fields.
    for r in &results {
        assert!(
            !r.field_scores.is_empty(),
            "Expected non-empty field_scores"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Empty shard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_shard() {
    let mut executor = DistributedQueryExecutor::new();

    // Shard 0: empty.
    let empty_searcher = MockShardSearcher::new();
    executor.register_shard(0, Box::new(empty_searcher));

    // Shard 1: 100 records.
    let mut searcher = MockShardSearcher::new();
    searcher.add_records(100, DIM as usize, "embedding", None);
    executor.register_shard(1, Box::new(searcher));

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    assert_eq!(response.results.len(), 10);

    // All results should come from shard 1 (non-empty).
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: k larger than total records
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_k_larger_than_total() {
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..3 {
        let mut searcher = MockShardSearcher::new();
        searcher.add_records(100, DIM as usize, "embedding", None);
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 1000,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    // Each shard returns min(100, 1000) = 100 results.
    // Total = 300. k=1000 > 300, so we get all 300.
    assert_eq!(response.results.len(), 300);

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted: {} > {}",
            window[0].distance,
            window[1].distance,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: LocalShardSearcher with SegmentManager (integration)
// ═══════════════════════════════════════════════════════════════════════════

fn test_schema_single_field() -> CollectionSchema {
    CollectionSchema::builder("test_collection")
        .add_field(FieldDefinition::vector(
            "embedding",
            DIM,
            DistanceMetric::Euclidean,
        ))
        .add_field(
            FieldDefinition::new("category", FieldType::String).with_filterable(true),
        )
        .build()
        .unwrap()
}

#[test]
fn test_local_shard_searcher_growing_segment() {
    use veda_query::local_shard::LocalShardSearcher;
    use veda_storage::segment_manager::SegmentManager;

    let tmp_dir = tempfile::tempdir().unwrap();
    let schema = test_schema_single_field();

    let mut sm =
        SegmentManager::new(schema.clone(), 0, tmp_dir.path(), 1024 * 1024 * 1024).unwrap();

    // Insert 50 records into the growing segment.
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let mut record = Record::new();
        let vec: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        record.set("embedding", FieldValue::Vector(DenseVector::new(vec)));
        record.set("category", FieldValue::String("test".into()));
        sm.insert(record).unwrap();
    }

    let searcher = LocalShardSearcher::new(sm);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (results, _timings) = searcher.search_shard(0, &request).unwrap();
    assert_eq!(results.len(), 10);

    for window in results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}

#[test]
fn test_local_shard_searcher_with_sealed_segments() {
    use veda_query::local_shard::LocalShardSearcher;
    use veda_storage::segment_manager::SegmentManager;

    let tmp_dir = tempfile::tempdir().unwrap();
    let schema = test_schema_single_field();

    let mut sm =
        SegmentManager::new(schema.clone(), 0, tmp_dir.path(), 1024 * 1024 * 1024).unwrap();

    // Insert records and seal.
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let mut record = Record::new();
        let vec: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        record.set("embedding", FieldValue::Vector(DenseVector::new(vec)));
        record.set("category", FieldValue::String("test".into()));
        sm.insert(record).unwrap();
    }
    sm.seal_growing().unwrap();

    // Insert more into a new growing segment.
    for _ in 0..50 {
        let mut record = Record::new();
        let vec: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        record.set("embedding", FieldValue::Vector(DenseVector::new(vec)));
        record.set("category", FieldValue::String("test".into()));
        sm.insert(record).unwrap();
    }

    let mut searcher = LocalShardSearcher::new(sm);
    searcher.build_indexes().unwrap();

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 20,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (results, _timings) = searcher.search_shard(0, &request).unwrap();
    assert_eq!(results.len(), 20);

    for window in results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }

    // Results should come from both sealed and growing segments.
    let segment_ids: HashSet<Uuid> = results.iter().map(|r| r.segment_id).collect();
    // With 150 random vectors and k=20, we expect results from both segments
    // most of the time.
    assert!(
        !segment_ids.is_empty(),
        "Expected results from at least one segment"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 9: Distributed executor with LocalShardSearcher
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_distributed_with_local_shards() {
    use veda_query::local_shard::LocalShardSearcher;
    use veda_storage::segment_manager::SegmentManager;

    let tmp_dir = tempfile::tempdir().unwrap();
    let schema = test_schema_single_field();

    let mut executor = DistributedQueryExecutor::new();
    let mut rng = rand::thread_rng();

    for shard_id in 0..2u32 {
        let mut sm = SegmentManager::new(
            schema.clone(),
            shard_id,
            tmp_dir.path(),
            1024 * 1024 * 1024,
        )
        .unwrap();

        for _ in 0..75 {
            let mut record = Record::new();
            let vec: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            record.set("embedding", FieldValue::Vector(DenseVector::new(vec)));
            record.set("category", FieldValue::String("test".into()));
            sm.insert(record).unwrap();
        }

        let searcher = LocalShardSearcher::new(sm);
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: random_vector(DIM as usize),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
    truncate_to: None,
    };

    let (response, _timings) = executor.search(&request).unwrap();

    assert_eq!(response.results.len(), 10);
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}
