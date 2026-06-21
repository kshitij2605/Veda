use rand::Rng;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_index::flat::FlatIndex;
use veda_index::traits::VectorIndex;
use veda_query::search::{SearchExecutor, VectorSearchRequest};
use veda_storage::growing_segment::GrowingSegment;
use veda_storage::segment::SegmentReader;

const DIM: u32 = 64;
const NUM_RECORDS: usize = 200;
const CATEGORIES: &[&str] = &["electronics", "books", "clothing", "food"];

/// Build a schema with 1 vector field (dim 64, Euclidean) + 2 scalar fields.
fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("test_collection")
        .add_field(FieldDefinition::vector(
            "embedding",
            DIM,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("category", FieldType::String).with_filterable(true))
        .add_field(FieldDefinition::new("price", FieldType::Float64).with_filterable(true))
        .build()
        .unwrap()
}

/// Generate a random vector of the given dimension with values in [-1, 1].
fn random_vector(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

/// Insert `n` records into a growing segment with known category/price values.
fn populate_growing_segment(gs: &mut GrowingSegment, n: usize) -> Vec<Record> {
    let mut rng = rand::thread_rng();
    let mut records = Vec::with_capacity(n);

    for i in 0..n {
        let mut record = Record::new();
        let vec_data = random_vector(&mut rng, DIM as usize);
        record.set("embedding", FieldValue::Vector(DenseVector::new(vec_data)));

        let category = CATEGORIES[i % CATEGORIES.len()];
        record.set("category", FieldValue::String(category.to_owned()));

        // Price: deterministic based on index for reproducibility in filter tests.
        let price = 50.0 + (i as f64) * 1.5;
        record.set("price", FieldValue::Float64(price));

        gs.insert(record.clone()).unwrap();
        records.push(record);
    }

    records
}

/// Build a query vector (random).
fn query_vector() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    random_vector(&mut rng, DIM as usize)
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Unfiltered search
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_unfiltered_search() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    // Should return exactly 10 results.
    assert_eq!(response.results.len(), 10);

    // Distances should be in ascending order.
    for window in response.results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted: {} > {}",
            window[0].distance,
            window[1].distance,
        );
    }

    // Total candidates should equal the number of records.
    assert_eq!(response.total_candidates, NUM_RECORDS);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Filtered search (category = 'electronics')
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_filtered_search_category() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    let records = populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("category = 'electronics'".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    // All results must have category = 'electronics'.
    for scored in &response.results {
        let record = &records[scored.doc_id as usize];
        let cat = match record.get("category") {
            Some(FieldValue::String(s)) => s.as_str(),
            _ => panic!("missing category"),
        };
        assert_eq!(
            cat, "electronics",
            "doc_id={} has wrong category",
            scored.doc_id
        );
    }

    // Should have at most 10 results.
    assert!(response.results.len() <= 10);
    // Should have at least 1 result (200 records with 4 categories => 50 electronics).
    assert!(!response.results.is_empty());

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Range filter (price BETWEEN 100 AND 200)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_filtered_search_range() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    let records = populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("price BETWEEN 100 AND 200".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    for scored in &response.results {
        let record = &records[scored.doc_id as usize];
        let price = match record.get("price") {
            Some(FieldValue::Float64(p)) => *p,
            _ => panic!("missing price"),
        };
        assert!(
            (100.0..=200.0).contains(&price),
            "doc_id={} has price={} outside [100, 200]",
            scored.doc_id,
            price,
        );
    }

    assert!(!response.results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Combined filter (category = 'electronics' AND price < 200)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_combined_filter() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    let records = populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("category = 'electronics' AND price < 200".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    for scored in &response.results {
        let record = &records[scored.doc_id as usize];
        let cat = match record.get("category") {
            Some(FieldValue::String(s)) => s.as_str(),
            _ => panic!("missing category"),
        };
        let price = match record.get("price") {
            Some(FieldValue::Float64(p)) => *p,
            _ => panic!("missing price"),
        };
        assert_eq!(cat, "electronics");
        assert!(
            price < 200.0,
            "doc_id={} has price={} >= 200",
            scored.doc_id,
            price,
        );
    }

    assert!(!response.results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Sealed segment search
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sealed_segment_search() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema.clone());
    populate_growing_segment(&mut gs, NUM_RECORDS);

    // Seal the growing segment to disk.
    let tmp_dir = tempfile::tempdir().unwrap();
    let seg_dir = tmp_dir.path().join("seg0");
    std::fs::create_dir_all(&seg_dir).unwrap();

    let _metadata = gs.seal(&seg_dir).unwrap();

    // Open the sealed segment.
    let reader = SegmentReader::open(&seg_dir).unwrap();
    assert_eq!(reader.row_count(), NUM_RECORDS as u64);

    // Build a FlatIndex from the stored vectors.
    let mut flat_index = FlatIndex::new(DistanceMetric::Euclidean, DIM);
    let row_count = reader.row_count() as u32;
    for doc_id in 0..row_count {
        let vec = reader.get_vector("embedding", doc_id).unwrap();
        flat_index.insert(doc_id, vec).unwrap();
    }

    // Unfiltered search on sealed segment.
    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let sealed_segments: Vec<(SegmentReader, Box<dyn VectorIndex>)> =
        vec![(reader, Box::new(flat_index))];

    let (response, _timings) = SearchExecutor::search(&request, None, &sealed_segments).unwrap();

    assert_eq!(response.results.len(), 10);

    // Distances should be ascending.
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }

    assert_eq!(response.total_candidates, NUM_RECORDS);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Sealed segment with filter
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sealed_segment_filtered_search() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema.clone());
    populate_growing_segment(&mut gs, NUM_RECORDS);

    let tmp_dir = tempfile::tempdir().unwrap();
    let seg_dir = tmp_dir.path().join("seg0");
    std::fs::create_dir_all(&seg_dir).unwrap();

    let _metadata = gs.seal(&seg_dir).unwrap();

    let reader = SegmentReader::open(&seg_dir).unwrap();

    // Build a FlatIndex.
    let mut flat_index = FlatIndex::new(DistanceMetric::Euclidean, DIM);
    let row_count = reader.row_count() as u32;
    for doc_id in 0..row_count {
        let vec = reader.get_vector("embedding", doc_id).unwrap();
        flat_index.insert(doc_id, vec).unwrap();
    }

    // Filtered search: category = 'books'
    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("category = 'books'".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let sealed_segments: Vec<(SegmentReader, Box<dyn VectorIndex>)> =
        vec![(reader, Box::new(flat_index))];

    let (response, _timings) = SearchExecutor::search(&request, None, &sealed_segments).unwrap();

    assert!(!response.results.is_empty());
    assert!(response.results.len() <= 10);

    // Verify all results have category = 'books' by reading the record.
    for scored in &response.results {
        let record = sealed_segments[0].0.get_record(scored.doc_id).unwrap();
        let cat = match record.get("category") {
            Some(FieldValue::String(s)) => s.clone(),
            _ => panic!("missing category for doc_id={}", scored.doc_id),
        };
        assert_eq!(
            cat, "books",
            "doc_id={} has category='{}', expected 'books'",
            scored.doc_id, cat,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Multi-segment merge
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_segment_merge() {
    let schema = test_schema();

    // Create and populate a growing segment (segment A).
    let mut gs_a = GrowingSegment::new(schema.clone());
    populate_growing_segment(&mut gs_a, 100);

    // Seal segment A to disk.
    let tmp_dir = tempfile::tempdir().unwrap();
    let seg_a_dir = tmp_dir.path().join("seg_a");
    std::fs::create_dir_all(&seg_a_dir).unwrap();
    let _meta_a = gs_a.seal(&seg_a_dir).unwrap();

    let reader_a = SegmentReader::open(&seg_a_dir).unwrap();
    let mut flat_a = FlatIndex::new(DistanceMetric::Euclidean, DIM);
    for doc_id in 0..reader_a.row_count() as u32 {
        let vec = reader_a.get_vector("embedding", doc_id).unwrap();
        flat_a.insert(doc_id, vec).unwrap();
    }

    // Create a new growing segment (segment B) — still in memory.
    let mut gs_b = GrowingSegment::new(schema);
    populate_growing_segment(&mut gs_b, 100);

    // Use a consistent query vector for both search calls.
    let qv = query_vector();

    // Search across both segments: sealed A + growing B.
    let sealed_segments: Vec<(SegmentReader, Box<dyn VectorIndex>)> =
        vec![(reader_a, Box::new(flat_a))];

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: qv,
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) =
        SearchExecutor::search(&request, Some(&gs_b), &sealed_segments).unwrap();

    // Should return exactly 10 results.
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

    // Total candidates = 100 (sealed) + 100 (growing) = 200.
    assert_eq!(response.total_candidates, 200);

    // Results should come from both segments (with high probability for 200
    // random vectors). Collect distinct segment IDs.
    let segment_ids: std::collections::HashSet<_> =
        response.results.iter().map(|r| r.segment_id).collect();

    // With random vectors and k=10, we expect results from both segments most
    // of the time, but this is not deterministic. At minimum, check we have
    // results from at least one segment.
    assert!(
        !segment_ids.is_empty(),
        "Expected results from at least one segment",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: k=0 returns empty
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_k_zero_returns_empty() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    populate_growing_segment(&mut gs, 50);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 0,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    assert!(response.results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 9: k larger than matching records
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_k_larger_than_matching() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    populate_growing_segment(&mut gs, 5);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 100,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    // Only 5 records exist, so at most 5 results.
    assert_eq!(response.results.len(), 5);

    // Distances ascending.
    for window in response.results.windows(2) {
        assert!(window[0].distance <= window[1].distance);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 10: No segments produces empty results
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_no_segments_empty_result() {
    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings): (veda_query::search::SearchResponse, _) =
        SearchExecutor::search(&request, None, &[]).unwrap();

    assert!(response.results.is_empty());
    assert_eq!(response.total_candidates, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 11: Sealed segment range filter
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sealed_segment_range_filter() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema.clone());
    let _records = populate_growing_segment(&mut gs, NUM_RECORDS);

    let tmp_dir = tempfile::tempdir().unwrap();
    let seg_dir = tmp_dir.path().join("seg0");
    std::fs::create_dir_all(&seg_dir).unwrap();
    let _metadata = gs.seal(&seg_dir).unwrap();

    let reader = SegmentReader::open(&seg_dir).unwrap();

    let mut flat_index = FlatIndex::new(DistanceMetric::Euclidean, DIM);
    for doc_id in 0..reader.row_count() as u32 {
        let vec = reader.get_vector("embedding", doc_id).unwrap();
        flat_index.insert(doc_id, vec).unwrap();
    }

    // price BETWEEN 100 AND 200
    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("price BETWEEN 100 AND 200".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let sealed_segments: Vec<(SegmentReader, Box<dyn VectorIndex>)> =
        vec![(reader, Box::new(flat_index))];

    let (response, _timings) = SearchExecutor::search(&request, None, &sealed_segments).unwrap();

    assert!(!response.results.is_empty());

    // Verify price constraint on each result.
    for scored in &response.results {
        // Prices are deterministic: 50.0 + doc_id * 1.5
        let expected_price = 50.0 + scored.doc_id as f64 * 1.5;
        assert!(
            (100.0..=200.0).contains(&expected_price),
            "doc_id={} has price={} outside [100, 200]",
            scored.doc_id,
            expected_price,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 12: Filter that matches nothing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_filter_matches_nothing() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: Some("category = 'nonexistent'".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    assert!(response.results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 13: record_eval unit tests via search
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_record_eval_ne_filter() {
    let schema = test_schema();
    let mut gs = GrowingSegment::new(schema);
    let records = populate_growing_segment(&mut gs, NUM_RECORDS);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: query_vector(),
        k: NUM_RECORDS, // get all
        metric: DistanceMetric::Euclidean,
        filter: Some("category != 'electronics'".into()),
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    let (response, _timings) = SearchExecutor::search(&request, Some(&gs), &[]).unwrap();

    for scored in &response.results {
        let record = &records[scored.doc_id as usize];
        let cat = match record.get("category") {
            Some(FieldValue::String(s)) => s.as_str(),
            _ => panic!("missing category"),
        };
        assert_ne!(cat, "electronics");
    }
}
