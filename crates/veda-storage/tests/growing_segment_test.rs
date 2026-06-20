use roaring::RoaringBitmap;
use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::simd;
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::growing_segment::GrowingSegment;
use veda_storage::segment::SegmentReader;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a test schema: 1 vector field (dim=128, Euclidean), 2 scalar fields.
fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("growing_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            128,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::new("score", FieldType::Float64))
        .build()
        .unwrap()
}

/// Generate a deterministic pseudo-random vector of the given dimension.
fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let val = ((state >> 16) as f32 / 32768.0) - 1.0;
        v.push(val);
    }
    v
}

/// Create a test record for the standard schema.
fn make_record(i: u32) -> Record {
    let mut record = Record::new();
    record.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(make_vector(128, i))),
    );
    record.set("title", FieldValue::String(format!("item_{i}")));
    record.set("score", FieldValue::Float64(i as f64 * 2.5));
    record
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_insert_100_records_and_verify_count() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..100u32 {
        let record = make_record(i);
        let doc_id = seg.insert(record).unwrap();
        assert_eq!(doc_id, i);
    }

    assert_eq!(seg.count(), 100);
    assert!(seg.size_bytes() > 0);
}

#[test]
fn test_get_record_returns_correct_record() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    let mut originals = Vec::new();
    for i in 0..20u32 {
        let record = make_record(i);
        originals.push(record.clone());
        seg.insert(record).unwrap();
    }

    for i in 0..20u32 {
        let record = seg.get_record(i).unwrap();
        assert_eq!(record.id(), originals[i as usize].id());
        assert_eq!(
            record.get("title"),
            Some(&FieldValue::String(format!("item_{i}")))
        );
        assert_eq!(
            record.get("score"),
            Some(&FieldValue::Float64(i as f64 * 2.5))
        );
    }

    // Out of range returns None.
    assert!(seg.get_record(20).is_none());
}

#[test]
fn test_get_vector_returns_correct_vector() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..10u32 {
        let record = make_record(i);
        seg.insert(record).unwrap();
    }

    for i in 0..10u32 {
        let expected = make_vector(128, i);
        let actual = seg.get_vector("embedding", i).unwrap();
        assert_eq!(actual, expected.as_slice(), "vector mismatch at doc {i}");
    }

    // Non-existent field.
    assert!(seg.get_vector("nonexistent", 0).is_none());
    // Out of range.
    assert!(seg.get_vector("embedding", 10).is_none());
}

#[test]
fn test_search_returns_top_k_nearest() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    let num_records = 100u32;
    for i in 0..num_records {
        let record = make_record(i);
        seg.insert(record).unwrap();
    }

    let query = make_vector(128, 42);
    let k = 10;

    // Search using the growing segment.
    let results = seg
        .search("embedding", &query, k, DistanceMetric::Euclidean, None)
        .unwrap();
    assert_eq!(results.len(), k);

    // Compute brute-force ground truth.
    let mut ground_truth: Vec<(u32, f32)> = (0..num_records)
        .map(|i| {
            let v = make_vector(128, i);
            let dist = simd::euclidean_distance(&query, &v);
            (i, dist)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let expected_top_k: Vec<(u32, f32)> = ground_truth.into_iter().take(k).collect();

    // Verify doc_ids match.
    let result_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    let expected_ids: Vec<u32> = expected_top_k.iter().map(|r| r.0).collect();
    assert_eq!(result_ids, expected_ids, "top-k doc_ids do not match");

    // Verify distances are sorted ascending.
    for w in results.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "results not sorted: {} > {}",
            w[0].1,
            w[1].1
        );
    }
}

#[test]
fn test_search_with_filter_bitmap() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..50u32 {
        let record = make_record(i);
        seg.insert(record).unwrap();
    }

    // Only allow even doc_ids.
    let mut filter = RoaringBitmap::new();
    for i in (0..50u32).step_by(2) {
        filter.insert(i);
    }

    let query = make_vector(128, 7);
    let results = seg
        .search(
            "embedding",
            &query,
            5,
            DistanceMetric::Euclidean,
            Some(&filter),
        )
        .unwrap();

    assert_eq!(results.len(), 5);
    for (doc_id, _dist) in &results {
        assert!(
            doc_id % 2 == 0,
            "doc_id {doc_id} is odd but should be filtered out"
        );
        assert!(filter.contains(*doc_id));
    }
}

#[test]
fn test_search_skips_deleted_records() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..20u32 {
        let record = make_record(i);
        seg.insert(record).unwrap();
    }

    // Delete some records.
    seg.mark_deleted(5);
    seg.mark_deleted(10);
    seg.mark_deleted(15);

    let query = make_vector(128, 5); // query is exactly record 5
    let results = seg
        .search("embedding", &query, 20, DistanceMetric::Euclidean, None)
        .unwrap();

    // Should return 17 results (20 - 3 deleted).
    assert_eq!(results.len(), 17);

    let result_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    assert!(!result_ids.contains(&5), "deleted doc 5 should not appear");
    assert!(
        !result_ids.contains(&10),
        "deleted doc 10 should not appear"
    );
    assert!(
        !result_ids.contains(&15),
        "deleted doc 15 should not appear"
    );
}

#[test]
fn test_mark_deleted_and_is_deleted() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..10u32 {
        seg.insert(make_record(i)).unwrap();
    }

    assert!(!seg.is_deleted(3));
    assert!(!seg.is_deleted(7));

    seg.mark_deleted(3);
    seg.mark_deleted(7);

    assert!(seg.is_deleted(3));
    assert!(seg.is_deleted(7));
    assert!(!seg.is_deleted(0));
    assert!(!seg.is_deleted(9));
}

#[test]
fn test_should_seal_threshold() {
    let schema = test_schema();

    // Use a very small threshold so we can trigger it easily.
    // Each record: 128 * 4 = 512 bytes for vectors + 200 overhead = 712 bytes.
    // 10 records: ~7120 bytes. Set threshold to 5000 bytes.
    let mut seg = GrowingSegment::with_max_size(schema, 5000);

    assert!(!seg.should_seal());

    // Insert records until we exceed the threshold.
    for i in 0..10u32 {
        // Once full, insert will fail.
        match seg.insert(make_record(i)) {
            Ok(_) => {}
            Err(_) => break,
        }
    }

    assert!(seg.should_seal());
}

#[test]
fn test_should_seal_false_for_small_data() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema); // 512 MiB threshold

    for i in 0..10u32 {
        seg.insert(make_record(i)).unwrap();
    }

    assert!(
        !seg.should_seal(),
        "10 records should not trigger seal at 512MiB threshold"
    );
}

#[test]
fn test_insert_fails_when_full() {
    let schema = test_schema();
    // Threshold of 1 byte — immediately full after first insert.
    let mut seg = GrowingSegment::with_max_size(schema, 1);

    // First insert should succeed because we check at the beginning.
    // Actually, size_bytes starts at 0, so the first insert goes through.
    // After the first insert, size_bytes > 1, so the second insert fails.
    let result = seg.insert(make_record(0));
    assert!(result.is_ok());

    let result = seg.insert(make_record(1));
    assert!(result.is_err());
}

#[test]
fn test_seal_produces_valid_sealed_segment() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("sealed_from_growing");
    let schema = test_schema();

    let mut seg = GrowingSegment::new(schema);

    let mut originals = Vec::new();
    for i in 0..50u32 {
        let record = make_record(i);
        originals.push(record.clone());
        seg.insert(record).unwrap();
    }

    let metadata = seg.seal(&segment_dir).unwrap();
    assert_eq!(metadata.row_count, 50);
    assert!(metadata.sealed);

    // Open with SegmentReader and verify.
    let reader = SegmentReader::open(&segment_dir).unwrap();
    assert_eq!(reader.row_count(), 50);

    // Verify all records can be read back.
    for i in 0..50u32 {
        let record = reader.get_record(i).unwrap();
        assert_eq!(record.id(), originals[i as usize].id());
        assert_eq!(
            record.get("title"),
            Some(&FieldValue::String(format!("item_{i}")))
        );
        assert_eq!(
            record.get("score"),
            Some(&FieldValue::Float64(i as f64 * 2.5))
        );

        // Verify vector data.
        let expected_vec = make_vector(128, i);
        let actual_vec = reader.get_vector("embedding", i).unwrap();
        assert_eq!(
            actual_vec,
            expected_vec.as_slice(),
            "vector mismatch at doc {i}"
        );
    }
}

#[test]
fn test_seal_excludes_deleted_records() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("sealed_with_deletes");
    let schema = test_schema();

    let mut seg = GrowingSegment::new(schema);

    for i in 0..20u32 {
        seg.insert(make_record(i)).unwrap();
    }

    // Delete records 5, 10, 15.
    seg.mark_deleted(5);
    seg.mark_deleted(10);
    seg.mark_deleted(15);

    let metadata = seg.seal(&segment_dir).unwrap();
    // 20 - 3 deleted = 17 records written.
    assert_eq!(metadata.row_count, 17);

    let reader = SegmentReader::open(&segment_dir).unwrap();
    assert_eq!(reader.row_count(), 17);
}

#[test]
fn test_segment_id_is_unique() {
    let schema = test_schema();
    let seg1 = GrowingSegment::new(schema.clone());
    let seg2 = GrowingSegment::new(schema);
    assert_ne!(seg1.id(), seg2.id());
}

#[test]
fn test_search_with_k_zero() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);

    for i in 0..10u32 {
        seg.insert(make_record(i)).unwrap();
    }

    let query = make_vector(128, 0);
    let results = seg
        .search("embedding", &query, 0, DistanceMetric::Euclidean, None)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_nonexistent_field_returns_error() {
    let schema = test_schema();
    let mut seg = GrowingSegment::new(schema);
    seg.insert(make_record(0)).unwrap();

    let query = make_vector(128, 0);
    let result = seg.search("nonexistent", &query, 5, DistanceMetric::Euclidean, None);
    assert!(result.is_err());
}

#[test]
fn test_search_cosine_metric() {
    let schema = CollectionSchema::builder("cosine_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            128,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::new("name", FieldType::String))
        .build()
        .unwrap();

    let mut seg = GrowingSegment::new(schema);

    for i in 0..50u32 {
        let mut record = Record::new();
        record.set(
            "embedding",
            FieldValue::Vector(DenseVector::new(make_vector(128, i))),
        );
        record.set("name", FieldValue::String(format!("item_{i}")));
        seg.insert(record).unwrap();
    }

    let query = make_vector(128, 25);
    let results = seg
        .search("embedding", &query, 5, DistanceMetric::Cosine, None)
        .unwrap();
    assert_eq!(results.len(), 5);

    // Verify distances are sorted ascending.
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }

    // Brute-force ground truth for cosine.
    let mut ground_truth: Vec<(u32, f32)> = (0..50)
        .map(|i| {
            let v = make_vector(128, i);
            let dist = simd::cosine_distance(&query, &v);
            (i, dist)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let expected_ids: Vec<u32> = ground_truth.iter().take(5).map(|r| r.0).collect();
    let result_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
    assert_eq!(result_ids, expected_ids);
}
