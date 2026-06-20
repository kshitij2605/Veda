use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::compaction::{
    CompactionConfig, compact_segments, select_segments_for_compaction,
};
use veda_storage::segment::{SegmentMetadata, SegmentReader, SegmentWriter};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("compaction_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            16,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::new("label", FieldType::String))
        .add_field(FieldDefinition::new("score", FieldType::Float64))
        .build()
        .unwrap()
}

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

fn make_record(i: u32) -> Record {
    let mut record = Record::new();
    record.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(make_vector(16, i))),
    );
    record.set("label", FieldValue::String(format!("item_{i}")));
    record.set("score", FieldValue::Float64(i as f64 * 2.5));
    record
}

/// Create a sealed segment with `count` records starting at seed `offset`.
/// Returns (SegmentMetadata, segment directory path within `base_dir`).
fn create_sealed_segment(
    base_dir: &std::path::Path,
    name: &str,
    schema: &CollectionSchema,
    offset: u32,
    count: u32,
) -> (SegmentMetadata, std::path::PathBuf) {
    let seg_dir = base_dir.join(name);
    let mut writer = SegmentWriter::create(&seg_dir, schema).unwrap();
    for i in offset..offset + count {
        let record = make_record(i);
        writer.insert(&record).unwrap();
    }
    let meta = writer.seal().unwrap();
    (meta, seg_dir)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_basic_merge_three_segments() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let (_, d1) = create_sealed_segment(dir.path(), "seg1", &schema, 0, 50);
    let (_, d2) = create_sealed_segment(dir.path(), "seg2", &schema, 50, 50);
    let (_, d3) = create_sealed_segment(dir.path(), "seg3", &schema, 100, 50);

    let r1 = SegmentReader::open(&d1).unwrap();
    let r2 = SegmentReader::open(&d2).unwrap();
    let r3 = SegmentReader::open(&d3).unwrap();

    let output_dir = dir.path().join("merged");
    let result = compact_segments(&[&r1, &r2, &r3], &schema, &output_dir).unwrap();

    assert_eq!(result.record_count, 150);
    assert_eq!(result.purged_count, 0);
    assert_eq!(result.merged_segment_ids.len(), 3);
    assert_eq!(result.new_segment.row_count, 150);
    assert!(result.new_segment.sealed);

    let merged = SegmentReader::open(&output_dir).unwrap();
    assert_eq!(merged.row_count(), 150);
    assert_eq!(merged.active_row_count(), 150);
}

#[test]
fn test_delete_purging() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let (_, d1) = create_sealed_segment(dir.path(), "seg1", &schema, 0, 50);
    let (_, d2) = create_sealed_segment(dir.path(), "seg2", &schema, 50, 50);

    let r1 = SegmentReader::open(&d1).unwrap();
    let mut r2 = SegmentReader::open(&d2).unwrap();

    // Delete 10 records (20%) from segment 2.
    for i in 0..10u32 {
        r2.delete_bitmap_mut().mark_deleted(i);
    }

    let output_dir = dir.path().join("merged");
    let result = compact_segments(&[&r1, &r2], &schema, &output_dir).unwrap();

    assert_eq!(result.record_count, 90);
    assert_eq!(result.purged_count, 10);

    let merged = SegmentReader::open(&output_dir).unwrap();
    assert_eq!(merged.row_count(), 90);
    assert_eq!(merged.active_row_count(), 90);
    // Delete bitmap should be empty in the newly compacted segment.
    assert_eq!(merged.delete_bitmap().deleted_count(), 0);
}

#[test]
fn test_selection_segment_count_threshold() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let mut segments: Vec<(SegmentMetadata, SegmentReader)> = Vec::new();
    for i in 0..5u32 {
        let (meta, d) = create_sealed_segment(dir.path(), &format!("seg{i}"), &schema, i * 20, 20);
        let reader = SegmentReader::open(&d).unwrap();
        segments.push((meta, reader));
    }

    let refs: Vec<(SegmentMetadata, &SegmentReader)> =
        segments.iter().map(|(m, r)| (m.clone(), r)).collect();

    let config = CompactionConfig {
        min_segments: 3,
        max_merge_segments: 10,
        delete_ratio_threshold: 0.2,
    };

    let selected = select_segments_for_compaction(&refs, &config);
    // With 5 segments and min_segments=3, all should be selected.
    assert_eq!(selected.len(), 5);
}

#[test]
fn test_selection_delete_ratio() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let (meta, d) = create_sealed_segment(dir.path(), "seg0", &schema, 0, 100);
    let mut reader = SegmentReader::open(&d).unwrap();

    // Delete 30 of 100 records = 30%, above threshold of 20%.
    for i in 0..30u32 {
        reader.delete_bitmap_mut().mark_deleted(i);
    }

    let segments = vec![(meta, &reader)];
    let config = CompactionConfig {
        min_segments: 5, // Higher than 1 so count-based selection doesn't trigger.
        max_merge_segments: 10,
        delete_ratio_threshold: 0.2,
    };

    let selected = select_segments_for_compaction(&segments, &config);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], 0);
}

#[test]
fn test_selection_respects_max_merge() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let mut segments: Vec<(SegmentMetadata, SegmentReader)> = Vec::new();
    for i in 0..10u32 {
        let (meta, d) = create_sealed_segment(dir.path(), &format!("seg{i}"), &schema, i * 10, 10);
        let reader = SegmentReader::open(&d).unwrap();
        segments.push((meta, reader));
    }

    let refs: Vec<(SegmentMetadata, &SegmentReader)> =
        segments.iter().map(|(m, r)| (m.clone(), r)).collect();

    let config = CompactionConfig {
        min_segments: 3,
        max_merge_segments: 5,
        delete_ratio_threshold: 0.2,
    };

    let selected = select_segments_for_compaction(&refs, &config);
    assert!(selected.len() <= 5);
}

#[test]
fn test_empty_segment_handling() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let (_, d) = create_sealed_segment(dir.path(), "seg0", &schema, 0, 50);
    let mut reader = SegmentReader::open(&d).unwrap();

    // Delete all records.
    for i in 0..50u32 {
        reader.delete_bitmap_mut().mark_deleted(i);
    }

    let output_dir = dir.path().join("merged");
    let result = compact_segments(&[&reader], &schema, &output_dir).unwrap();

    assert_eq!(result.record_count, 0);
    assert_eq!(result.purged_count, 50);
    assert_eq!(result.new_segment.row_count, 0);
}

#[test]
fn test_data_integrity_after_compaction() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    // Create original records so we can verify values after compaction.
    let mut original_records: Vec<Record> = Vec::new();
    for i in 0..100u32 {
        original_records.push(make_record(i));
    }

    // Write into two segments: [0..50) and [50..100).
    let seg1_dir = dir.path().join("seg1");
    {
        let mut writer = SegmentWriter::create(&seg1_dir, &schema).unwrap();
        for rec in &original_records[..50] {
            writer.insert(rec).unwrap();
        }
        writer.seal().unwrap();
    }

    let seg2_dir = dir.path().join("seg2");
    {
        let mut writer = SegmentWriter::create(&seg2_dir, &schema).unwrap();
        for rec in &original_records[50..] {
            writer.insert(rec).unwrap();
        }
        writer.seal().unwrap();
    }

    let r1 = SegmentReader::open(&seg1_dir).unwrap();
    let r2 = SegmentReader::open(&seg2_dir).unwrap();

    let output_dir = dir.path().join("merged");
    let result = compact_segments(&[&r1, &r2], &schema, &output_dir).unwrap();
    assert_eq!(result.record_count, 100);

    let merged = SegmentReader::open(&output_dir).unwrap();

    // Verify every record's data matches the original.
    for i in 0..100u32 {
        let record = merged.get_record(i).unwrap();
        let orig = &original_records[i as usize];

        // Record ID must match (we used the same Record objects).
        assert_eq!(record.id(), orig.id(), "record id mismatch at {i}");

        // Scalar fields.
        assert_eq!(
            record.get("label"),
            orig.get("label"),
            "label mismatch at {i}"
        );
        assert_eq!(
            record.get("score"),
            orig.get("score"),
            "score mismatch at {i}"
        );

        // Vector field.
        let merged_vec = merged.get_vector("embedding", i).unwrap();
        let orig_vec = make_vector(16, i);
        assert_eq!(merged_vec, orig_vec.as_slice(), "vector mismatch at {i}");
    }
}

#[test]
fn test_compaction_result_correctness() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let (_, d1) = create_sealed_segment(dir.path(), "seg1", &schema, 0, 60);
    let (_, d2) = create_sealed_segment(dir.path(), "seg2", &schema, 60, 40);

    let r1 = SegmentReader::open(&d1).unwrap();
    let mut r2 = SegmentReader::open(&d2).unwrap();

    let seg1_id = r1.metadata().id;
    let seg2_id = r2.metadata().id;

    // Delete 5 records in segment 2.
    for i in 0..5u32 {
        r2.delete_bitmap_mut().mark_deleted(i);
    }

    let output_dir = dir.path().join("merged");
    let result = compact_segments(&[&r1, &r2], &schema, &output_dir).unwrap();

    assert_eq!(result.merged_segment_ids.len(), 2);
    assert!(result.merged_segment_ids.contains(&seg1_id));
    assert!(result.merged_segment_ids.contains(&seg2_id));
    assert_eq!(result.record_count, 95);
    assert_eq!(result.purged_count, 5);
    assert_eq!(result.new_segment.row_count, 95);
    assert!(result.new_segment.sealed);
}
