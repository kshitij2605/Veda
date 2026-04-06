use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::segment_manager::SegmentManager;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a test schema: 1 vector field (dim=32, Euclidean), 1 scalar field.
fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("manager_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            32,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("tag", FieldType::String))
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
        FieldValue::Vector(DenseVector::new(make_vector(32, i))),
    );
    record.set("tag", FieldValue::String(format!("item_{i}")));
    record
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_basic_insert_and_count() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    for i in 0..50 {
        mgr.insert(make_record(i)).unwrap();
    }

    assert_eq!(mgr.total_records(), 50);
    assert_eq!(mgr.growing_segment().count(), 50);
    assert!(mgr.sealed_segments().is_empty());
}

#[test]
fn test_auto_seal_on_threshold() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    // Each record: 32 * 4 = 128 bytes for vector + 200 overhead = 328 bytes.
    // Threshold of 1000 bytes should trigger seal after ~3 records.
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 1000).unwrap();

    // Insert records until auto-seal triggers. We'll insert 10 — enough to
    // trigger at least one seal.
    for i in 0..10 {
        mgr.insert(make_record(i)).unwrap();
    }

    assert!(
        !mgr.sealed_segments().is_empty(),
        "auto-seal should have triggered"
    );
    // The growing segment should have fewer records than 10 (rest are sealed).
    assert!(mgr.growing_segment().count() < 10);
    assert_eq!(mgr.total_records(), 10);
}

#[test]
fn test_explicit_seal() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    for i in 0..20 {
        mgr.insert(make_record(i)).unwrap();
    }

    let metadata = mgr.seal_growing().unwrap();
    assert!(metadata.is_some());
    let metadata = metadata.unwrap();
    assert_eq!(metadata.row_count, 20);
    assert!(metadata.sealed);

    assert_eq!(mgr.sealed_segments().len(), 1);
    assert_eq!(mgr.growing_segment().count(), 0);
    assert_eq!(mgr.total_records(), 20);
}

#[test]
fn test_multiple_seals() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    // First batch + seal.
    for i in 0..15 {
        mgr.insert(make_record(i)).unwrap();
    }
    let meta1 = mgr.seal_growing().unwrap().unwrap();
    assert_eq!(meta1.row_count, 15);

    // Second batch + seal.
    for i in 100..125 {
        mgr.insert(make_record(i)).unwrap();
    }
    let meta2 = mgr.seal_growing().unwrap().unwrap();
    assert_eq!(meta2.row_count, 25);

    assert_eq!(mgr.sealed_segments().len(), 2);
    assert_eq!(mgr.growing_segment().count(), 0);
    assert_eq!(mgr.total_records(), 40);
}

#[test]
fn test_sealed_segment_readable() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let mut originals = Vec::new();
    for i in 0..30 {
        let rec = make_record(i);
        originals.push(rec.clone());
        mgr.insert(rec).unwrap();
    }

    mgr.seal_growing().unwrap();
    assert_eq!(mgr.sealed_segments().len(), 1);

    let handle = &mgr.sealed_segments()[0];

    // Verify vectors.
    for i in 0..30u32 {
        let vec = handle.reader.get_vector("embedding", i).unwrap();
        let expected = make_vector(32, i);
        assert_eq!(vec, expected.as_slice(), "vector mismatch at doc {i}");
    }

    // Verify record reconstruction.
    for i in [0u32, 10, 29] {
        let record = handle.reader.get_record(i).unwrap();
        assert_eq!(record.id(), originals[i as usize].id());
        assert_eq!(
            record.get("tag"),
            Some(&FieldValue::String(format!("item_{i}")))
        );
    }
}

#[test]
fn test_delete_record() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let mut ids = Vec::new();
    for i in 0..10 {
        let rec = make_record(i);
        ids.push(rec.id());
        mgr.insert(rec).unwrap();
    }

    // Delete record 5 from growing segment.
    let found = mgr.delete_record(ids[5]).unwrap();
    assert!(found);
    assert!(mgr.growing_segment().is_deleted(5));

    // Deleting a non-existent record returns false.
    let found = mgr.delete_record(uuid::Uuid::new_v4()).unwrap();
    assert!(!found);
}

#[test]
fn test_delete_record_in_sealed_segment() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let mut ids = Vec::new();
    for i in 0..10 {
        let rec = make_record(i);
        ids.push(rec.id());
        mgr.insert(rec).unwrap();
    }

    mgr.seal_growing().unwrap();

    // Delete a record from the sealed segment.
    let found = mgr.delete_record(ids[3]).unwrap();
    assert!(found);
    assert!(mgr.sealed_segments()[0].reader.delete_bitmap().is_deleted(3));
}

#[test]
fn test_load_from_disk() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    // Create manager, insert, seal.
    {
        let mut mgr = SegmentManager::new(schema.clone(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
        for i in 0..25 {
            mgr.insert(make_record(i)).unwrap();
        }
        mgr.seal_growing().unwrap();
        assert_eq!(mgr.sealed_segments().len(), 1);
    }

    // Create a new manager with the same base_dir and recover.
    let mut mgr2 = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();
    assert!(mgr2.sealed_segments().is_empty());

    let loaded = mgr2.load_sealed_segments().unwrap();
    assert_eq!(loaded, 1);
    assert_eq!(mgr2.sealed_segments().len(), 1);
    assert_eq!(mgr2.sealed_segments()[0].metadata.row_count, 25);
}

#[test]
fn test_empty_seal_returns_none() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    // Seal with no records.
    let result = mgr.seal_growing().unwrap();
    assert!(result.is_none());
    assert!(mgr.sealed_segments().is_empty());
}

#[test]
fn test_segment_ids_unique() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut mgr = SegmentManager::new(schema, 0, dir.path(), 512 * 1024 * 1024).unwrap();

    for i in 0..5 {
        mgr.insert(make_record(i)).unwrap();
    }
    mgr.seal_growing().unwrap();

    for i in 10..15 {
        mgr.insert(make_record(i)).unwrap();
    }
    mgr.seal_growing().unwrap();

    // Insert some in growing too.
    mgr.insert(make_record(99)).unwrap();

    let ids = mgr.segment_ids();
    // Should have: 1 growing + 2 sealed = 3 unique IDs.
    assert_eq!(ids.len(), 3);

    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 3, "all segment IDs must be unique");
}
