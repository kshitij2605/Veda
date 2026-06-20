//! Recovery edge-case tests for `SegmentManager::load_sealed_segments`.
//!
//! Each test sets up a shard directory with one good sealed segment
//! and one degenerate one (mid-seal crash, corrupt metadata, etc.),
//! then loads the shard and checks that:
//! - The good segment is recovered.
//! - The bad one is skipped, not panic.
//! - The returned `loaded` count is exactly the number that succeeded.

use std::fs;
use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::segment_manager::SegmentManager;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("recovery_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            16,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::new("title", FieldType::String))
        .build()
        .unwrap()
}

fn make_record(title: &str) -> Record {
    let mut r = Record::new();
    r.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(vec![0.1; 16])),
    );
    r.set("title", FieldValue::String(title.to_owned()));
    r
}

/// Seal one good segment in `manager`, returning the shard dir for
/// the tests to mess with.
fn seal_one_good_segment(manager: &mut SegmentManager) {
    for i in 0..5 {
        manager.insert(make_record(&format!("title-{i}"))).unwrap();
    }
    let meta = manager
        .seal_growing()
        .expect("seal succeeds")
        .expect("not empty");
    assert!(meta.sealed);
}

#[test]
fn partial_segment_without_metadata_json_is_skipped() {
    let dir = TempDir::new().unwrap();
    let mut manager = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    seal_one_good_segment(&mut manager);
    assert_eq!(manager.sealed_segments().len(), 1);

    // Drop a half-written segment dir alongside the good one.
    let shard_dir = dir.path().join("shard_0");
    let zombie = shard_dir.join("segment_00000000-0000-0000-0000-000000000001");
    fs::create_dir_all(zombie.join("vectors")).unwrap();
    // Intentionally NO metadata.json — simulates mid-seal crash.

    // Build a fresh manager pointing at the same dir and recover.
    let mut recover = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    let loaded = recover.load_sealed_segments().unwrap();
    assert_eq!(loaded, 1);
    assert_eq!(recover.sealed_segments().len(), 1);
}

#[test]
fn segment_with_corrupt_metadata_is_skipped() {
    let dir = TempDir::new().unwrap();
    let mut manager = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    seal_one_good_segment(&mut manager);

    let shard_dir = dir.path().join("shard_0");
    let bad = shard_dir.join("segment_00000000-0000-0000-0000-000000000002");
    fs::create_dir_all(&bad).unwrap();
    // metadata.json is present but unparseable.
    fs::write(bad.join("metadata.json"), b"not json").unwrap();

    let mut recover = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    let loaded = recover.load_sealed_segments().unwrap();
    assert_eq!(loaded, 1);
}

#[test]
fn segment_with_missing_records_jsonl_is_skipped() {
    let dir = TempDir::new().unwrap();
    let mut manager = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    seal_one_good_segment(&mut manager);

    let shard_dir = dir.path().join("shard_0");
    let bad = shard_dir.join("segment_00000000-0000-0000-0000-000000000003");
    fs::create_dir_all(&bad).unwrap();
    // metadata.json + schema.json present, but the JSONL was never
    // flushed — SegmentReader::open will fail when trying to mmap it.
    fs::write(
        bad.join("metadata.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "id": "00000000-0000-0000-0000-000000000003",
            "row_count": 0,
            "schema_version": 1,
            "created_at": 0,
            "sealed": true,
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        bad.join("schema.json"),
        serde_json::to_vec(&schema()).unwrap(),
    )
    .unwrap();

    let mut recover = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    let loaded = recover.load_sealed_segments().unwrap();
    assert_eq!(loaded, 1);
}

#[test]
fn unrelated_dirs_are_ignored() {
    let dir = TempDir::new().unwrap();
    let mut manager = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    seal_one_good_segment(&mut manager);

    let shard_dir = dir.path().join("shard_0");
    // Drop a couple of bogus siblings that don't match the
    // segment_{uuid} naming convention.
    fs::create_dir_all(shard_dir.join("tmp_compact")).unwrap();
    fs::write(shard_dir.join("note.txt"), b"hello").unwrap();

    let mut recover = SegmentManager::new(schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();
    let loaded = recover.load_sealed_segments().unwrap();
    assert_eq!(loaded, 1);
}
