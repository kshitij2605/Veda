//! Phase F integration: segment manager with MVCC overlay.

use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::schema::FieldDefinition;
use veda_storage::segment_manager::SegmentManager;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("mvcc_test")
        .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
        .add_field(FieldDefinition::new("label", FieldType::String))
        .shard_count(1)
        .build()
        .unwrap()
}

fn record(label: &str) -> Record {
    let mut r = Record::new();
    r.set("v", FieldValue::Vector(veda_core::vector::DenseVector::new(vec![1.0, 0.0])));
    r.set("label", FieldValue::String(label.to_owned()));
    r
}

#[test]
fn insert_mvcc_requires_enable_mvcc() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    let r = record("a");
    let err = sm.insert_mvcc(100, r).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("enable_mvcc"), "expected enable_mvcc hint, got {msg}");
}

#[test]
fn insert_mvcc_then_read_at_ts_returns_visible_version() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    let r = record("v1");
    let rid = r.id();
    sm.insert_mvcc(100, r).unwrap();
    // Re-use the same UUID: simulate updating the same logical row.
    let mut r2 = Record::with_id(rid);
    r2.set("v", FieldValue::Vector(veda_core::vector::DenseVector::new(vec![1.0, 0.0])));
    r2.set("label", FieldValue::String("v2".to_owned()));
    sm.insert_mvcc(200, r2).unwrap();

    let r150_bytes = sm.read_at_ts(&rid, 150).unwrap();
    let r250_bytes = sm.read_at_ts(&rid, 250).unwrap();
    let r150: Record = serde_json::from_slice(&r150_bytes).unwrap();
    let r250: Record = serde_json::from_slice(&r250_bytes).unwrap();
    let label_150 = match r150.get("label") {
        Some(FieldValue::String(s)) => s.clone(),
        other => panic!("unexpected label@150: {other:?}"),
    };
    let label_250 = match r250.get("label") {
        Some(FieldValue::String(s)) => s.clone(),
        other => panic!("unexpected label@250: {other:?}"),
    };
    assert_eq!(label_150, "v1");
    assert_eq!(label_250, "v2");
}

#[test]
fn delete_mvcc_tombstones_at_ts() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    let r = record("live");
    let rid = r.id();
    sm.insert_mvcc(100, r).unwrap();
    sm.delete_mvcc(200, &rid).unwrap();
    // Read at 150 sees the live row; at 250 sees None.
    assert!(sm.read_at_ts(&rid, 150).is_some());
    assert!(sm.read_at_ts(&rid, 250).is_none());
}

#[test]
fn mvcc_gc_runs_when_safe_ts_advances() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm.enable_mvcc();
    let r = record("a");
    let rid = r.id();
    for ts in [100u128, 200, 300] {
        let mut clone = Record::with_id(rid);
        clone.set("v", FieldValue::Vector(veda_core::vector::DenseVector::new(vec![1.0, 0.0])));
        clone.set("label", FieldValue::String(format!("upd{ts}")));
        sm.insert_mvcc(ts, clone).unwrap();
    }
    let _ = r;
    sm.set_mvcc_safe_ts(250);
    let dropped = sm.mvcc_gc().unwrap();
    assert!(dropped >= 1);
}

#[test]
fn mvcc_durable_survives_drop_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let wal_path = tmp.path().join("mvcc.wal");
    let rid;
    // Lifetime 1: enable durable MVCC, write 2 versions, drop.
    {
        let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
        sm.enable_mvcc_durable(&wal_path).unwrap();
        let r = record("v1");
        rid = r.id();
        sm.insert_mvcc(100, r).unwrap();
        let mut r2 = Record::with_id(rid);
        r2.set("v", FieldValue::Vector(veda_core::vector::DenseVector::new(vec![1.0, 0.0])));
        r2.set("label", FieldValue::String("v2".into()));
        sm.insert_mvcc(200, r2).unwrap();
    }
    // Lifetime 2: a fresh manager opens the SAME WAL; the
    // versions are replayed and visible.
    let mut sm2 = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm2.enable_mvcc_durable(&wal_path).unwrap();
    let bytes_150 = sm2.read_at_ts(&rid, 150).expect("v1 visible at 150");
    let bytes_250 = sm2.read_at_ts(&rid, 250).expect("v2 visible at 250");
    let r150: Record = serde_json::from_slice(&bytes_150).unwrap();
    let r250: Record = serde_json::from_slice(&bytes_250).unwrap();
    let label_150 = match r150.get("label") {
        Some(FieldValue::String(s)) => s.clone(),
        other => panic!("label@150: {other:?}"),
    };
    let label_250 = match r250.get("label") {
        Some(FieldValue::String(s)) => s.clone(),
        other => panic!("label@250: {other:?}"),
    };
    assert_eq!(label_150, "v1");
    assert_eq!(label_250, "v2");
}

#[test]
fn mvcc_durable_replays_set_safe_ts() {
    let tmp = tempfile::tempdir().unwrap();
    let wal_path = tmp.path().join("mvcc.wal");
    {
        let mut sm = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
        sm.enable_mvcc_durable(&wal_path).unwrap();
        sm.set_mvcc_safe_ts(777);
    }
    // Re-open and run mvcc_gc — the watermark survived so GC
    // honours it.
    let mut sm2 = SegmentManager::new(schema(), 0, tmp.path(), 512 * 1024).unwrap();
    sm2.enable_mvcc_durable(&wal_path).unwrap();
    // Insert a version at 100 (≤ safe_ts), then at 900 (>
    // safe_ts). GC should be eligible to drop the version at
    // 100 once a newer one exists.
    let r1 = record("a");
    let rid = r1.id();
    sm2.insert_mvcc(100, r1).unwrap();
    let mut r2 = Record::with_id(rid);
    r2.set("v", FieldValue::Vector(veda_core::vector::DenseVector::new(vec![1.0, 0.0])));
    r2.set("label", FieldValue::String("b".into()));
    sm2.insert_mvcc(900, r2).unwrap();
    let dropped = sm2.mvcc_gc().unwrap();
    // The version at 100 is OLDER than the floor (the newest ≤
    // safe_ts=777 is at 100; nothing else is ≤ 777). So the
    // floor IS the 100 version and it must be kept. The 900
    // version is > safe_ts, also kept. Net dropped = 0.
    assert_eq!(dropped, 0);
}
