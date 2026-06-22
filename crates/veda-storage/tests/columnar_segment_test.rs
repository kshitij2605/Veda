//! End-to-end tests for columnar-format sealed segments:
//! write a segment via `SegmentWriter::create_columnar`, seal, reopen
//! via `SegmentReader`, verify records round-trip and the on-disk
//! marker is present.

use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::segment::{SegmentReader, SegmentWriter, StoredFormat};

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("columnar_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            16,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::new("price", FieldType::Float64))
        .add_field(FieldDefinition::new("year", FieldType::Int64))
        .build()
        .unwrap()
}

fn mk_record(title: &str, price: f64, year: i64) -> Record {
    let mut r = Record::new();
    r.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(vec![0.1; 16])),
    );
    r.set("title", FieldValue::String(title.to_owned()));
    r.set("price", FieldValue::Float64(price));
    r.set("year", FieldValue::Int64(year));
    r
}

#[test]
fn columnar_segment_round_trips_via_reader() {
    let dir = TempDir::new().unwrap();
    let schema = test_schema();

    let mut w = SegmentWriter::create_columnar(dir.path(), &schema).unwrap();
    let rec_a = mk_record("alpha", 9.99, 2024);
    let rec_b = mk_record("ブラボー", 1.50, 2026);
    let rec_c = mk_record("charlie", 0.0, 1999);
    let id_a = rec_a.id();
    let id_b = rec_b.id();
    let id_c = rec_c.id();
    w.insert(&rec_a).unwrap();
    w.insert(&rec_b).unwrap();
    w.insert(&rec_c).unwrap();
    let meta = w.seal().unwrap();

    // Metadata recorded the columnar format.
    assert_eq!(meta.stored_format, StoredFormat::Columnar);
    assert_eq!(meta.row_count, 3);
    // The format marker file landed.
    assert!(dir.path().join("storage_format.txt").exists());
    assert!(dir.path().join("columns").is_dir());

    let r = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(r.metadata().stored_format, StoredFormat::Columnar);
    assert_eq!(r.metadata().row_count, 3);
    // doc_id_for_uuid works across the columnar backend.
    assert_eq!(r.doc_id_for_uuid(&id_a), Some(0));
    assert_eq!(r.doc_id_for_uuid(&id_b), Some(1));
    assert_eq!(r.doc_id_for_uuid(&id_c), Some(2));

    // Records reconstruct with scalar values intact.
    let got_a = r.get_record(0).unwrap();
    assert_eq!(got_a.id(), id_a);
    assert_eq!(
        got_a.get("title"),
        Some(&FieldValue::String("alpha".into()))
    );
    assert_eq!(got_a.get("price"), Some(&FieldValue::Float64(9.99)));
    assert_eq!(got_a.get("year"), Some(&FieldValue::Int64(2024)));

    let got_b = r.get_record(1).unwrap();
    assert_eq!(
        got_b.get("title"),
        Some(&FieldValue::String("ブラボー".into()))
    );

    let got_c = r.get_record(2).unwrap();
    assert_eq!(got_c.get("price"), Some(&FieldValue::Float64(0.0)));
}

#[test]
fn get_record_id_matches_full_record_id_both_backends() {
    // The query hot path only needs a candidate's UUID, not its whole
    // record. `get_record_id` is the cheap accessor (an O(1) mmap slice
    // on the columnar backend) and MUST return exactly the UUID that the
    // heavy `get_record(..).id()` would, for every doc, on both storage
    // backends.
    for columnar in [true, false] {
        let dir = TempDir::new().unwrap();
        let schema = test_schema();
        let mut w = if columnar {
            SegmentWriter::create_columnar(dir.path(), &schema).unwrap()
        } else {
            SegmentWriter::create(dir.path(), &schema).unwrap()
        };
        let recs = [
            mk_record("alpha", 9.99, 2024),
            mk_record("ブラボー", 1.50, 2026),
            mk_record("charlie", 0.0, 1999),
        ];
        let ids: Vec<_> = recs.iter().map(|r| r.id()).collect();
        for r in &recs {
            w.insert(r).unwrap();
        }
        w.seal().unwrap();

        let r = SegmentReader::open(dir.path()).unwrap();
        for (doc_id, expected) in ids.iter().enumerate() {
            let cheap = r.get_record_id(doc_id as u32).unwrap();
            let heavy = r.get_record(doc_id as u32).unwrap().id();
            assert_eq!(cheap, *expected, "columnar={columnar} doc {doc_id}");
            assert_eq!(cheap, heavy, "columnar={columnar} doc {doc_id}");
        }
    }
}

#[test]
fn jsonl_default_still_works_after_columnar_landing() {
    // Sanity: the default `SegmentWriter::create` path is unchanged.
    // A segment built with it loads as Jsonl format and reads back
    // every field correctly.
    let dir = TempDir::new().unwrap();
    let schema = test_schema();
    let mut w = SegmentWriter::create(dir.path(), &schema).unwrap();
    let rec = mk_record("default-jsonl", 42.0, 2026);
    let id = rec.id();
    w.insert(&rec).unwrap();
    let meta = w.seal().unwrap();
    assert_eq!(meta.stored_format, StoredFormat::Jsonl);
    assert!(!dir.path().join("storage_format.txt").exists());

    let r = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(r.metadata().stored_format, StoredFormat::Jsonl);
    assert_eq!(r.doc_id_for_uuid(&id), Some(0));
    let got = r.get_record(0).unwrap();
    assert_eq!(
        got.get("title"),
        Some(&FieldValue::String("default-jsonl".into()))
    );
}
