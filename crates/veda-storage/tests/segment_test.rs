use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::segment::{SegmentReader, SegmentWriter};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a test schema with 2 vector fields and 3 scalar fields.
fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("test_collection")
        .add_field(FieldDefinition::vector(
            "text_embedding",
            128,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::vector(
            "image_embedding",
            64,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::new("price", FieldType::Float64))
        .add_field(FieldDefinition::new("is_active", FieldType::Bool))
        .build()
        .unwrap()
}

/// Build a schema that includes a blob field.
fn schema_with_blob() -> CollectionSchema {
    CollectionSchema::builder("blob_collection")
        .add_field(FieldDefinition::vector(
            "embedding",
            32,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::new("name", FieldType::String))
        .add_field(FieldDefinition::blob("image_data"))
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
        "text_embedding",
        FieldValue::Vector(DenseVector::new(make_vector(128, i))),
    );
    record.set(
        "image_embedding",
        FieldValue::Vector(DenseVector::new(make_vector(64, i + 10000))),
    );
    record.set("title", FieldValue::String(format!("item_{i}")));
    record.set("price", FieldValue::Float64(i as f64 * 1.5));
    record.set("is_active", FieldValue::Bool(i % 2 == 0));
    record
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_segment_write_seal_and_read_500_records() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_001");
    let schema = test_schema();

    // Keep originals for verification.
    let mut original_records: Vec<Record> = Vec::new();
    let mut original_text_vecs: Vec<Vec<f32>> = Vec::new();
    let mut original_image_vecs: Vec<Vec<f32>> = Vec::new();

    // Write 500 records.
    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..500u32 {
            let record = make_record(i);
            original_text_vecs.push(make_vector(128, i));
            original_image_vecs.push(make_vector(64, i + 10000));
            original_records.push(record.clone());
            let doc_id = writer.insert(&record).unwrap();
            assert_eq!(doc_id, i);
        }
        assert_eq!(writer.count(), 500);
        let metadata = writer.seal().unwrap();
        assert_eq!(metadata.row_count, 500);
        assert!(metadata.sealed);
    }

    // Open reader and verify.
    let reader = SegmentReader::open(&segment_dir).unwrap();

    // Row count.
    assert_eq!(reader.row_count(), 500);
    assert_eq!(reader.active_row_count(), 500);

    // Verify vectors for both fields match originals.
    for i in 0..500u32 {
        let text_vec = reader.get_vector("text_embedding", i).unwrap();
        assert_eq!(
            text_vec,
            original_text_vecs[i as usize].as_slice(),
            "text_embedding mismatch at doc {i}"
        );

        let image_vec = reader.get_vector("image_embedding", i).unwrap();
        assert_eq!(
            image_vec,
            original_image_vecs[i as usize].as_slice(),
            "image_embedding mismatch at doc {i}"
        );
    }

    // Verify full record reconstruction.
    for i in [0u32, 1, 42, 250, 499] {
        let record = reader.get_record(i).unwrap();
        let orig = &original_records[i as usize];

        // Record ID matches.
        assert_eq!(record.id(), orig.id());

        // Scalar fields match.
        assert_eq!(
            record.get("title"),
            orig.get("title"),
            "title mismatch at {i}"
        );
        assert_eq!(
            record.get("price"),
            orig.get("price"),
            "price mismatch at {i}"
        );
        assert_eq!(
            record.get("is_active"),
            orig.get("is_active"),
            "is_active mismatch at {i}"
        );

        // Vector fields match.
        if let (Some(FieldValue::Vector(a)), Some(FieldValue::Vector(b))) =
            (record.get("text_embedding"), orig.get("text_embedding"))
        {
            assert_eq!(a.as_slice(), b.as_slice(), "text vec mismatch at {i}");
        } else {
            panic!("missing text_embedding at {i}");
        }
    }
}

#[test]
fn test_inverted_index_search() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_inv");
    let schema = test_schema();

    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..100u32 {
            let record = make_record(i);
            writer.insert(&record).unwrap();
        }
        writer.seal().unwrap();
    }

    let reader = SegmentReader::open(&segment_dir).unwrap();

    // Search for a known title.
    let inv = reader.inverted_index("title").expect("title index");
    let bitmap = inv.search_exact("item_42");
    assert!(bitmap.is_some());
    let bitmap = bitmap.unwrap();
    assert!(bitmap.contains(42));
    assert_eq!(bitmap.len(), 1);

    // Search for a title that does not exist.
    let bitmap = inv.search_exact("item_999");
    assert!(bitmap.is_none());
}

#[test]
fn test_range_index_search() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_rng");
    let schema = test_schema();

    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..100u32 {
            let record = make_record(i);
            writer.insert(&record).unwrap();
        }
        writer.seal().unwrap();
    }

    let reader = SegmentReader::open(&segment_dir).unwrap();
    let rng = reader.range_index("price").expect("price index");

    // price = i * 1.5, so doc 10 has price 15.0, doc 20 has price 30.0
    // Search range [15.0, 30.0] should include docs 10..=20.
    let bitmap = rng.search_range(15.0, 30.0);
    for i in 10..=20u32 {
        assert!(bitmap.contains(i), "doc {i} should be in range [15, 30]");
    }
    // Doc 9 has price 13.5, should not be included.
    assert!(!bitmap.contains(9));
    // Doc 21 has price 31.5, should not be included.
    assert!(!bitmap.contains(21));
}

#[test]
fn test_bitmap_index_search() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_bmp");
    let schema = test_schema();

    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..100u32 {
            let record = make_record(i);
            writer.insert(&record).unwrap();
        }
        writer.seal().unwrap();
    }

    let reader = SegmentReader::open(&segment_dir).unwrap();
    let bmp = reader.bitmap_index("is_active").expect("is_active index");

    // Even docs have is_active=true.
    let true_bitmap = bmp.search_eq("true").expect("true bitmap");
    for i in (0..100u32).step_by(2) {
        assert!(true_bitmap.contains(i), "doc {i} should be active");
    }
    for i in (1..100u32).step_by(2) {
        assert!(!true_bitmap.contains(i), "doc {i} should not be active");
    }
}

#[test]
fn test_delete_bitmap_initially_empty() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_del");
    let schema = test_schema();

    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..10u32 {
            let record = make_record(i);
            writer.insert(&record).unwrap();
        }
        writer.seal().unwrap();
    }

    let reader = SegmentReader::open(&segment_dir).unwrap();
    assert_eq!(reader.delete_bitmap().deleted_count(), 0);
    for i in 0..10u32 {
        assert!(!reader.delete_bitmap().is_deleted(i));
    }
    assert_eq!(reader.active_row_count(), 10);
}

#[test]
fn test_blob_ref_fields() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_blob");
    let schema = schema_with_blob();

    let mut originals: Vec<Record> = Vec::new();
    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        for i in 0..20u32 {
            let mut record = Record::new();
            record.set(
                "embedding",
                FieldValue::Vector(DenseVector::new(make_vector(32, i))),
            );
            record.set("name", FieldValue::String(format!("blob_{i}")));
            record.set(
                "image_data",
                FieldValue::BlobRef(format!("s3://bucket/image_{i}.png")),
            );
            originals.push(record.clone());
            writer.insert(&record).unwrap();
        }
        writer.seal().unwrap();
    }

    let reader = SegmentReader::open(&segment_dir).unwrap();
    assert_eq!(reader.row_count(), 20);

    // Check blob ref is stored and can be reconstructed.
    for i in [0u32, 5, 19] {
        let record = reader.get_record(i).unwrap();
        assert_eq!(record.id(), originals[i as usize].id());
        assert_eq!(
            record.get("image_data"),
            Some(&FieldValue::BlobRef(format!(
                "s3://bucket/image_{i}.png"
            )))
        );
        assert_eq!(
            record.get("name"),
            Some(&FieldValue::String(format!("blob_{i}")))
        );
    }
}

#[test]
fn test_seal_produces_correct_directory_structure() {
    let dir = TempDir::new().unwrap();
    let segment_dir = dir.path().join("seg_structure");
    let schema = test_schema();

    {
        let mut writer = SegmentWriter::create(&segment_dir, &schema).unwrap();
        let record = make_record(0);
        writer.insert(&record).unwrap();
        writer.seal().unwrap();
    }

    // Check directory structure.
    assert!(segment_dir.join("vectors").is_dir());
    assert!(segment_dir.join("scalars").is_dir());

    // Vector files.
    assert!(segment_dir.join("vectors/text_embedding.vec").is_file());
    assert!(segment_dir.join("vectors/image_embedding.vec").is_file());

    // Scalar index files.
    assert!(segment_dir.join("scalars/title.inv").is_file());
    assert!(segment_dir.join("scalars/price.rng").is_file());
    assert!(segment_dir.join("scalars/is_active.bmp").is_file());

    // Metadata files.
    assert!(segment_dir.join("metadata.json").is_file());
    assert!(segment_dir.join("delete_bitmap.bin").is_file());
    assert!(segment_dir.join("blob_refs.json").is_file());
    assert!(segment_dir.join("records.jsonl").is_file());
    assert!(segment_dir.join("schema.json").is_file());
}

#[test]
fn test_delete_bitmap_standalone() {
    use veda_storage::delete_bitmap::DeleteBitmap;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_delete.bin");

    let mut bm = DeleteBitmap::new();
    assert_eq!(bm.deleted_count(), 0);
    assert!(!bm.is_deleted(5));

    bm.mark_deleted(5);
    bm.mark_deleted(10);
    bm.mark_deleted(42);
    assert_eq!(bm.deleted_count(), 3);
    assert!(bm.is_deleted(5));
    assert!(bm.is_deleted(10));
    assert!(bm.is_deleted(42));
    assert!(!bm.is_deleted(0));

    // Persist and reload.
    bm.save(&path).unwrap();
    let loaded = DeleteBitmap::load(&path).unwrap();
    assert_eq!(loaded.deleted_count(), 3);
    assert!(loaded.is_deleted(5));
    assert!(loaded.is_deleted(10));
    assert!(loaded.is_deleted(42));
    assert!(!loaded.is_deleted(0));
}
