use uuid::Uuid;

use veda_core::record::{FieldValue, Record};
use veda_core::vector::DenseVector;

// ── DenseVector basics ──

#[test]
fn test_dense_vector_creation_and_dimension() {
    let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.dimension(), 3);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_dense_vector_as_bytes() {
    let v = DenseVector::new(vec![1.0]);
    let bytes = v.as_bytes();
    assert_eq!(bytes.len(), 4); // one f32 = 4 bytes
    // Verify the byte representation matches the IEEE 754 encoding of 1.0f32
    assert_eq!(bytes, &1.0_f32.to_ne_bytes());
}

// ── Cosine distance ──

#[test]
fn test_cosine_distance_identical() {
    let a = DenseVector::new(vec![1.0, 0.0]);
    let b = DenseVector::new(vec![1.0, 0.0]);
    let d = a.cosine_distance(&b);
    assert!((d - 0.0).abs() < 1e-5, "expected ~0.0, got {d}");
}

#[test]
fn test_cosine_distance_orthogonal() {
    let a = DenseVector::new(vec![1.0, 0.0]);
    let b = DenseVector::new(vec![0.0, 1.0]);
    let d = a.cosine_distance(&b);
    assert!((d - 1.0).abs() < 1e-5, "expected ~1.0, got {d}");
}

#[test]
fn test_cosine_distance_opposite() {
    let a = DenseVector::new(vec![1.0, 0.0]);
    let b = DenseVector::new(vec![-1.0, 0.0]);
    let d = a.cosine_distance(&b);
    assert!((d - 2.0).abs() < 1e-5, "expected ~2.0, got {d}");
}

// ── Euclidean distance ──

#[test]
fn test_euclidean_distance_known() {
    let a = DenseVector::new(vec![0.0, 0.0]);
    let b = DenseVector::new(vec![3.0, 4.0]);
    let d = a.euclidean_distance(&b);
    assert!((d - 5.0).abs() < 1e-5, "expected 5.0, got {d}");
}

#[test]
fn test_euclidean_distance_same_point() {
    let a = DenseVector::new(vec![2.0, 3.0]);
    let d = a.euclidean_distance(&a);
    assert!((d - 0.0).abs() < 1e-5, "expected 0.0, got {d}");
}

// ── Dot product ──

#[test]
fn test_dot_product_known() {
    let a = DenseVector::new(vec![1.0, 2.0, 3.0]);
    let b = DenseVector::new(vec![4.0, 5.0, 6.0]);
    let d = a.dot_product(&b);
    // 1*4 + 2*5 + 3*6 = 32
    assert!((d - 32.0).abs() < 1e-5, "expected 32.0, got {d}");
}

#[test]
fn test_dot_product_orthogonal() {
    let a = DenseVector::new(vec![1.0, 0.0]);
    let b = DenseVector::new(vec![0.0, 1.0]);
    let d = a.dot_product(&b);
    assert!((d - 0.0).abs() < 1e-5, "expected 0.0, got {d}");
}

// ── Manhattan distance ──

#[test]
fn test_manhattan_distance_known() {
    let a = DenseVector::new(vec![1.0, 2.0, 3.0]);
    let b = DenseVector::new(vec![4.0, 6.0, 3.0]);
    let d = a.manhattan_distance(&b);
    // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
    assert!((d - 7.0).abs() < 1e-5, "expected 7.0, got {d}");
}

// ── Normalize ──

#[test]
fn test_normalize_produces_unit_vector() {
    let mut v = DenseVector::new(vec![3.0, 4.0]);
    v.normalize();
    let norm: f32 = v.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "expected norm ~1.0, got {norm}");
}

#[test]
fn test_normalized_returns_copy() {
    let v = DenseVector::new(vec![3.0, 4.0]);
    let n = v.normalized();
    // Original unchanged
    assert_eq!(v.as_slice(), &[3.0, 4.0]);
    // Normalized copy has unit norm
    let norm: f32 = n.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_is_normalized() {
    let v = DenseVector::new(vec![3.0, 4.0]);
    assert!(!v.is_normalized());

    let n = v.normalized();
    assert!(n.is_normalized());
}

// ── Dimension mismatch ──

#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_cosine_distance_dimension_mismatch() {
    let a = DenseVector::new(vec![1.0, 2.0]);
    let b = DenseVector::new(vec![1.0, 2.0, 3.0]);
    a.cosine_distance(&b);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_euclidean_distance_dimension_mismatch() {
    let a = DenseVector::new(vec![1.0]);
    let b = DenseVector::new(vec![1.0, 2.0]);
    a.euclidean_distance(&b);
}

// ── Record basics ──

#[test]
fn test_record_new_has_uuid() {
    let r = Record::new();
    // UUID should not be nil
    assert!(!r.id().is_nil());
}

#[test]
fn test_record_auto_generated_uuids_are_unique() {
    let r1 = Record::new();
    let r2 = Record::new();
    assert_ne!(r1.id(), r2.id());
}

#[test]
fn test_record_with_explicit_id() {
    let id = Uuid::new_v4();
    let r = Record::with_id(id);
    assert_eq!(r.id(), id);
}

// ── Record set / get / remove / has_field ──

#[test]
fn test_record_set_and_get() {
    let mut r = Record::new();
    r.set("name", FieldValue::String("alice".to_string()));
    let val = r.get("name").expect("field should exist");
    assert_eq!(val, &FieldValue::String("alice".to_string()));
}

#[test]
fn test_record_remove() {
    let mut r = Record::new();
    r.set("x", FieldValue::Int32(42));
    assert!(r.has_field("x"));

    let removed = r.remove("x");
    assert_eq!(removed, Some(FieldValue::Int32(42)));
    assert!(!r.has_field("x"));
}

#[test]
fn test_record_has_field() {
    let mut r = Record::new();
    assert!(!r.has_field("missing"));
    r.set("present", FieldValue::Bool(true));
    assert!(r.has_field("present"));
}

#[test]
fn test_record_field_names() {
    let mut r = Record::new();
    r.set("a", FieldValue::Null);
    r.set("b", FieldValue::Null);
    let mut names = r.field_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
}

// ── Record with multiple field types ──

#[test]
fn test_record_multiple_field_types() {
    let mut r = Record::new();
    r.set("title", FieldValue::String("hello".to_string()));
    r.set("score", FieldValue::Float64(0.95));
    r.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(vec![0.1, 0.2, 0.3])),
    );
    r.set(
        "image_uri",
        FieldValue::BlobRef("s3://bucket/image.png".to_string()),
    );

    assert!(matches!(r.get("title"), Some(FieldValue::String(_))));
    assert!(matches!(r.get("score"), Some(FieldValue::Float64(_))));
    assert!(matches!(r.get("embedding"), Some(FieldValue::Vector(_))));
    assert!(matches!(r.get("image_uri"), Some(FieldValue::BlobRef(_))));
    assert_eq!(r.fields().len(), 4);
}

// ── FieldValue serialization round-trip ──

#[test]
fn test_field_value_serde_roundtrip() {
    let values = vec![
        FieldValue::String("test".to_string()),
        FieldValue::Int32(42),
        FieldValue::Int64(123_456_789),
        FieldValue::Float32(1.5),
        FieldValue::Float64(2.25),
        FieldValue::Bool(true),
        FieldValue::Timestamp(1_700_000_000_000),
        FieldValue::Vector(DenseVector::new(vec![1.0, 2.0, 3.0])),
        FieldValue::Json(serde_json::json!({"key": "value"})),
        FieldValue::BlobRef("s3://bucket/obj".to_string()),
        FieldValue::Uuid(Uuid::new_v4()),
        FieldValue::Array(vec![FieldValue::Int32(1), FieldValue::Int32(2)]),
        FieldValue::Null,
    ];

    for val in &values {
        let json = serde_json::to_string(val).expect("serialization should succeed");
        let deserialized: FieldValue =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(
            val, &deserialized,
            "round-trip failed for {val:?} -> {json}"
        );
    }
}

#[test]
fn test_record_serde_roundtrip() {
    let mut r = Record::new();
    r.set("name", FieldValue::String("test".to_string()));
    r.set("count", FieldValue::Int64(7));

    let json = serde_json::to_string(&r).expect("serialize record");
    let deserialized: Record = serde_json::from_str(&json).expect("deserialize record");

    assert_eq!(deserialized.id(), r.id());
    assert_eq!(deserialized.get("name"), r.get("name"));
    assert_eq!(deserialized.get("count"), r.get("count"));
}
