use veda_core::error::VedaError;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType, IndexType};

#[test]
fn test_mixed_field_schema() {
    let schema = CollectionSchema::builder("my_collection")
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::new("score", FieldType::Float64))
        .add_field(FieldDefinition::vector(
            "embedding",
            768,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::blob("image"))
        .build()
        .expect("schema should build successfully");

    assert_eq!(schema.name(), "my_collection");
    assert_eq!(schema.fields().len(), 4);
    assert_eq!(schema.vector_fields().len(), 1);
    // scalar fields: title (String) + score (Float64) = 2
    assert_eq!(schema.scalar_fields().len(), 2);
    assert_eq!(schema.blob_fields().len(), 1);
}

#[test]
fn test_get_field_by_name() {
    let schema = CollectionSchema::builder("lookup_test")
        .add_field(FieldDefinition::new("name", FieldType::String))
        .add_field(FieldDefinition::vector(
            "vec",
            128,
            DistanceMetric::Euclidean,
        ))
        .build()
        .unwrap();

    let field = schema.get_field("vec").expect("field 'vec' should exist");
    assert_eq!(field.name(), "vec");
    assert!(matches!(
        field.field_type(),
        FieldType::Vector {
            dimension: 128,
            metric: DistanceMetric::Euclidean
        }
    ));
    assert_eq!(field.index_type(), IndexType::Hnsw);

    assert!(schema.get_field("nonexistent").is_none());
}

#[test]
fn test_builder_rejects_empty_name() {
    let result = CollectionSchema::builder("").build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(&err, VedaError::InvalidSchema(msg) if msg.contains("empty")),
        "Expected InvalidSchema about empty name, got: {err}"
    );
}

#[test]
fn test_builder_rejects_whitespace_only_name() {
    let result = CollectionSchema::builder("   ").build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(&err, VedaError::InvalidSchema(msg) if msg.contains("empty")),
        "Expected InvalidSchema about empty name, got: {err}"
    );
}

#[test]
fn test_builder_rejects_duplicate_fields() {
    let result = CollectionSchema::builder("dup_test")
        .add_field(FieldDefinition::new("name", FieldType::String))
        .add_field(FieldDefinition::new("name", FieldType::Int32))
        .build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(&err, VedaError::InvalidSchema(msg) if msg.contains("Duplicate")),
        "Expected InvalidSchema about duplicate fields, got: {err}"
    );
}

#[test]
fn test_builder_rejects_zero_dimension_vector() {
    let result = CollectionSchema::builder("zero_dim")
        .add_field(FieldDefinition::vector(
            "bad_vec",
            0,
            DistanceMetric::Cosine,
        ))
        .build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(&err, VedaError::InvalidSchema(msg) if msg.contains("dimension")),
        "Expected InvalidSchema about dimension, got: {err}"
    );
}

#[test]
fn test_distance_metric_equality() {
    assert_eq!(DistanceMetric::Cosine, DistanceMetric::Cosine);
    assert_eq!(DistanceMetric::Euclidean, DistanceMetric::Euclidean);
    assert_ne!(DistanceMetric::Cosine, DistanceMetric::Euclidean);
    assert_ne!(DistanceMetric::DotProduct, DistanceMetric::Hamming);
    assert_ne!(DistanceMetric::Manhattan, DistanceMetric::Jaccard);
}

#[test]
fn test_serde_roundtrip_field_type() {
    let types = vec![
        FieldType::Vector {
            dimension: 256,
            metric: DistanceMetric::DotProduct,
        },
        FieldType::SparseVector,
        FieldType::String,
        FieldType::Int32,
        FieldType::Int64,
        FieldType::Float32,
        FieldType::Float64,
        FieldType::Bool,
        FieldType::Timestamp,
        FieldType::Array(Box::new(FieldType::String)),
        FieldType::Json,
        FieldType::Blob,
        FieldType::Uuid,
    ];

    for ft in &types {
        let json = serde_json::to_string(ft).expect("serialization should succeed");
        let deserialized: FieldType =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(
            ft, &deserialized,
            "Round-trip failed for {:?} -> {json}",
            ft
        );
    }
}

#[test]
fn test_serde_roundtrip_schema() {
    let schema = CollectionSchema::builder("serde_test")
        .add_field(FieldDefinition::new("name", FieldType::String))
        .add_field(FieldDefinition::vector(
            "emb",
            384,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::blob("data"))
        .add_field(
            FieldDefinition::new("tags", FieldType::Array(Box::new(FieldType::String)))
                .with_filterable(true)
                .with_index_type(IndexType::Inverted),
        )
        .shard_count(4)
        .replication_factor(3)
        .build()
        .unwrap();

    let json = serde_json::to_string(&schema).expect("serialize schema");
    let deserialized: CollectionSchema = serde_json::from_str(&json).expect("deserialize schema");

    assert_eq!(schema, deserialized);
    assert_eq!(deserialized.shard_count(), 4);
    assert_eq!(deserialized.replication_factor(), 3);
}

#[test]
fn test_field_definition_with_builders() {
    let field = FieldDefinition::new("tags", FieldType::Array(Box::new(FieldType::String)))
        .with_filterable(true)
        .with_index_type(IndexType::Inverted);

    assert_eq!(field.name(), "tags");
    assert!(field.filterable());
    assert_eq!(field.index_type(), IndexType::Inverted);
}

#[test]
fn test_default_shard_and_replication() {
    let schema = CollectionSchema::builder("defaults")
        .add_field(FieldDefinition::new("id", FieldType::Uuid))
        .build()
        .unwrap();

    assert_eq!(schema.shard_count(), 1);
    assert_eq!(schema.replication_factor(), 1);
}

#[test]
fn test_vector_field_defaults_to_hnsw() {
    let field = FieldDefinition::vector("emb", 512, DistanceMetric::Cosine);
    assert_eq!(field.index_type(), IndexType::Hnsw);
    assert!(!field.filterable());
}

#[test]
fn test_blob_field_defaults() {
    let field = FieldDefinition::blob("raw_image");
    assert_eq!(field.name(), "raw_image");
    assert!(matches!(field.field_type(), FieldType::Blob));
    assert_eq!(field.index_type(), IndexType::None);
    assert!(!field.filterable());
}

#[test]
fn test_serde_roundtrip_distance_metrics() {
    let metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Hamming,
        DistanceMetric::Manhattan,
        DistanceMetric::Jaccard,
    ];

    for m in &metrics {
        let json = serde_json::to_string(m).unwrap();
        let deserialized: DistanceMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(m, &deserialized);
    }
}

#[test]
fn test_serde_roundtrip_index_types() {
    let index_types = vec![
        IndexType::Hnsw,
        IndexType::DiskAnn,
        IndexType::IvfPq,
        IndexType::Flat,
        IndexType::SuffixArray,
        IndexType::Inverted,
        IndexType::Range,
        IndexType::Bitmap,
        IndexType::Hash,
        IndexType::None,
    ];

    for it in &index_types {
        let json = serde_json::to_string(it).unwrap();
        let deserialized: IndexType = serde_json::from_str(&json).unwrap();
        assert_eq!(it, &deserialized);
    }
}

#[test]
fn test_multiple_vector_fields() {
    let schema = CollectionSchema::builder("multi_vec")
        .add_field(FieldDefinition::vector(
            "text_emb",
            768,
            DistanceMetric::Cosine,
        ))
        .add_field(FieldDefinition::vector(
            "image_emb",
            512,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("label", FieldType::String))
        .build()
        .unwrap();

    assert_eq!(schema.vector_fields().len(), 2);
    assert_eq!(schema.scalar_fields().len(), 1);
    assert_eq!(schema.blob_fields().len(), 0);
}

#[test]
fn test_schema_id_is_unique() {
    let schema1 = CollectionSchema::builder("a")
        .add_field(FieldDefinition::new("f", FieldType::Bool))
        .build()
        .unwrap();
    let schema2 = CollectionSchema::builder("a")
        .add_field(FieldDefinition::new("f", FieldType::Bool))
        .build()
        .unwrap();

    // UUIDs should be different for each build
    assert_ne!(schema1.id(), schema2.id());
}
