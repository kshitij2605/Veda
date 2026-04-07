use veda_core::embedding::{
    AutoEmbedConfig, EmbeddingModelConfig, EmbeddingModelRegistry, Modality,
    validate_auto_embed_config,
};
use veda_core::embedding_client::EmbeddingClient;
use veda_core::error::VedaError;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;

fn clip_model() -> EmbeddingModelConfig {
    EmbeddingModelConfig {
        name: "clip-vit-l-14".into(),
        endpoint: "http://localhost:8080/embed".into(),
        supported_modalities: vec![Modality::Image, Modality::Text],
        output_dimension: 768,
        max_input_size_bytes: 10 * 1024 * 1024,
        timeout_seconds: 30,
        batch_size: 32,
    }
}

fn text_only_model() -> EmbeddingModelConfig {
    EmbeddingModelConfig {
        name: "bge-large".into(),
        endpoint: "http://localhost:8081/embed".into(),
        supported_modalities: vec![Modality::Text],
        output_dimension: 1024,
        max_input_size_bytes: 512 * 1024,
        timeout_seconds: 10,
        batch_size: 64,
    }
}

// ── 1. Register model, verify get/find ──────────────────────────────────

#[test]
fn test_register_and_get() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());
    registry.register(text_only_model());

    assert_eq!(registry.list().len(), 2);

    let clip = registry.get("clip-vit-l-14").expect("should find clip model");
    assert_eq!(clip.output_dimension, 768);

    let bge = registry.get("bge-large").expect("should find bge model");
    assert_eq!(bge.output_dimension, 1024);

    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_find_by_modality() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());
    registry.register(text_only_model());

    let text_models = registry.find_by_modality(Modality::Text);
    assert_eq!(text_models.len(), 2);

    let image_models = registry.find_by_modality(Modality::Image);
    assert_eq!(image_models.len(), 1);
    assert_eq!(image_models[0].name, "clip-vit-l-14");

    let audio_models = registry.find_by_modality(Modality::Audio);
    assert!(audio_models.is_empty());
}

// ── 2. Validate modality — supported → Ok, unsupported → error ─────────

#[test]
fn test_validate_modality_supported() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());

    assert!(registry.validate_modality("clip-vit-l-14", Modality::Image).is_ok());
    assert!(registry.validate_modality("clip-vit-l-14", Modality::Text).is_ok());
}

#[test]
fn test_validate_modality_unsupported_with_suggestion() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());
    registry.register(text_only_model());

    let err = registry
        .validate_modality("bge-large", Modality::Image)
        .unwrap_err();

    match &err {
        VedaError::ModalityNotSupported { model, modality } => {
            assert_eq!(model, "bge-large");
            assert!(modality.contains("Image"));
            assert!(modality.contains("clip-vit-l-14"), "should suggest clip model");
        }
        other => panic!("expected ModalityNotSupported, got: {other}"),
    }
}

#[test]
fn test_validate_modality_model_not_found() {
    let registry = EmbeddingModelRegistry::new();
    let err = registry.validate_modality("no-such-model", Modality::Text).unwrap_err();
    assert!(matches!(err, VedaError::InvalidSchema(_)));
}

// ── 3. Mock embedding client — text returns correct dimension ───────────

#[test]
fn test_embed_text_correct_dimension() {
    let client = EmbeddingClient::new(clip_model());
    let vec = client.embed_text("hello world").unwrap();
    assert_eq!(vec.len(), 768);

    // Should be normalized (L2 norm ≈ 1.0)
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "vector should be normalized, got norm={norm}");
}

#[test]
fn test_embed_text_deterministic() {
    let client = EmbeddingClient::new(clip_model());
    let v1 = client.embed_text("hello world").unwrap();
    let v2 = client.embed_text("hello world").unwrap();
    assert_eq!(v1, v2, "same input should produce same vector");

    let v3 = client.embed_text("goodbye world").unwrap();
    assert_ne!(v1, v3, "different input should produce different vector");
}

// ── 4. Mock embedding client — validate rejects wrong modality ──────────

#[test]
fn test_validate_input_rejects_wrong_modality() {
    let client = EmbeddingClient::new(text_only_model());
    let err = client.validate_input(b"fake image bytes", Modality::Image).unwrap_err();
    assert!(matches!(err, VedaError::ModalityNotSupported { .. }));
}

// ── 5. Mock embedding client — validate rejects oversized input ─────────

#[test]
fn test_validate_input_rejects_oversized() {
    let config = EmbeddingModelConfig {
        name: "tiny".into(),
        endpoint: "http://localhost/embed".into(),
        supported_modalities: vec![Modality::Text],
        output_dimension: 4,
        max_input_size_bytes: 100,
        timeout_seconds: 5,
        batch_size: 1,
    };
    let client = EmbeddingClient::new(config);

    // 100 bytes is ok
    assert!(client.validate_input(&[0u8; 100], Modality::Text).is_ok());
    // 101 bytes exceeds limit
    let err = client.validate_input(&[0u8; 101], Modality::Text).unwrap_err();
    assert!(matches!(err, VedaError::InvalidSchema(ref msg) if msg.contains("exceeds")), "got: {err}");
}

// ── 6. Auto-embed config validation ────────────────────────────────────

#[test]
fn test_auto_embed_config_valid() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());

    let schema = CollectionSchema::builder("products")
        .add_field(FieldDefinition::new("image", veda_core::types::FieldType::Blob))
        .add_field(FieldDefinition::vector("image_embedding", 768, DistanceMetric::Cosine))
        .build()
        .unwrap();

    let config = AutoEmbedConfig {
        source_field: "image".into(),
        target_field: "image_embedding".into(),
        model: "clip-vit-l-14".into(),
        modality: Modality::Image,
    };

    assert!(validate_auto_embed_config(&config, &registry, &schema).is_ok());
}

#[test]
fn test_auto_embed_config_missing_source_field() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());

    let schema = CollectionSchema::builder("products")
        .add_field(FieldDefinition::vector("image_embedding", 768, DistanceMetric::Cosine))
        .build()
        .unwrap();

    let config = AutoEmbedConfig {
        source_field: "image".into(),
        target_field: "image_embedding".into(),
        model: "clip-vit-l-14".into(),
        modality: Modality::Image,
    };

    let err = validate_auto_embed_config(&config, &registry, &schema).unwrap_err();
    assert!(matches!(err, VedaError::FieldNotFound(ref f) if f == "image"), "got: {err}");
}

#[test]
fn test_auto_embed_config_dimension_mismatch() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model()); // output_dimension = 768

    let schema = CollectionSchema::builder("products")
        .add_field(FieldDefinition::new("image", veda_core::types::FieldType::Blob))
        .add_field(FieldDefinition::vector("image_embedding", 512, DistanceMetric::Cosine))
        .build()
        .unwrap();

    let config = AutoEmbedConfig {
        source_field: "image".into(),
        target_field: "image_embedding".into(),
        model: "clip-vit-l-14".into(),
        modality: Modality::Image,
    };

    let err = validate_auto_embed_config(&config, &registry, &schema).unwrap_err();
    assert!(matches!(err, VedaError::DimensionMismatch { expected: 512, actual: 768 }), "got: {err}");
}

#[test]
fn test_auto_embed_config_target_not_vector() {
    let mut registry = EmbeddingModelRegistry::new();
    registry.register(clip_model());

    let schema = CollectionSchema::builder("products")
        .add_field(FieldDefinition::new("image", veda_core::types::FieldType::Blob))
        .add_field(FieldDefinition::new("image_embedding", veda_core::types::FieldType::String))
        .build()
        .unwrap();

    let config = AutoEmbedConfig {
        source_field: "image".into(),
        target_field: "image_embedding".into(),
        model: "clip-vit-l-14".into(),
        modality: Modality::Image,
    };

    let err = validate_auto_embed_config(&config, &registry, &schema).unwrap_err();
    assert!(matches!(err, VedaError::InvalidSchema(ref msg) if msg.contains("not a vector")), "got: {err}");
}

// ── 7. Modality detection from MIME types ───────────────────────────────

#[test]
fn test_modality_from_mime() {
    assert_eq!(Modality::from_mime("text/plain"), Some(Modality::Text));
    assert_eq!(Modality::from_mime("text/html"), Some(Modality::Text));
    assert_eq!(Modality::from_mime("image/png"), Some(Modality::Image));
    assert_eq!(Modality::from_mime("image/jpeg"), Some(Modality::Image));
    assert_eq!(Modality::from_mime("audio/mp3"), Some(Modality::Audio));
    assert_eq!(Modality::from_mime("audio/wav"), Some(Modality::Audio));
    assert_eq!(Modality::from_mime("video/mp4"), Some(Modality::Video));
    assert_eq!(Modality::from_mime("application/pdf"), Some(Modality::Pdf));
    assert_eq!(Modality::from_mime("application/json"), None);
    assert_eq!(Modality::from_mime("APPLICATION/PDF"), Some(Modality::Pdf));
}

#[test]
fn test_modality_from_field_name() {
    assert_eq!(Modality::from_field_name("text_content"), Some(Modality::Text));
    assert_eq!(Modality::from_field_name("title"), Some(Modality::Text));
    assert_eq!(Modality::from_field_name("description"), Some(Modality::Text));
    assert_eq!(Modality::from_field_name("image"), Some(Modality::Image));
    assert_eq!(Modality::from_field_name("product_photo"), Some(Modality::Image));
    assert_eq!(Modality::from_field_name("audio_clip"), Some(Modality::Audio));
    assert_eq!(Modality::from_field_name("video_file"), Some(Modality::Video));
    assert_eq!(Modality::from_field_name("pdf_data"), Some(Modality::Pdf));
    assert_eq!(Modality::from_field_name("random_field"), None);
}
