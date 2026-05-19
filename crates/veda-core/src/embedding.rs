use serde::{Deserialize, Serialize};

use crate::error::VedaError;
use crate::schema::CollectionSchema;
use crate::types::FieldType;

/// Supported data modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Pdf,
}

impl Modality {
    /// Detect modality from a MIME type string.
    pub fn from_mime(mime: &str) -> Option<Self> {
        let lower = mime.to_ascii_lowercase();
        if lower.starts_with("text/") {
            Some(Modality::Text)
        } else if lower.starts_with("image/") {
            Some(Modality::Image)
        } else if lower.starts_with("audio/") {
            Some(Modality::Audio)
        } else if lower.starts_with("video/") {
            Some(Modality::Video)
        } else if lower == "application/pdf" {
            Some(Modality::Pdf)
        } else {
            None
        }
    }

    /// Detect modality from a field name convention.
    pub fn from_field_name(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        if lower.contains("text") || lower.contains("title") || lower.contains("description") {
            Some(Modality::Text)
        } else if lower.contains("image") || lower.contains("photo") || lower.contains("picture") {
            Some(Modality::Image)
        } else if lower.contains("audio") || lower.contains("sound") {
            Some(Modality::Audio)
        } else if lower.contains("video") || lower.contains("clip") {
            Some(Modality::Video)
        } else if lower.contains("pdf") || lower.contains("document") {
            Some(Modality::Pdf)
        } else {
            None
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "Text",
            Modality::Image => "Image",
            Modality::Audio => "Audio",
            Modality::Video => "Video",
            Modality::Pdf => "Pdf",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelConfig {
    pub name: String,
    pub endpoint: String,
    pub api_key: String,
    pub supported_modalities: Vec<Modality>,
    pub output_dimension: u32,
    pub max_input_size_bytes: usize,
    pub timeout_seconds: u32,
    pub batch_size: u32,
}

impl EmbeddingModelConfig {
    /// Curated catalog of 2026 text embedding presets.
    ///
    /// Each preset ships the model's dimension / modality / size
    /// envelope — everything but `endpoint` and `api_key`, which
    /// the deployment supplies. See
    /// `research/2026-05-19-sota-audit/03-text-embeddings.md` for
    /// the per-model rationale.
    pub fn text_presets() -> Vec<Self> {
        vec![
            // BAAI/bge-m3 — the default. Uniquely ships dense +
            // sparse + multi-vector from one forward pass.
            Self {
                name: "bge-m3".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 1024,
                max_input_size_bytes: 32 * 1024,
                timeout_seconds: 30,
                batch_size: 32,
            },
            // Qwen/Qwen3-Embedding-8B — 2026 open-weights frontier.
            // MTEB-EN 70.58, Apache-2.0, 32K context, MRL.
            Self {
                name: "qwen3-embedding-8b".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 4096,
                max_input_size_bytes: 128 * 1024,
                timeout_seconds: 60,
                batch_size: 16,
            },
            // Qwen3-Embedding-4B — smaller sibling, ~3x cheaper at
            // similar MTEB-EN scores. Apache-2.0.
            Self {
                name: "qwen3-embedding-4b".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 2560,
                max_input_size_bytes: 128 * 1024,
                timeout_seconds: 30,
                batch_size: 32,
            },
            // Qwen3-Embedding-0.6B — small, CPU-deployable.
            Self {
                name: "qwen3-embedding-0.6b".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 1024,
                max_input_size_bytes: 128 * 1024,
                timeout_seconds: 30,
                batch_size: 64,
            },
            // voyage-3-large — proprietary frontier (Anthropic-
            // recommended). 32K context, int8/binary quantization-
            // aware. Endpoint = Voyage API base URL.
            Self {
                name: "voyage-3-large".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 1024,
                max_input_size_bytes: 128 * 1024,
                timeout_seconds: 30,
                batch_size: 128,
            },
            // voyage-3.5 — newer + cheaper than voyage-3-large.
            Self {
                name: "voyage-3.5".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text],
                output_dimension: 1024,
                max_input_size_bytes: 128 * 1024,
                timeout_seconds: 30,
                batch_size: 128,
            },
        ]
    }

    /// Curated catalog of 2026 multimodal embedding presets. See
    /// `research/2026-05-19-sota-audit/04-multimodal-embeddings.md`.
    pub fn multimodal_presets() -> Vec<Self> {
        vec![
            // Qwen/Qwen3-VL-Embedding-2B — the existing default.
            Self {
                name: "qwen3-vl-embedding-2b".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text, Modality::Image, Modality::Video],
                output_dimension: 2048,
                max_input_size_bytes: 20 * 1024 * 1024,
                timeout_seconds: 60,
                batch_size: 8,
            },
            // Qwen3-VL-Embedding-8B — opt-in "quality" tier. Same
            // Apache-2.0, ~4x VRAM. MMEB-V2 77.8.
            Self {
                name: "qwen3-vl-embedding-8b".to_owned(),
                endpoint: String::new(),
                api_key: String::new(),
                supported_modalities: vec![Modality::Text, Modality::Image, Modality::Video],
                output_dimension: 3584,
                max_input_size_bytes: 32 * 1024 * 1024,
                timeout_seconds: 120,
                batch_size: 4,
            },
        ]
    }

    /// Look up a preset by name across both text and multimodal
    /// catalogs. Returns a clone with empty endpoint/api_key.
    pub fn preset(name: &str) -> Option<Self> {
        Self::text_presets()
            .into_iter()
            .chain(Self::multimodal_presets())
            .find(|c| c.name == name)
    }

    /// Builder-style override for endpoint + api_key on a preset.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self.api_key = api_key.into();
        self
    }
}

/// Registry of configured embedding models.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingModelRegistry {
    models: Vec<EmbeddingModelConfig>,
}

impl EmbeddingModelRegistry {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Register a model.
    pub fn register(&mut self, config: EmbeddingModelConfig) {
        self.models.push(config);
    }

    /// Get a model by name.
    pub fn get(&self, name: &str) -> Option<&EmbeddingModelConfig> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Find models that support a given modality.
    pub fn find_by_modality(&self, modality: Modality) -> Vec<&EmbeddingModelConfig> {
        self.models
            .iter()
            .filter(|m| m.supported_modalities.contains(&modality))
            .collect()
    }

    /// Validate that a model supports a given modality.
    /// Returns `Ok(())` or `Err(VedaError::ModalityNotSupported)` with a suggestion.
    pub fn validate_modality(
        &self,
        model_name: &str,
        modality: Modality,
    ) -> Result<(), VedaError> {
        let model = self.get(model_name).ok_or_else(|| {
            VedaError::InvalidSchema(format!("model '{model_name}' not found in registry"))
        })?;

        if model.supported_modalities.contains(&modality) {
            return Ok(());
        }

        // Find an alternative model that does support this modality.
        let suggestion = self
            .find_by_modality(modality)
            .first()
            .map(|m| format!(". Try '{}' instead", m.name))
            .unwrap_or_default();

        Err(VedaError::ModalityNotSupported {
            model: model_name.to_owned(),
            modality: format!("{modality}{suggestion}"),
        })
    }

    /// List all registered models.
    pub fn list(&self) -> &[EmbeddingModelConfig] {
        &self.models
    }
}

/// Configuration for auto-embedding on a collection field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoEmbedConfig {
    /// Raw data field name.
    pub source_field: String,
    /// Vector field name to store embedding.
    pub target_field: String,
    /// Model name from registry.
    pub model: String,
    /// Expected modality of source data.
    pub modality: Modality,
}

/// Validate auto-embed configuration against model registry and schema.
pub fn validate_auto_embed_config(
    config: &AutoEmbedConfig,
    registry: &EmbeddingModelRegistry,
    schema: &CollectionSchema,
) -> Result<(), VedaError> {
    // Validate source field exists in schema.
    if schema.get_field(&config.source_field).is_none() {
        return Err(VedaError::FieldNotFound(config.source_field.clone()));
    }

    // Validate target field is a vector field.
    let target = schema.get_field(&config.target_field).ok_or_else(|| {
        VedaError::FieldNotFound(config.target_field.clone())
    })?;
    let target_dim = match target.field_type() {
        FieldType::Vector { dimension, .. } => *dimension,
        _ => {
            return Err(VedaError::InvalidSchema(format!(
                "target field '{}' is not a vector field",
                config.target_field
            )));
        }
    };

    // Validate model exists and supports the modality.
    let model = registry.get(&config.model).ok_or_else(|| {
        VedaError::InvalidSchema(format!("model '{}' not found in registry", config.model))
    })?;

    registry.validate_modality(&config.model, config.modality)?;

    // Validate dimension match.
    if model.output_dimension != target_dim {
        return Err(VedaError::DimensionMismatch {
            expected: target_dim,
            actual: model.output_dimension,
        });
    }

    Ok(())
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn known_presets_resolve() {
        assert!(EmbeddingModelConfig::preset("bge-m3").is_some());
        assert!(EmbeddingModelConfig::preset("qwen3-embedding-8b").is_some());
        assert!(EmbeddingModelConfig::preset("qwen3-embedding-4b").is_some());
        assert!(EmbeddingModelConfig::preset("qwen3-embedding-0.6b").is_some());
        assert!(EmbeddingModelConfig::preset("voyage-3-large").is_some());
        assert!(EmbeddingModelConfig::preset("voyage-3.5").is_some());
        assert!(EmbeddingModelConfig::preset("qwen3-vl-embedding-2b").is_some());
        assert!(EmbeddingModelConfig::preset("qwen3-vl-embedding-8b").is_some());
    }

    #[test]
    fn unknown_preset_is_none() {
        assert!(EmbeddingModelConfig::preset("not-a-real-model").is_none());
    }

    #[test]
    fn preset_has_dimensions_set() {
        let qwen = EmbeddingModelConfig::preset("qwen3-embedding-8b").unwrap();
        assert_eq!(qwen.output_dimension, 4096);
        assert!(qwen.supported_modalities.contains(&Modality::Text));
        assert!(qwen.endpoint.is_empty(), "preset endpoint must be deployment-supplied");
    }

    #[test]
    fn with_endpoint_builder_fills_deployment_fields() {
        let cfg = EmbeddingModelConfig::preset("voyage-3-large")
            .unwrap()
            .with_endpoint("https://api.voyageai.com", "sk-test");
        assert_eq!(cfg.endpoint, "https://api.voyageai.com");
        assert_eq!(cfg.api_key, "sk-test");
        assert_eq!(cfg.output_dimension, 1024);
    }

    #[test]
    fn multimodal_preset_supports_image_and_text() {
        let cfg = EmbeddingModelConfig::preset("qwen3-vl-embedding-2b").unwrap();
        assert!(cfg.supported_modalities.contains(&Modality::Text));
        assert!(cfg.supported_modalities.contains(&Modality::Image));
    }
}
