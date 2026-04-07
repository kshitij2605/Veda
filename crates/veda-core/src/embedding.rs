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
