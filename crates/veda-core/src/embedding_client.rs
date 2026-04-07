use crate::embedding::{EmbeddingModelConfig, Modality};
use crate::error::{Result, VedaError};

/// Client for calling embedding model endpoints.
///
/// Uses a mock implementation that returns deterministic vectors
/// based on input hashing, enabling full pipeline testing without
/// external services.
pub struct EmbeddingClient {
    config: EmbeddingModelConfig,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingModelConfig) -> Self {
        Self { config }
    }

    /// Embed a text string. Returns a deterministic, normalized vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.validate_input(text.as_bytes(), Modality::Text)?;
        Ok(deterministic_vector(text.as_bytes(), self.config.output_dimension))
    }

    /// Embed raw bytes (image, audio, video, pdf).
    /// Returns a deterministic, normalized vector.
    pub fn embed_bytes(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
        self.validate_input(data, modality)?;
        Ok(deterministic_vector(data, self.config.output_dimension))
    }

    /// Validate input before embedding.
    pub fn validate_input(&self, data: &[u8], modality: Modality) -> Result<()> {
        if !self.config.supported_modalities.contains(&modality) {
            return Err(VedaError::ModalityNotSupported {
                model: self.config.name.clone(),
                modality: modality.to_string(),
            });
        }

        if data.len() > self.config.max_input_size_bytes {
            return Err(VedaError::InvalidSchema(format!(
                "input size {} bytes exceeds max {} bytes for model '{}'",
                data.len(),
                self.config.max_input_size_bytes,
                self.config.name,
            )));
        }

        Ok(())
    }
}

/// Generate a deterministic, normalized vector from input data.
///
/// Uses a simple hash-seeded PRNG (xorshift32) to produce
/// `dimension` floats, then L2-normalizes the result.
fn deterministic_vector(data: &[u8], dimension: u32) -> Vec<f32> {
    let seed = simple_hash(data);
    let mut state = if seed == 0 { 1u32 } else { seed };
    let dim = dimension as usize;
    let mut vec = Vec::with_capacity(dim);

    for _ in 0..dim {
        state = xorshift32(state);
        // Map u32 to [-1.0, 1.0]
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        vec.push(f);
    }

    // L2 normalize
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }

    vec
}

/// Simple FNV-1a-like hash producing a u32.
fn simple_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Xorshift32 PRNG step.
fn xorshift32(mut state: u32) -> u32 {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    state
}
