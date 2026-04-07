use crate::embedding::{EmbeddingModelConfig, Modality};
use crate::error::{Result, VedaError};

/// Client for calling embedding model endpoints.
///
/// Uses a real HTTP client for endpoints starting with "http" (excluding "mock"),
/// and falls back to a deterministic mock for testing.
pub struct EmbeddingClient {
    config: EmbeddingModelConfig,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingModelConfig) -> Self {
        Self { config }
    }

    /// Embed a text string. Uses HTTP for real endpoints, mock otherwise.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.validate_input(text.as_bytes(), Modality::Text)?;
        if self.is_real_endpoint() {
            self.embed_text_http(text)
        } else {
            Ok(deterministic_vector(text.as_bytes(), self.config.output_dimension))
        }
    }

    /// Embed raw bytes (image, audio, video, pdf).
    /// Uses mock for now; real HTTP support for non-text modalities can be added later.
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

    fn is_real_endpoint(&self) -> bool {
        self.config.endpoint.starts_with("http") && !self.config.endpoint.contains("mock")
            && !self.config.endpoint.contains("localhost")
    }

    fn embed_text_http(&self, text: &str) -> Result<Vec<f32>> {
        let client = reqwest::blocking::Client::new();
        let body = serde_json::json!({
            "model": self.config.name,
            "input": text
        });

        let url = format!("{}/v1/embeddings", self.config.endpoint.trim_end_matches('/'));
        let mut request = client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.config.timeout_seconds as u64));

        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }

        let resp = request
            .send()
            .map_err(|e| VedaError::Storage(format!("embedding request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(VedaError::Storage(format!(
                "embedding endpoint returned {status}: {body}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| VedaError::Storage(format!("embedding response parse failed: {e}")))?;

        // Parse OpenAI-compatible response: {"data": [{"embedding": [...]}]}
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| VedaError::Storage("invalid embedding response format".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
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
