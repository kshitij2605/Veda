use std::time::Instant;

use base64::prelude::*;
use tracing::instrument;

use crate::embedding::{EmbeddingModelConfig, Modality};
use crate::error::{Result, VedaError};
use crate::metrics as m;

/// Client for calling embedding model endpoints.
///
/// Uses a real HTTP client for endpoints starting with "http" (excluding "mock"),
/// and falls back to a deterministic mock for testing.
pub struct EmbeddingClient {
    config: EmbeddingModelConfig,
    http: reqwest::Client,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingModelConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    /// Embed a text string. Uses HTTP for real endpoints, mock otherwise.
    #[instrument(name = "embed_text", skip(self, text), fields(model = %self.config.name, input_len = text.len()))]
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.validate_input(text.as_bytes(), Modality::Text)?;
        if self.is_real_endpoint() {
            self.embed_text_http(text).await
        } else {
            Ok(deterministic_vector(text.as_bytes(), self.config.output_dimension))
        }
    }

    /// Embed raw bytes (image, audio, video, pdf).
    /// Uses HTTP for real endpoints, mock otherwise.
    #[instrument(name = "embed_bytes", skip(self, data), fields(model = %self.config.name, modality = %modality, input_size = data.len()))]
    pub async fn embed_bytes(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
        self.validate_input(data, modality)?;
        if self.is_real_endpoint() {
            self.embed_bytes_http(data, modality).await
        } else {
            Ok(deterministic_vector(data, self.config.output_dimension))
        }
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
    }

    fn build_multimodal_payload(&self, data: &[u8], modality: Modality) -> Result<serde_json::Value> {
        let b64 = BASE64_STANDARD.encode(data);
        let (media_part, instruction) = match modality {
            Modality::Image => (
                serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:image/jpeg;base64,{b64}") }
                }),
                "Represent the given image.",
            ),
            Modality::Video => (
                serde_json::json!({
                    "type": "video_url",
                    "video_url": { "url": format!("data:video/mp4;base64,{b64}") }
                }),
                "Represent the given video.",
            ),
            Modality::Pdf => {
                return Err(VedaError::Storage(
                    "PDF modality cannot be embedded directly; use OCR to extract text/images first".into(),
                ));
            }
            _ => {
                return Err(VedaError::Storage(format!(
                    "unsupported modality for multimodal embedding: {modality:?}"
                )));
            }
        };

        Ok(serde_json::json!({
            "model": self.config.name,
            "messages": [{
                "role": "user",
                "content": [media_part, {"type": "text", "text": instruction}]
            }],
            "encoding_format": "float"
        }))
    }

    async fn embed_bytes_http(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
        let start = Instant::now();
        let body = self.build_multimodal_payload(data, modality)?;

        let url = format!("{}/v1/embeddings", self.config.endpoint.trim_end_matches('/'));
        let mut request = self
            .http
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.config.timeout_seconds as u64));

        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| VedaError::Storage(format!("embedding request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(VedaError::Storage(format!(
                "embedding endpoint returned {status}: {body}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VedaError::Storage(format!("embedding response parse failed: {e}")))?;

        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| VedaError::Storage("invalid embedding response format".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        metrics::histogram!(m::EMBEDDING_DURATION, "model" => self.config.name.clone(), "modality" => format!("{:?}", modality))
            .record(start.elapsed().as_secs_f64());

        Ok(embedding)
    }

    async fn embed_text_http(&self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();
        let body = serde_json::json!({
            "model": self.config.name,
            "input": text
        });

        let url = format!("{}/v1/embeddings", self.config.endpoint.trim_end_matches('/'));
        let mut request = self
            .http
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.config.timeout_seconds as u64));

        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| VedaError::Storage(format!("embedding request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(VedaError::Storage(format!(
                "embedding endpoint returned {status}: {body}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VedaError::Storage(format!("embedding response parse failed: {e}")))?;

        // Parse OpenAI-compatible response: {"data": [{"embedding": [...]}]}
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| VedaError::Storage("invalid embedding response format".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        metrics::histogram!(m::EMBEDDING_DURATION, "model" => self.config.name.clone(), "modality" => "text")
            .record(start.elapsed().as_secs_f64());

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModelConfig;

    fn mock_config() -> EmbeddingModelConfig {
        EmbeddingModelConfig {
            name: "test-vlm".to_string(),
            endpoint: "http://mock:8000".to_string(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Image, Modality::Video],
            output_dimension: 128,
            max_input_size_bytes: 10 * 1024 * 1024,
            timeout_seconds: 30,
            batch_size: 1,
        }
    }

    #[test]
    fn test_build_multimodal_payload_image() {
        let client = EmbeddingClient::new(mock_config());
        let data = b"fake-image-bytes";
        let payload = client.build_multimodal_payload(data, Modality::Image).unwrap();

        // Verify top-level structure
        assert_eq!(payload["model"], "test-vlm");
        assert_eq!(payload["encoding_format"], "float");

        // Verify messages structure
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");

        // Verify content contains image_url + text instruction
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[1]["type"], "text");

        let url = content[0]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));

        // Verify base64 round-trips correctly
        let b64_part = url.strip_prefix("data:image/jpeg;base64,").unwrap();
        let decoded = BASE64_STANDARD.decode(b64_part).unwrap();
        assert_eq!(decoded, data);
    }

    #[tokio::test]
    async fn test_embed_bytes_mock_returns_deterministic_vector() {
        let client = EmbeddingClient::new(mock_config());
        let data = b"fake-image-bytes";

        let v1 = client.embed_bytes(data, Modality::Image).await.unwrap();
        let v2 = client.embed_bytes(data, Modality::Image).await.unwrap();

        // Right dimension and deterministic across calls (mock path).
        assert_eq!(v1.len(), 128);
        assert_eq!(v1, v2);
    }
}
