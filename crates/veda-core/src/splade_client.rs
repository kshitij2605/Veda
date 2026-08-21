//! HTTP client for SPLADE-style sparse text encoders.
//!
//! SPLADE (Sparse Lexical AnD Expansion) produces a high-dimensional
//! sparse vector — typically `dimension == vocab_size` (≈ 30 K for
//! BERT-style tokenizers), with most entries zero. The non-zero
//! entries are weighted lexical activations that fold in vocabulary
//! expansion ("car" → also activates "vehicle", "auto", etc.). The
//! combination is what makes SPLADE competitive with BM25 + kNN
//! hybrids on heterogeneous corpora.
//!
//! Veda's [`sparse_vector`](crate::types::FieldType::SparseVector)
//! field type already stores this shape; this module is the missing
//! piece for getting SPLADE output INTO Veda automatically.
//!
//! ## Wire shape
//!
//! Several SPLADE deployments use slightly different request /
//! response shapes (HuggingFace TEI, vLLM with a SPLADE head, FastAPI
//! shims). We pick the TEI shape because that's what Veda's existing
//! BGE-M3 embedder uses, and TEI now supports SPLADE checkpoints:
//!
//! ```text
//! POST {endpoint}/embed_sparse
//! Authorization: Bearer <token>
//! Content-Type: application/json
//! Body: {"inputs": ["query text"]}
//!
//! 200 OK
//! Body: [
//!   [
//!     {"index": 1234, "value": 0.87},
//!     {"index": 5678, "value": 0.42},
//!     ...
//!   ]
//! ]
//! ```
//!
//! For server endpoints that follow a different shape (e.g. an
//! `indices` + `values` parallel-array variant), add a second decoder
//! branch in `parse_response`.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::{Result, VedaError};

/// Configuration for a SPLADE-encoder endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpladeClientConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Per-request timeout in seconds. Sparse encoders are typically
    /// faster than dense — sub-100ms — so 30s is generous.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    /// Vocabulary upper bound; rejects responses with indices >= this.
    /// Mirrors the `dimension` of the destination sparse_vector field.
    pub vocab_size: u32,
}

fn default_timeout() -> u64 {
    30
}

impl SpladeClientConfig {
    /// Read config from `VEDA_SPLADE_*` env vars. Returns `None` when
    /// the endpoint isn't set so callers can default to "no sparse
    /// encoding" cleanly.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("VEDA_SPLADE_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())?;
        let model = std::env::var("VEDA_SPLADE_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "naver/splade-cocondenser-ensembledistil".to_owned());
        let api_key = std::env::var("VEDA_SPLADE_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let vocab_size = std::env::var("VEDA_SPLADE_VOCAB_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30522); // BERT-base vocab
        let timeout_seconds = std::env::var("VEDA_SPLADE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        Some(Self {
            endpoint,
            model,
            api_key,
            timeout_seconds,
            vocab_size,
        })
    }
}

/// Sparse encoding result: parallel arrays of vocab indices and their
/// non-zero weights. Mirrors `veda_core::sparse::SparseVec`'s shape.
#[derive(Debug, Clone)]
pub struct SpladeEncoding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// SPLADE encoder client.
pub struct SpladeClient {
    config: SpladeClientConfig,
    http: reqwest::Client,
}

impl SpladeClient {
    pub fn new(config: SpladeClientConfig) -> Self {
        // One async client, built once and reused, exactly as
        // [`crate::embedding_client::EmbeddingClient`] does. It must be the
        // async client: every caller is inside a Tokio runtime, and
        // `reqwest::blocking` panics there with "Cannot drop a runtime in a
        // context where blocking is not allowed" the moment its internal
        // runtime is dropped. That panic was reachable from `ingest_markdown`
        // for any collection with a sparse auto-embed field.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .unwrap_or_default();
        Self { config, http }
    }

    /// Encode `text` into a sparse vector. Returns an error wrapping
    /// the HTTP status / response body when the call fails — callers
    /// should treat that as "skip SPLADE for this record" rather than
    /// aborting the whole ingest.
    #[instrument(
        name = "splade_encode",
        skip(self, text),
        fields(model = %self.config.model, input_len = text.len())
    )]
    pub async fn encode(&self, text: &str) -> Result<SpladeEncoding> {
        let url = format!(
            "{}/embed_sparse",
            self.config.endpoint.trim_end_matches('/')
        );
        let body = serde_json::json!({"inputs": [text]});
        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VedaError::Index(format!("splade: request failed: {e}")))?;
        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| VedaError::Index(format!("splade: read body: {e}")))?;
        if !status.is_success() {
            return Err(VedaError::Index(format!(
                "splade: HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(300)
                    .collect::<String>()
            )));
        }
        parse_response(&bytes, self.config.vocab_size)
    }
}

/// Decode the TEI sparse response shape. The body is a list-of-lists:
/// the outer dim is one entry per input string in the request.
fn parse_response(bytes: &[u8], vocab_size: u32) -> Result<SpladeEncoding> {
    #[derive(Deserialize)]
    struct Entry {
        index: u32,
        value: f32,
    }
    let outer: Vec<Vec<Entry>> = serde_json::from_slice(bytes)
        .map_err(|e| VedaError::Index(format!("splade: parse: {e}")))?;
    let first = outer
        .into_iter()
        .next()
        .ok_or_else(|| VedaError::Index("splade: empty response array".into()))?;
    // The TEI sparse contract orders entries by descending weight, but
    // `SparseVec::new` requires indices sorted ascending without
    // duplicates. Sort here so the encoding always satisfies that
    // invariant regardless of the encoder's output order.
    let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(first.len());
    for e in first {
        if e.index >= vocab_size {
            return Err(VedaError::Index(format!(
                "splade: index {} exceeds vocab_size {}",
                e.index, vocab_size
            )));
        }
        pairs.push((e.index, e.value));
    }
    pairs.sort_by_key(|p| p.0);
    let indices = pairs.iter().map(|p| p.0).collect();
    let values = pairs.iter().map(|p| p.1).collect();
    Ok(SpladeEncoding { indices, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn config_reads_minimal_env() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VEDA_SPLADE_ENDPOINT", "http://splade:8000");
        }
        let cfg = SpladeClientConfig::from_env().unwrap();
        assert_eq!(cfg.endpoint, "http://splade:8000");
        assert_eq!(cfg.vocab_size, 30522);
        unsafe {
            std::env::remove_var("VEDA_SPLADE_ENDPOINT");
        }
    }

    #[test]
    fn config_none_when_endpoint_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("VEDA_SPLADE_ENDPOINT");
        }
        assert!(SpladeClientConfig::from_env().is_none());
    }

    #[test]
    fn parse_tei_response_shape() {
        let body = br#"[[{"index": 5, "value": 0.7}, {"index": 11, "value": 0.4}]]"#;
        let out = parse_response(body, 32).unwrap();
        assert_eq!(out.indices, vec![5, 11]);
        assert_eq!(out.values, vec![0.7, 0.4]);
    }

    #[test]
    fn parse_rejects_out_of_range_index() {
        let body = br#"[[{"index": 999, "value": 0.5}]]"#;
        let err = parse_response(body, 32).unwrap_err();
        assert!(format!("{err}").contains("exceeds vocab_size"));
    }

    #[test]
    fn parse_sorts_indices_ascending_for_sparsevec_invariant() {
        // Real SPLADE/TEI output is ordered by descending weight, not by
        // index. The parsed encoding must come back sorted ascending so
        // `SparseVec::new` accepts it.
        let body = br#"[[{"index": 20, "value": 2.9}, {"index": 5, "value": 1.4}, {"index": 11, "value": 0.6}]]"#;
        let out = parse_response(body, 32).unwrap();
        assert_eq!(out.indices, vec![5, 11, 20]);
        assert_eq!(out.values, vec![1.4, 0.6, 2.9]);
        // Must satisfy the SparseVec contract.
        crate::sparse::SparseVec::new(out.indices, out.values).unwrap();
    }

    /// A one-shot HTTP server returning a canned TEI sparse body.
    ///
    /// Runs on its own OS thread with no async runtime of its own, so the only
    /// runtime in play is the one the test itself runs under — which is the
    /// whole point of the test below.
    fn spawn_stub() -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf);
                let body = br#"[[{"index": 11, "value": 0.4}, {"index": 5, "value": 0.7}]]"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes());
                let _ = sock.write_all(body);
                let _ = sock.flush();
            }
        });
        format!("http://{addr}")
    }

    fn stub_config(endpoint: String) -> SpladeClientConfig {
        SpladeClientConfig {
            endpoint,
            model: "test".to_owned(),
            api_key: None,
            timeout_seconds: 10,
            vocab_size: 32,
        }
    }

    /// Encoding must work from inside a Tokio runtime.
    ///
    /// Every caller in the product is async — `ingest_markdown` calls this
    /// immediately after an `.await` — so this is the only context that
    /// matters. Every other test here drives `parse_response` directly and
    /// never makes an HTTP call, which is precisely how a blocking client
    /// survived in this module undetected.
    #[tokio::test]
    async fn encode_works_from_an_async_context() {
        let client = SpladeClient::new(stub_config(spawn_stub()));
        let enc = client
            .encode("pressure valves")
            .await
            .expect("encode must succeed from async");
        assert_eq!(enc.indices, vec![5, 11]);
        assert_eq!(enc.values, vec![0.7, 0.4]);
    }
}
