//! Thin client for offloading HNSW build to an NVIDIA NIM container.
//!
//! Companion to `documentation/gpu-indexing.md`. The intent is to let
//! deployments without an in-binary CUDA dependency still benefit
//! from GPU-accelerated index build by pointing Veda at a NIM service
//! that hosts cuVS CAGRA.
//!
//! ## Status
//!
//! Preview / scaffold. The client speaks a generic
//! "POST vectors → get back HNSW bytes" shape that matches the
//! `cuvs-server` style of NIM endpoint — NVIDIA's exact wire format
//! per microservice is opinionated and likely needs adjustment
//! against the concrete NIM image. Activated via the `gpu_nim` cargo
//! feature; off by default to keep the dependency tree small.
//!
//! ## Wire shape (proposed)
//!
//! ```text
//! POST {endpoint}/v1/index/cagra/build
//! Authorization: Bearer <token>
//! Content-Type: application/octet-stream
//! X-Veda-Metric: cosine | euclidean | dotproduct
//! X-Veda-Dim:    <u32>
//! X-Veda-Count:  <u32>
//! Body: contiguous f32 little-endian, count * dim * 4 bytes
//!
//! 200 OK
//! Content-Type: application/octet-stream
//! Body: serialized HNSW (cuVS export format → Veda HnswIndex::load)
//! ```
//!
//! The response body is whatever bytes Veda's existing
//! `HnswIndex::load` can ingest — i.e. NIM does the cuVS→HNSW
//! conversion server-side via `cuvs::to_hnsw()`.

#[cfg(feature = "gpu_nim")]
use std::time::Duration;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

/// Configuration for the NIM GPU build client. Drawn from env vars on
/// startup; see `from_env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuNimConfig {
    /// Base URL of the NIM container, e.g. `http://nim-cagra:8000`.
    pub endpoint: String,
    /// Optional bearer token for auth.
    pub api_key: Option<String>,
    /// Build timeout per call. NIM CAGRA build on a 100K-row corpus
    /// is sub-second; this guards against a stuck endpoint.
    pub timeout_seconds: u64,
    /// If true, fall back to local CPU build when NIM is unreachable
    /// or returns an error. Recommended for production.
    pub fallback_to_local: bool,
}

impl GpuNimConfig {
    /// Read config from `VEDA_GPU_NIM_*` env vars. Returns `None` if
    /// the endpoint isn't set, so callers default to local build.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("VEDA_GPU_NIM_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())?;
        let api_key = std::env::var("VEDA_GPU_NIM_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let timeout_seconds = std::env::var("VEDA_GPU_NIM_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let fallback_to_local = std::env::var("VEDA_GPU_NIM_FALLBACK")
            .ok()
            .map(|s| s != "0" && s.to_lowercase() != "false")
            .unwrap_or(true);
        Some(Self {
            endpoint,
            api_key,
            timeout_seconds,
            fallback_to_local,
        })
    }
}

/// Build a CAGRA → HNSW index on a NIM endpoint and return the
/// serialized bytes Veda can `HnswIndex::load` straight from disk.
///
/// `vectors` is a flat `count * dim * 4` byte buffer of contiguous
/// little-endian f32. Caller is responsible for serializing in that
/// shape — typically that's just `bytemuck::cast_slice` over a
/// `&[f32]` accumulated by row.
/// **Blocking on purpose — call it from a blocking context.** This uses
/// `reqwest::blocking`, which panics ("Cannot drop a runtime in a context
/// where blocking is not allowed") if its runtime is dropped inside a Tokio
/// worker. That is the bug PR #22 fixed in `splade_client` and that sat
/// latent in `chunking::contextualize_chunks`.
///
/// The fix here is NOT to make this async. The whole index-build path is
/// synchronous — `LocalShardSearcher::build_indexes` is a `pub fn`, and every
/// other index type builds on the calling thread — so an async variant would
/// have no caller able to await it. When this is wired up, the caller must be
/// inside `tokio::task::spawn_blocking`, which is both correct for a
/// CPU/IO-heavy index build and safe for a blocking client.
///
/// It has no callers today, so this is a constraint to honour rather than a
/// live defect.
#[cfg(feature = "gpu_nim")]
pub fn build_on_nim(
    cfg: &GpuNimConfig,
    metric: DistanceMetric,
    dimension: u32,
    count: u32,
    vectors: &[u8],
) -> Result<Vec<u8>> {
    use reqwest::blocking::Client;

    let url = format!(
        "{}/v1/index/cagra/build",
        cfg.endpoint.trim_end_matches('/')
    );
    let client = Client::builder()
        .timeout(Duration::from_secs(cfg.timeout_seconds))
        .build()
        .map_err(|e| VedaError::Index(format!("gpu_nim: client build: {e}")))?;
    let metric_header = match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::Euclidean => "euclidean",
        DistanceMetric::DotProduct => "dotproduct",
        m => {
            return Err(VedaError::Index(format!(
                "gpu_nim: unsupported metric {m:?}"
            )));
        }
    };
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/octet-stream")
        .header("X-Veda-Metric", metric_header)
        .header("X-Veda-Dim", dimension.to_string())
        .header("X-Veda-Count", count.to_string())
        .body(vectors.to_vec());
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .map_err(|e| VedaError::Index(format!("gpu_nim: request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(VedaError::Index(format!(
            "gpu_nim: HTTP {}: {}",
            resp.status(),
            resp.text().unwrap_or_default()
        )));
    }
    let bytes = resp
        .bytes()
        .map_err(|e| VedaError::Index(format!("gpu_nim: read response: {e}")))?
        .to_vec();
    Ok(bytes)
}

#[cfg(not(feature = "gpu_nim"))]
pub fn build_on_nim(
    _cfg: &GpuNimConfig,
    _metric: DistanceMetric,
    _dimension: u32,
    _count: u32,
    _vectors: &[u8],
) -> Result<Vec<u8>> {
    Err(VedaError::Index(
        "gpu_nim: build_on_nim called but the gpu_nim cargo feature is off; \
         rebuild with --features veda-index/gpu_nim"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize env-mutating tests so cargo's parallel runner doesn't
    // interleave set_var / remove_var across them. A simple Mutex is
    // enough — these tests are cheap and run in process.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn config_reads_minimal_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: env mutation is intentional and the lock makes it
        // observable only to this test at a time.
        unsafe {
            std::env::set_var("VEDA_GPU_NIM_ENDPOINT", "http://nim:8000");
        }
        let cfg = GpuNimConfig::from_env().unwrap();
        assert_eq!(cfg.endpoint, "http://nim:8000");
        assert!(cfg.fallback_to_local);
        unsafe {
            std::env::remove_var("VEDA_GPU_NIM_ENDPOINT");
        }
    }

    #[test]
    fn config_none_when_endpoint_missing() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("VEDA_GPU_NIM_ENDPOINT");
        }
        assert!(GpuNimConfig::from_env().is_none());
    }

    #[test]
    #[cfg(not(feature = "gpu_nim"))]
    fn build_returns_clear_error_when_feature_off() {
        let cfg = GpuNimConfig {
            endpoint: "http://localhost".into(),
            api_key: None,
            timeout_seconds: 1,
            fallback_to_local: false,
        };
        let err = build_on_nim(&cfg, DistanceMetric::Cosine, 8, 0, &[]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_nim cargo feature"), "got: {msg}");
    }
}
