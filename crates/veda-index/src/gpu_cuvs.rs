//! In-binary GPU-accelerated index build via NVIDIA cuVS.
//!
//! Companion to [`crate::gpu_nim_client`]. The NIM client offloads
//! to an external container — this module would link cuvs-rs
//! directly so the Veda binary itself runs the GPU build.
//!
//! ## Status
//!
//! Scaffold only. The `gpu_cuvs` cargo feature gates the cuvs-rs
//! dependency. Real activation requires:
//!
//! 1. A working CUDA toolkit at build time (CUDA 12.x recommended).
//! 2. An NVIDIA GPU + recent driver at runtime (≥ 535).
//! 3. The `cuvs` crate published with a stable Rust API. As of the
//!    last review, `cuvs-rs` was still pre-1.0 and the ergonomics
//!    are rough. The scaffolding below is the plug point; flesh it
//!    out once the upstream crate stabilises (or vendor a wrapper).
//!
//! We chose the NIM client (`crates/veda-index/src/gpu_nim_client.rs`)
//! as the primary GPU path because it doesn't force CUDA into the
//! Veda binary — operators can keep Veda CPU-only and run a sidecar
//! NIM container for the build burst. This in-binary variant is the
//! tighter integration story for deployments that want a single
//! binary and have CUDA available.
//!
//! ## Intended wire
//!
//! ```rust,ignore
//! // With the gpu_cuvs feature:
//! let mut idx = CagraIndex::new(DistanceMetric::Cosine, dim);
//! idx.build(vectors)?;            // GPU build via cuVS CAGRA
//! let hnsw_bytes = idx.export_hnsw()?; // CAGRA → HNSW conversion
//! // Persist via the normal HnswIndex::load entry point.
//! ```

use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

/// GPU CAGRA index handle. When the `gpu_cuvs` feature is off, the
/// struct exists but every method returns `VedaError::Index` with a
/// clear "feature not enabled" message so callers can detect at
/// runtime whether GPU acceleration is available.
pub struct CagraIndex {
    metric: DistanceMetric,
    dimension: u32,
    #[cfg(feature = "gpu_cuvs")]
    inner: Option<CagraInner>,
}

#[cfg(feature = "gpu_cuvs")]
struct CagraInner {
    // Placeholder for the cuvs::Index handle and any GPU state.
    // Filled in when the cuvs-rs API stabilizes.
}

impl CagraIndex {
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            metric,
            dimension,
            #[cfg(feature = "gpu_cuvs")]
            inner: None,
        }
    }

    /// Export the CAGRA graph in HNSW-compatible bytes so the rest of
    /// Veda can serve it from the existing `HnswIndex::load` path.
    /// Returns a clear error when the feature is off.
    pub fn export_hnsw(&self) -> Result<Vec<u8>> {
        #[cfg(feature = "gpu_cuvs")]
        {
            // TODO: wire cuvs::Index::to_hnsw_bytes() when the upstream
            // API exposes it; until then, this is the explicit gap.
            Err(VedaError::Index(
                "gpu_cuvs: export_hnsw not yet wired against cuvs-rs (upstream API \
                 is still pre-1.0). Track this in documentation/gpu-indexing.md."
                    .into(),
            ))
        }
        #[cfg(not(feature = "gpu_cuvs"))]
        {
            Err(VedaError::Index(
                "gpu_cuvs: CagraIndex called but the gpu_cuvs cargo feature is off; \
                 rebuild with --features veda-index/gpu_cuvs and ensure the CUDA \
                 toolkit is on the build PATH"
                    .into(),
            ))
        }
    }
}

impl VectorIndex for CagraIndex {
    fn build(&mut self, _vectors: &[&[f32]]) -> Result<()> {
        #[cfg(feature = "gpu_cuvs")]
        {
            // Placeholder — the cuvs-rs call would live here.
            return Err(VedaError::Index(
                "gpu_cuvs: build not yet wired against cuvs-rs (upstream API is \
                 still pre-1.0). The scaffold + feature flag are ready; cuvs \
                 integration is the next step once the upstream crate stabilises."
                    .into(),
            ));
        }
        #[cfg(not(feature = "gpu_cuvs"))]
        {
            Err(VedaError::Index(
                "gpu_cuvs: build called but the gpu_cuvs cargo feature is off"
                    .into(),
            ))
        }
    }

    fn search(&self, _query: &[f32], _k: usize) -> Result<Vec<SearchResult>> {
        Err(VedaError::Index(
            "gpu_cuvs: search not implemented (see documentation/gpu-indexing.md)".into(),
        ))
    }

    fn search_filtered(
        &self,
        _query: &[f32],
        _k: usize,
        _filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        self.search(_query, _k)
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: insert not implemented; CAGRA is a batch-build index".into(),
        ))
    }

    fn delete(&mut self, _id: u32) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: delete not implemented".into(),
        ))
    }

    fn len(&self) -> usize {
        0
    }

    fn save(&self, _path: &Path) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: save not implemented; export via export_hnsw and persist \
             through HnswIndex::save instead"
                .into(),
        ))
    }

    fn load(_path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::new(metric, dimension))
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_off_returns_clear_error() {
        let mut idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.build(&[]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }

    #[test]
    fn export_hnsw_error_is_actionable() {
        let idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.export_hnsw().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }
}
