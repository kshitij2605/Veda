//! In-binary GPU-accelerated index build via NVIDIA cuVS.
//!
//! Companion to [`crate::gpu_nim_client`]. The NIM client offloads to an
//! external container; this module links `cuvs-rs` directly so the Veda
//! binary itself runs the CAGRA build on the GPU.
//!
//! ## Build prereqs (`--features veda-index/gpu_cuvs`)
//!
//! - CUDA toolkit (12.x recommended) on the build host. `cuvs-sys`'s
//!   `build.rs` invokes `nvcc` and links against `libcuvs`.
//! - An NVIDIA GPU + driver ≥ 535 at runtime.
//!
//! Default `cargo build` does not enable this feature and stays
//! CUDA-free; the no-op `CagraIndex` below still compiles and returns
//! actionable errors when called.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // With --features gpu_cuvs:
//! let mut idx = CagraIndex::new(DistanceMetric::Cosine, dim);
//! idx.build(vectors)?;            // CAGRA build on GPU
//! let hits = idx.search(query, 10)?;     // GPU search
//! idx.serialize_to_hnswlib(path)?;       // persist via cuVS HNSW
//! ```

use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

#[cfg(feature = "gpu_cuvs")]
use cuvs::cagra::{Index as CuvsCagraIndex, IndexParams, SearchParams};
#[cfg(feature = "gpu_cuvs")]
use cuvs::{ManagedTensor, Resources};
#[cfg(feature = "gpu_cuvs")]
use ndarray::{Array2, s};

/// GPU CAGRA index handle.
///
/// With the `gpu_cuvs` feature on, this owns a `cuvs::Resources` plus the
/// underlying CAGRA index built on the GPU. Without the feature, every
/// method returns a clear `VedaError::Index` so callers can detect at
/// runtime that GPU acceleration was not compiled in.
pub struct CagraIndex {
    metric: DistanceMetric,
    dimension: u32,
    count: usize,
    #[cfg(feature = "gpu_cuvs")]
    resources: Option<Resources>,
    #[cfg(feature = "gpu_cuvs")]
    index: Option<CuvsCagraIndex>,
}

impl CagraIndex {
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            metric,
            dimension,
            count: 0,
            #[cfg(feature = "gpu_cuvs")]
            resources: None,
            #[cfg(feature = "gpu_cuvs")]
            index: None,
        }
    }

    /// Persist the CAGRA graph in cuVS's hnswlib-compatible format.
    /// The bytes can be loaded back with [`Self::deserialize_from_hnswlib`].
    /// Note: per cuVS, the serialised file is only readable by cuVS's own
    /// hnswlib wrapper — it is not byte-compatible with upstream hnswlib.
    pub fn serialize_to_hnswlib(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "gpu_cuvs")]
        {
            let res = self
                .resources
                .as_ref()
                .ok_or_else(|| feature_off_error("serialize_to_hnswlib: index not built"))?;
            let idx = self
                .index
                .as_ref()
                .ok_or_else(|| feature_off_error("serialize_to_hnswlib: index not built"))?;
            idx.serialize_to_hnswlib(res, path)
                .map_err(|e| VedaError::Index(format!("cuvs serialize_to_hnswlib failed: {e:?}")))
        }
        #[cfg(not(feature = "gpu_cuvs"))]
        {
            let _ = path;
            Err(feature_off_error("serialize_to_hnswlib"))
        }
    }

    /// Load a previously-serialised CAGRA graph from cuVS's hnswlib format.
    pub fn deserialize_from_hnswlib(
        path: &Path,
        metric: DistanceMetric,
        dimension: u32,
    ) -> Result<Self> {
        #[cfg(feature = "gpu_cuvs")]
        {
            let res = Resources::new()
                .map_err(|e| VedaError::Index(format!("cuvs Resources init failed: {e:?}")))?;
            let idx = CuvsCagraIndex::deserialize(&res, path)
                .map_err(|e| VedaError::Index(format!("cuvs deserialize failed: {e:?}")))?;
            Ok(Self {
                metric,
                dimension,
                count: 0, // unknown after load — populated only by build()
                resources: Some(res),
                index: Some(idx),
            })
        }
        #[cfg(not(feature = "gpu_cuvs"))]
        {
            let _ = (path, metric, dimension);
            Err(feature_off_error("deserialize_from_hnswlib"))
        }
    }
}

#[allow(dead_code)]
fn feature_off_error(op: &str) -> VedaError {
    VedaError::Index(format!(
        "gpu_cuvs: {op} called but the gpu_cuvs cargo feature is off; \
         rebuild with --features veda-index/gpu_cuvs and ensure the CUDA \
         toolkit is on the build PATH"
    ))
}

impl VectorIndex for CagraIndex {
    #[cfg(feature = "gpu_cuvs")]
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let n = vectors.len();
        if n == 0 {
            return Err(VedaError::Index(
                "gpu_cuvs: cannot build over empty vector set".into(),
            ));
        }
        let dim = self.dimension as usize;
        // Validate dimensions match.
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VedaError::Index(format!(
                    "gpu_cuvs: vector {i} has dim {} but index dim is {dim}",
                    v.len()
                )));
            }
        }
        // Flatten to a row-major ndarray Array2 on the host. cuvs takes
        // care of staging to device memory.
        let mut flat = Vec::with_capacity(n * dim);
        for v in vectors {
            flat.extend_from_slice(v);
        }
        let dataset: Array2<f32> = Array2::from_shape_vec((n, dim), flat)
            .map_err(|e| VedaError::Index(format!("gpu_cuvs: ndarray reshape failed: {e}")))?;

        let res = Resources::new()
            .map_err(|e| VedaError::Index(format!("cuvs Resources init failed: {e:?}")))?;
        let params = IndexParams::new()
            .map_err(|e| VedaError::Index(format!("cuvs IndexParams init failed: {e:?}")))?;
        let idx = CuvsCagraIndex::build(&res, &params, &dataset)
            .map_err(|e| VedaError::Index(format!("cuvs CAGRA build failed: {e:?}")))?;

        self.resources = Some(res);
        self.index = Some(idx);
        self.count = n;
        Ok(())
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn build(&mut self, _vectors: &[&[f32]]) -> Result<()> {
        Err(feature_off_error("build"))
    }

    #[cfg(feature = "gpu_cuvs")]
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let res = self
            .resources
            .as_ref()
            .ok_or_else(|| VedaError::Index("gpu_cuvs: search before build".into()))?;
        let idx = self
            .index
            .as_ref()
            .ok_or_else(|| VedaError::Index("gpu_cuvs: search before build".into()))?;
        let dim = self.dimension as usize;
        if query.len() != dim {
            return Err(VedaError::Index(format!(
                "gpu_cuvs: query dim {} != index dim {dim}",
                query.len()
            )));
        }

        let query_host: Array2<f32> = Array2::from_shape_vec((1, dim), query.to_vec())
            .map_err(|e| VedaError::Index(format!("gpu_cuvs: query reshape failed: {e}")))?;
        let queries_dev = ManagedTensor::from(&query_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs query to_device failed: {e:?}")))?;

        let mut neighbors_host: Array2<u32> = Array2::zeros((1, k));
        let neighbors_dev = ManagedTensor::from(&neighbors_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs neighbors to_device failed: {e:?}")))?;

        let mut distances_host: Array2<f32> = Array2::zeros((1, k));
        let distances_dev = ManagedTensor::from(&distances_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs distances to_device failed: {e:?}")))?;

        let search_params = SearchParams::new()
            .map_err(|e| VedaError::Index(format!("cuvs SearchParams init failed: {e:?}")))?;

        idx.search(
            res,
            &search_params,
            &queries_dev,
            &neighbors_dev,
            &distances_dev,
        )
        .map_err(|e| VedaError::Index(format!("cuvs CAGRA search failed: {e:?}")))?;

        // Copy results back to host.
        neighbors_dev
            .to_host(res, &mut neighbors_host)
            .map_err(|e| VedaError::Index(format!("cuvs neighbors to_host failed: {e:?}")))?;
        distances_dev
            .to_host(res, &mut distances_host)
            .map_err(|e| VedaError::Index(format!("cuvs distances to_host failed: {e:?}")))?;

        let neighbors_row = neighbors_host.slice(s![0, ..]);
        let distances_row = distances_host.slice(s![0, ..]);
        let mut results = Vec::with_capacity(k);
        for (id, dist) in neighbors_row.iter().zip(distances_row.iter()) {
            results.push(SearchResult {
                id: *id,
                distance: *dist,
            });
        }
        Ok(results)
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn search(&self, _query: &[f32], _k: usize) -> Result<Vec<SearchResult>> {
        Err(feature_off_error("search"))
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        _filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        // cuVS exposes a filtered search via cuvsCagraSearchWithFilter, but
        // the bitset format is cuvs-specific (`cuvsFilter`). For now,
        // delegate to the unfiltered search; callers can post-filter
        // against the returned ids. Wiring the cuvs bitset is a follow-up
        // once the Rust binding stabilises that surface.
        self.search(query, k)
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: CAGRA is a batch-build index — incremental insert not supported".into(),
        ))
    }

    fn delete(&mut self, _id: u32) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: CAGRA delete not supported (immutable graph)".into(),
        ))
    }

    fn len(&self) -> usize {
        self.count
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.serialize_to_hnswlib(path)
    }

    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        Self::deserialize_from_hnswlib(path, metric, dimension)
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
    #[cfg(not(feature = "gpu_cuvs"))]
    fn feature_off_returns_clear_error() {
        let mut idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.build(&[]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }

    #[test]
    #[cfg(not(feature = "gpu_cuvs"))]
    fn search_off_returns_clear_error() {
        let idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.search(&[0.0; 16], 5).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }

    // Smoke test for the real GPU path. Gated behind both the cargo
    // feature and `--ignored` so it only runs on hosts with CUDA +
    // an NVIDIA GPU: `cargo test --features gpu_cuvs -- --ignored`.
    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn build_and_search_roundtrip_on_gpu() {
        let dim = 64u32;
        let n = 1024usize;
        let mut data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0; dim as usize];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = ((i * 31 + j * 17) % 1000) as f32 / 1000.0;
                }
                v
            })
            .collect();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut idx = CagraIndex::new(DistanceMetric::Cosine, dim);
        idx.build(&refs).expect("CAGRA build should succeed");
        assert_eq!(idx.len(), n);

        // Query with the first vector — its nearest neighbor should be itself.
        let q = data.remove(0);
        let hits = idx.search(&q, 5).expect("CAGRA search should succeed");
        assert_eq!(hits.len(), 5);
        assert_eq!(
            hits[0].id, 0,
            "self should be nearest neighbor; got {hits:?}"
        );
    }
}
