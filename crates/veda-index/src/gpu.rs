//! GPU-accelerated index building via NVIDIA cuVS.
//!
//! ## Status
//!
//! This module is a placeholder. The integration plan and trade-off
//! analysis live in `documentation/gpu-indexing.md`.
//!
//! ## What this would do
//!
//! Build HNSW / CAGRA / IVF-PQ indexes on GPU via the NVIDIA cuVS library,
//! exposed as Rust bindings (`cuvs-rs`). cuVS reports 10-50× index-build
//! speedups vs CPU HNSW on equivalent hardware. The query path can stay
//! on CPU (search is typically not the bottleneck at single-node scale).
//!
//! ## Why this isn't wired up
//!
//! cuVS requires the CUDA toolkit + a compatible NVIDIA driver at build
//! time, plus a GPU at runtime. Adding the dependency unconditionally
//! breaks the build for CPU-only users. Feature-gating works, but the
//! `cuvs-rs` API is still iterating in mid-2026; building a stable
//! wrapper at this layer needs the kind of focused review that fits a
//! dedicated session rather than being squeezed into a session-end.
//!
//! ## Trait shape (when implemented)
//!
//! ```ignore
//! pub trait GpuIndexBuilder {
//!     fn build_cagra(
//!         vectors: &[&[f32]],
//!         params: CagraParams,
//!     ) -> Result<Box<dyn VectorIndex>>;
//!
//!     fn build_ivf_pq_gpu(
//!         vectors: &[&[f32]],
//!         params: IvfPqGpuParams,
//!     ) -> Result<Box<dyn VectorIndex>>;
//! }
//! ```

#![cfg(feature = "gpu")]

// Intentionally empty — feature flag exists so callers can detect a GPU
// build, but no implementation yet. See documentation/gpu-indexing.md for
// the integration plan.
