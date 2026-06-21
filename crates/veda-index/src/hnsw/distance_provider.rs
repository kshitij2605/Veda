//! Pluggable distance backend for HNSW traversal.
//!
//! The HNSW graph itself is just topology — neighbour lists per layer
//! plus an entry point. Building and searching it requires answering
//! `distance(query, doc_id)` for arbitrary `doc_id`s. Historically
//! Veda's `HnswSearcher` took `vectors: &[Vec<f32>]` + a `DistanceMetric`
//! and computed `compute_distance(query, vectors[id], metric)`. That
//! works for raw-f32 storage but forecloses on quantized backends.
//!
//! `DistanceProvider` lifts that into a trait so the same searcher
//! powers either:
//! - [`RawVectorsProvider`] — the original f32 path.
//! - [`QuantizedDistanceProvider`] — RaBitQ asymmetric distance over
//!   1-bit-per-dim codes. Roughly 32× smaller working set, slightly
//!   lossier distance, same graph structure.
//!
//! The trait stays object-safe (`&dyn DistanceProvider`) so the
//! searcher can take a single parameter instead of generic bounds
//! everywhere.

use veda_core::simd;
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::quantization::rabitq::RaBitQuantizer;

/// L2-normalize a vector in place into a fresh `Vec<f32>`.
///
/// Used by the Cosine HNSW path: the index keeps an internally
/// normalized copy of every vector so traversal distance reduces to
/// `1 - dot` (see [`RawVectorsProvider`]). A zero-norm vector is
/// returned unchanged (its cosine distance is 1.0 by convention,
/// which `1 - dot` with an all-zero vector also yields).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = simd::l2_norm_squared(v).sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    let inv = 1.0 / norm;
    v.iter().map(|&x| x * inv).collect()
}

/// Read-only view that answers `distance(query, doc_id)` for HNSW
/// traversal. The graph code never sees the raw vectors — it only
/// asks for distances against a doc_id it picked from the neighbour
/// list.
pub trait DistanceProvider: Sync {
    /// Distance from a full-precision query to the document at
    /// `doc_id`. Lower = more similar by Veda convention.
    fn distance(&self, query: &[f32], doc_id: u32) -> f32;

    /// Number of documents accessible through this provider. Builders
    /// use this for entry-point bounds checks; searchers don't need
    /// it for the traversal itself.
    fn len(&self) -> usize;

    /// Whether the provider exposes zero documents.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Metric the provider was constructed with. Surfaces through to
    /// `HnswIndex::metric()` so callers can pick a downstream rerank.
    fn metric(&self) -> DistanceMetric;
}

/// Wraps a `&[Vec<f32>]` corpus and the distance metric to use. This
/// is the historical path; semantically equivalent to the old
/// `compute_distance(query, vectors[id], metric)` calls.
///
/// `normalized_cosine` is the Cosine fast path: when set, both the
/// stored `vectors` and the query handed to [`Self::distance`] are
/// assumed L2-normalized, so cosine distance reduces to `1 - dot`.
/// This skips the two norm + one division that `simd::cosine_distance`
/// recomputes on every call (~3× the FLOPs of a bare dot). The
/// returned value and ranking are identical to the Cosine path within
/// fp tolerance because for unit vectors `1 - dot == 1 - cos`.
pub struct RawVectorsProvider<'a> {
    pub vectors: &'a [Vec<f32>],
    pub metric: DistanceMetric,
    pub normalized_cosine: bool,
}

impl<'a> RawVectorsProvider<'a> {
    /// Construct a provider over raw vectors with the given metric
    /// (no normalization fast path).
    pub fn new(vectors: &'a [Vec<f32>], metric: DistanceMetric) -> Self {
        Self {
            vectors,
            metric,
            normalized_cosine: false,
        }
    }

    /// Construct a provider that treats `vectors` (and the query) as
    /// L2-normalized and computes cosine distance as `1 - dot`. Only
    /// valid for [`DistanceMetric::Cosine`].
    pub fn new_normalized_cosine(vectors: &'a [Vec<f32>]) -> Self {
        Self {
            vectors,
            metric: DistanceMetric::Cosine,
            normalized_cosine: true,
        }
    }
}

impl<'a> DistanceProvider for RawVectorsProvider<'a> {
    fn distance(&self, query: &[f32], doc_id: u32) -> f32 {
        let doc = &self.vectors[doc_id as usize];
        if self.normalized_cosine {
            1.0 - simd::dot_product(query, doc)
        } else {
            compute_distance(query, doc, self.metric)
        }
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Wraps a `RaBitQuantizer` for quantized distance lookups. Distance
/// is the asymmetric `query_f32 · sign_bits_of(doc)` reconstruction —
/// see `crates/veda-index/src/quantization/rabitq.rs`. The query
/// stays full-precision; only the indexed corpus is quantized.
pub struct QuantizedDistanceProvider<'a> {
    pub quantizer: &'a RaBitQuantizer,
}

impl<'a> QuantizedDistanceProvider<'a> {
    pub fn new(quantizer: &'a RaBitQuantizer) -> Self {
        Self { quantizer }
    }
}

impl<'a> DistanceProvider for QuantizedDistanceProvider<'a> {
    fn distance(&self, query: &[f32], doc_id: u32) -> f32 {
        self.quantizer.estimate_distance(query, doc_id as usize)
    }

    fn len(&self) -> usize {
        self.quantizer.len()
    }

    fn metric(&self) -> DistanceMetric {
        self.quantizer.metric()
    }
}
