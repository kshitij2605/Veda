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

use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::quantization::rabitq::RaBitQuantizer;

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
pub struct RawVectorsProvider<'a> {
    pub vectors: &'a [Vec<f32>],
    pub metric: DistanceMetric,
}

impl<'a> RawVectorsProvider<'a> {
    pub fn new(vectors: &'a [Vec<f32>], metric: DistanceMetric) -> Self {
        Self { vectors, metric }
    }
}

impl<'a> DistanceProvider for RawVectorsProvider<'a> {
    fn distance(&self, query: &[f32], doc_id: u32) -> f32 {
        compute_distance(query, &self.vectors[doc_id as usize], self.metric)
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
