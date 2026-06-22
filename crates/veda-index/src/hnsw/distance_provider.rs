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
use crate::hnsw::slab::VectorSlab;
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

/// Wraps a contiguous [`VectorSlab`] corpus and the distance metric to
/// use. This is the historical raw-f32 path; semantically equivalent to
/// the old `compute_distance(query, vectors[id], metric)` calls, but the
/// row is read from a single flat slab (`slab.row(id)`) so each probe
/// hits a predictable, cache-friendly location instead of chasing a
/// per-row heap pointer.
///
/// `normalized_cosine` is the Cosine fast path: when set, both the
/// stored rows and the query handed to [`Self::distance`] are
/// assumed L2-normalized, so cosine distance reduces to `1 - dot`.
/// This skips the two norm + one division that `simd::cosine_distance`
/// recomputes on every call (~3× the FLOPs of a bare dot). The
/// returned value and ranking are identical to the Cosine path within
/// fp tolerance because for unit vectors `1 - dot == 1 - cos`.
pub struct RawVectorsProvider<'a> {
    pub vectors: &'a VectorSlab,
    pub metric: DistanceMetric,
    pub normalized_cosine: bool,
}

impl<'a> RawVectorsProvider<'a> {
    /// Construct a provider over a raw-f32 slab with the given metric
    /// (no normalization fast path).
    pub fn new(vectors: &'a VectorSlab, metric: DistanceMetric) -> Self {
        Self {
            vectors,
            metric,
            normalized_cosine: false,
        }
    }

    /// Construct a provider that treats the slab rows (and the query) as
    /// L2-normalized and computes cosine distance as `1 - dot`. Only
    /// valid for [`DistanceMetric::Cosine`].
    pub fn new_normalized_cosine(vectors: &'a VectorSlab) -> Self {
        Self {
            vectors,
            metric: DistanceMetric::Cosine,
            normalized_cosine: true,
        }
    }
}

impl<'a> DistanceProvider for RawVectorsProvider<'a> {
    fn distance(&self, query: &[f32], doc_id: u32) -> f32 {
        let doc = self.vectors.row(doc_id as usize);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    /// The contiguous-slab `RawVectorsProvider` must return distances
    /// identical (within fp tolerance) to the naive per-row
    /// `compute_distance(query, rows[id], metric)` it replaced. This is
    /// the equivalence guard for the memory-layout change: same numbers,
    /// just read from a cache-friendly location.
    #[test]
    fn slab_provider_matches_naive_rows_all_metrics() {
        let dim = 17;
        let n = 64;
        let rows: Vec<Vec<f32>> = (0..n).map(|i| vec_of(i as u32, dim)).collect();

        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ] {
            let mut slab = VectorSlab::new(dim);
            for r in &rows {
                slab.push(r);
            }
            let provider = RawVectorsProvider::new(&slab, metric);

            for q_seed in [100u32, 200, 300] {
                let query = vec_of(q_seed, dim);
                for (id, row) in rows.iter().enumerate() {
                    let from_slab = provider.distance(&query, id as u32);
                    let naive = compute_distance(&query, row, metric);
                    assert!(
                        (from_slab - naive).abs() <= 1e-5 * (1.0 + naive.abs()),
                        "metric {metric:?} id {id}: slab {from_slab} vs naive {naive}",
                    );
                }
            }
        }
    }

    /// The normalized-cosine fast path over the slab must equal
    /// `1 - dot` over L2-normalized rows — the same value the index
    /// stores and searches against.
    #[test]
    fn slab_normalized_cosine_matches_one_minus_dot() {
        let dim = 13;
        let rows: Vec<Vec<f32>> = (0..32).map(|i| l2_normalize(&vec_of(i, dim))).collect();
        let mut slab = VectorSlab::new(dim);
        for r in &rows {
            slab.push(r);
        }
        let provider = RawVectorsProvider::new_normalized_cosine(&slab);
        let query = l2_normalize(&vec_of(999, dim));
        for (id, row) in rows.iter().enumerate() {
            let got = provider.distance(&query, id as u32);
            let expect = 1.0 - simd::dot_product(&query, row);
            assert!((got - expect).abs() <= 1e-6, "id {id}: {got} vs {expect}");
        }
    }
}
