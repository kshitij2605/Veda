use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::Result;
use veda_core::types::DistanceMetric;

/// Result of a nearest neighbor search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Internal document ID.
    pub id: u32,
    /// Distance to the query vector. Lower values indicate greater similarity
    /// for L2, cosine, and Manhattan metrics. For dot product, the raw score is
    /// negated so that the same "lower = more similar" convention applies.
    pub distance: f32,
}

/// Common trait for all vector index types.
pub trait VectorIndex: Send + Sync {
    /// Build the index from a batch of vectors. Each vector is identified by
    /// its positional index (0, 1, 2, ...).
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()>;

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Returns results sorted by distance in ascending order (closest first).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search with a filter bitmap. Only results whose IDs are in `filter`
    /// are returned.
    ///
    /// The default implementation oversamples by 10x and post-filters.
    /// Index implementations may override this with a more efficient strategy.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        let results = self.search(query, k * 10)?;
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| filter.contains(r.id))
            .take(k)
            .collect();
        Ok(filtered)
    }

    /// Insert a single vector with the given ID.
    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()>;

    /// Mark a vector as deleted (logical delete).
    fn delete(&mut self, id: u32) -> Result<()>;

    /// Number of non-deleted vectors in the index.
    fn len(&self) -> usize;

    /// Returns `true` if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist the index to disk at `path`.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load a previously persisted index from `path`.
    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized;

    /// The distance metric used by this index.
    fn metric(&self) -> DistanceMetric;

    /// The dimensionality of vectors stored in this index.
    fn dimension(&self) -> u32;
}
