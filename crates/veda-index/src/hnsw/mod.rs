pub mod builder;
pub mod distance_provider;
pub mod graph;
pub mod quantized;
pub mod searcher;

pub use builder::HnswBuilder;
pub use graph::HnswGraph;
pub use searcher::HnswSearcher;

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::hnsw::distance_provider::DistanceProvider;
use crate::traits::{SearchResult, VectorIndex};

/// HNSW (Hierarchical Navigable Small World) index implementing VectorIndex.
///
/// This provides approximate nearest neighbor search with logarithmic search
/// complexity while achieving high recall rates (typically > 95% with default
/// parameters).
pub struct HnswIndex {
    graph: HnswGraph,
    // TODO(perf): flat contiguous `Vec<f32>` slab + CSR adjacency layout.
    // The current `Vec<Vec<f32>>` vectors and per-node `Vec<Vec<u32>>`
    // neighbour lists (graph.rs) scatter every distance probe and edge
    // walk across the heap, costing a cache miss per access on the search
    // hot path. A single contiguous vector slab (id*dim slice) + CSR
    // adjacency would be the biggest remaining structural win, but it is a
    // rewrite plus an on-disk format change — deferred to a later cycle.
    /// Internal vector copies. For Cosine metric these are kept
    /// **L2-normalized** so traversal distance reduces to `1 - dot`
    /// (see `normalized` and `distance_provider::RawVectorsProvider`).
    /// This is an index-internal representation only — the record store
    /// is untouched, so `get_record` still returns original vectors.
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    /// True iff `metric == Cosine`: enables the normalized-dot fast path.
    normalized: bool,
    // Configuration
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    deleted: HashSet<u32>,
}

/// Serializable form of `HnswIndex` for save/load round-trips.
#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    graph: HnswGraph,
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    deleted: Vec<u32>,
}

impl HnswIndex {
    /// Create a new HNSW index with default parameters.
    ///
    /// Defaults: m=16, m_max0=32, ef_construction=200, ef_search=96.
    /// `ef_search=96` is the lowest fixed traversal budget that holds
    /// recall@10 ≥ 0.99 on the at-scale single-segment benchmark while
    /// roughly halving per-query work vs the old 200 default. Per-query
    /// callers can still override via [`Self::search_with_ef`].
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        let m = 16;
        Self {
            graph: HnswGraph::new(),
            vectors: Vec::new(),
            metric,
            dimension,
            normalized: metric == DistanceMetric::Cosine,
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ef_search: 96,
            ml: 1.0 / (m as f64).ln(),
            deleted: HashSet::new(),
        }
    }

    /// Create a new HNSW index with custom parameters.
    pub fn with_params(
        metric: DistanceMetric,
        dimension: u32,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        Self {
            graph: HnswGraph::new(),
            vectors: Vec::new(),
            metric,
            dimension,
            normalized: metric == DistanceMetric::Cosine,
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            deleted: HashSet::new(),
        }
    }

    /// Build the traversal `DistanceProvider` over the internal vectors,
    /// selecting the normalized-dot fast path for Cosine indexes.
    fn make_provider(&self) -> crate::hnsw::distance_provider::RawVectorsProvider<'_> {
        use crate::hnsw::distance_provider::RawVectorsProvider;
        if self.normalized {
            RawVectorsProvider::new_normalized_cosine(&self.vectors)
        } else {
            RawVectorsProvider::new(&self.vectors, self.metric)
        }
    }

    /// Prepare a query for traversal: L2-normalize for Cosine indexes
    /// (so `1 - dot` matches cosine distance), pass through otherwise.
    fn prepare_query(&self, query: &[f32]) -> Vec<f32> {
        if self.normalized {
            crate::hnsw::distance_provider::l2_normalize(query)
        } else {
            query.to_vec()
        }
    }

    /// Core layer-0 traversal shared by `search` / `search_with_ef`,
    /// parameterized by the per-query `ef` budget. `query_internal` must
    /// already be query-prepared (normalized for Cosine).
    fn search_internal(&self, query_internal: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let ep = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let dp = self.make_provider();
        let mut ep_dist = dp.distance(query_internal, ep_id);

        // Phase 1: Greedily descend from the top level down to layer 1.
        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(query_internal, ep_id, layer, &self.graph, &dp);
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        // Phase 2: Search at layer 0 with the requested ef (>= k).
        let entry_points = vec![(ep_id, ep_dist)];
        let results = HnswSearcher::search_layer(
            query_internal,
            &entry_points,
            ef.max(k),
            0,
            &self.graph,
            &dp,
            &self.deleted,
        );

        results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }

    /// Set the ef_search parameter at runtime to tune the recall/speed trade-off.
    ///
    /// Higher ef_search values yield better recall at the cost of slower queries.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    /// Get the current ef_search value.
    pub fn ef_search(&self) -> usize {
        self.ef_search
    }
}

impl VectorIndex for HnswIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        // Reset state.
        self.graph = HnswGraph::new();
        self.vectors.clear();
        self.deleted.clear();

        // Validate dimensions and store vectors. For Cosine, store an
        // L2-normalized copy so traversal distance is `1 - dot` (the
        // record store keeps the originals — this copy is index-internal).
        for v in vectors {
            if v.len() as u32 != self.dimension {
                return Err(VedaError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len() as u32,
                });
            }
            if self.normalized {
                self.vectors
                    .push(crate::hnsw::distance_provider::l2_normalize(v));
            } else {
                self.vectors.push(v.to_vec());
            }
        }

        // Insert each vector into the HNSW graph.
        for id in 0..self.vectors.len() {
            HnswBuilder::insert(
                id as u32,
                &self.vectors[id],
                &mut self.graph,
                &self.vectors,
                self.metric,
                self.normalized,
                self.m,
                self.m_max0,
                self.ef_construction,
                self.ml,
            );
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.ef_search)
    }

    fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }

        if k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let q = self.prepare_query(query);
        Ok(self.search_internal(&q, k, ef))
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }

        if k == 0 || self.vectors.is_empty() || filter.is_empty() {
            return Ok(Vec::new());
        }

        let ep = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let q = self.prepare_query(query);
        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let dp = self.make_provider();
        let mut ep_dist = dp.distance(&q, ep_id);

        // Phase 1: Greedily descend from the top level down to layer 1.
        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(&q, ep_id, layer, &self.graph, &dp);
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        // Phase 2: Filtered search at layer 0.
        let entry_points = vec![(ep_id, ep_dist)];
        let results = HnswSearcher::search_layer_filtered(
            &self.graph,
            &dp,
            &q,
            &entry_points,
            self.ef_search.max(k),
            0,
            &self.deleted,
            filter,
            k,
        );

        Ok(results)
    }

    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()> {
        if vector.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len() as u32,
            });
        }

        // Grow the vectors array if needed.
        if id as usize >= self.vectors.len() {
            self.vectors
                .resize(id as usize + 1, vec![0.0; self.dimension as usize]);
        }
        // Store the index-internal (normalized for Cosine) copy.
        let stored = if self.normalized {
            crate::hnsw::distance_provider::l2_normalize(vector)
        } else {
            vector.to_vec()
        };
        self.vectors[id as usize] = stored;
        self.deleted.remove(&id);

        // Insert into the HNSW graph using the stored representation so
        // build-time distances match the normalized search-time path.
        let inserted = self.vectors[id as usize].clone();
        HnswBuilder::insert(
            id,
            &inserted,
            &mut self.graph,
            &self.vectors,
            self.metric,
            self.normalized,
            self.m,
            self.m_max0,
            self.ef_construction,
            self.ml,
        );

        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        if (id as usize) >= self.vectors.len() {
            return Err(VedaError::Index(format!(
                "Cannot delete vector with id {id}: id out of range"
            )));
        }
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        // Count actual non-None nodes minus deleted.
        self.graph.node_count() - self.deleted.len()
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Round-trip the graph via bincode so we can embed it in HnswIndexData
        // without requiring HnswGraph: Clone.
        let graph_bytes = bincode::serialize(&self.graph)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        let graph_copy: HnswGraph = bincode::deserialize(&graph_bytes)
            .map_err(|e| VedaError::Index(format!("bincode deserialize failed: {e}")))?;

        let data = HnswIndexData {
            graph: graph_copy,
            vectors: self.vectors.clone(),
            metric: self.metric,
            dimension: self.dimension,
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            ml: self.ml,
            deleted: self.deleted.iter().copied().collect(),
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let bytes = fs::read(path)?;
        let data: HnswIndexData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("bincode deserialize failed: {e}")))?;

        if data.metric != metric {
            return Err(VedaError::Index(format!(
                "Metric mismatch: file has {:?}, expected {:?}",
                data.metric, metric
            )));
        }
        if data.dimension != dimension {
            return Err(VedaError::DimensionMismatch {
                expected: dimension,
                actual: data.dimension,
            });
        }

        let normalized = data.metric == DistanceMetric::Cosine;
        // Re-normalize on load for Cosine so the internal `1 - dot` fast
        // path is correct even for indexes persisted before normalized
        // storage existed. Normalizing an already-unit vector is a no-op
        // within fp tolerance, so this is safe + idempotent for new files.
        let vectors = if normalized {
            data.vectors
                .iter()
                .map(|v| crate::hnsw::distance_provider::l2_normalize(v))
                .collect()
        } else {
            data.vectors
        };

        Ok(Self {
            graph: data.graph,
            vectors,
            metric: data.metric,
            dimension: data.dimension,
            normalized,
            m: data.m,
            m_max0: data.m_max0,
            ef_construction: data.ef_construction,
            ef_search: data.ef_search,
            ml: data.ml,
            deleted: data.deleted.into_iter().collect(),
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}
