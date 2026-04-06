pub mod builder;
pub mod graph;
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

use crate::distance::compute_distance;
use crate::traits::{SearchResult, VectorIndex};

/// HNSW (Hierarchical Navigable Small World) index implementing VectorIndex.
///
/// This provides approximate nearest neighbor search with logarithmic search
/// complexity while achieving high recall rates (typically > 95% with default
/// parameters).
pub struct HnswIndex {
    graph: HnswGraph,
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
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
    /// Defaults: m=16, m_max0=32, ef_construction=200, ef_search=200.
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        let m = 16;
        Self {
            graph: HnswGraph::new(),
            vectors: Vec::new(),
            metric,
            dimension,
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ef_search: 200,
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
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            deleted: HashSet::new(),
        }
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

        // Validate dimensions and store vectors.
        for v in vectors {
            if v.len() as u32 != self.dimension {
                return Err(VedaError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len() as u32,
                });
            }
            self.vectors.push(v.to_vec());
        }

        // Insert each vector into the HNSW graph.
        for id in 0..self.vectors.len() {
            HnswBuilder::insert(
                id as u32,
                &self.vectors[id],
                &mut self.graph,
                &self.vectors,
                self.metric,
                self.m,
                self.m_max0,
                self.ef_construction,
                self.ml,
            );
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }

        if k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let ep = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let mut ep_dist = compute_distance(query, &self.vectors[ep_id as usize], self.metric);

        // Phase 1: Greedily descend from the top level down to layer 1.
        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) = HnswSearcher::greedy_search(
                    query,
                    ep_id,
                    layer,
                    &self.graph,
                    &self.vectors,
                    self.metric,
                );
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        // Phase 2: Search at layer 0 with ef_search.
        let entry_points = vec![(ep_id, ep_dist)];
        let results = HnswSearcher::search_layer(
            query,
            &entry_points,
            self.ef_search.max(k), // ef must be >= k
            0,
            &self.graph,
            &self.vectors,
            self.metric,
            &self.deleted,
        );

        // Return top-k results.
        let results: Vec<SearchResult> = results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect();

        Ok(results)
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

        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let mut ep_dist = compute_distance(query, &self.vectors[ep_id as usize], self.metric);

        // Phase 1: Greedily descend from the top level down to layer 1.
        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) = HnswSearcher::greedy_search(
                    query,
                    ep_id,
                    layer,
                    &self.graph,
                    &self.vectors,
                    self.metric,
                );
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        // Phase 2: Filtered search at layer 0.
        let entry_points = vec![(ep_id, ep_dist)];
        let results = HnswSearcher::search_layer_filtered(
            &self.graph,
            &self.vectors,
            query,
            &entry_points,
            self.ef_search.max(k),
            0,
            self.metric,
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
        self.vectors[id as usize] = vector.to_vec();
        self.deleted.remove(&id);

        // Insert into the HNSW graph.
        HnswBuilder::insert(
            id,
            vector,
            &mut self.graph,
            &self.vectors,
            self.metric,
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
        // Serialize the graph via a round-trip so we can embed it in HnswIndexData
        // without requiring HnswGraph: Clone.
        let graph_json = serde_json::to_string(&self.graph)?;
        let graph_copy: HnswGraph = serde_json::from_str(&graph_json)?;

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
        let json = serde_json::to_string(&data)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let json = fs::read_to_string(path)?;
        let data: HnswIndexData = serde_json::from_str(&json)?;

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

        Ok(Self {
            graph: data.graph,
            vectors: data.vectors,
            metric: data.metric,
            dimension: data.dimension,
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
