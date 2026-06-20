pub mod builder;
pub mod graph;
pub mod searcher;

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

/// DiskANN (Vamana graph) index for approximate nearest neighbor search.
///
/// This is a batch-build index optimized for high recall on large datasets.
/// It does not support incremental inserts; use `build` to construct the index.
pub struct DiskAnnIndex {
    graph: graph::VamanaGraph,
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    r: usize,
    l_build: usize,
    alpha: f32,
    l_search: usize,
    deleted: HashSet<u32>,
}

/// Serializable form of `DiskAnnIndex` for save/load round-trips.
#[derive(Serialize, Deserialize)]
struct DiskAnnIndexData {
    graph: graph::VamanaGraph,
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    r: usize,
    l_build: usize,
    alpha: f32,
    l_search: usize,
    deleted: Vec<u32>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index with default parameters.
    ///
    /// Defaults: r=64, l_build=100, alpha=1.2, l_search=100.
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            graph: graph::VamanaGraph::new(0),
            vectors: Vec::new(),
            metric,
            dimension,
            r: 64,
            l_build: 100,
            alpha: 1.2,
            l_search: 100,
            deleted: HashSet::new(),
        }
    }

    /// Create a new DiskANN index with custom parameters.
    pub fn with_params(
        metric: DistanceMetric,
        dimension: u32,
        r: usize,
        l_build: usize,
        alpha: f32,
        l_search: usize,
    ) -> Self {
        Self {
            graph: graph::VamanaGraph::new(0),
            vectors: Vec::new(),
            metric,
            dimension,
            r,
            l_build,
            alpha,
            l_search,
            deleted: HashSet::new(),
        }
    }

    /// Set the L_search parameter at runtime to tune recall/speed trade-off.
    pub fn set_l_search(&mut self, l: usize) {
        self.l_search = l;
    }
}

impl VectorIndex for DiskAnnIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.vectors.clear();
        self.deleted.clear();

        if vectors.is_empty() {
            self.graph = graph::VamanaGraph::new(0);
            return Ok(());
        }

        for v in vectors {
            if v.len() as u32 != self.dimension {
                return Err(VedaError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len() as u32,
                });
            }
            self.vectors.push(v.to_vec());
        }

        if self.vectors.len() == 1 {
            let mut g = graph::VamanaGraph::new(1);
            g.set_medoid(0);
            self.graph = g;
            return Ok(());
        }

        self.graph =
            builder::build_vamana(&self.vectors, self.metric, self.r, self.l_build, self.alpha);

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

        let medoid = match self.graph.medoid() {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let l = self.l_search.max(k);
        let candidates =
            searcher::greedy_search(query, medoid, l, &self.graph, &self.vectors, self.metric);

        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|&(id, _)| !self.deleted.contains(&id))
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect();

        Ok(results)
    }

    /// Phase F: native filtered search. DiskANN's greedy graph
    /// search needs to traverse non-filtered nodes to navigate
    /// the graph, so we can't skip them in the walk itself — but
    /// we CAN oversample the L search list aggressively, then
    /// filter the candidates BEFORE truncating to k. This avoids
    /// the default's blind 10x oversample of the final result
    /// set in favour of growing the search frontier inside the
    /// graph walk where DiskANN's recall is best.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &roaring::RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }
        if k == 0 || self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        let medoid = match self.graph.medoid() {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };
        // Oversample inside the graph walk. 4× the default L
        // is a reasonable starting point; for very sparse filters
        // the caller can rerun if recall is too low (this is the
        // same tradeoff that ACORN-style filtered ANN papers
        // discuss).
        let l = self.l_search.max(k * 4);
        let candidates =
            searcher::greedy_search(query, medoid, l, &self.graph, &self.vectors, self.metric);
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|&(id, _)| !self.deleted.contains(&id) && filter.contains(id))
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect();
        Ok(results)
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        Err(VedaError::Index(
            "DiskANN does not support incremental inserts; use build() instead".to_string(),
        ))
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
        self.vectors.len() - self.deleted.len()
    }

    fn save(&self, path: &Path) -> Result<()> {
        let graph_bytes = bincode::serialize(&self.graph)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        let graph_copy: graph::VamanaGraph = bincode::deserialize(&graph_bytes)
            .map_err(|e| VedaError::Index(format!("bincode deserialize failed: {e}")))?;

        let data = DiskAnnIndexData {
            graph: graph_copy,
            vectors: self.vectors.clone(),
            metric: self.metric,
            dimension: self.dimension,
            r: self.r,
            l_build: self.l_build,
            alpha: self.alpha,
            l_search: self.l_search,
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
        let data: DiskAnnIndexData = bincode::deserialize(&bytes)
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

        Ok(Self {
            graph: data.graph,
            vectors: data.vectors,
            metric: data.metric,
            dimension: data.dimension,
            r: data.r,
            l_build: data.l_build,
            alpha: data.alpha,
            l_search: data.l_search,
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
