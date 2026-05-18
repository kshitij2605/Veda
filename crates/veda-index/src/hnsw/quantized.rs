//! Deep-quantized HNSW: a graph trained on raw f32 vectors but
//! persisted and searched with 1-bit-per-dim RaBitQ codes.
//!
//! The trade vs the lighter alternatives:
//! - [`HnswIndex`](super::HnswIndex): raw f32 vectors stored alongside
//!   the graph. Best recall, biggest memory footprint.
//! - [`RaBitQIndex`](crate::quantization::rabitq_vector_index::RaBitQIndex):
//!   no graph, brute-force scan of quantized codes. Smallest memory,
//!   weakest recall, fine at huge N where graph pruning + raw f32
//!   memory becomes the bottleneck.
//! - [`RaBitQRerankIndex`](crate::quantization::rabitq_rerank::RaBitQRerankIndex):
//!   keep raw vectors *and* quantized codes; brute-force the codes
//!   then rerank with raw. Highest recall of any quantized variant,
//!   doesn't save memory.
//! - **This module**: HNSW graph for pruning + RaBitQ codes for
//!   distance. ~32× smaller than `HnswIndex`, much faster than
//!   `RaBitQIndex` at large N because the graph skips most
//!   comparisons.
//!
//! Build: we still need raw vectors to compute high-quality graph
//! edges during insert (the prune step in `HnswBuilder::insert`
//! looks up neighbours' raw values). Raw vectors are kept in memory
//! during `build` and dropped right after — only the graph and the
//! quantizer persist.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::hnsw::builder::HnswBuilder;
use crate::hnsw::distance_provider::{DistanceProvider, QuantizedDistanceProvider};
use crate::hnsw::graph::HnswGraph;
use crate::hnsw::searcher::HnswSearcher;
use crate::quantization::rabitq::RaBitQuantizer;
use crate::traits::{SearchResult, VectorIndex};

/// HNSW graph + RaBitQ codes (no raw f32 vectors). Search uses
/// asymmetric distance against the codes; the graph still gives
/// O(log N) pruning.
pub struct QuantizedHnswIndex {
    graph: HnswGraph,
    quantizer: Option<RaBitQuantizer>,
    metric: DistanceMetric,
    dimension: u32,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    deleted: HashSet<u32>,
}

#[derive(Serialize, Deserialize)]
struct QuantizedHnswIndexData {
    graph: HnswGraph,
    metric: DistanceMetric,
    dimension: u32,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    deleted: Vec<u32>,
}

impl QuantizedHnswIndex {
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        let m = 16;
        Self {
            graph: HnswGraph::new(),
            quantizer: None,
            metric,
            dimension,
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 200,
            ml: 1.0 / (m as f64).ln(),
            deleted: HashSet::new(),
        }
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }
}

impl VectorIndex for QuantizedHnswIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        // 1. Train the quantizer on the full corpus.
        let mut quantizer = RaBitQuantizer::new(self.metric);
        let owned: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
        let refs: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
        quantizer.train(&refs);

        // 2. Build the HNSW graph using raw vectors (better edges).
        //    The builder uses RawVectorsProvider internally so this is
        //    just the existing path — we ditch the raw vectors after.
        self.graph = HnswGraph::new();
        let max_level_cap = 16;
        for (id, v) in owned.iter().enumerate() {
            let _ = max_level_cap; // future hyperparameter
            HnswBuilder::insert(
                id as u32,
                v,
                &mut self.graph,
                &owned,
                self.metric,
                self.m,
                self.m_max0,
                self.ef_construction,
                self.ml,
            );
        }

        // 3. Stash the quantizer; raw vectors go out of scope.
        self.quantizer = Some(quantizer);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let Some(q) = self.quantizer.as_ref() else {
            return Ok(Vec::new());
        };
        if k == 0 || q.is_empty() {
            return Ok(Vec::new());
        }
        let ep = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };
        let dp = QuantizedDistanceProvider::new(q);
        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let mut ep_dist = dp.distance(query, ep_id);

        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(query, ep_id, layer, &self.graph, &dp);
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        let entry_points = vec![(ep_id, ep_dist)];
        let results = HnswSearcher::search_layer(
            query,
            &entry_points,
            self.ef_search.max(k),
            0,
            &self.graph,
            &dp,
            &self.deleted,
        );
        Ok(results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        let Some(q) = self.quantizer.as_ref() else {
            return Ok(Vec::new());
        };
        if k == 0 || q.is_empty() || filter.is_empty() {
            return Ok(Vec::new());
        }
        let ep = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };
        let dp = QuantizedDistanceProvider::new(q);
        let max_level = self.graph.max_level();
        let mut ep_id = ep;
        let mut ep_dist = dp.distance(query, ep_id);
        if max_level > 0 {
            for layer in (1..=max_level).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(query, ep_id, layer, &self.graph, &dp);
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }
        let entry_points = vec![(ep_id, ep_dist)];
        Ok(HnswSearcher::search_layer_filtered(
            &self.graph,
            &dp,
            query,
            &entry_points,
            self.ef_search.max(k),
            0,
            &self.deleted,
            filter,
            k,
        ))
    }

    fn insert(&mut self, id: u32, _vector: &[f32]) -> Result<()> {
        // Post-build inserts would require re-encoding through the
        // existing quantizer + rebuilding graph edges from quantized
        // distance. Defer until a user actually needs append. The
        // VectorIndex trait still requires the method, so make it a
        // soft no-op that just clears the tombstone flag.
        self.deleted.remove(&id);
        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.quantizer
            .as_ref()
            .map(|q| q.len())
            .unwrap_or(0)
            .saturating_sub(self.deleted.len())
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Persist as a directory: graph metadata + quantizer side by
        // side. Path is the directory root.
        fs::create_dir_all(path)?;
        let data = QuantizedHnswIndexData {
            graph: self.graph.clone(),
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
            .map_err(|e| VedaError::Index(format!("quantized_hnsw save: {e}")))?;
        fs::write(path.join("hnsw.bin"), bytes)?;
        if let Some(q) = self.quantizer.as_ref() {
            q.save(&path.join("rabitq.bin"))?;
        }
        Ok(())
    }

    fn load(path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let bytes = fs::read(path.join("hnsw.bin"))?;
        let data: QuantizedHnswIndexData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("quantized_hnsw load: {e}")))?;
        let quantizer = RaBitQuantizer::load(&path.join("rabitq.bin")).ok();
        Ok(Self {
            graph: data.graph,
            quantizer,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn build_then_search_returns_top_k() {
        let mut idx = QuantizedHnswIndex::new(DistanceMetric::Cosine, 32);
        let vecs: Vec<Vec<f32>> = (0..200).map(|i| vec_of(i, 32)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        idx.build(&refs).unwrap();

        let q = vecs[42].clone();
        let hits = idx.search(&q, 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(
            hits.iter().any(|h| h.id == 42),
            "exact match should be in top-5; got {hits:?}"
        );
    }

    #[test]
    fn delete_then_search_excludes_id() {
        let mut idx = QuantizedHnswIndex::new(DistanceMetric::Cosine, 16);
        let vecs: Vec<Vec<f32>> = (0..30).map(|i| vec_of(i, 16)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        idx.build(&refs).unwrap();
        idx.delete(7).unwrap();
        let hits = idx.search(&vecs[7].clone(), 10).unwrap();
        assert!(hits.iter().all(|h| h.id != 7));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("idx");
        let mut idx = QuantizedHnswIndex::new(DistanceMetric::Cosine, 16);
        let vecs: Vec<Vec<f32>> = (0..40).map(|i| vec_of(i, 16)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        idx.build(&refs).unwrap();
        idx.save(&dir).unwrap();

        let loaded = QuantizedHnswIndex::load(&dir, DistanceMetric::Cosine, 16).unwrap();
        let q = vecs[3].clone();
        let h1 = idx.search(&q, 3).unwrap();
        let h2 = loaded.search(&q, 3).unwrap();
        assert_eq!(h1.len(), h2.len());
        assert_eq!(h1[0].id, h2[0].id);
    }
}

// HnswGraph needs Clone for save's data struct. The graph already
// derives Serialize/Deserialize but not Clone — keep it implicit so
// we don't widen the API surface unless other callers need it.
// (Implemented in graph.rs.)
