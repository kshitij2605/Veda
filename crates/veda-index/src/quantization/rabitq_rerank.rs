//! RaBitQ + raw-vector rerank.
//!
//! Two-stage retrieval inspired by the classic IVF-PQ + rerank
//! pattern. Stage 1 retrieves `candidates` (default `k * 10`) results
//! from the 1-bit-per-dim quantized index — cheap distance op, can
//! scan a large pool. Stage 2 re-scores those candidates using the
//! original full-precision f32 vectors and returns the top `k`.
//!
//! The intuition: RaBitQ alone trades some recall for huge memory
//! savings; the rerank step restores precision where it matters most
//! (the top of the result set) by paying the f32-distance cost on
//! only `candidates` rows, not the full corpus.
//!
//! Memory profile: keep both the quantized codes (1 bit / dim) and
//! the raw vectors. That's more memory than pure RaBitQ but the codes
//! are still 32× smaller than the vectors, so the working set during
//! the coarse stage fits in cache. Above ~1 M rows this becomes a
//! meaningful win over pure HNSW; below that, pure HNSW still wins.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::quantization::rabitq::RaBitQuantizer;
use crate::traits::{SearchResult, VectorIndex};

/// Default candidate-pool multiplier for the rerank stage. `k=10`
/// pulls 100 candidates from the RaBitQ scan and reranks them by raw
/// distance. Tuned empirically on similar quantizer + rerank stacks
/// (FAISS, Milvus); we expose `set_rerank_multiplier` for workloads
/// that want to dial it.
const DEFAULT_RERANK_MULTIPLIER: usize = 10;

pub struct RaBitQRerankIndex {
    quantizer: RaBitQuantizer,
    /// Full-precision vectors kept alongside the quantized codes so
    /// stage 2 can re-score exactly. Indexed by doc_id.
    raw_vectors: Vec<Vec<f32>>,
    deleted: HashSet<u32>,
    metric: DistanceMetric,
    rerank_multiplier: usize,
}

impl RaBitQRerankIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            quantizer: RaBitQuantizer::new(metric),
            raw_vectors: Vec::new(),
            deleted: HashSet::new(),
            metric,
            rerank_multiplier: DEFAULT_RERANK_MULTIPLIER,
        }
    }

    pub fn set_rerank_multiplier(&mut self, m: usize) {
        self.rerank_multiplier = m.max(1);
    }
}

impl VectorIndex for RaBitQRerankIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.quantizer.train(vectors);
        self.raw_vectors = vectors.iter().map(|v| v.to_vec()).collect();
        self.deleted.clear();
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if k == 0 || self.raw_vectors.is_empty() {
            return Ok(Vec::new());
        }
        // Stage 1: coarse RaBitQ scan, oversampled to `k * multiplier`.
        let candidates = (k * self.rerank_multiplier).min(self.raw_vectors.len());
        let coarse = self.quantizer.search(query, candidates, &self.deleted);

        // Stage 2: re-score with exact f32 distance.
        let mut reranked: Vec<(u32, f32)> = coarse
            .into_iter()
            .map(|(id, _)| {
                let d = compute_distance(query, &self.raw_vectors[id as usize], self.metric);
                (id, d)
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        // Push the filter into stage 1 by widening `skip` so RaBitQ's
        // scan already excludes non-matching IDs, then rerank the
        // surviving candidates exactly.
        let mut skip = self.deleted.clone();
        for id in 0..self.raw_vectors.len() as u32 {
            if !filter.contains(id) {
                skip.insert(id);
            }
        }
        let candidates = (k * self.rerank_multiplier).min(self.raw_vectors.len());
        let coarse = self.quantizer.search(query, candidates, &skip);
        let mut reranked: Vec<(u32, f32)> = coarse
            .into_iter()
            .map(|(id, _)| {
                let d = compute_distance(query, &self.raw_vectors[id as usize], self.metric);
                (id, d)
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn insert(&mut self, id: u32, _vector: &[f32]) -> Result<()> {
        // Post-build insert is a no-op for now (see RaBitQIndex).
        self.deleted.remove(&id);
        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.raw_vectors.len().saturating_sub(self.deleted.len())
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Persist both the quantizer and the raw vectors side by side
        // under `path` (a directory). The composite layout keeps stage
        // 2 cheap on segment open.
        let dir: PathBuf = path.into();
        std::fs::create_dir_all(&dir)?;
        self.quantizer.save(&dir.join("rabitq.bin"))?;
        // raw_vectors → contiguous f32 bytes + a row_count header so
        // reload is a single read + reshape.
        let mut buf: Vec<u8> = Vec::with_capacity(8 + self.raw_vectors.len() * 4 * 32);
        buf.extend_from_slice(&(self.raw_vectors.len() as u64).to_le_bytes());
        let dim = self
            .raw_vectors
            .first()
            .map(|v| v.len() as u64)
            .unwrap_or(0);
        buf.extend_from_slice(&dim.to_le_bytes());
        for v in &self.raw_vectors {
            for f in v {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        std::fs::write(dir.join("raw.bin"), &buf)?;
        Ok(())
    }

    fn load(path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let quantizer = RaBitQuantizer::load(&path.join("rabitq.bin"))?;
        let raw_bytes = std::fs::read(path.join("raw.bin"))?;
        if raw_bytes.len() < 16 {
            return Err(VedaError::Storage(
                "rabitq_rerank: raw.bin truncated".into(),
            ));
        }
        let row_count = u64::from_le_bytes(raw_bytes[0..8].try_into().unwrap()) as usize;
        let dim = u64::from_le_bytes(raw_bytes[8..16].try_into().unwrap()) as usize;
        let mut raw_vectors = Vec::with_capacity(row_count);
        let stride = dim * 4;
        for i in 0..row_count {
            let start = 16 + i * stride;
            let end = start + stride;
            if end > raw_bytes.len() {
                return Err(VedaError::Storage(
                    "rabitq_rerank: raw.bin truncated row".into(),
                ));
            }
            let mut row = Vec::with_capacity(dim);
            for j in 0..dim {
                let s = start + j * 4;
                row.push(f32::from_le_bytes(raw_bytes[s..s + 4].try_into().unwrap()));
            }
            raw_vectors.push(row);
        }
        let metric = quantizer.metric();
        Ok(Self {
            quantizer,
            raw_vectors,
            deleted: HashSet::new(),
            metric,
            rerank_multiplier: DEFAULT_RERANK_MULTIPLIER,
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.quantizer.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn rerank_recovers_exact_match() {
        let mut idx = RaBitQRerankIndex::new(DistanceMetric::Cosine);
        let dim = 64;
        let vecs: Vec<Vec<f32>> = (0..200).map(|i| vec_of(i, dim)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        idx.build(&refs).unwrap();

        // Query is an exact corpus member — the rerank stage should
        // pick it as rank 0 even though the coarse RaBitQ stage might
        // shuffle near-equal candidates.
        let target: u32 = 173;
        let q = vecs[target as usize].clone();
        let hits = idx.search(&q, 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].id, target);
    }

    #[test]
    fn rerank_beats_raw_rabitq_on_exact_match_rate() {
        // Probabilistic test: over many queries, the rerank top-1
        // should match the exact answer at least as often as raw
        // RaBitQ would (which can shuffle near-ties due to 1-bit
        // quantization).
        use crate::quantization::rabitq_vector_index::RaBitQIndex;

        let dim = 32;
        let n = 100;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| vec_of(i, dim)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut bare = RaBitQIndex::new(DistanceMetric::Cosine);
        bare.build(&refs).unwrap();
        let mut rerank = RaBitQRerankIndex::new(DistanceMetric::Cosine);
        rerank.build(&refs).unwrap();

        let mut bare_hits = 0;
        let mut rerank_hits = 0;
        for i in 0..n {
            let q = vecs[i as usize].clone();
            if bare.search(&q, 1).unwrap().first().map(|h| h.id) == Some(i) {
                bare_hits += 1;
            }
            if rerank.search(&q, 1).unwrap().first().map(|h| h.id) == Some(i) {
                rerank_hits += 1;
            }
        }
        assert!(
            rerank_hits >= bare_hits,
            "rerank ({rerank_hits}) should ≥ bare ({bare_hits})"
        );
        // Sanity: rerank essentially aces the exact-match recall.
        let n_usize = n as usize;
        assert!(
            rerank_hits + 5 >= n_usize,
            "rerank exact-match too low: {rerank_hits} of {n_usize}"
        );
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("idx");
        let mut idx = RaBitQRerankIndex::new(DistanceMetric::Cosine);
        let dim = 32;
        let vecs: Vec<Vec<f32>> = (0..50).map(|i| vec_of(i, dim)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        idx.build(&refs).unwrap();
        idx.save(&dir).unwrap();

        let loaded = RaBitQRerankIndex::load(&dir, DistanceMetric::Cosine, dim as u32).unwrap();
        let q = vecs[7].clone();
        let h1 = idx.search(&q, 3).unwrap();
        let h2 = loaded.search(&q, 3).unwrap();
        assert_eq!(h1.len(), h2.len());
        assert_eq!(h1[0].id, h2[0].id);
    }
}
