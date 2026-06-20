//! RaBitQ (1-bit coarse) + Extended-RaBitQ (4-bit rerank).
//!
//! Two-stage retrieval that swaps the raw-f32 rerank stage of
//! [`crate::quantization::rabitq_rerank::RaBitQRerankIndex`] for a
//! 4-bit Extended-RaBitQ pass. Same recall envelope as the
//! raw-float rerank for typical embeddings (BGE-M3 / Qwen3-Embedding-8B
//! / Voyage) but with **~8× less memory** in the rerank stage.
//!
//! Per the 2026 SOTA audit
//! (`research/2026-05-19-sota-audit/02-quantization.md`) this is
//! Veda's recommended default rerank stage. Falls back to raw-f32
//! rerank semantics via the existing [`RaBitQRerankIndex`] when a
//! caller specifically wants the precise full-precision pass.
//!
//! ## Memory profile
//!
//! - **Stage-1 codes** (1-bit RaBitQ): N × ⌈dim/8⌉ bytes.
//! - **Stage-2 codes** (4-bit Extended-RaBitQ): N × ⌈dim/2⌉ bytes,
//!   plus the per-vector norm (4 bytes) and the centroid (dim × 4
//!   bytes, one-shot).
//!
//! For dim=1024:
//!   raw-float rerank → ~4 KB per row
//!   ext-rabitq rerank → ~512 B per row + 128 B centroid amortised
//!
//! ## What v0 ships vs the SIGMOD 2025 paper
//!
//! The current Extended-RaBitQ implementation
//! (`crates/veda-index/src/quantization/rabitq_ext.rs`) is a
//! simplified uniform 4-bit grid over the per-component centered
//! residual. The SIGMOD '25 paper additionally specifies:
//!   - 3/5/7/8/9-bit ladders (we ship the 4-bit rung only).
//!   - An optimised codebook from the centered-residual
//!     distribution (we use a uniform grid).
//!   - A random rotation before quantisation that flattens the
//!     per-component variance.
//!
//! Filling those in is the path to 4-bit recall matching raw-f32
//! at <½ the storage of SQ8. Today we capture the storage win;
//! the recall-parity tightening is a follow-up.

use std::collections::HashSet;
use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::quantization::rabitq::RaBitQuantizer;
use crate::quantization::rabitq_ext::ExtRaBitQuantizer;
use crate::traits::{SearchResult, VectorIndex};

const DEFAULT_RERANK_MULTIPLIER: usize = 10;
/// Fixed seed for the Hadamard rotation so the index reloads deterministically.
const ROTATION_SEED: u64 = 0x5EED_1B17_4A11_0CD5;

pub struct RaBitQExtRerankIndex {
    /// Stage-1 1-bit quantizer producing the candidate pool.
    coarse: RaBitQuantizer,
    /// Stage-2 4-bit quantizer producing the final scores.
    rerank: ExtRaBitQuantizer,
    deleted: HashSet<u32>,
    metric: DistanceMetric,
    dimension: u32,
    rerank_multiplier: usize,
}

impl RaBitQExtRerankIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            coarse: RaBitQuantizer::new(metric),
            rerank: ExtRaBitQuantizer::new(metric),
            deleted: HashSet::new(),
            metric,
            dimension: 0,
            rerank_multiplier: DEFAULT_RERANK_MULTIPLIER,
        }
    }

    pub fn set_rerank_multiplier(&mut self, m: usize) {
        self.rerank_multiplier = m.max(1);
    }

    /// Opt-in variant with Hadamard rotation + Lloyd-Max codebook on the
    /// rerank stage. Benchmarks (synthetic isotropic AND anisotropic) showed
    /// this REGRESSES recall vs the plain 4-bit default at equal memory, so it
    /// is NOT the default — exposed for experimentation / future real-embedding
    /// evaluation. See documentation/2026-06-10-rabitq-recall-design.md.
    #[doc(hidden)]
    pub fn new_with_rotation(metric: DistanceMetric) -> Self {
        Self {
            coarse: RaBitQuantizer::new(metric),
            rerank: ExtRaBitQuantizer::new(metric)
                .with_rotation(ROTATION_SEED)
                .with_optimized_codebook(),
            deleted: HashSet::new(),
            metric,
            dimension: 0,
            rerank_multiplier: DEFAULT_RERANK_MULTIPLIER,
        }
    }
}

impl VectorIndex for RaBitQExtRerankIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.coarse.train(vectors);
        self.rerank.train(vectors);
        self.deleted.clear();
        if let Some(first) = vectors.first() {
            self.dimension = first.len() as u32;
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if k == 0 || self.rerank.is_empty() {
            return Ok(Vec::new());
        }
        let candidates = (k * self.rerank_multiplier).min(self.rerank.len());
        let coarse = self.coarse.search(query, candidates, &self.deleted);

        let mut reranked: Vec<(u32, f32)> = coarse
            .into_iter()
            .map(|(id, _)| {
                let d = self.rerank.estimate_distance(query, id as usize);
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
        if k == 0 || self.rerank.is_empty() || filter.is_empty() {
            return Ok(Vec::new());
        }
        // Use the filter as an additional skip set against the
        // coarse stage so we don't waste candidates on filtered-out
        // rows. The rerank stage then only sees survivors.
        let candidates = (k * self.rerank_multiplier).min(self.rerank.len());
        let mut skip = self.deleted.clone();
        for id in 0..self.rerank.len() as u32 {
            if !filter.contains(id) {
                skip.insert(id);
            }
        }
        let coarse = self.coarse.search(query, candidates, &skip);

        let mut reranked: Vec<(u32, f32)> = coarse
            .into_iter()
            .map(|(id, _)| {
                let d = self.rerank.estimate_distance(query, id as usize);
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

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        // Like RaBitQRerankIndex, this is a batch-only index —
        // call `build()` with the full vector set. Incremental
        // insert would require re-encoding the affected codes and
        // potentially retraining the quantizer, which is enough
        // work to be its own follow-up.
        Err(VedaError::Index(
            "RaBitQExtRerank does not support incremental insert; rebuild instead".into(),
        ))
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.rerank.len().saturating_sub(self.deleted.len())
    }

    fn save(&self, path: &Path) -> Result<()> {
        // `path` is the directory the segment manager owns for
        // this index. Inside it we write coarse + rerank + a
        // small manifest carrying the metric, dimension, and
        // tombstones.
        std::fs::create_dir_all(path)?;
        self.coarse.save(&path.join("coarse.q"))?;
        self.rerank.save(&path.join("rerank.q"))?;
        let manifest = ExtRerankManifest {
            metric: self.metric,
            dimension: self.dimension,
            deleted: self.deleted.iter().copied().collect(),
            rerank_multiplier: self.rerank_multiplier,
        };
        let bytes = bincode::serialize(&manifest)
            .map_err(|e| VedaError::Index(format!("ext-rerank manifest serialise: {e}")))?;
        std::fs::write(path.join("manifest.bin"), bytes)?;
        Ok(())
    }

    fn load(path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self> {
        let coarse = RaBitQuantizer::load(&path.join("coarse.q"))?;
        let rerank = ExtRaBitQuantizer::load(&path.join("rerank.q"))?;
        // Manifest is the source of truth for metric/dim/deleted;
        // the trait's `_metric` / `_dimension` arguments are
        // ignored here. (Other quantizers don't persist them and
        // rely on the caller to pass them; we persist for
        // self-containment.)
        let manifest_bytes = std::fs::read(path.join("manifest.bin"))?;
        let manifest: ExtRerankManifest = bincode::deserialize(&manifest_bytes)
            .map_err(|e| VedaError::Index(format!("ext-rerank manifest deserialise: {e}")))?;
        Ok(Self {
            coarse,
            rerank,
            deleted: manifest.deleted.into_iter().collect(),
            metric: manifest.metric,
            dimension: manifest.dimension,
            rerank_multiplier: manifest.rerank_multiplier,
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExtRerankManifest {
    metric: DistanceMetric,
    dimension: u32,
    deleted: Vec<u32>,
    rerank_multiplier: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0_f32; dim];
                for (d, slot) in v.iter_mut().enumerate() {
                    *slot = ((i + d) as f32 * 0.07).sin();
                }
                v
            })
            .collect()
    }

    #[test]
    fn build_then_self_search_finds_self() {
        let vectors = synthetic_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut idx = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
        idx.build(&refs).unwrap();

        // The 1-bit coarse stage is approximate but the 4-bit
        // rerank should pull the true self-match into the top.
        let r = idx.search(&vectors[7], 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 7);
    }

    #[test]
    fn rerank_multiplier_controls_oversampling() {
        let vectors = synthetic_vectors(200, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut idx = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
        idx.build(&refs).unwrap();
        idx.set_rerank_multiplier(20);
        let r = idx.search(&vectors[0], 5).unwrap();
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn delete_excludes_from_results() {
        let vectors = synthetic_vectors(20, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut idx = RaBitQExtRerankIndex::new(DistanceMetric::Cosine);
        idx.build(&refs).unwrap();
        idx.delete(3).unwrap();
        let r = idx.search(&vectors[3], 5).unwrap();
        assert!(
            r.iter().all(|h| h.id != 3),
            "deleted row should not appear in results"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let vectors = synthetic_vectors(10, 4);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut idx = RaBitQExtRerankIndex::new(DistanceMetric::Euclidean);
        idx.build(&refs).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        idx.save(tmp.path()).unwrap();
        let loaded = RaBitQExtRerankIndex::load(tmp.path(), DistanceMetric::Euclidean, 4).unwrap();
        assert_eq!(loaded.len(), 10);
        assert_eq!(loaded.dimension(), 4);
        assert_eq!(loaded.metric(), DistanceMetric::Euclidean);
        // A self-query should still find the true match.
        let r = loaded.search(&vectors[5], 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 5);
    }

    /// Reload-safety: rotation + Lloyd-Max codebook survive a save/load cycle.
    ///
    /// Uses ≥256 vectors so the codebook trains on a non-trivial sample.
    /// Asserts that search results (ids + distances) are identical before
    /// and after the round-trip — proving the rotation and codebook are
    /// fully persisted and re-applied on reload.
    #[test]
    fn rotation_and_codebook_survive_reload() {
        let vectors = synthetic_vectors(300, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut idx = RaBitQExtRerankIndex::new_with_rotation(DistanceMetric::Cosine);
        idx.build(&refs).unwrap();

        // Verify the rerank stage has rotation and codebook active.
        assert!(
            idx.rerank.has_rotation(),
            "rotation must be active after build"
        );
        assert!(
            idx.rerank.has_optimized_codebook(),
            "Lloyd-Max codebook must be active after build"
        );

        // Collect search results for several probe vectors before saving.
        let probes: Vec<&[f32]> = vectors[0..5].iter().map(|v| v.as_slice()).collect();
        let before: Vec<Vec<SearchResult>> =
            probes.iter().map(|q| idx.search(q, 5).unwrap()).collect();

        // Save → reload via the VectorIndex::save / VectorIndex::load API
        // (directory-based: coarse.q + rerank.q + manifest.bin).
        let tmp = tempfile::tempdir().unwrap();
        idx.save(tmp.path()).unwrap();
        let loaded = RaBitQExtRerankIndex::load(tmp.path(), DistanceMetric::Cosine, 64).unwrap();

        // Reloaded index must still carry rotation + codebook.
        assert!(loaded.rerank.has_rotation(), "rotation must survive reload");
        assert!(
            loaded.rerank.has_optimized_codebook(),
            "Lloyd-Max codebook must survive reload"
        );

        // Search results must be bit-identical after reload.
        for (i, q) in probes.iter().enumerate() {
            let after = loaded.search(q, 5).unwrap();
            assert_eq!(
                before[i].len(),
                after.len(),
                "result count changed after reload for probe {i}"
            );
            for (b, a) in before[i].iter().zip(after.iter()) {
                assert_eq!(b.id, a.id, "result id mismatch after reload for probe {i}");
                assert!(
                    (b.distance - a.distance).abs() < 1e-6,
                    "distance changed after reload for probe {i}: before={} after={}",
                    b.distance,
                    a.distance
                );
            }
        }
    }
}
