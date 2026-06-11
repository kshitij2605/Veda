//! `VectorIndex` adapter over `ScalarQuantizer` — exposes 4-bit (`Sq4`)
//! and 8-bit (`Sq8`) per-dimension scalar quantization as standalone
//! brute-force indexes. ~D/2 bytes/vec (4-bit) or D bytes/vec (8-bit).

use std::collections::HashSet;
use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::Result;
use veda_core::types::DistanceMetric;

use crate::quantization::scalar::ScalarQuantizer;
use crate::traits::{SearchResult, VectorIndex};

/// `VectorIndex` impl over `ScalarQuantizer`. Holds the trained
/// quantizer plus a runtime soft-delete set (not persisted).
pub struct ScalarQuantIndex {
    quantizer: ScalarQuantizer,
    deleted: HashSet<u32>,
}

impl ScalarQuantIndex {
    pub fn with_bits(metric: DistanceMetric, bits: u8) -> Self {
        Self {
            quantizer: ScalarQuantizer::with_bits(metric, bits),
            deleted: HashSet::new(),
        }
    }

    pub fn sq4(metric: DistanceMetric) -> Self {
        Self::with_bits(metric, 4)
    }

    pub fn sq8(metric: DistanceMetric) -> Self {
        Self::with_bits(metric, 8)
    }
}

impl VectorIndex for ScalarQuantIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.quantizer.train(vectors);
        self.deleted.clear();
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        Ok(self
            .quantizer
            .search(query, k, &self.deleted)
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
        // SQ search is brute-force; merge deleted + complement-of-filter
        // into a single skip set rather than oversampling.
        let mut skip = self.deleted.clone();
        for id in 0..self.quantizer.len() as u32 {
            if !filter.contains(id) {
                skip.insert(id);
            }
        }
        Ok(self
            .quantizer
            .search(query, k, &skip)
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn insert(&mut self, id: u32, _vector: &[f32]) -> Result<()> {
        // ScalarQuantizer doesn't support post-train append.
        // Sealed segments build once; for those, this is a no-op.
        self.deleted.remove(&id);
        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.quantizer.len().saturating_sub(self.deleted.len())
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.quantizer.save(path)
    }

    fn load(path: &Path, _metric: DistanceMetric, _dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        // ScalarQuantizer::load reads metric + bits + dimension from the
        // persisted bytes; the trait's extra args are ignored.
        Ok(Self {
            quantizer: ScalarQuantizer::load(path)?,
            deleted: HashSet::new(),
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.quantizer.metric()
    }

    fn dimension(&self) -> u32 {
        self.quantizer.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761);
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn build_search_delete_save_load() {
        for bits in [4u8, 8] {
            let mut idx = ScalarQuantIndex::with_bits(DistanceMetric::Cosine, bits);
            let v: Vec<Vec<f32>> = (0..100).map(|i| seeded(i, 48)).collect();
            let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
            idx.build(&refs).unwrap();

            let hits = idx.search(&v[10], 5).unwrap();
            assert!(
                hits.iter().any(|h| h.id == 10),
                "bits={bits} exact match in top-5"
            );

            idx.delete(10).unwrap();
            assert!(
                idx.search(&v[10], 5).unwrap().iter().all(|h| h.id != 10),
                "bits={bits} deleted id absent from results"
            );

            let dir = tempfile::tempdir().unwrap();
            let p = dir.path().join("idx.bin");
            idx.save(&p).unwrap();
            let re = ScalarQuantIndex::load(&p, DistanceMetric::Cosine, 48).unwrap();
            assert_eq!(re.dimension(), 48, "bits={bits} dimension preserved after load");
        }
    }
}
