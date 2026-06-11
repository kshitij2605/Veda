//! Scalar quantization: quantize each dimension to 4-bit or 8-bit.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;

/// Scalar quantizer: maps each float dimension to a quantized value.
/// Supports 8-bit (256 levels, 1 byte/dim) and 4-bit (16 levels, 2 dims/byte).
pub struct ScalarQuantizer {
    mins: Vec<f32>,
    maxs: Vec<f32>,
    codes: Vec<Vec<u8>>,
    dimension: u32,
    bits: u8,
    metric: DistanceMetric,
}

#[derive(Serialize, Deserialize)]
struct ScalarQuantizerData {
    dimension: u32,
    bits: u8,
    metric: DistanceMetric,
    mins: Vec<f32>,
    maxs: Vec<f32>,
    codes: Vec<Vec<u8>>,
}

impl ScalarQuantizer {
    /// Create a new ScalarQuantizer with the given metric and bit-width.
    /// `bits` must be 4 or 8.
    pub fn with_bits(metric: DistanceMetric, bits: u8) -> Self {
        assert!(bits == 4 || bits == 8, "bits must be 4 or 8, got {bits}");
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            codes: Vec::new(),
            dimension: 0,
            bits,
            metric,
        }
    }

    /// Create a default 8-bit quantizer with Cosine metric (backward compat).
    pub fn new() -> Self {
        Self::with_bits(DistanceMetric::Cosine, 8)
    }

    /// Train: compute per-dimension min/max, then encode all vectors.
    pub fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].len();
        self.dimension = dim as u32;

        // Per-dimension min and max
        self.mins = vec![f32::MAX; dim];
        self.maxs = vec![f32::MIN; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < self.mins[i] {
                    self.mins[i] = val;
                }
                if val > self.maxs[i] {
                    self.maxs[i] = val;
                }
            }
        }

        // Encode all vectors
        self.codes.clear();
        self.codes.reserve(vectors.len());
        for v in vectors {
            self.codes.push(self.encode(v));
        }
    }

    /// Encode a single vector.
    /// 8-bit: one byte per dimension (0..=255).
    /// 4-bit: two dimensions per byte; even dim in low nibble, odd dim in high nibble.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let levels = (1u32 << self.bits) - 1; // 255 for 8-bit, 15 for 4-bit

        match self.bits {
            8 => vector
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let range = self.maxs[i] - self.mins[i];
                    if range < 1e-10 {
                        (levels / 2) as u8
                    } else {
                        let normalized = (val - self.mins[i]) / range;
                        (normalized.clamp(0.0, 1.0) * levels as f32).round() as u8
                    }
                })
                .collect(),
            4 => {
                let dim = vector.len();
                let packed_len = dim.div_ceil(2);
                let mut out = vec![0u8; packed_len];
                for (d, &val) in vector.iter().enumerate() {
                    let range = self.maxs[d] - self.mins[d];
                    let nibble: u8 = if range < 1e-10 {
                        (levels / 2) as u8
                    } else {
                        let normalized = (val - self.mins[d]) / range;
                        (normalized.clamp(0.0, 1.0) * levels as f32).round() as u8
                    };
                    let byte_idx = d / 2;
                    if d % 2 == 0 {
                        out[byte_idx] |= nibble;       // low nibble
                    } else {
                        out[byte_idx] |= nibble << 4;  // high nibble
                    }
                }
                out
            }
            _ => unreachable!(),
        }
    }

    /// Decode a code back to float values (approximate reconstruction).
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let dim = self.dimension as usize;
        match self.bits {
            8 => code
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let range = self.maxs[i] - self.mins[i];
                    self.mins[i] + (val as f32 / 255.0) * range
                })
                .collect(),
            4 => {
                let mut out = Vec::with_capacity(dim);
                for d in 0..dim {
                    let byte_idx = d / 2;
                    let nibble = if d % 2 == 0 {
                        code[byte_idx] & 0x0F
                    } else {
                        (code[byte_idx] >> 4) & 0x0F
                    };
                    let range = self.maxs[d] - self.mins[d];
                    out.push(self.mins[d] + (nibble as f32 / 15.0) * range);
                }
                out
            }
            _ => unreachable!(),
        }
    }

    /// Estimate distance by decoding the stored code and computing distance on the reconstruction.
    /// The `metric` parameter is accepted for backward compatibility; self.metric is used.
    pub fn estimate_distance(
        &self,
        query: &[f32],
        code_idx: usize,
        metric: DistanceMetric,
    ) -> f32 {
        let reconstructed = self.decode(&self.codes[code_idx]);
        compute_distance(query, &reconstructed, metric)
    }

    /// Search: scan all encoded vectors, return top-k by ascending distance (skipping `skip`).
    pub fn search(&self, query: &[f32], k: usize, skip: &HashSet<u32>) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = (0..self.codes.len())
            .filter(|&i| !skip.contains(&(i as u32)))
            .map(|i| (i as u32, self.estimate_distance(query, i, self.metric)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = ScalarQuantizerData {
            dimension: self.dimension,
            bits: self.bits,
            metric: self.metric,
            mins: self.mins.clone(),
            maxs: self.maxs.clone(),
            codes: self.codes.clone(),
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: ScalarQuantizerData = bincode::deserialize(&bytes).map_err(|e| {
            VedaError::Index(format!("Failed to deserialize ScalarQuantizer: {e}"))
        })?;
        Ok(Self {
            dimension: data.dimension,
            bits: data.bits,
            metric: data.metric,
            mins: data.mins,
            maxs: data.maxs,
            codes: data.codes,
        })
    }
}

impl Default for ScalarQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_core::types::DistanceMetric;

    fn seeded(seed: u32, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..dim)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn roundtrip_8bit() {
        let v: Vec<Vec<f32>> = (0..50).map(|i| seeded(i, 17)).collect();
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 8);
        q.train(&refs);
        let dec = q.decode(&q.encode(&v[3]));
        for (a, b) in v[3].iter().zip(&dec) {
            assert!((a - b).abs() < 0.02, "8-bit recon {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_4bit_odd_dim() {
        let v: Vec<Vec<f32>> = (0..50).map(|i| seeded(i, 17)).collect(); // odd dim
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let code = q.encode(&v[3]);
        assert_eq!(code.len(), 17usize.div_ceil(2), "4-bit packs 2 dims/byte");
        let dec = q.decode(&code);
        for (a, b) in v[3].iter().zip(&dec) {
            assert!((a - b).abs() < 0.15, "4-bit recon {a} vs {b}");
        }
    }

    #[test]
    fn search_finds_exact() {
        let v: Vec<Vec<f32>> = (0..100).map(|i| seeded(i, 32)).collect();
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let hits = q.search(&v[42], 5, &std::collections::HashSet::new());
        assert!(hits.iter().any(|(id, _)| *id == 42));
    }

    #[test]
    fn save_load_roundtrip() {
        let v: Vec<Vec<f32>> = (0..40).map(|i| seeded(i, 24)).collect();
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let mut q = ScalarQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q.train(&refs);
        let before = q.search(&v[5], 5, &std::collections::HashSet::new());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sq.bin");
        q.save(&p).unwrap();
        let q2 = ScalarQuantizer::load(&p).unwrap();
        let after = q2.search(&v[5], 5, &std::collections::HashSet::new());
        assert_eq!(
            before.iter().map(|x| x.0).collect::<Vec<_>>(),
            after.iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }
}
