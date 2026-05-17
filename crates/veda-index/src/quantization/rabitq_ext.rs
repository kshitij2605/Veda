//! Extended RaBitQ — multi-bit centered residual quantizer.
//!
//! Where the original [`crate::quantization::rabitq::RaBitQuantizer`] uses
//! 1 bit per dimension (sign only), this variant uses 4 bits per dimension
//! with a uniform grid over the centered residual range. Trade-off:
//!
//! - **4× the storage** of 1-bit RaBitQ
//! - **~8× smaller than full f32** vectors
//! - **Noticeably better recall** than 1-bit, especially at higher dims and
//!   for fields where the components are not near-Gaussian.
//!
//! Inspired by Extended RaBitQ (SIGMOD 2025) — this is a simplified uniform
//! grid version rather than the paper's optimised codebook. Use when you
//! need more accuracy than 1-bit RaBitQ but can't afford SQ8 / float
//! storage.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

/// 4-bit-per-component residual quantizer.
pub struct ExtRaBitQuantizer {
    dimension: u32,
    norms: Vec<f32>,
    /// Packed 4-bit codes: byte `i` holds `(low_nibble = component 2i, high_nibble = component 2i+1)`.
    codes: Vec<Vec<u8>>,
    centroid: Vec<f32>,
    /// Per-dimension scale: `code = ((c + max_abs) * (15 / (2*max_abs)))`
    /// → `c_decoded = code * (2*max_abs / 15) - max_abs`.
    max_abs: f32,
    metric: DistanceMetric,
}

#[derive(Serialize, Deserialize)]
struct ExtRaBitData {
    dimension: u32,
    norms: Vec<f32>,
    codes: Vec<Vec<u8>>,
    centroid: Vec<f32>,
    max_abs: f32,
    metric: DistanceMetric,
}

/// Number of bytes needed to pack `dim` 4-bit codes.
fn bytes_for_dim(dim: u32) -> usize {
    (dim as usize).div_ceil(2)
}

const LEVELS: u8 = 15; // 4-bit max
const STEPS: f32 = LEVELS as f32;

impl ExtRaBitQuantizer {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            dimension: 0,
            norms: Vec::new(),
            codes: Vec::new(),
            centroid: Vec::new(),
            max_abs: 1.0,
            metric,
        }
    }

    pub fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }
        let dim = vectors[0].len();
        self.dimension = dim as u32;

        // Centroid.
        let n = vectors.len() as f32;
        self.centroid = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                self.centroid[i] += val;
            }
        }
        for c in &mut self.centroid {
            *c /= n;
        }

        // Find per-component max abs in centered space to set the
        // quantization range. We use the 99th-percentile to clip outliers.
        let mut samples: Vec<f32> = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                samples.push((val - self.centroid[i]).abs());
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((samples.len() as f32 * 0.99) as usize).min(samples.len() - 1);
        self.max_abs = samples[p99_idx].max(1e-6);

        // Encode each vector.
        self.norms.clear();
        self.codes.clear();
        for v in vectors {
            let centered: Vec<f32> = v.iter().zip(&self.centroid).map(|(a, b)| a - b).collect();
            let norm = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.norms.push(norm);
            self.codes.push(self.encode(&centered));
        }
    }

    pub fn encode(&self, centered: &[f32]) -> Vec<u8> {
        let dim = self.dimension as usize;
        let mut out = vec![0u8; bytes_for_dim(self.dimension)];
        for i in 0..dim {
            let normed = ((centered[i] + self.max_abs) * (STEPS / (2.0 * self.max_abs)))
                .round()
                .clamp(0.0, STEPS) as u8;
            let byte_idx = i / 2;
            if i % 2 == 0 {
                out[byte_idx] = (out[byte_idx] & 0xF0) | (normed & 0x0F);
            } else {
                out[byte_idx] = (out[byte_idx] & 0x0F) | ((normed & 0x0F) << 4);
            }
        }
        out
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let dim = self.dimension as usize;
        let step = 2.0 * self.max_abs / STEPS;
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                codes[byte_idx] & 0x0F
            } else {
                (codes[byte_idx] >> 4) & 0x0F
            };
            out.push(nibble as f32 * step - self.max_abs);
        }
        out
    }

    pub fn estimate_distance(&self, query: &[f32], code_idx: usize) -> f32 {
        let centered_query: Vec<f32> = query
            .iter()
            .zip(&self.centroid)
            .map(|(a, b)| a - b)
            .collect();
        let q_norm_sq: f32 = centered_query.iter().map(|x| x * x).sum();

        let decoded = self.decode(&self.codes[code_idx]);
        // decoded is the centered residual estimate; the stored vector
        // norm is in `self.norms[code_idx]`. For cosine/L2/dot, compute on
        // the centered space then add the centroid offset only when needed.
        let dot: f32 = centered_query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let v_norm = self.norms[code_idx];
        let decoded_norm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

        match self.metric {
            DistanceMetric::Cosine => {
                let q_norm = q_norm_sq.sqrt();
                let denom = q_norm * v_norm.max(decoded_norm);
                if denom < 1e-10 {
                    1.0
                } else {
                    1.0 - (dot / denom)
                }
            }
            DistanceMetric::Euclidean => {
                let l2_sq = q_norm_sq + decoded.iter().map(|x| x * x).sum::<f32>() - 2.0 * dot;
                l2_sq.max(0.0).sqrt()
            }
            DistanceMetric::DotProduct => -dot,
            _ => 1.0 - dot,
        }
    }

    pub fn search(&self, query: &[f32], k: usize, skip: &HashSet<u32>) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = (0..self.codes.len())
            .filter(|&i| !skip.contains(&(i as u32)))
            .map(|i| (i as u32, self.estimate_distance(query, i)))
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

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = ExtRaBitData {
            dimension: self.dimension,
            norms: self.norms.clone(),
            codes: self.codes.clone(),
            centroid: self.centroid.clone(),
            max_abs: self.max_abs,
            metric: self.metric,
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("ext-rabitq serialise: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: ExtRaBitData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("ext-rabitq deserialise: {e}")))?;
        Ok(Self {
            dimension: data.dimension,
            norms: data.norms,
            codes: data.codes,
            centroid: data.centroid,
            max_abs: data.max_abs,
            metric: data.metric,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_search_recovers_self() {
        let v0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let v1 = vec![0.0f32, 1.0, 0.0, 0.0];
        let v2 = vec![0.0f32, 0.0, 1.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&v0, &v1, &v2];

        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Cosine);
        q.train(&vectors);

        let results = q.search(&v0, 1, &HashSet::new());
        assert_eq!(results.len(), 1);
        // The nearest match for v0 should be v0 itself (id=0).
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_4bit_codes_pack_correctly() {
        let v = vec![0.5f32, -0.3, 0.1, -0.4];
        let vectors: Vec<&[f32]> = vec![&v];
        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Cosine);
        q.train(&vectors);
        // For dim=4, we need 2 bytes (4-bit codes × 4 dims = 16 bits).
        assert_eq!(q.codes[0].len(), 2);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let v0 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let v1 = vec![0.5f32, 0.4, 0.3, 0.2, 0.1];
        let vectors: Vec<&[f32]> = vec![&v0, &v1];
        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Euclidean);
        q.train(&vectors);

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ext.q");
        q.save(&path).unwrap();
        let q2 = ExtRaBitQuantizer::load(&path).unwrap();
        assert_eq!(q2.len(), 2);
        assert_eq!(q2.dimension, 5);
    }
}
