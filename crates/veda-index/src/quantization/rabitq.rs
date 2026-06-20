//! RaBitQ: Random Bit Quantization for fast approximate distance computation.
//!
//! Quantizes D-dimensional float vectors into compact binary codes (1 bit per dimension).
//! Uses asymmetric distance estimation: the query stays in full precision while database
//! vectors are represented by their binary codes and stored norms.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

/// RaBitQ quantizer: binary codes with norm-based correction for distance estimation.
pub struct RaBitQuantizer {
    dimension: u32,
    norms: Vec<f32>,
    binary_codes: Vec<Vec<u64>>,
    centroid: Vec<f32>,
    /// Average absolute value of normalized centered vector components.
    /// Used to scale the asymmetric dot product estimate.
    mean_abs_component: f32,
    metric: DistanceMetric,
}

#[derive(Serialize, Deserialize)]
struct RaBitQuantizerData {
    dimension: u32,
    norms: Vec<f32>,
    binary_codes: Vec<Vec<u64>>,
    centroid: Vec<f32>,
    mean_abs_component: f32,
    metric: DistanceMetric,
}

/// Number of u64 words needed to pack `dim` bits.
fn words_for_dim(dim: u32) -> usize {
    (dim as usize).div_ceil(64)
}

/// Compute the asymmetric inner product between a full-precision vector and a binary code.
///
/// The binary code encodes signs: bit=1 means +1, bit=0 means -1.
/// Returns sum_i(query_i * sign_i).
fn asymmetric_dot(query: &[f32], code: &[u64]) -> f32 {
    let sum_all: f32 = query.iter().sum();
    let mut sum_pos: f32 = 0.0;

    for (word_idx, &word) in code.iter().enumerate() {
        let base = word_idx * 64;
        let mut bits = word;
        while bits != 0 {
            let bit_pos = bits.trailing_zeros() as usize;
            let idx = base + bit_pos;
            if idx < query.len() {
                sum_pos += query[idx];
            }
            bits &= bits - 1;
        }
    }

    2.0 * sum_pos - sum_all
}

impl RaBitQuantizer {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            dimension: 0,
            norms: Vec::new(),
            binary_codes: Vec::new(),
            centroid: Vec::new(),
            mean_abs_component: 0.0,
            metric,
        }
    }

    /// Train the quantizer on a set of vectors.
    pub fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].len();
        self.dimension = dim as u32;

        // Compute centroid
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

        // First pass: compute norms and mean absolute component of normalized vectors
        let mut total_abs = 0.0f64;
        let mut total_components = 0u64;

        self.norms.clear();
        self.binary_codes.clear();
        self.norms.reserve(vectors.len());
        self.binary_codes.reserve(vectors.len());

        for v in vectors {
            let centered: Vec<f32> = v.iter().zip(&self.centroid).map(|(a, b)| a - b).collect();
            let norm_sq: f32 = centered.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            self.norms.push(norm);

            if norm > 1e-10 {
                for &c in &centered {
                    total_abs += (c / norm).abs() as f64;
                }
                total_components += dim as u64;
            }

            self.binary_codes.push(binarize(self.dimension, &centered));
        }

        // mean_abs_component: average |v_hat_i| across all vectors and dimensions.
        // The sign approximation replaces each v_hat_i with sign(v_hat_i) * mean_abs,
        // so dot(q, v_hat) ≈ mean_abs * sum(q_i * sign_i) = mean_abs * raw_dot.
        self.mean_abs_component = if total_components > 0 {
            (total_abs / total_components as f64) as f32
        } else {
            1.0 / (dim as f32).sqrt()
        };
    }

    /// Encode a single vector into binary code.
    pub fn encode(&self, vector: &[f32]) -> Vec<u64> {
        let centered: Vec<f32> = vector
            .iter()
            .zip(&self.centroid)
            .map(|(a, b)| a - b)
            .collect();
        binarize(self.dimension, &centered)
    }

    /// Estimate distance between a full-precision query and an encoded vector.
    pub fn estimate_distance(&self, query: &[f32], code_idx: usize) -> f32 {
        let v_norm = self.norms[code_idx];

        // Center query
        let q_centered: Vec<f32> = query
            .iter()
            .zip(&self.centroid)
            .map(|(a, b)| a - b)
            .collect();
        let q_norm_sq: f32 = q_centered.iter().map(|x| x * x).sum();

        // Asymmetric dot: sum(q'_i * sign(v'_i))
        let raw_dot = asymmetric_dot(&q_centered, &self.binary_codes[code_idx]);

        // The normalized db vector v_hat has components approximated by sign(v_hat_i).
        // E[v_hat_i | sign(v_hat_i)] ≈ sign * mean_abs_component.
        // So dot(q', v_hat) ≈ mean_abs_component * raw_dot.
        // And dot(q', v') = ||v'|| * dot(q', v_hat) ≈ ||v'|| * mean_abs * raw_dot.
        let estimated_dot = v_norm * self.mean_abs_component * raw_dot;

        match self.metric {
            DistanceMetric::Cosine => {
                let q_norm = q_norm_sq.sqrt();
                let denom = q_norm * v_norm;
                if denom < 1e-10 {
                    return 1.0;
                }
                let cos_sim = (estimated_dot / denom).clamp(-1.0, 1.0);
                1.0 - cos_sim
            }
            DistanceMetric::Euclidean => {
                let l2_sq = q_norm_sq + v_norm * v_norm - 2.0 * estimated_dot;
                l2_sq.max(0.0).sqrt()
            }
            DistanceMetric::DotProduct => -estimated_dot,
            _ => -raw_dot,
        }
    }

    /// Search: estimate distances from query to all encoded vectors, return top-k.
    pub fn search(&self, query: &[f32], k: usize, skip: &HashSet<u32>) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = (0..self.binary_codes.len())
            .filter(|&i| !skip.contains(&(i as u32)))
            .map(|i| (i as u32, self.estimate_distance(query, i)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.binary_codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.binary_codes.is_empty()
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = RaBitQuantizerData {
            dimension: self.dimension,
            norms: self.norms.clone(),
            binary_codes: self.binary_codes.clone(),
            centroid: self.centroid.clone(),
            mean_abs_component: self.mean_abs_component,
            metric: self.metric,
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: RaBitQuantizerData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("Failed to deserialize RaBitQuantizer: {e}")))?;
        Ok(Self {
            dimension: data.dimension,
            norms: data.norms,
            binary_codes: data.binary_codes,
            centroid: data.centroid,
            mean_abs_component: data.mean_abs_component,
            metric: data.metric,
        })
    }
}

/// Binarize a vector: bit = 1 if component >= 0, 0 otherwise.
fn binarize(dimension: u32, values: &[f32]) -> Vec<u64> {
    let num_words = words_for_dim(dimension);
    let mut code = vec![0u64; num_words];
    for (i, &val) in values.iter().enumerate() {
        if val >= 0.0 {
            code[i / 64] |= 1u64 << (i % 64);
        }
    }
    code
}
