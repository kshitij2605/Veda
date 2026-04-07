//! Scalar quantization (SQ8): quantize each dimension to uint8.

use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;

/// Scalar quantizer: maps each float dimension to a uint8 value in [0, 255].
pub struct ScalarQuantizer {
    mins: Vec<f32>,
    maxs: Vec<f32>,
    codes: Vec<Vec<u8>>,
    dimension: u32,
}

impl ScalarQuantizer {
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            codes: Vec::new(),
            dimension: 0,
        }
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

    /// Encode a single vector to uint8 per dimension.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let range = self.maxs[i] - self.mins[i];
                if range < 1e-10 {
                    128u8
                } else {
                    let normalized = (val - self.mins[i]) / range;
                    (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
                }
            })
            .collect()
    }

    /// Decode (approximate reconstruction) from uint8 back to float.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        code.iter()
            .enumerate()
            .map(|(i, &val)| {
                let range = self.maxs[i] - self.mins[i];
                self.mins[i] + (val as f32 / 255.0) * range
            })
            .collect()
    }

    /// Estimate distance by decoding the stored code and computing exact distance on the reconstruction.
    pub fn estimate_distance(
        &self,
        query: &[f32],
        code_idx: usize,
        metric: DistanceMetric,
    ) -> f32 {
        let reconstructed = self.decode(&self.codes[code_idx]);
        compute_distance(query, &reconstructed, metric)
    }
}

impl Default for ScalarQuantizer {
    fn default() -> Self {
        Self::new()
    }
}
