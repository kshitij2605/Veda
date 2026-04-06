use serde::{Deserialize, Serialize};

use crate::simd;

/// A dense floating-point vector used for similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseVector {
    data: Vec<f32>,
}

impl DenseVector {
    /// Create a new dense vector from the given data.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Return the number of dimensions.
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Return the underlying data as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Reinterpret the vector data as raw bytes (for storage).
    pub fn as_bytes(&self) -> &[u8] {
        let ptr = self.data.as_ptr() as *const u8;
        let len = self.data.len() * std::mem::size_of::<f32>();
        // SAFETY: f32 slice and u8 slice share the same memory layout;
        // the resulting slice lives as long as &self.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// L2 normalize the vector in place.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 0.0 {
            for v in &mut self.data {
                *v /= norm;
            }
        }
    }

    /// Return a new vector that is L2-normalized.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Check whether the vector is approximately L2-normalized (norm ~ 1.0).
    pub fn is_normalized(&self) -> bool {
        let norm = self.l2_norm();
        (norm - 1.0).abs() < 1e-5
    }

    /// Cosine distance: `1.0 - cosine_similarity`.
    /// Returns 0.0 for identical directions, 2.0 for opposite directions.
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);
        simd::cosine_distance(&self.data, &other.data)
    }

    /// Euclidean (L2) distance.
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);
        simd::euclidean_distance(&self.data, &other.data)
    }

    /// Inner (dot) product.
    pub fn dot_product(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);
        simd::dot_product(&self.data, &other.data)
    }

    /// Manhattan (L1) distance.
    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);
        simd::manhattan_distance(&self.data, &other.data)
    }

    // ── private helpers ──

    fn l2_norm(&self) -> f32 {
        simd::l2_norm_squared(&self.data).sqrt()
    }

    fn assert_same_dimension(&self, other: &Self) {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "dimension mismatch: {} vs {}",
            self.dimension(),
            other.dimension()
        );
    }
}
