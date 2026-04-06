use serde::{Deserialize, Serialize};

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

        let dot = self.dot_product_raw(other);
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        let denom = norm_a * norm_b;

        if denom == 0.0 {
            return 1.0;
        }

        1.0 - (dot / denom)
    }

    /// Euclidean (L2) distance.
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);

        let sum_sq: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        sum_sq.sqrt()
    }

    /// Inner (dot) product.
    pub fn dot_product(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);
        self.dot_product_raw(other)
    }

    /// Manhattan (L1) distance.
    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        self.assert_same_dimension(other);

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    // ── private helpers ──

    fn l2_norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn dot_product_raw(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
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
