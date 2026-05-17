//! Sparse vector type for high-dimensional, mostly-zero embeddings
//! (e.g. SPLADE, BM25 term-frequency vectors, BGE-M3 sparse output).
//!
//! Stored as parallel `indices` (sorted ascending, unique) and `values`
//! arrays. Distance functions are merge-style and run in O(nnz_a + nnz_b).

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VedaError};

/// A sparse f32 vector: index → value pairs for non-zero components.
///
/// Indices are sorted ascending and unique. `dimension` (the field's
/// declared upper bound) is not stored on the vector itself; it lives on
/// the schema and is validated at insert time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVec {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVec {
    /// Construct from parallel arrays. Validates length match and that
    /// indices are sorted ascending without duplicates.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(VedaError::InvalidSchema(format!(
                "sparse vector indices ({}) and values ({}) must have the same length",
                indices.len(),
                values.len()
            )));
        }
        for w in indices.windows(2) {
            if w[0] >= w[1] {
                return Err(VedaError::InvalidSchema(
                    "sparse vector indices must be sorted ascending without duplicates".into(),
                ));
            }
        }
        Ok(Self { indices, values })
    }

    /// Construct from an unordered iterator of (index, value) pairs.
    /// Sorts and validates uniqueness.
    pub fn from_pairs(mut pairs: Vec<(u32, f32)>) -> Result<Self> {
        pairs.sort_by_key(|p| p.0);
        for w in pairs.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(VedaError::InvalidSchema(format!(
                    "duplicate index {} in sparse vector",
                    w[0].0
                )));
            }
        }
        let (indices, values) = pairs.into_iter().unzip();
        Ok(Self { indices, values })
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of non-zero components.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Maximum index + 1 (or 0 for empty), useful for dimension checks.
    pub fn max_index(&self) -> u32 {
        self.indices.last().copied().map_or(0, |i| i + 1)
    }

    /// L2 norm.
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Sparse-sparse dot product. O(nnz_a + nnz_b) merge join on indices.
    pub fn dot(&self, other: &Self) -> f32 {
        let mut i = 0usize;
        let mut j = 0usize;
        let mut sum = 0.0f32;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    /// Cosine distance: `1.0 - dot / (||a|| * ||b||)`. Returns 1.0 if either
    /// vector has zero norm.
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let denom = self.l2_norm() * other.l2_norm();
        if denom < 1e-12 {
            1.0
        } else {
            1.0 - (dot / denom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_validates_length() {
        assert!(SparseVec::new(vec![1, 2], vec![0.1]).is_err());
    }

    #[test]
    fn test_new_rejects_unsorted() {
        assert!(SparseVec::new(vec![2, 1], vec![0.1, 0.2]).is_err());
    }

    #[test]
    fn test_new_rejects_duplicates() {
        assert!(SparseVec::new(vec![1, 1], vec![0.1, 0.2]).is_err());
    }

    #[test]
    fn test_dot_disjoint() {
        let a = SparseVec::new(vec![1, 3, 5], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVec::new(vec![2, 4, 6], vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_dot_overlap() {
        let a = SparseVec::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVec::new(vec![3, 5, 7], vec![4.0, 5.0, 6.0]).unwrap();
        // overlap at 3 (2*4=8) and 5 (3*5=15)
        assert_eq!(a.dot(&b), 23.0);
    }

    #[test]
    fn test_cosine_identical_is_zero() {
        let a = SparseVec::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
        let d = a.cosine_distance(&a);
        assert!(d.abs() < 1e-6, "expected ~0, got {d}");
    }

    #[test]
    fn test_cosine_orthogonal_is_one() {
        let a = SparseVec::new(vec![0, 1], vec![1.0, 0.0]).unwrap();
        let b = SparseVec::new(vec![0, 1], vec![0.0, 1.0]).unwrap();
        assert!((a.cosine_distance(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_pairs_sorts() {
        let v = SparseVec::from_pairs(vec![(5, 0.1), (1, 0.3), (3, 0.2)]).unwrap();
        assert_eq!(v.indices(), &[1, 3, 5]);
        assert_eq!(v.values(), &[0.3, 0.2, 0.1]);
    }
}
