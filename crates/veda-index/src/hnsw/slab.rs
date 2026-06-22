//! Contiguous vector slab for cache-friendly HNSW distance probes.
//!
//! The HNSW search hot path computes `distance(query, doc_id)` for many
//! candidate `doc_id`s in quick succession. Storing each row as its own
//! heap allocation (`Vec<Vec<f32>>`) means every probe dereferences a
//! pointer into a random heap location — a near-guaranteed cache miss
//! per candidate before the SIMD dot product even runs. At 30k×1024
//! that scatters ~120MB of rows across the heap.
//!
//! [`VectorSlab`] packs every row end-to-end into a single `Vec<f32>`,
//! so row `id` lives at `data[id*dim .. (id+1)*dim]` — a predictable,
//! contiguous location the prefetcher can follow. This keeps the
//! on-disk segment format and the graph adjacency untouched: the slab
//! is built in memory at index build/load from the existing per-vector
//! data, and serialized back out as the same row-major `Vec<Vec<f32>>`.
//!
//! For the Cosine metric the index stores L2-normalized rows in the
//! slab (matching the prior `Vec<Vec<f32>>` behaviour) so traversal
//! distance reduces to `1 - dot`.

/// A flat, row-major slab of fixed-dimension vectors.
///
/// Row `id` occupies `data[id*dim .. (id+1)*dim]`. The number of rows
/// is `data.len() / dim`. A `dim` of zero is permitted only for an
/// empty slab (no rows can be stored).
#[derive(Clone, Debug)]
pub struct VectorSlab {
    data: Vec<f32>,
    dim: usize,
}

impl VectorSlab {
    /// Create an empty slab for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
        }
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of rows currently stored.
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Whether the slab holds zero rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Drop all rows (keeps the configured dimension).
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Borrow row `id` as a contiguous slice. Panics if `id` is out of
    /// range — callers (the graph) only ask for ids they inserted.
    #[inline]
    pub fn row(&self, id: usize) -> &[f32] {
        let start = id * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Append a row to the end of the slab. `row.len()` must equal `dim`.
    pub fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dim, "row dimension mismatch");
        self.data.extend_from_slice(row);
    }

    /// Set row `id` to `row`, growing the slab with zero-filled rows as
    /// needed so that intermediate ids are valid (matching the prior
    /// `Vec<Vec<f32>>::resize` + index-assign insert semantics).
    /// `row.len()` must equal `dim`.
    pub fn set_row(&mut self, id: usize, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dim, "row dimension mismatch");
        let needed = (id + 1) * self.dim;
        if self.data.len() < needed {
            self.data.resize(needed, 0.0);
        }
        let start = id * self.dim;
        self.data[start..start + self.dim].copy_from_slice(row);
    }

    /// Reserve capacity for `rows` additional rows.
    pub fn reserve(&mut self, rows: usize) {
        self.data.reserve(rows * self.dim);
    }

    /// Materialize the slab back into the row-major `Vec<Vec<f32>>`
    /// on-disk representation (unchanged serialization format).
    pub fn to_rows(&self) -> Vec<Vec<f32>> {
        (0..self.len()).map(|i| self.row(i).to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_then_row_roundtrips() {
        let mut slab = VectorSlab::new(3);
        slab.push(&[1.0, 2.0, 3.0]);
        slab.push(&[4.0, 5.0, 6.0]);
        assert_eq!(slab.len(), 2);
        assert_eq!(slab.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(slab.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn set_row_grows_and_zero_fills_gaps() {
        let mut slab = VectorSlab::new(2);
        slab.set_row(2, &[9.0, 9.0]);
        assert_eq!(slab.len(), 3);
        assert_eq!(slab.row(0), &[0.0, 0.0]);
        assert_eq!(slab.row(1), &[0.0, 0.0]);
        assert_eq!(slab.row(2), &[9.0, 9.0]);
    }

    #[test]
    fn set_row_overwrites_existing() {
        let mut slab = VectorSlab::new(2);
        slab.push(&[1.0, 1.0]);
        slab.set_row(0, &[7.0, 8.0]);
        assert_eq!(slab.row(0), &[7.0, 8.0]);
        assert_eq!(slab.len(), 1);
    }

    #[test]
    fn to_rows_matches_pushed_data() {
        let mut slab = VectorSlab::new(2);
        slab.push(&[1.0, 2.0]);
        slab.push(&[3.0, 4.0]);
        assert_eq!(slab.to_rows(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn empty_slab_reports_empty() {
        let slab = VectorSlab::new(4);
        assert!(slab.is_empty());
        assert_eq!(slab.len(), 0);
    }
}
