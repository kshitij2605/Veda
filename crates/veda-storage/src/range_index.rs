use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::Result;

// ── Serialization helper ────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct RangeIndexData {
    entries: Vec<(f64, u32)>,
}

// ── RangeIndex ──────────────────────────────────────────────────────────────

/// Range index for numeric fields (Int32, Int64, Float32, Float64, Timestamp).
///
/// Stores (value, doc_id) pairs sorted by value and uses binary search for
/// efficient range queries. All numeric types are converted to `f64`.
pub struct RangeIndex {
    entries: Vec<(f64, u32)>,
    sorted: bool,
}

impl RangeIndex {
    /// Create an empty range index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            sorted: true,
        }
    }

    /// Insert a (value, doc_id) pair. Marks the index as unsorted.
    pub fn insert(&mut self, doc_id: u32, value: f64) {
        self.entries.push((value, doc_id));
        self.sorted = false;
    }

    /// Sort the entries by value. Must be called before any query methods.
    pub fn build(&mut self) {
        self.entries
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.sorted = true;
    }

    /// Return doc IDs whose value is exactly `value`.
    ///
    /// Because f64 comparisons can be tricky, this checks for bitwise
    /// equality via `to_bits`.
    pub fn search_eq(&self, value: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        // Find the start of the range where entries could equal `value`.
        let start = self.entries.partition_point(|e| e.0 < value);
        for entry in &self.entries[start..] {
            if entry.0.to_bits() == value.to_bits() {
                result.insert(entry.1);
            } else if entry.0 > value {
                break;
            }
        }
        result
    }

    /// Return doc IDs with value in `[min, max]` (inclusive on both ends).
    pub fn search_range(&self, min: f64, max: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        let start = self.entries.partition_point(|e| e.0 < min);
        for entry in &self.entries[start..] {
            if entry.0 > max {
                break;
            }
            result.insert(entry.1);
        }
        result
    }

    /// Return doc IDs with value strictly greater than `value`.
    pub fn search_gt(&self, value: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        let start = self.entries.partition_point(|e| e.0 <= value);
        for entry in &self.entries[start..] {
            result.insert(entry.1);
        }
        result
    }

    /// Return doc IDs with value greater than or equal to `value`.
    pub fn search_gte(&self, value: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        let start = self.entries.partition_point(|e| e.0 < value);
        for entry in &self.entries[start..] {
            result.insert(entry.1);
        }
        result
    }

    /// Return doc IDs with value strictly less than `value`.
    pub fn search_lt(&self, value: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        let end = self.entries.partition_point(|e| e.0 < value);
        for entry in &self.entries[..end] {
            result.insert(entry.1);
        }
        result
    }

    /// Return doc IDs with value less than or equal to `value`.
    pub fn search_lte(&self, value: f64) -> RoaringBitmap {
        assert!(self.sorted, "RangeIndex::build() must be called before queries");
        let mut result = RoaringBitmap::new();
        let end = self.entries.partition_point(|e| e.0 <= value);
        for entry in &self.entries[..end] {
            result.insert(entry.1);
        }
        result
    }

    /// The total number of unique document IDs in the index.
    pub fn doc_count(&self) -> u32 {
        let mut all = RoaringBitmap::new();
        for entry in &self.entries {
            all.insert(entry.1);
        }
        all.len() as u32
    }

    // ── Persistence ─────────────────────────────────────────────────────

    /// Serialize the index to JSON and write it to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = RangeIndexData {
            entries: self.entries.clone(),
        };
        let json = serde_json::to_vec(&data)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved index from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: RangeIndexData = serde_json::from_slice(&bytes)?;
        Ok(Self {
            entries: data.entries,
            sorted: true, // saved data is always sorted (save is called after build)
        })
    }
}

impl Default for RangeIndex {
    fn default() -> Self {
        Self::new()
    }
}
