use std::collections::HashMap;
use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::Result;

// ── Serialization helper ────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct BitmapIndexData {
    /// Each entry is (value, serialized RoaringBitmap bytes).
    bitmaps: Vec<(String, Vec<u8>)>,
    total_docs: u32,
}

// ── BitmapIndex ─────────────────────────────────────────────────────────────

/// Bitmap index for low-cardinality fields (bool, enum-like strings).
///
/// Stores one roaring bitmap per distinct value. Efficient for fields with
/// few unique values, enabling equality, IN, and NOT queries.
pub struct BitmapIndex {
    bitmaps: HashMap<String, RoaringBitmap>,
    total_docs: u32,
}

impl BitmapIndex {
    /// Create an empty bitmap index.
    pub fn new() -> Self {
        Self {
            bitmaps: HashMap::new(),
            total_docs: 0,
        }
    }

    /// Associate `doc_id` with the given `value`. Tracks the highest doc_id
    /// seen to maintain a count of total documents.
    pub fn insert(&mut self, doc_id: u32, value: &str) {
        self.bitmaps
            .entry(value.to_owned())
            .or_default()
            .insert(doc_id);
        // Track total docs as max(doc_id) + 1 seen so far.
        if doc_id >= self.total_docs {
            self.total_docs = doc_id + 1;
        }
    }

    /// Return the bitmap for an exact match on `value`.
    pub fn search_eq(&self, value: &str) -> Option<&RoaringBitmap> {
        self.bitmaps.get(value)
    }

    /// Return the union of bitmaps for each value in `values`.
    pub fn search_in(&self, values: &[&str]) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for &v in values {
            if let Some(bitmap) = self.bitmaps.get(v) {
                result |= bitmap;
            }
        }
        result
    }

    /// Return all doc IDs that do NOT match `value`.
    ///
    /// Computed as the set of all known doc IDs minus those matching `value`.
    pub fn search_not(&self, value: &str) -> RoaringBitmap {
        let all = self.all_doc_ids();
        match self.bitmaps.get(value) {
            Some(matching) => {
                let mut result = all;
                result -= matching;
                result
            }
            None => all,
        }
    }

    /// Return all distinct values in the index.
    pub fn distinct_values(&self) -> Vec<&str> {
        let mut values: Vec<&str> = self.bitmaps.keys().map(|s| s.as_str()).collect();
        values.sort();
        values
    }

    /// The total number of unique document IDs across all values.
    pub fn doc_count(&self) -> u32 {
        self.all_doc_ids().len() as u32
    }

    // ── Persistence ─────────────────────────────────────────────────────

    /// Serialize the index to JSON and write it to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = BitmapIndexData {
            bitmaps: self
                .bitmaps
                .iter()
                .map(|(value, bitmap)| {
                    let mut bytes = Vec::new();
                    bitmap.serialize_into(&mut bytes).expect("bitmap serialize");
                    (value.clone(), bytes)
                })
                .collect(),
            total_docs: self.total_docs,
        };
        let json = serde_json::to_vec(&data)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved index from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: BitmapIndexData = serde_json::from_slice(&bytes)?;
        let mut bitmaps = HashMap::new();
        for (value, bitmap_bytes) in data.bitmaps {
            let bitmap = RoaringBitmap::deserialize_from(bitmap_bytes.as_slice())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            bitmaps.insert(value, bitmap);
        }
        Ok(Self {
            bitmaps,
            total_docs: data.total_docs,
        })
    }

    // ── private ─────────────────────────────────────────────────────────

    fn all_doc_ids(&self) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for bitmap in self.bitmaps.values() {
            result |= bitmap;
        }
        result
    }
}

impl Default for BitmapIndex {
    fn default() -> Self {
        Self::new()
    }
}
