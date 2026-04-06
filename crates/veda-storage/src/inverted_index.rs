use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::Result;

// ── Serialization helper ────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct InvertedIndexData {
    /// Each entry is (term, serialized RoaringBitmap bytes).
    postings: Vec<(String, Vec<u8>)>,
}

// ── InvertedIndex ───────────────────────────────────────────────────────────

/// Inverted index for string/keyword fields.
///
/// Maps string values to roaring bitmaps of document IDs, enabling exact
/// match, prefix, and multi-value (IN) queries.
pub struct InvertedIndex {
    postings: BTreeMap<String, RoaringBitmap>,
}

impl InvertedIndex {
    /// Create an empty inverted index.
    pub fn new() -> Self {
        Self {
            postings: BTreeMap::new(),
        }
    }

    /// Associate `doc_id` with the given string `value`.
    pub fn insert(&mut self, doc_id: u32, value: &str) {
        self.postings
            .entry(value.to_owned())
            .or_default()
            .insert(doc_id);
    }

    /// Return the bitmap for an exact match on `value`, if any documents match.
    pub fn search_exact(&self, value: &str) -> Option<&RoaringBitmap> {
        self.postings.get(value)
    }

    /// Return the union of all bitmaps whose key starts with `prefix`.
    pub fn search_prefix(&self, prefix: &str) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        // BTreeMap range gives us all keys starting from `prefix`.
        for (key, bitmap) in self.postings.range(prefix.to_owned()..) {
            if key.starts_with(prefix) {
                result |= bitmap;
            } else {
                break;
            }
        }
        result
    }

    /// Return the union of exact matches for each value in `values`.
    pub fn search_in(&self, values: &[&str]) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for &v in values {
            if let Some(bitmap) = self.postings.get(v) {
                result |= bitmap;
            }
        }
        result
    }

    /// Return the union of all bitmaps whose key contains `substring`.
    pub fn search_contains(&self, substring: &str) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for (key, bitmap) in &self.postings {
            if key.contains(substring) {
                result |= bitmap;
            }
        }
        result
    }

    /// Check whether `doc_id` is associated with `value`.
    pub fn contains(&self, value: &str, doc_id: u32) -> bool {
        self.postings
            .get(value)
            .is_some_and(|bitmap| bitmap.contains(doc_id))
    }

    /// Return the union of all document IDs across every term.
    pub fn all_doc_ids(&self) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for bitmap in self.postings.values() {
            result |= bitmap;
        }
        result
    }

    /// The number of distinct terms (keys) in the index.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// The total number of unique document IDs across all terms.
    pub fn doc_count(&self) -> u32 {
        self.all_doc_ids().len() as u32
    }

    // ── Persistence ─────────────────────────────────────────────────────

    /// Serialize the index to JSON and write it to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = InvertedIndexData {
            postings: self
                .postings
                .iter()
                .map(|(term, bitmap)| {
                    let mut bytes = Vec::new();
                    bitmap.serialize_into(&mut bytes).expect("bitmap serialize");
                    (term.clone(), bytes)
                })
                .collect(),
        };
        let json = serde_json::to_vec(&data)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved index from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: InvertedIndexData = serde_json::from_slice(&bytes)?;
        let mut postings = BTreeMap::new();
        for (term, bitmap_bytes) in data.postings {
            let bitmap = RoaringBitmap::deserialize_from(bitmap_bytes.as_slice())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            postings.insert(term, bitmap);
        }
        Ok(Self { postings })
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}
