pub mod builder;
pub mod searcher;

use std::fs;
use std::path::Path;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use veda_core::error::Result;

/// A suffix array index over text content of a field.
#[derive(Serialize, Deserialize)]
pub struct SuffixArrayIndex {
    /// Concatenated text from all documents, separated by sentinel characters.
    text: Vec<u8>,
    /// The suffix array (indices into text, sorted by suffix lexicographic order).
    sa: Vec<u32>,
    /// Document boundaries: (start_offset, end_offset, doc_id) for each document.
    doc_boundaries: Vec<(usize, usize, u32)>,
    /// Total number of documents.
    doc_count: u32,
}

/// Result of a pattern search.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub doc_id: u32,
    pub offset: usize,
    pub length: usize,
}

/// Search configuration.
#[derive(Debug, Clone)]
pub struct PatternSearchConfig {
    pub fuzzy: bool,
    pub max_edits: usize,
    pub preserve_order: bool,
    pub max_results: usize,
}

impl Default for PatternSearchConfig {
    fn default() -> Self {
        Self {
            fuzzy: false,
            max_edits: 0,
            preserve_order: true,
            max_results: 100,
        }
    }
}

impl SuffixArrayIndex {
    /// Save the index to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load the index from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let index: SuffixArrayIndex = serde_json::from_str(&json)?;
        Ok(index)
    }

    /// Look up which document a text offset belongs to.
    fn doc_for_offset(&self, offset: usize) -> Option<(u32, usize)> {
        for &(start, end, doc_id) in &self.doc_boundaries {
            if offset >= start && offset < end {
                return Some((doc_id, start));
            }
        }
        None
    }

    /// Search and return unique matching document IDs as a RoaringBitmap.
    pub fn search_doc_ids(&self, pattern: &str, config: &PatternSearchConfig) -> RoaringBitmap {
        let matches = if config.fuzzy {
            self.search_fuzzy(pattern, config)
        } else {
            self.search_exact(pattern)
        };
        let mut bitmap = RoaringBitmap::new();
        for m in &matches {
            bitmap.insert(m.doc_id);
        }
        bitmap
    }
}
