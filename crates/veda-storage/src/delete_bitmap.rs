use std::fs;
use std::io::Write;
use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::Result;

/// Tracks soft-deleted rows within a segment using a roaring bitmap.
///
/// Each bit position corresponds to a doc_id. A set bit means that document
/// has been logically deleted and should be excluded from query results.
pub struct DeleteBitmap {
    bitmap: RoaringBitmap,
}

impl DeleteBitmap {
    /// Create a new, empty delete bitmap (no documents deleted).
    pub fn new() -> Self {
        Self {
            bitmap: RoaringBitmap::new(),
        }
    }

    /// Mark a document as deleted.
    pub fn mark_deleted(&mut self, doc_id: u32) {
        self.bitmap.insert(doc_id);
    }

    /// Check whether a document has been deleted.
    pub fn is_deleted(&self, doc_id: u32) -> bool {
        self.bitmap.contains(doc_id)
    }

    /// Return the number of deleted documents.
    pub fn deleted_count(&self) -> u32 {
        self.bitmap.len() as u32
    }

    /// Serialize the bitmap and write it to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::new();
        self.bitmap
            .serialize_into(&mut bytes)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut file = fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load a previously saved delete bitmap from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(Self { bitmap })
    }
}

impl Default for DeleteBitmap {
    fn default() -> Self {
        Self::new()
    }
}
