//! Sparse vector storage format.
//!
//! Stores all sparse vectors for a single field in one bincode-serialised
//! `Vec<SparseVec>`. Read on open into an in-memory Vec for O(1) doc_id lookup.
//!
//! This is intentionally simple — appropriate for collections where each
//! sealed segment fits comfortably in memory (typical: ~100K records × ~1KB
//! per sparse vector = ~100MB). A streaming append-only binary format with
//! per-record offsets would be a later optimisation.

use std::fs;
use std::path::Path;

use veda_core::error::{Result, VedaError};
use veda_core::sparse::SparseVec;

/// Writes sparse vectors to disk, one file per field per segment.
pub struct SparseVectorStorageWriter {
    vectors: Vec<SparseVec>,
}

impl SparseVectorStorageWriter {
    pub fn create() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    /// Append a sparse vector. Returns the 0-indexed doc id.
    pub fn append(&mut self, sv: SparseVec) -> u32 {
        let id = self.vectors.len() as u32;
        self.vectors.push(sv);
        id
    }

    pub fn count(&self) -> u64 {
        self.vectors.len() as u64
    }

    /// Serialize and write to `path`.
    pub fn close(self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(&self.vectors)
            .map_err(|e| VedaError::Storage(format!("sparse serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }
}

/// Reads sparse vectors from disk. Loads the entire file into memory on open.
pub struct SparseVectorStorageReader {
    vectors: Vec<SparseVec>,
}

impl SparseVectorStorageReader {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let vectors: Vec<SparseVec> = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Storage(format!("sparse deserialize failed: {e}")))?;
        Ok(Self { vectors })
    }

    pub fn count(&self) -> u64 {
        self.vectors.len() as u64
    }

    /// Get a sparse vector by its 0-indexed doc id.
    pub fn get(&self, id: u32) -> Result<&SparseVec> {
        self.vectors.get(id as usize).ok_or_else(|| {
            VedaError::Storage(format!(
                "sparse vector id {id} out of range (count={})",
                self.vectors.len()
            ))
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &SparseVec)> {
        self.vectors.iter().enumerate().map(|(i, v)| (i as u32, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.spr");

        let mut w = SparseVectorStorageWriter::create();
        let id0 = w.append(SparseVec::new(vec![1, 5, 9], vec![0.1, 0.2, 0.3]).unwrap());
        let id1 = w.append(SparseVec::new(vec![2, 6], vec![0.5, 0.6]).unwrap());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(w.count(), 2);
        w.close(&path).unwrap();

        let r = SparseVectorStorageReader::open(&path).unwrap();
        assert_eq!(r.count(), 2);
        let v0 = r.get(0).unwrap();
        assert_eq!(v0.indices(), &[1, 5, 9]);
        assert_eq!(v0.values(), &[0.1, 0.2, 0.3]);
        let v1 = r.get(1).unwrap();
        assert_eq!(v1.indices(), &[2, 6]);
        assert_eq!(v1.values(), &[0.5, 0.6]);
    }

    #[test]
    fn test_get_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.spr");
        let mut w = SparseVectorStorageWriter::create();
        w.append(SparseVec::new(vec![0], vec![1.0]).unwrap());
        w.close(&path).unwrap();
        let r = SparseVectorStorageReader::open(&path).unwrap();
        assert!(r.get(5).is_err());
    }
}
