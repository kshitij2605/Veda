//! Multi-vector storage format (ColBERT/ColPali style).
//!
//! Each record stores a `Vec<Vec<f32>>` — a variable number of dense vectors,
//! all the same dimension. Stored as a single bincode-serialised
//! `Vec<Vec<Vec<f32>>>` per field per segment.

use std::fs;
use std::path::Path;

use veda_core::error::{Result, VedaError};

pub struct MultiVectorStorageWriter {
    rows: Vec<Vec<Vec<f32>>>,
}

impl MultiVectorStorageWriter {
    pub fn create() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn append(&mut self, rows: Vec<Vec<f32>>) -> u32 {
        let id = self.rows.len() as u32;
        self.rows.push(rows);
        id
    }

    pub fn count(&self) -> u64 {
        self.rows.len() as u64
    }

    pub fn close(self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(&self.rows)
            .map_err(|e| VedaError::Storage(format!("mvec serialise: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }
}

pub struct MultiVectorStorageReader {
    rows: Vec<Vec<Vec<f32>>>,
}

impl MultiVectorStorageReader {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let rows: Vec<Vec<Vec<f32>>> = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Storage(format!("mvec deserialise: {e}")))?;
        Ok(Self { rows })
    }

    pub fn count(&self) -> u64 {
        self.rows.len() as u64
    }

    pub fn get(&self, id: u32) -> Result<&Vec<Vec<f32>>> {
        self.rows.get(id as usize).ok_or_else(|| {
            VedaError::Storage(format!(
                "multi-vector id {id} out of range (count={})",
                self.rows.len()
            ))
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &Vec<Vec<f32>>)> {
        self.rows.iter().enumerate().map(|(i, v)| (i as u32, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.mvec");

        let mut w = MultiVectorStorageWriter::create();
        let id = w.append(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
        assert_eq!(id, 0);
        assert_eq!(w.count(), 1);
        w.close(&path).unwrap();

        let r = MultiVectorStorageReader::open(&path).unwrap();
        assert_eq!(r.count(), 1);
        let rows = r.get(0).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![0.1, 0.2, 0.3]);
    }
}
