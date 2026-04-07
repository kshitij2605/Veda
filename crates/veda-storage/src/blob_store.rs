use std::fs;
use std::path::{Path, PathBuf};

use veda_core::error::{Result, VedaError};

/// Trait for blob storage backends.
pub trait BlobStore: Send + Sync {
    /// Store a blob at the given key.
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Retrieve a blob by key.
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Delete a blob by key.
    fn delete(&self, key: &str) -> Result<()>;

    /// Check if a blob exists.
    fn exists(&self, key: &str) -> Result<bool>;

    /// List blobs with a given prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

/// Local filesystem blob store (for standalone/dev mode).
pub struct LocalBlobStore {
    base_dir: PathBuf,
}

impl LocalBlobStore {
    pub fn new(base_dir: &Path) -> Result<Self> {
        fs::create_dir_all(base_dir)?;
        Ok(Self {
            base_dir: base_dir.to_path_buf(),
        })
    }

    fn key_path(&self, key: &str) -> PathBuf {
        self.base_dir.join(key)
    }
}

impl BlobStore for LocalBlobStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.key_path(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, data)?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.key_path(key);
        if !path.exists() {
            return Err(VedaError::Storage(format!("Blob not found: {key}")));
        }
        Ok(fs::read(&path)?)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.key_path(key);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.key_path(key).exists())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let search_dir = self.key_path(prefix);
        // Walk up to find an existing ancestor to start listing from.
        let start_dir = if search_dir.is_dir() {
            search_dir.clone()
        } else if let Some(parent) = search_dir.parent() {
            if parent.is_dir() {
                parent.to_path_buf()
            } else {
                return Ok(Vec::new());
            }
        } else {
            return Ok(Vec::new());
        };

        let mut results = Vec::new();
        collect_files(&start_dir, &self.base_dir, prefix, &mut results)?;
        results.sort();
        Ok(results)
    }
}

/// Recursively collect files whose key starts with the given prefix.
fn collect_files(
    dir: &Path,
    base_dir: &Path,
    prefix: &str,
    results: &mut Vec<String>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, base_dir, prefix, results)?;
        } else if let Ok(rel) = path.strip_prefix(base_dir) {
            let key = rel.to_string_lossy().to_string();
            if key.starts_with(prefix) {
                results.push(key);
            }
        }
    }
    Ok(())
}

/// Placeholder for S3/MinIO blob store (requires async + real credentials).
pub struct S3BlobStore {
    _bucket: String,
}

impl S3BlobStore {
    #[allow(dead_code)]
    pub fn new(_bucket: &str, _endpoint: &str, _region: &str) -> Result<Self> {
        Err(VedaError::Storage(
            "S3BlobStore is not yet implemented".to_string(),
        ))
    }
}
