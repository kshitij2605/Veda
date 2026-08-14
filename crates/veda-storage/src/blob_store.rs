use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tracing::instrument;
use veda_core::error::{Result, VedaError};

/// Cheap identity of a stored object, used to detect that a source changed
/// without transferring it.
///
/// `version` is opaque: only equality is meaningful. Local stores put a
/// modification time in it; object stores put an ETag, or Last-Modified
/// when the backend supplies no ETag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobStat {
    pub size: u64,
    pub version: String,
}

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

    /// Metadata for a key without transferring the object.
    fn stat(&self, key: &str) -> Result<BlobStat>;
}

/// Wrapper to use an `Arc<dyn BlobStore>` as a `Box<dyn BlobStore>`.
pub struct ArcBlobStore(pub Arc<dyn BlobStore>);

impl BlobStore for ArcBlobStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.0.put(key, data)
    }
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.0.get(key)
    }
    fn delete(&self, key: &str) -> Result<()> {
        self.0.delete(key)
    }
    fn exists(&self, key: &str) -> Result<bool> {
        self.0.exists(key)
    }
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.0.list(prefix)
    }
    fn stat(&self, key: &str) -> Result<BlobStat> {
        self.0.stat(key)
    }
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

    fn stat(&self, key: &str) -> Result<BlobStat> {
        let path = self.key_path(key);
        let metadata = fs::metadata(&path)
            .map_err(|_| VedaError::Storage(format!("Blob not found: {key}")))?;
        let modified_ms = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| VedaError::Storage(format!("invalid modified time: {e}")))?
            .as_millis();
        Ok(BlobStat {
            size: metadata.len(),
            version: modified_ms.to_string(),
        })
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

/// Configuration for S3-compatible object storage (MinIO or AWS S3).
#[derive(Debug, Clone)]
pub struct ObjectStoreConfig {
    /// Backend type: "minio" (default) or "s3".
    pub backend: String,
    /// Endpoint URL, e.g. "http://minio:9000".
    pub endpoint: String,
    /// Bucket name, e.g. "veda-blobs".
    pub bucket: String,
    /// AWS access key ID.
    pub access_key: String,
    /// AWS secret access key.
    pub secret_key: String,
    /// AWS region, default "us-east-1".
    pub region: String,
}

/// Object storage blob store backed by MinIO or AWS S3.
pub struct ObjectBlobStore {
    bucket: Box<s3::Bucket>,
}

impl ObjectBlobStore {
    pub fn new(config: &ObjectStoreConfig) -> Result<Self> {
        let credentials = s3::creds::Credentials {
            access_key: Some(config.access_key.clone()),
            secret_key: Some(config.secret_key.clone()),
            security_token: None,
            session_token: None,
            expiration: None,
        };

        let region = if config.backend == "minio" {
            s3::Region::Custom {
                region: config.region.clone(),
                endpoint: config.endpoint.clone(),
            }
        } else {
            config
                .region
                .parse()
                .map_err(|e| VedaError::Storage(format!("invalid S3 region: {e}")))?
        };

        let bucket = if config.backend == "minio" {
            // Try to create the bucket (ignore AlreadyExists / errors).
            let _ = s3::Bucket::create_with_path_style(
                &config.bucket,
                region.clone(),
                credentials.clone(),
                s3::BucketConfiguration::default(),
            );
            s3::Bucket::new(&config.bucket, region, credentials)
                .map_err(|e| VedaError::Storage(format!("failed to create bucket handle: {e}")))?
                .with_path_style()
        } else {
            let _ = s3::Bucket::create(
                &config.bucket,
                region.clone(),
                credentials.clone(),
                s3::BucketConfiguration::default(),
            );
            s3::Bucket::new(&config.bucket, region, credentials)
                .map_err(|e| VedaError::Storage(format!("failed to create bucket handle: {e}")))?
        };

        Ok(Self { bucket })
    }
}

impl BlobStore for ObjectBlobStore {
    #[instrument(name = "blob_put", skip(self, data), fields(key = %key, size = data.len()))]
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let start = Instant::now();
        let resp = self
            .bucket
            .put_object(key, data)
            .map_err(|e| VedaError::Storage(format!("object store put failed: {e}")))?;
        let code = resp.status_code();
        if code != 200 {
            return Err(VedaError::Storage(format!(
                "object store put returned status {code}"
            )));
        }
        metrics::histogram!(veda_core::metrics::BLOB_OP_DURATION, "operation" => "put")
            .record(start.elapsed().as_secs_f64());
        Ok(())
    }

    #[instrument(name = "blob_get", skip(self), fields(key = %key))]
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let start = Instant::now();
        let resp = self
            .bucket
            .get_object(key)
            .map_err(|e| VedaError::Storage(format!("object store get failed: {e}")))?;
        if resp.status_code() == 404 {
            return Err(VedaError::Storage(format!("Blob not found: {key}")));
        }
        let data = resp.to_vec();
        metrics::histogram!(veda_core::metrics::BLOB_OP_DURATION, "operation" => "get")
            .record(start.elapsed().as_secs_f64());
        Ok(data)
    }

    #[instrument(name = "blob_delete", skip(self), fields(key = %key))]
    fn delete(&self, key: &str) -> Result<()> {
        let start = Instant::now();
        self.bucket
            .delete_object(key)
            .map_err(|e| VedaError::Storage(format!("object store delete failed: {e}")))?;
        metrics::histogram!(veda_core::metrics::BLOB_OP_DURATION, "operation" => "delete")
            .record(start.elapsed().as_secs_f64());
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        match self.bucket.head_object(key) {
            Ok((_head, code)) => Ok(code == 200),
            Err(_) => Ok(false),
        }
    }

    fn stat(&self, key: &str) -> Result<BlobStat> {
        let (head, code) = self
            .bucket
            .head_object(key)
            .map_err(|e| VedaError::Storage(format!("object store head failed: {e}")))?;
        if code != 200 {
            return Err(VedaError::Storage(format!("Blob not found: {key}")));
        }
        let size = head.content_length.unwrap_or(0).max(0) as u64;
        let version = head.e_tag.or(head.last_modified).unwrap_or_default();
        Ok(BlobStat { size, version })
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let results = self
            .bucket
            .list(prefix.to_string(), None)
            .map_err(|e| VedaError::Storage(format!("object store list failed: {e}")))?;
        let mut keys = Vec::new();
        for result in &results {
            for obj in &result.contents {
                keys.push(obj.key.clone());
            }
        }
        keys.sort();
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_stat_reports_size_and_changes_with_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalBlobStore::new(dir.path()).unwrap();
        store.put("a.bin", b"hello").unwrap();
        let first = store.stat("a.bin").unwrap();
        assert_eq!(first.size, 5);

        // A different payload of a different length must produce a different
        // stat, or cache invalidation cannot work at all.
        store.put("a.bin", b"hello world").unwrap();
        let second = store.stat("a.bin").unwrap();
        assert_eq!(second.size, 11);
        assert_ne!(first, second);
    }

    #[test]
    fn local_stat_on_a_missing_key_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalBlobStore::new(dir.path()).unwrap();
        assert!(store.stat("nope.bin").is_err());
    }

    #[test]
    fn arc_blob_store_forwards_stat() {
        let dir = tempfile::tempdir().unwrap();
        let inner = std::sync::Arc::new(LocalBlobStore::new(dir.path()).unwrap());
        inner.put("a.bin", b"xyz").unwrap();
        let wrapped = ArcBlobStore(inner.clone());
        assert_eq!(wrapped.stat("a.bin").unwrap(), inner.stat("a.bin").unwrap());
    }
}
