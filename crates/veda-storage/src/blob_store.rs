use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let resp = self
            .bucket
            .put_object(key, data)
            .map_err(|e| VedaError::Storage(format!("S3 put failed: {e}")))?;
        let code = resp.status_code();
        if code != 200 {
            return Err(VedaError::Storage(format!(
                "S3 put returned status {code}"
            )));
        }
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let resp = self
            .bucket
            .get_object(key)
            .map_err(|e| VedaError::Storage(format!("S3 get failed: {e}")))?;
        if resp.status_code() == 404 {
            return Err(VedaError::Storage(format!("Blob not found: {key}")));
        }
        Ok(resp.to_vec())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.bucket
            .delete_object(key)
            .map_err(|e| VedaError::Storage(format!("S3 delete failed: {e}")))?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        match self.bucket.head_object(key) {
            Ok((_head, code)) => Ok(code == 200),
            Err(_) => Ok(false),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let results = self
            .bucket
            .list(prefix.to_string(), None)
            .map_err(|e| VedaError::Storage(format!("S3 list failed: {e}")))?;
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
