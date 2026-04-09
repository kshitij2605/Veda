use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;
use veda_core::embedding::{EmbeddingModelConfig, EmbeddingModelRegistry, Modality};
use veda_core::schema::CollectionSchema;
use veda_query::distributed::DistributedQueryExecutor;
use metrics_exporter_prometheus::PrometheusHandle;
use veda_storage::blob_store::{BlobStore, LocalBlobStore, ObjectBlobStore, ObjectStoreConfig};
use veda_storage::segment_manager::SegmentManager;

use crate::auth::Role;

/// Represents a user in the in-memory user store.
#[derive(Clone)]
pub struct UserRecord {
    pub password: String,
    pub role: Role,
}

/// Shared application state accessible from all handlers.
#[derive(Clone)]
pub struct AppState {
    pub(crate) inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    /// Collection schemas keyed by collection name.
    pub schemas: RwLock<HashMap<String, CollectionSchema>>,
    /// Segment managers per collection, keyed by (collection_name, shard_id).
    pub segment_managers: RwLock<HashMap<String, HashMap<u32, SegmentManager>>>,
    /// Query executors per collection.
    pub query_executors: RwLock<HashMap<String, DistributedQueryExecutor>>,
    /// Embedding model registry.
    pub embedding_registry: RwLock<EmbeddingModelRegistry>,
    /// Base data directory.
    pub data_dir: PathBuf,
    /// FlashParse OCR endpoint (None = document processing disabled).
    pub flashparse_endpoint: Option<String>,
    /// Whether JWT authentication is enabled.
    pub auth_enabled: bool,
    /// Secret used for JWT signing and verification.
    pub jwt_secret: String,
    /// In-memory user store: username -> UserRecord.
    pub users: HashMap<String, UserRecord>,
    /// Blob storage backend.
    pub blob_store: RwLock<Arc<dyn BlobStore>>,
    /// Prometheus metrics handle (None when recorder is not installed).
    pub prometheus_handle: Option<PrometheusHandle>,
}

impl AppState {
    /// Create a new AppState with auth **disabled** (default for tests).
    pub fn new(data_dir: &Path) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: false,
                jwt_secret: "default-test-secret".to_owned(),
                users: HashMap::new(),
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
            }),
        }
    }

    /// Create a new AppState with auth **enabled** and a default admin user.
    pub fn with_auth(data_dir: &Path, jwt_secret: &str) -> Self {
        let mut users = HashMap::new();
        users.insert(
            "admin".to_owned(),
            UserRecord {
                password: "admin".to_owned(),
                role: Role::Admin,
            },
        );
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
            }),
        }
    }

    /// Create an AppState with auth enabled and custom users.
    pub fn with_auth_and_users(
        data_dir: &Path,
        jwt_secret: &str,
        users: HashMap<String, UserRecord>,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
            }),
        }
    }

    /// Create a new AppState with a Prometheus handle.
    pub fn with_prometheus(data_dir: &Path, prometheus_handle: PrometheusHandle) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: false,
                jwt_secret: "default-test-secret".to_owned(),
                users: HashMap::new(),
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: Some(prometheus_handle),
            }),
        }
    }

    /// Get a reference to the embedding model registry for configuration.
    pub fn embedding_registry(
        &self,
    ) -> &RwLock<EmbeddingModelRegistry> {
        &self.inner.embedding_registry
    }

    /// Load embedding model configuration from environment variables and register it.
    ///
    /// Registers two models when configured:
    /// 1. Text model from `EMBEDDING_*` env vars (registered first so `find_by_modality(Text)` returns it first)
    /// 2. Multimodal model from `MULTIMODAL_EMBEDDING_*` env vars
    pub async fn load_embedding_from_env(&self) {
        // --- Text embedding model (EMBEDDING_*) ---
        let text_config = match (
            std::env::var("EMBEDDING_ENDPOINT").ok().filter(|s| !s.is_empty()),
            std::env::var("EMBEDDING_MODEL").ok().filter(|s| !s.is_empty()),
        ) {
            (Some(endpoint), Some(model)) => {
                let api_key = std::env::var("EMBEDDING_API_KEY").unwrap_or_default();
                let dim = std::env::var("EMBEDDING_DIMENSION")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1024);
                Some((endpoint, model, api_key, dim))
            }
            _ => None,
        };

        // --- Multimodal embedding model (MULTIMODAL_EMBEDDING_*) ---
        let mm_config = match (
            std::env::var("MULTIMODAL_EMBEDDING_ENDPOINT").ok().filter(|s| !s.is_empty()),
            std::env::var("MULTIMODAL_EMBEDDING_MODEL").ok().filter(|s| !s.is_empty()),
        ) {
            (Some(endpoint), Some(model)) => {
                let api_key = std::env::var("MULTIMODAL_EMBEDDING_API_KEY").unwrap_or_default();
                let dim = std::env::var("MULTIMODAL_EMBEDDING_DIMENSION")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2048);
                Some((endpoint, model, api_key, dim))
            }
            _ => None,
        };

        if text_config.is_none() && mm_config.is_none() {
            return;
        }

        let mut registry = self.inner.embedding_registry.write().await;

        if let Some((endpoint, model, api_key, dim)) = text_config {
            registry.register(EmbeddingModelConfig {
                name: model.clone(),
                endpoint: endpoint.clone(),
                api_key,
                supported_modalities: vec![Modality::Text],
                output_dimension: dim,
                max_input_size_bytes: 32768,
                timeout_seconds: 30,
                batch_size: 32,
            });
            tracing::info!(model = %model, endpoint = %endpoint, dimension = dim, "registered text embedding model from environment");
        }

        if let Some((endpoint, model, api_key, dim)) = mm_config {
            registry.register(EmbeddingModelConfig {
                name: model.clone(),
                endpoint: endpoint.clone(),
                api_key,
                supported_modalities: vec![Modality::Text, Modality::Image, Modality::Video],
                output_dimension: dim,
                max_input_size_bytes: 20 * 1024 * 1024,
                timeout_seconds: 60,
                batch_size: 8,
            });
            tracing::info!(model = %model, endpoint = %endpoint, dimension = dim, "registered multimodal embedding model from environment");
        }
    }

    /// Load blob store configuration from environment variables.
    ///
    /// If `VEDA_BLOB_BACKEND` is set, creates an `ObjectBlobStore` and swaps it
    /// into the RwLock (replacing the default `LocalBlobStore`).
    pub async fn load_blob_store_from_env(&self) {
        let backend = match std::env::var("VEDA_BLOB_BACKEND").ok().filter(|s| !s.is_empty()) {
            Some(b) => b,
            None => {
                tracing::info!("VEDA_BLOB_BACKEND not set, keeping local blob store");
                return;
            }
        };

        let config = ObjectStoreConfig {
            backend,
            endpoint: std::env::var("VEDA_BLOB_ENDPOINT")
                .unwrap_or_else(|_| "http://minio:9000".to_owned()),
            bucket: std::env::var("VEDA_BLOB_BUCKET")
                .unwrap_or_else(|_| "veda-blobs".to_owned()),
            access_key: std::env::var("VEDA_BLOB_ACCESS_KEY").unwrap_or_default(),
            secret_key: std::env::var("VEDA_BLOB_SECRET_KEY").unwrap_or_default(),
            region: std::env::var("VEDA_BLOB_REGION")
                .unwrap_or_else(|_| "us-east-1".to_owned()),
        };

        match ObjectBlobStore::new(&config) {
            Ok(store) => {
                let mut guard = self.inner.blob_store.write().await;
                *guard = Arc::new(store);
                tracing::info!(
                    backend = %config.backend,
                    endpoint = %config.endpoint,
                    bucket = %config.bucket,
                    "switched to object blob store"
                );
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to create object blob store, keeping local");
            }
        }
    }
}
