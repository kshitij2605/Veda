use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;
use veda_core::embedding::{EmbeddingModelConfig, EmbeddingModelRegistry, Modality};
use veda_core::llm_client::LlmConfig;
use veda_core::splade_client::{SpladeClient, SpladeClientConfig};
use veda_core::schema::CollectionSchema;
use veda_query::distributed::DistributedQueryExecutor;
use metrics_exporter_prometheus::PrometheusHandle;
use veda_storage::blob_store::{BlobStore, LocalBlobStore, ObjectBlobStore, ObjectStoreConfig};
use veda_storage::segment_manager::SegmentManager;

use crate::auth::Role;
use crate::rest::saved_objects::SavedObjectStore;

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
    /// Optional SPLADE sparse-encoder client, configured from
    /// `VEDA_SPLADE_*` env vars at startup. None when no endpoint is
    /// set — sparse_vector fields with `auto_embed` then fall back to
    /// "leave empty" with a one-shot warn.
    pub splade_client: RwLock<Option<SpladeClient>>,
    /// Optional LLM endpoint config (OpenAI-compatible chat completions),
    /// from `VEDA_LLM_*` env vars at startup. When unset the
    /// `POST /v1/llm/complete` route returns 503 and LLM-backed UI
    /// affordances (cluster labels, NL→DSL, failure classifier) hide.
    pub llm_config: RwLock<Option<LlmConfig>>,
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
    /// SavedObject store, persisted to `{data_dir}/_saved_objects.json`.
    pub saved_objects: SavedObjectStore,
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
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: false,
                jwt_secret: "default-test-secret".to_owned(),
                users: HashMap::new(),
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
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
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
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
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
            }),
        }
    }

    /// Create a new AppState with a Prometheus handle (auth disabled, for backwards compat).
    pub fn with_prometheus(data_dir: &Path, prometheus_handle: PrometheusHandle) -> Self {
        let mut state = Self::new(data_dir);
        Arc::get_mut(&mut state.inner)
            .expect("no other references to AppStateInner")
            .prometheus_handle = Some(prometheus_handle);
        state
    }

    /// Create a new AppState with auth enabled and a Prometheus handle.
    pub fn with_auth_and_prometheus(
        data_dir: &Path,
        jwt_secret: &str,
        prometheus_handle: PrometheusHandle,
    ) -> Self {
        let mut state = Self::with_auth(data_dir, jwt_secret);
        Arc::get_mut(&mut state.inner)
            .expect("no other references to AppStateInner")
            .prometheus_handle = Some(prometheus_handle);
        state
    }

    /// Get a reference to the embedding model registry for configuration.
    pub fn embedding_registry(
        &self,
    ) -> &RwLock<EmbeddingModelRegistry> {
        &self.inner.embedding_registry
    }

    /// Load a SPLADE sparse-encoder client from `VEDA_SPLADE_*` env
    /// vars. No-op when the endpoint isn't configured; sparse_vector
    /// fields with `auto_embed.modality = text` will then leave their
    /// values empty at insert time.
    pub async fn load_splade_from_env(&self) {
        let Some(cfg) = SpladeClientConfig::from_env() else {
            return;
        };
        let endpoint = cfg.endpoint.clone();
        let model = cfg.model.clone();
        let mut guard = self.inner.splade_client.write().await;
        *guard = Some(SpladeClient::new(cfg));
        tracing::info!(model = %model, endpoint = %endpoint, "registered SPLADE encoder from environment");
    }

    /// Load an OpenAI-compatible LLM endpoint from `VEDA_LLM_*` env
    /// vars. No-op when unset — LLM-backed UI affordances then hide
    /// (cluster labels, failure classification, NL→DSL).
    pub async fn load_llm_from_env(&self) {
        let Some(cfg) = LlmConfig::from_env() else { return };
        let endpoint = cfg.endpoint.clone();
        let model = cfg.model.clone();
        let mut guard = self.inner.llm_config.write().await;
        *guard = Some(cfg);
        tracing::info!(model = %model, endpoint = %endpoint, "registered LLM endpoint from environment");
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

    /// Recover collections from `data_dir` on startup.
    ///
    /// For each subdirectory of `data_dir` that contains a `collection.json`,
    /// re-build the in-memory state (schema, segment managers, query
    /// executor) and load any sealed segments under each shard. This is
    /// what makes Veda survive restarts; without it the in-memory
    /// `schemas` map starts empty and any prior data is orphaned.
    ///
    /// Returns the number of collections recovered.
    pub async fn recover_collections_from_disk(&self) -> usize {
        use veda_storage::segment_manager::SegmentManager;
        use veda_query::distributed::DistributedQueryExecutor;
        use veda_query::local_shard::LocalShardSearcher;

        let data_dir = &self.inner.data_dir;
        let Ok(entries) = std::fs::read_dir(data_dir) else {
            return 0;
        };

        let mut recovered = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // Skip Veda-managed dirs that aren't collections.
            if let Some(name) = path.file_name().and_then(|s| s.to_str())
                && (name == "blobs" || name.starts_with('.'))
            {
                continue;
            }
            let marker = path.join("collection.json");
            if !marker.exists() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()).map(String::from) else {
                continue;
            };

            let schema_bytes = match std::fs::read(&marker) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(collection = %name, error = %e, "recover: read schema failed");
                    continue;
                }
            };
            let schema: CollectionSchema = match serde_json::from_slice(&schema_bytes) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(collection = %name, error = %e, "recover: parse schema failed");
                    continue;
                }
            };

            let mut shard_managers: HashMap<u32, SegmentManager> = HashMap::new();
            let mut executor = DistributedQueryExecutor::new();
            let mut total_sealed = 0usize;

            for shard_id in 0..schema.shard_count() {
                let mut sm = match SegmentManager::new(
                    schema.clone(),
                    shard_id,
                    &path,
                    512 * 1024 * 1024,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: build segment manager failed");
                        continue;
                    }
                };
                match sm.load_sealed_segments() {
                    Ok(n) => total_sealed += n,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: load_sealed_segments failed");
                    }
                }
                // Build a parallel manager for the searcher (same pattern as
                // create_collection — the executor wants its own owned
                // copy of the segments since it builds ANN indexes lazily).
                let mut searcher_sm = match SegmentManager::new(
                    schema.clone(),
                    shard_id,
                    &path,
                    512 * 1024 * 1024,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: build searcher manager failed");
                        continue;
                    }
                };
                let _ = searcher_sm.load_sealed_segments();
                let searcher = LocalShardSearcher::new(searcher_sm);
                executor.register_shard(shard_id, Box::new(searcher));
                shard_managers.insert(shard_id, sm);
            }

            self.inner.schemas.write().await.insert(name.clone(), schema);
            self.inner
                .segment_managers
                .write()
                .await
                .insert(name.clone(), shard_managers);
            self.inner
                .query_executors
                .write()
                .await
                .insert(name.clone(), executor);
            recovered += 1;
            tracing::info!(collection = %name, sealed_segments = total_sealed, "recovered collection from disk");
        }
        recovered
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
