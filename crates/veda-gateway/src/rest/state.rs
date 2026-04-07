use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;
use veda_core::embedding::EmbeddingModelRegistry;
use veda_core::schema::CollectionSchema;
use veda_query::distributed::DistributedQueryExecutor;
use veda_storage::segment_manager::SegmentManager;

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
}

impl AppState {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                data_dir: data_dir.to_owned(),
            }),
        }
    }

    /// Get a reference to the embedding model registry for configuration.
    pub fn embedding_registry(
        &self,
    ) -> &RwLock<EmbeddingModelRegistry> {
        &self.inner.embedding_registry
    }
}
