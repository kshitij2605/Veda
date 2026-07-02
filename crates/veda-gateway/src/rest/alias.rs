//! Non-Raft (single-node) collection-name alias resolution.
//!
//! `set_alias`/`get_alias` (`collections.rs`) already work end-to-end in
//! Raft mode via `ClusterCommand::SetCollectionAlias` on the replicated
//! state machine. In non-Raft (single-node) mode that command has nowhere
//! to go — this module is the local, persisted equivalent, mirroring
//! `SavedObjectStore`'s `Arc<RwLock<HashMap>>` + write-through-JSON pattern
//! (`crates/veda-gateway/src/rest/saved_objects.rs`).
//!
//! `resolve` is the identity function when no alias is set, matching
//! Raft-mode `get_alias`'s documented behavior ("alias = target when no
//! alias is set").

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AliasMap {
    inner: Arc<RwLock<HashMap<String, String>>>,
    path: PathBuf,
}

impl AliasMap {
    /// Load from `{data_dir}/_aliases.json`, starting empty if absent or
    /// unparseable (mirrors `SavedObjectStore::new` / `JobRegistry::new`).
    pub fn new(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("_aliases.json");
        let inner: HashMap<String, String> = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "alias: parse failed, starting empty"
                );
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
            path,
        }
    }

    /// Set (or clear, by pointing an alias at itself) an alias -> target
    /// mapping and persist immediately.
    pub async fn set(&self, alias: String, target: String) {
        {
            let mut m = self.inner.write().await;
            m.insert(alias, target);
        }
        self.write_through().await;
    }

    /// The raw alias entry, if one exists (used by `get_alias`'s HTTP
    /// handler to distinguish "no alias" from "alias = self").
    pub async fn get(&self, alias: &str) -> Option<String> {
        self.inner.read().await.get(alias).cloned()
    }

    /// Resolve `name` through the alias map. Returns the target if an
    /// alias entry exists, else returns `name` unchanged (identity
    /// default — a collection with no alias resolves to itself).
    pub async fn resolve(&self, name: &str) -> String {
        self.inner
            .read()
            .await
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    }

    async fn write_through(&self) {
        let path = self.path.clone();
        let snapshot: HashMap<String, String> = self.inner.read().await.clone();
        let result = tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_vec_pretty(&snapshot) {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&path, bytes) {
                        tracing::warn!(error = %e, path = %path.display(), "alias: write failed");
                    }
                }
                Err(e) => tracing::warn!(error = %e, "alias: serialize failed"),
            }
        })
        .await;
        if let Err(e) = result {
            tracing::warn!(error = %e, "alias: write task panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_returns_identity_when_no_alias_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.resolve("rag_anything").await, "rag_anything");
    }

    #[tokio::test]
    async fn resolve_returns_target_after_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("rag_anything".to_owned(), "rag_anything@v2".to_owned())
            .await;
        assert_eq!(map.resolve("rag_anything").await, "rag_anything@v2");
        // Unrelated names still resolve to themselves.
        assert_eq!(map.resolve("other_collection").await, "other_collection");
    }

    #[tokio::test]
    async fn set_persists_and_reloads() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let map = AliasMap::new(tmp.path());
            map.set("a".to_owned(), "a@v2".to_owned()).await;
        }
        // Fresh instance over the same data_dir picks up the persisted alias.
        let map2 = AliasMap::new(tmp.path());
        assert_eq!(map2.resolve("a").await, "a@v2");
    }

    #[tokio::test]
    async fn get_returns_none_when_unset() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.get("a").await, None);
    }

    #[tokio::test]
    async fn set_can_be_reverted_by_setting_back() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("a".to_owned(), "a@v2".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a@v2");
        // Revert: point the alias back at the original name.
        map.set("a".to_owned(), "a".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a");
    }
}
