//! Registry of per-shard [`ShardLog`]s. Owns the on-disk layout
//! convention `{base}/{collection}/{shard:04}/` and caches open
//! `Arc<ShardLog>` handles. Replaces the gateway's informal
//! `RwLock<HashMap<String, Arc<ShardLog>>>` + `ensure_shard_log`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::shard_log::{ShardLog, ShardLogError};

pub struct ShardLogRegistry {
    base_dir: PathBuf,
    logs: RwLock<HashMap<(String, u32), Arc<ShardLog>>>,
}

impl ShardLogRegistry {
    /// `base_dir` is the directory under which per-shard log dirs
    /// live. The gateway passes `{data_dir}/_logs`.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_owned(),
            logs: RwLock::new(HashMap::new()),
        }
    }

    /// On-disk directory for one shard's log:
    /// `{base}/{collection}/{shard:04}`.
    pub fn shard_dir(&self, collection: &str, shard: u32) -> PathBuf {
        self.base_dir.join(collection).join(format!("{shard:04}"))
    }

    /// Return the cached log for `(collection, shard)`, opening
    /// (and caching) it on first access.
    pub fn get_or_open(
        &self,
        collection: &str,
        shard: u32,
    ) -> Result<Arc<ShardLog>, ShardLogError> {
        let key = (collection.to_owned(), shard);
        if let Some(log) = self.logs.read().unwrap().get(&key) {
            return Ok(log.clone());
        }
        let dir = self.shard_dir(collection, shard);
        let log = Arc::new(ShardLog::open(&dir)?);
        let mut guard = self.logs.write().unwrap();
        Ok(guard.entry(key).or_insert(log).clone())
    }

    /// Evict every cached shard handle for `collection`. Returns
    /// the count evicted. Does not touch on-disk files.
    pub fn remove_collection(&self, collection: &str) -> usize {
        let mut guard = self.logs.write().unwrap();
        let before = guard.len();
        guard.retain(|(c, _), _| c != collection);
        before - guard.len()
    }

    /// Remove the on-disk log directory `{base}/{collection}` and
    /// all its shard subdirs. No-op if the dir is absent.
    pub fn purge_collection_dir(&self, collection: &str) -> std::io::Result<()> {
        let dir = self.base_dir.join(collection);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_open_caches_same_arc() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        let a = reg.get_or_open("coll", 0).unwrap();
        let b = reg.get_or_open("coll", 0).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b), "same (collection,shard) returns the cached Arc");
    }

    #[test]
    fn shard_dir_matches_legacy_layout() {
        let reg = ShardLogRegistry::new(std::path::PathBuf::from("/data/_logs"));
        assert_eq!(
            reg.shard_dir("rag_anything", 3),
            std::path::PathBuf::from("/data/_logs/rag_anything/0003")
        );
    }

    #[test]
    fn remove_collection_drops_all_shards_for_that_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        reg.get_or_open("a", 0).unwrap();
        reg.get_or_open("a", 1).unwrap();
        reg.get_or_open("b", 0).unwrap();
        assert_eq!(reg.remove_collection("a"), 2, "both shards of 'a' evicted");
        assert_eq!(reg.remove_collection("a"), 0, "idempotent: already gone");
    }

    #[test]
    fn purge_collection_dir_removes_on_disk_logs() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        reg.get_or_open("doomed", 0).unwrap();
        let dir = tmp.path().join("_logs").join("doomed");
        assert!(dir.exists());
        reg.purge_collection_dir("doomed").unwrap();
        assert!(!dir.exists(), "collection log dir removed");
    }
}
