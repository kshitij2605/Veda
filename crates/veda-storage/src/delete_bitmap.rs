use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use roaring::RoaringBitmap;
use veda_core::error::Result;

/// Tracks soft-deleted rows within a segment using a roaring bitmap.
///
/// Each bit position corresponds to a doc_id. A set bit means that document
/// has been logically deleted and should be excluded from query results.
pub struct DeleteBitmap {
    bitmap: RoaringBitmap,
    /// Set by `mark_deleted`, cleared by `save`/`save_if_dirty`. Lets
    /// `SegmentManager::persist_sealed_delete_bitmaps` skip segments that
    /// weren't touched by the current operation instead of rewriting every
    /// sealed segment's bitmap file unconditionally.
    dirty: bool,
}

impl DeleteBitmap {
    /// Create a new, empty delete bitmap (no documents deleted).
    pub fn new() -> Self {
        Self {
            bitmap: RoaringBitmap::new(),
            dirty: false,
        }
    }

    /// Mark a document as deleted.
    pub fn mark_deleted(&mut self, doc_id: u32) {
        self.bitmap.insert(doc_id);
        self.dirty = true;
    }

    /// Check whether a document has been deleted.
    pub fn is_deleted(&self, doc_id: u32) -> bool {
        self.bitmap.contains(doc_id)
    }

    /// Return the number of deleted documents.
    pub fn deleted_count(&self) -> u32 {
        self.bitmap.len() as u32
    }

    /// Whether this bitmap has changes since the last successful
    /// `save`/`save_if_dirty`/`load`.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Serialize the bitmap and write it to `path`, atomically.
    ///
    /// Writes to a sibling `<path>.tmp` file, `fsync`s it, then renames it
    /// over `path`. A crash or ENOSPC mid-write now leaves the previous
    /// `path` (if any) untouched instead of a truncated file —
    /// `DeleteBitmap::load` would fail to deserialize a truncated file, and
    /// `SegmentManager::load_sealed_segments` skips segments it can't open,
    /// which would silently drop every record in that sealed segment from
    /// query results. Does NOT clear the dirty flag; prefer
    /// `save_if_dirty` for callers that track it.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::new();
        self.bitmap
            .serialize_into(&mut bytes)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp_path = PathBuf::from(tmp_name);
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&tmp_path, path)?;
        // `file.sync_all()` above only guarantees the FILE's contents are
        // durable; on filesystems like ext4 the RENAME that makes it
        // visible at `path` is a separate metadata operation that can
        // still be lost on a power-cut, reverting a persisted delete back
        // to whatever `path` held before (or to nothing, on a first
        // write). Fsync the containing directory too, so the rename
        // itself survives. Best-effort: if the directory can't be opened
        // (e.g. no read access, or a platform where opening a directory
        // as a file isn't supported), the just-renamed file is almost
        // certainly still intact — this narrows an already-rare crash
        // window rather than being the load-bearing guarantee the
        // temp+fsync+rename above is.
        if let Some(parent) = path.parent()
            && let Ok(dir) = fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    /// `save`, but only when `is_dirty()` — and clears the dirty flag on
    /// success. Lets a caller that persists many segments' bitmaps in one
    /// pass (e.g. after a batch of deletes) skip the ones it never
    /// touched.
    pub fn save_if_dirty(&mut self, path: &Path) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.save(path)?;
        self.dirty = false;
        Ok(())
    }

    /// Load a previously saved delete bitmap from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let bitmap = RoaringBitmap::deserialize_from(bytes.as_slice())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(Self {
            bitmap,
            dirty: false,
        })
    }
}

impl Default for DeleteBitmap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_is_atomic_and_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("delete_bitmap.bin");
        let mut bm = DeleteBitmap::new();
        bm.mark_deleted(3);
        bm.mark_deleted(7);
        bm.save(&path).unwrap();

        // No leftover .tmp file after a successful save.
        let tmp_path = tmp.path().join("delete_bitmap.bin.tmp");
        assert!(!tmp_path.exists());

        let loaded = DeleteBitmap::load(&path).unwrap();
        assert!(loaded.is_deleted(3));
        assert!(loaded.is_deleted(7));
        assert!(!loaded.is_deleted(1));
        assert!(!loaded.is_dirty());
    }

    #[test]
    fn save_if_dirty_skips_a_clean_bitmap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("delete_bitmap.bin");
        let mut bm = DeleteBitmap::new();
        // Never marked deleted -> not dirty -> save_if_dirty must not
        // write anything.
        bm.save_if_dirty(&path).unwrap();
        assert!(!path.exists(), "a clean bitmap must not be written");
    }

    #[test]
    fn save_if_dirty_writes_and_clears_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("delete_bitmap.bin");
        let mut bm = DeleteBitmap::new();
        bm.mark_deleted(1);
        assert!(bm.is_dirty());
        bm.save_if_dirty(&path).unwrap();
        assert!(!bm.is_dirty());
        assert!(path.exists());

        let loaded = DeleteBitmap::load(&path).unwrap();
        assert!(loaded.is_deleted(1));
    }
}
