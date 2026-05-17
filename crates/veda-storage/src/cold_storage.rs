//! Cold-tier archival to and from a `BlobStore`.
//!
//! Walks a sealed-segment directory, uploads each file as a separate blob
//! under `cold/{collection}/{segment_id}/{relative_path}`, then optionally
//! removes the local copy. Restore reverses the process.
//!
//! ## Why per-file rather than a single tar
//!
//! - Avoids pulling in a tar dependency.
//! - Lets the blob store deduplicate at the file granularity (some
//!   segments share schema.json content).
//! - Restore can stream files concurrently.
//! - Failed mid-archive uploads can resume by listing what's already there.

use std::fs;
use std::path::{Path, PathBuf};

use veda_core::error::{Result, VedaError};

use crate::blob_store::BlobStore;

/// Archive every file under `segment_dir` to the blob store using the
/// prefix `cold/{collection}/{segment_id}/`. Does NOT touch the local
/// directory; the caller decides whether to remove it after the upload
/// succeeds.
pub fn archive_segment_to_blob(
    blob_store: &dyn BlobStore,
    segment_dir: &Path,
    collection: &str,
    segment_id: &str,
) -> Result<usize> {
    let prefix = format!("cold/{collection}/{segment_id}/");
    let mut uploaded = 0usize;
    walk_dir(segment_dir, segment_dir, |rel, full| {
        let key = format!("{prefix}{}", rel.to_string_lossy().replace('\\', "/"));
        let bytes = fs::read(full).map_err(|e| {
            VedaError::Storage(format!("cold archive: read {full:?} failed: {e}"))
        })?;
        blob_store.put(&key, &bytes)?;
        uploaded += 1;
        Ok(())
    })?;
    Ok(uploaded)
}

/// Restore a segment directory from the blob store. The directory must
/// not already exist (or must be empty); files are written under it
/// preserving relative paths.
pub fn restore_segment_from_blob(
    blob_store: &dyn BlobStore,
    segment_dir: &Path,
    collection: &str,
    segment_id: &str,
) -> Result<usize> {
    let prefix = format!("cold/{collection}/{segment_id}/");
    fs::create_dir_all(segment_dir)?;
    let keys = blob_store.list(&prefix)?;
    let mut count = 0usize;
    for key in keys {
        let Some(rel) = key.strip_prefix(&prefix) else {
            continue;
        };
        let dest = segment_dir.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = blob_store.get(&key)?;
        fs::write(&dest, bytes).map_err(|e| {
            VedaError::Storage(format!("cold restore: write {dest:?} failed: {e}"))
        })?;
        count += 1;
    }
    Ok(count)
}

/// Walk `dir` recursively, calling `f(relative_path, full_path)` for each
/// regular file. `base` is the root to compute relative paths against.
fn walk_dir<F>(dir: &Path, base: &Path, mut f: F) -> Result<()>
where
    F: FnMut(&Path, &Path) -> Result<()>,
{
    walk_dir_inner(dir, base, &mut f)
}

fn walk_dir_inner<F>(dir: &Path, base: &Path, f: &mut F) -> Result<()>
where
    F: FnMut(&Path, &Path) -> Result<()>,
{
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk_dir_inner(&path, base, f)?;
        } else if ft.is_file() {
            let rel: PathBuf = path
                .strip_prefix(base)
                .map_err(|e| VedaError::Storage(format!("strip_prefix: {e}")))?
                .to_path_buf();
            f(&rel, &path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::LocalBlobStore;

    #[test]
    fn archive_and_restore_roundtrips_a_segment_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("seg_abc");
        let dst = tmp.path().join("seg_abc_restored");
        let blob_dir = tmp.path().join("blobs");

        // Create a fake segment dir with a few files.
        fs::create_dir_all(src.join("vectors")).unwrap();
        fs::create_dir_all(src.join("scalars")).unwrap();
        fs::write(src.join("metadata.json"), b"{\"id\":\"abc\"}").unwrap();
        fs::write(src.join("vectors").join("emb.vec"), b"VEDA-fake-bytes").unwrap();
        fs::write(src.join("scalars").join("title.inv"), b"inv-bytes").unwrap();

        let store = LocalBlobStore::new(&blob_dir).unwrap();
        let n_up = archive_segment_to_blob(&store, &src, "products", "abc").unwrap();
        assert_eq!(n_up, 3);

        let n_dl = restore_segment_from_blob(&store, &dst, "products", "abc").unwrap();
        assert_eq!(n_dl, 3);

        let restored_meta = fs::read(dst.join("metadata.json")).unwrap();
        assert_eq!(restored_meta, b"{\"id\":\"abc\"}");
        let restored_vec = fs::read(dst.join("vectors").join("emb.vec")).unwrap();
        assert_eq!(restored_vec, b"VEDA-fake-bytes");
    }
}
