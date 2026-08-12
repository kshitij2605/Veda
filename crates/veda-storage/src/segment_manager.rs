use std::fs;
use std::path::{Path, PathBuf};

use tracing::instrument;
use uuid::Uuid;
use veda_core::error::{Result, VedaError};
use veda_core::record::Record;
use veda_core::schema::CollectionSchema;
use veda_core::types::SegmentId;

use crate::compaction::{CompactionConfig, compact_segments, select_segments_for_compaction};
use crate::growing_segment::GrowingSegment;
use crate::segment::{SegmentMetadata, SegmentReader};

/// Phase F: companion-snapshot path for an MVCC WAL. Given a
/// `path/to/mvcc.wal`, the snapshot lives next to it at
/// `path/to/mvcc.wal.snap`. Keeping them sibling-named makes
/// the recovery story trivial — one ls of the shard dir tells
/// the operator everything.
fn mvcc_snap_path(wal_path: &Path) -> PathBuf {
    let mut p = wal_path.to_path_buf();
    if let Some(name) = p.file_name().map(|n| n.to_os_string()) {
        let mut new_name = name;
        new_name.push(".snap");
        p.set_file_name(new_name);
    } else {
        p.set_extension("snap");
    }
    p
}

/// Handle to a sealed segment with its associated reader.
pub struct SealedSegmentHandle {
    pub metadata: SegmentMetadata,
    pub reader: SegmentReader,
    pub dir: PathBuf,
}

/// Manages the lifecycle of segments for a single shard of a collection.
///
/// Handles: growing segment writes, auto-seal on threshold, explicit seal,
/// delete propagation, and recovery of sealed segments from disk.
pub struct SegmentManager {
    schema: CollectionSchema,
    shard_id: u32,
    base_dir: PathBuf,
    growing: GrowingSegment,
    sealed_segments: Vec<SealedSegmentHandle>,
    seal_threshold_bytes: usize,
    /// Optional MVCC overlay. When `enable_mvcc` is called, the
    /// segment manager mirrors every `insert_mvcc(...)` write
    /// into this in-memory versioned store. Snapshot reads via
    /// `read_at_ts` then return the version visible at the
    /// requested timestamp. The existing single-version paths
    /// (`insert`, `growing_segment` reads) are unaffected.
    mvcc: Option<crate::mvcc::MvccStore>,
    /// Optional MVCC durability WAL. When
    /// `enable_mvcc_durable(path)` opens this, every
    /// `insert_mvcc` / `delete_mvcc` / `set_mvcc_safe_ts` call
    /// also appends to the WAL. Recovery on restart replays the
    /// WAL into a fresh `MvccStore`.
    mvcc_wal: Option<crate::mvcc_wal::MvccWal>,
}

impl SegmentManager {
    /// Create a new segment manager for a shard.
    ///
    /// Creates `base_dir/shard_{shard_id}/` and initialises a fresh growing segment.
    pub fn new(
        schema: CollectionSchema,
        shard_id: u32,
        base_dir: &Path,
        seal_threshold_bytes: usize,
    ) -> Result<Self> {
        let shard_dir = base_dir.join(format!("shard_{shard_id}"));
        fs::create_dir_all(&shard_dir)?;

        let growing = GrowingSegment::with_max_size(schema.clone(), seal_threshold_bytes);

        Ok(Self {
            schema,
            shard_id,
            base_dir: shard_dir,
            growing,
            sealed_segments: Vec::new(),
            seal_threshold_bytes,
            mvcc: None,
            mvcc_wal: None,
        })
    }

    /// Turn on the MVCC overlay. Idempotent — calling twice is
    /// a no-op (preserves existing versions). In-memory only;
    /// for crash-safe durability use `enable_mvcc_durable()`.
    pub fn enable_mvcc(&mut self) {
        if self.mvcc.is_none() {
            self.mvcc = Some(crate::mvcc::MvccStore::new());
        }
    }

    /// Turn on the MVCC overlay with WAL-backed durability. The
    /// WAL lives at `path` (typically
    /// `{base_dir}/mvcc.wal`); a sibling snapshot at
    /// `{path}.snap` (if present) is loaded first, then the
    /// WAL is replayed on top. Subsequent `insert_mvcc` /
    /// `delete_mvcc` / `set_mvcc_safe_ts` calls append + fsync
    /// to the WAL. Call `mvcc_checkpoint()` to rewrite the
    /// snapshot and truncate the WAL.
    pub fn enable_mvcc_durable(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        let snap_path = mvcc_snap_path(&path);
        let mut store = crate::mvcc::MvccStore::new();
        // 1. Load snapshot (if any) — gets us to the last
        //    checkpoint point.
        let _snap_n = crate::mvcc_wal::MvccWal::load_snapshot_into(&snap_path, &mut store)
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc snap load: {e}")))?;
        // 2. Replay WAL on top — picks up everything after the
        //    last checkpoint.
        let (_applied, _dropped) = crate::mvcc_wal::MvccWal::replay_into(&path, &mut store)
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc replay: {e}")))?;
        let wal = crate::mvcc_wal::MvccWal::open(&path)
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc wal open: {e}")))?;
        self.mvcc = Some(store);
        self.mvcc_wal = Some(wal);
        Ok(())
    }

    /// Phase F: write a full MVCC snapshot to disk and truncate
    /// the WAL. Returns the count of versions snapshotted, or
    /// `None` if MVCC durability isn't enabled. Idempotent on a
    /// quiescent store — repeated checkpoints just overwrite the
    /// snapshot file.
    pub fn mvcc_checkpoint(&mut self) -> Result<Option<usize>> {
        let (Some(store), Some(wal)) = (self.mvcc.as_ref(), self.mvcc_wal.as_mut()) else {
            return Ok(None);
        };
        let snap_path = mvcc_snap_path(wal.path());
        let n = wal
            .checkpoint(store, &snap_path)
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc checkpoint: {e}")))?;
        Ok(Some(n))
    }

    /// Whether the MVCC overlay is active on this shard.
    pub fn has_mvcc(&self) -> bool {
        self.mvcc.is_some()
    }

    /// Per-record MVCC visibility check. Returns true if some
    /// non-tombstone version with `commit_ts <= read_ts` exists.
    /// Collections without MVCC always return true.
    pub fn is_visible_at(&self, record_id: &uuid::Uuid, read_ts: u128) -> bool {
        match self.mvcc.as_ref() {
            None => true,
            Some(m) => m.get_at(record_id.as_bytes(), read_ts).is_some(),
        }
    }

    /// Phase F: per-doc-id visibility check. The growing segment
    /// may have multiple doc_ids for the same record (each
    /// version is a separate doc_id, stamped with commit_ts).
    /// This routes the visibility decision to the right doc.
    /// Returns true on legacy / non-MVCC docs.
    pub fn doc_visible_at(&self, doc_id: u32, read_ts: u128) -> bool {
        self.growing.is_visible_at(doc_id, read_ts)
    }

    /// MVCC-aware insert. Mirrors the record into the MVCC
    /// overlay at `commit_ts` AND into the existing growing
    /// segment (stamped with the same commit_ts so per-version
    /// visibility filtering can pick the right doc_id). Returns
    /// `Err` if `enable_mvcc()` hasn't been called.
    pub fn insert_mvcc(&mut self, commit_ts: u128, record: Record) -> Result<()> {
        let mvcc = self.mvcc.as_mut().ok_or_else(|| {
            veda_core::error::VedaError::Index("insert_mvcc requires enable_mvcc()".into())
        })?;
        let key = record.id().as_bytes().to_vec();
        let payload = bytes::Bytes::from(
            serde_json::to_vec(&record)
                .map_err(|e| veda_core::error::VedaError::Index(format!("serde: {e}")))?,
        );
        mvcc.put_payload(&key, commit_ts, payload.clone())
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc.put: {e}")))?;
        if let Some(wal) = self.mvcc_wal.as_mut() {
            wal.append(&crate::mvcc_wal::MvccWalEntry::Put {
                user_key: key,
                commit_ts,
                payload,
            })
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc wal append: {e}")))?;
        }
        // Phase F: write into the growing segment with the
        // commit_ts so per-version visibility filtering works
        // for search hits and growing-segment reads.
        if self.growing.should_seal() {
            self.seal_growing()?;
        }
        self.growing.insert_with_commit_ts(record, commit_ts)?;
        Ok(())
    }

    /// MVCC-aware tombstone. Same shape as `insert_mvcc`.
    pub fn delete_mvcc(&mut self, commit_ts: u128, record_id: &uuid::Uuid) -> Result<()> {
        let mvcc = self.mvcc.as_mut().ok_or_else(|| {
            veda_core::error::VedaError::Index("delete_mvcc requires enable_mvcc()".into())
        })?;
        let key = record_id.as_bytes().to_vec();
        mvcc.delete(&key, commit_ts)
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc.delete: {e}")))?;
        if let Some(wal) = self.mvcc_wal.as_mut() {
            wal.append(&crate::mvcc_wal::MvccWalEntry::Delete {
                user_key: key,
                commit_ts,
            })
            .map_err(|e| veda_core::error::VedaError::Index(format!("mvcc wal append: {e}")))?;
        }
        // Tombstone the growing segment too.
        let _ = self.delete_record(*record_id)?;
        Ok(())
    }

    /// MVCC snapshot read. Returns the record JSON visible at
    /// `read_ts`. `None` means no version exists at that ts OR
    /// the visible version is a tombstone.
    pub fn read_at_ts(&self, record_id: &uuid::Uuid, read_ts: u128) -> Option<bytes::Bytes> {
        let mvcc = self.mvcc.as_ref()?;
        mvcc.get_at(record_id.as_bytes(), read_ts)
    }

    /// Advance the MVCC safe-ts watermark. Callers (typically
    /// the gateway's GC scheduler) wire this from the TSO's
    /// `safe_ts` watermark.
    pub fn set_mvcc_safe_ts(&mut self, ts: u128) -> bool {
        if let Some(mvcc) = self.mvcc.as_mut() {
            mvcc.set_safe_ts(ts);
            if let Some(wal) = self.mvcc_wal.as_mut() {
                // Best-effort: if the WAL append fails we still
                // advanced the in-memory watermark. Log + go.
                if let Err(e) = wal.append(&crate::mvcc_wal::MvccWalEntry::SetSafeTs { ts }) {
                    tracing::warn!(error = %e, ts = ts, "mvcc wal set_safe_ts append failed");
                }
            }
            true
        } else {
            false
        }
    }

    /// Run MVCC GC on this shard. Returns the count of versions
    /// dropped, or `None` if MVCC isn't enabled.
    pub fn mvcc_gc(&mut self) -> Option<usize> {
        self.mvcc.as_mut().map(|m| m.gc())
    }

    /// Insert a record into the growing segment.
    ///
    /// If the growing segment has reached the seal threshold, it is sealed first
    /// and a new growing segment is created before inserting.
    #[instrument(name = "segment_insert", skip(self, record), fields(shard_id = self.shard_id))]
    pub fn insert(&mut self, record: Record) -> Result<()> {
        if self.growing.should_seal() {
            self.seal_growing()?;
        }
        self.growing.insert(record)?;
        Ok(())
    }

    /// Explicitly seal the current growing segment and create a new one.
    ///
    /// Returns the metadata of the newly sealed segment, or `None` if the
    /// growing segment was empty (no empty segments are created).
    #[instrument(name = "seal_growing", skip(self), fields(shard_id = self.shard_id))]
    pub fn seal_growing(&mut self) -> Result<Option<SegmentMetadata>> {
        if self.growing.count() == 0 {
            return Ok(None);
        }

        let segment_id = self.growing.id();
        let segment_dir = self.base_dir.join(format!("segment_{segment_id}"));

        let new_growing =
            GrowingSegment::with_max_size(self.schema.clone(), self.seal_threshold_bytes);
        let old_growing = std::mem::replace(&mut self.growing, new_growing);

        let metadata = old_growing.seal(&segment_dir)?;
        let reader = SegmentReader::open(&segment_dir)?;

        self.sealed_segments.push(SealedSegmentHandle {
            metadata: metadata.clone(),
            reader,
            dir: segment_dir,
        });

        Ok(Some(metadata))
    }

    /// Return the shard ID this manager belongs to.
    pub fn shard_id(&self) -> u32 {
        self.shard_id
    }

    /// Get the current growing segment (for search).
    pub fn growing_segment(&self) -> &GrowingSegment {
        &self.growing
    }

    /// Get all sealed segments (for search).
    pub fn sealed_segments(&self) -> &[SealedSegmentHandle] {
        &self.sealed_segments
    }

    /// Number of records across all segments (growing + sealed).
    pub fn total_records(&self) -> u64 {
        let growing = self.growing.count() as u64;
        let sealed: u64 = self
            .sealed_segments
            .iter()
            .map(|s| s.metadata.row_count)
            .sum();
        growing + sealed
    }

    /// List all segment IDs (growing + sealed).
    pub fn segment_ids(&self) -> Vec<SegmentId> {
        let mut ids = vec![self.growing.id()];
        for handle in &self.sealed_segments {
            ids.push(handle.metadata.id);
        }
        ids
    }

    /// O(1) record lookup by record UUID across this shard's growing +
    /// sealed segments. Used by the gateway's search-field hydration so
    /// every hit returned by ANN search can be materialized without
    /// walking the entire collection. Returns `None` if the record isn't
    /// in this manager's segments (e.g. it lives in another shard).
    pub fn lookup_record(&self, id: uuid::Uuid) -> Option<Record> {
        if let Some(doc_id) = self.growing.doc_id_for_uuid(&id)
            && let Some(rec) = self.growing.get_record(doc_id)
        {
            return Some(rec.clone());
        }
        for handle in &self.sealed_segments {
            if let Some(doc_id) = handle.reader.doc_id_for_uuid(&id) {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    return None;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    return Some(rec);
                }
            }
        }
        None
    }

    /// TTL-aware record lookup. Identical to `lookup_record` but also filters
    /// out records whose `_expires_at` field is `<= now_secs`.
    ///
    /// `now_secs` is the caller-supplied Unix epoch SECONDS (injected for
    /// determinism in tests; use `SystemTime::now()` at query start in prod).
    ///
    /// Records without `_expires_at` are always returned (backward compat).
    pub fn lookup_record_at(&self, id: uuid::Uuid, now_secs: i64) -> Option<Record> {
        let rec = self.lookup_record(id)?;
        if rec.is_expired_at(now_secs) {
            None
        } else {
            Some(rec)
        }
    }

    /// Scan all segments and mark every record whose `_expires_at <= now_secs`
    /// as deleted in the delete_bitmap.  Reuses the existing `delete_record`
    /// path so compaction will later purge these rows physically.
    ///
    /// Returns the count of records newly marked deleted.
    ///
    /// `now_secs` is injected (not `SystemTime::now()`) for determinism in tests.
    pub fn sweep_expired(&mut self, now_secs: i64) -> usize {
        let mut swept = 0usize;

        // Growing segment — iterate doc_ids directly.
        let count = self.growing.count();
        for doc_id in 0..count {
            if self.growing.is_deleted(doc_id) {
                continue;
            }
            let expired = self
                .growing
                .get_record(doc_id)
                .map(|r| r.is_expired_at(now_secs))
                .unwrap_or(false);
            if expired {
                self.growing.mark_deleted(doc_id);
                swept += 1;
            }
        }

        // Sealed segments.
        for handle in &mut self.sealed_segments {
            let row_count = handle.metadata.row_count as u32;
            for doc_id in 0..row_count {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                let expired = handle
                    .reader
                    .get_record(doc_id)
                    .map(|r| r.is_expired_at(now_secs))
                    .unwrap_or(false);
                if expired {
                    handle.reader.delete_bitmap_mut().mark_deleted(doc_id);
                    swept += 1;
                }
            }
        }

        if swept > 0 {
            tracing::info!(swept, now_secs, "ttl: swept expired records");
        }
        swept
    }

    /// Move a sealed segment to a new storage tier. Hot ↔ Warm transitions
    /// only affect index-build behaviour at the next executor rebuild —
    /// no data is moved. Returns `true` if the segment was found and updated.
    ///
    /// **Not for Cold** — use [`demote_to_cold`] which also uploads and
    /// removes local files.
    pub fn set_segment_tier(
        &mut self,
        segment_id: SegmentId,
        tier: crate::segment::SegmentTier,
    ) -> Result<bool> {
        for handle in &mut self.sealed_segments {
            if handle.metadata.id == segment_id {
                handle.metadata.tier = tier;
                let meta_path = handle.dir.join("metadata.json");
                let json = serde_json::to_vec_pretty(&handle.metadata)?;
                fs::write(&meta_path, json)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Archive a sealed segment to cold blob storage and remove the local
    /// copy. The segment is removed from this manager's `sealed_segments`
    /// list — subsequent searches will not see it until it is promoted.
    ///
    /// Returns the segment's metadata so the caller can track it
    /// out-of-band (Veda does not yet maintain a cold-segment registry).
    pub fn demote_to_cold(
        &mut self,
        segment_id: SegmentId,
        blob_store: &dyn crate::blob_store::BlobStore,
        collection: &str,
    ) -> Result<Option<crate::segment::SegmentMetadata>> {
        let pos = self
            .sealed_segments
            .iter()
            .position(|h| h.metadata.id == segment_id);
        let Some(pos) = pos else {
            return Ok(None);
        };

        let handle = &self.sealed_segments[pos];
        crate::cold_storage::archive_segment_to_blob(
            blob_store,
            &handle.dir,
            collection,
            &segment_id.to_string(),
        )?;

        let mut metadata = handle.metadata.clone();
        metadata.tier = crate::segment::SegmentTier::Cold;
        let dir = handle.dir.clone();

        // Drop the reader (and mmaps) before deleting the directory.
        self.sealed_segments.remove(pos);
        fs::remove_dir_all(&dir).ok();
        Ok(Some(metadata))
    }

    /// Restore a previously cold-archived segment from blob storage and
    /// reattach it as a Warm sealed segment. The segment dir is recreated
    /// under this shard's base dir using the original `segment_{id}` name.
    pub fn promote_from_cold(
        &mut self,
        segment_id: SegmentId,
        blob_store: &dyn crate::blob_store::BlobStore,
        collection: &str,
    ) -> Result<()> {
        let dir = self.base_dir.join(format!("segment_{segment_id}"));
        crate::cold_storage::restore_segment_from_blob(
            blob_store,
            &dir,
            collection,
            &segment_id.to_string(),
        )?;
        let reader = crate::segment::SegmentReader::open(&dir)?;
        let mut metadata = reader.metadata().clone();
        metadata.tier = crate::segment::SegmentTier::Warm;
        // Persist the new tier marker so a restart keeps it Warm.
        fs::write(
            dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        self.sealed_segments.push(SealedSegmentHandle {
            metadata,
            reader,
            dir,
        });
        Ok(())
    }

    /// Delete a record by marking it in the appropriate segment's delete bitmap.
    ///
    /// Scans the growing segment first, then sealed segments. Returns `true` if
    /// the record was found and marked deleted.
    pub fn delete_record(&mut self, record_id: Uuid) -> Result<bool> {
        // Search growing segment.
        for doc_id in 0..self.growing.count() {
            if let Some(rec) = self.growing.get_record(doc_id)
                && rec.id() == record_id
            {
                self.growing.mark_deleted(doc_id);
                return Ok(true);
            }
        }

        // Search sealed segments.
        for handle in &mut self.sealed_segments {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                match handle.reader.get_record(doc_id) {
                    Ok(rec) if rec.id() == record_id => {
                        handle.reader.delete_bitmap_mut().mark_deleted(doc_id);
                        return Ok(true);
                    }
                    _ => continue,
                }
            }
        }

        Ok(false)
    }

    /// Flush every sealed segment's in-memory delete bitmap back to its
    /// `delete_bitmap.bin` file.
    ///
    /// `delete_record` only mutates a sealed segment's bitmap in memory.
    /// Search that goes through the query executor (dense/ANN search)
    /// operates on a `SegmentManager` rebuilt fresh from disk (see
    /// `rebuild_executors`), which re-reads `delete_bitmap.bin` via
    /// `SegmentReader::open` — it never sees an in-memory-only deletion.
    /// Callers that soft-delete a sealed record and need that deletion to
    /// survive such a rebuild must call this afterward.
    pub fn persist_sealed_delete_bitmaps(&self) -> Result<()> {
        for handle in &self.sealed_segments {
            handle
                .reader
                .delete_bitmap()
                .save(&handle.dir.join(crate::segment::DELETE_BITMAP_FILE))?;
        }
        Ok(())
    }

    /// Load existing sealed segments from disk (for recovery).
    ///
    /// Scans `base_dir/shard_{shard_id}/` for directories matching the
    /// `segment_{uuid}` naming convention and opens each as a `SegmentReader`.
    /// Returns the number of segments loaded.
    pub fn load_sealed_segments(&mut self) -> Result<usize> {
        // A mid-seal crash can leave a `segment_*` dir without
        // metadata.json, schema.json, or records.jsonl. Skip-and-log
        // such segments so one bad dir doesn't make the whole shard
        // unrecoverable — that was the failure mode flagged in the
        // last review of recover_collections_from_disk.
        let mut loaded = 0;

        let entries = fs::read_dir(&self.base_dir).map_err(|e| {
            VedaError::Storage(format!("failed to read shard dir {:?}: {e}", self.base_dir))
        })?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(error = %e, "load_sealed_segments: skip unreadable entry");
                    continue;
                }
            };
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(name = %name_str, error = %e, "load_sealed_segments: file_type failed");
                    continue;
                }
            };

            if !name_str.starts_with("segment_") || !file_type.is_dir() {
                continue;
            }

            // Skip segments we already have loaded.
            let seg_dir = entry.path();
            if self.sealed_segments.iter().any(|s| s.dir == seg_dir) {
                continue;
            }

            // Skip partial segments (mid-seal crash leaves them
            // without metadata.json). The reader open below would
            // also fail, but this check makes the log message clearer.
            if !seg_dir.join("metadata.json").exists() {
                tracing::warn!(
                    dir = %seg_dir.display(),
                    "load_sealed_segments: skipping segment without metadata.json (mid-seal crash?)"
                );
                continue;
            }

            let reader = match SegmentReader::open(&seg_dir) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        dir = %seg_dir.display(),
                        error = %e,
                        "load_sealed_segments: skipping segment that failed to open"
                    );
                    continue;
                }
            };
            let metadata = reader.metadata().clone();

            self.sealed_segments.push(SealedSegmentHandle {
                metadata,
                reader,
                dir: seg_dir,
            });
            loaded += 1;
        }

        Ok(loaded)
    }

    /// Run compaction if segment selection criteria are met.
    ///
    /// Returns the number of segments removed (0 if no compaction needed).
    pub fn compact_if_needed(&mut self, config: &CompactionConfig) -> Result<usize> {
        let segment_pairs: Vec<_> = self
            .sealed_segments
            .iter()
            .map(|h| (h.metadata.clone(), &h.reader))
            .collect();

        let selected = select_segments_for_compaction(&segment_pairs, config);
        if selected.is_empty() {
            return Ok(0);
        }

        let readers: Vec<&SegmentReader> = selected
            .iter()
            .map(|&i| &self.sealed_segments[i].reader)
            .collect();
        let output_dir = self.base_dir.join(format!("segment_{}", Uuid::new_v4()));

        let result = compact_segments(&readers, &self.schema, &output_dir)?;

        // Remove merged segments.
        let merged_ids: std::collections::HashSet<Uuid> =
            result.merged_segment_ids.iter().copied().collect();
        let removed_count = merged_ids.len();

        // Remove old segment directories.
        let old_dirs: Vec<PathBuf> = self
            .sealed_segments
            .iter()
            .filter(|h| merged_ids.contains(&h.metadata.id))
            .map(|h| h.dir.clone())
            .collect();
        for dir in &old_dirs {
            let _ = fs::remove_dir_all(dir);
        }

        self.sealed_segments
            .retain(|h| !merged_ids.contains(&h.metadata.id));

        // Add the new merged segment.
        let new_reader = SegmentReader::open(&output_dir)?;
        self.sealed_segments.push(SealedSegmentHandle {
            metadata: result.new_segment,
            reader: new_reader,
            dir: output_dir,
        });

        Ok(removed_count)
    }

    /// Force-merge **all** sealed segments into a single segment, bypassing
    /// the `max_merge_segments` / `min_segments` caps that `compact_if_needed`
    /// honours. Used by the `/optimize` endpoint to fully consolidate a
    /// fragmented bulk load (e.g. ~38 small sealed segments) into one, so a
    /// single ANN graph serves the whole shard. Returns the number of sealed
    /// segments that existed before the merge (0 or 1 → no-op).
    pub fn force_merge_all(&mut self) -> Result<usize> {
        let before = self.sealed_segments.len();
        if before <= 1 {
            return Ok(before);
        }

        let readers: Vec<&SegmentReader> = self.sealed_segments.iter().map(|h| &h.reader).collect();
        let output_dir = self.base_dir.join(format!("segment_{}", Uuid::new_v4()));
        let result = compact_segments(&readers, &self.schema, &output_dir)?;

        // Remove every old segment directory (all of them were merged).
        let old_dirs: Vec<PathBuf> = self.sealed_segments.iter().map(|h| h.dir.clone()).collect();
        for dir in &old_dirs {
            let _ = fs::remove_dir_all(dir);
        }
        self.sealed_segments.clear();

        let new_reader = SegmentReader::open(&output_dir)?;
        self.sealed_segments.push(SealedSegmentHandle {
            metadata: result.new_segment,
            reader: new_reader,
            dir: output_dir,
        });

        Ok(before)
    }
}
