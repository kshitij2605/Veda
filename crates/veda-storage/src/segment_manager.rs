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
    /// `{base_dir}/mvcc.wal`); existing entries are replayed
    /// into a fresh `MvccStore` before the manager accepts new
    /// writes. Subsequent `insert_mvcc` / `delete_mvcc` /
    /// `set_mvcc_safe_ts` calls append + fsync to the WAL.
    pub fn enable_mvcc_durable(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        let mut store = crate::mvcc::MvccStore::new();
        let (_applied, _dropped) =
            crate::mvcc_wal::MvccWal::replay_into(&path, &mut store).map_err(|e| {
                veda_core::error::VedaError::Index(format!("mvcc replay: {e}"))
            })?;
        let wal = crate::mvcc_wal::MvccWal::open(&path).map_err(|e| {
            veda_core::error::VedaError::Index(format!("mvcc wal open: {e}"))
        })?;
        self.mvcc = Some(store);
        self.mvcc_wal = Some(wal);
        Ok(())
    }

    /// Whether the MVCC overlay is active on this shard.
    pub fn has_mvcc(&self) -> bool {
        self.mvcc.is_some()
    }

    /// MVCC-aware insert. Mirrors the record into the MVCC
    /// overlay at `commit_ts` AND into the existing growing
    /// segment so that legacy reads continue to work. Returns
    /// `Err` if `enable_mvcc()` hasn't been called.
    pub fn insert_mvcc(&mut self, commit_ts: u128, record: Record) -> Result<()> {
        let mvcc = self.mvcc.as_mut().ok_or_else(|| {
            veda_core::error::VedaError::Index(
                "insert_mvcc requires enable_mvcc()".into(),
            )
        })?;
        let key = record.id().as_bytes().to_vec();
        let payload = bytes::Bytes::from(
            serde_json::to_vec(&record)
                .map_err(|e| veda_core::error::VedaError::Index(format!("serde: {e}")))?,
        );
        mvcc.put_payload(&key, commit_ts, payload.clone()).map_err(|e| {
            veda_core::error::VedaError::Index(format!("mvcc.put: {e}"))
        })?;
        if let Some(wal) = self.mvcc_wal.as_mut() {
            wal.append(&crate::mvcc_wal::MvccWalEntry::Put {
                user_key: key,
                commit_ts,
                payload,
            })
            .map_err(|e| {
                veda_core::error::VedaError::Index(format!("mvcc wal append: {e}"))
            })?;
        }
        // Still write into the growing segment so existing
        // search/list paths work. v1 will read those paths via
        // the MVCC overlay too.
        self.insert(record)?;
        Ok(())
    }

    /// MVCC-aware tombstone. Same shape as `insert_mvcc`.
    pub fn delete_mvcc(&mut self, commit_ts: u128, record_id: &uuid::Uuid) -> Result<()> {
        let mvcc = self.mvcc.as_mut().ok_or_else(|| {
            veda_core::error::VedaError::Index(
                "delete_mvcc requires enable_mvcc()".into(),
            )
        })?;
        let key = record_id.as_bytes().to_vec();
        mvcc.delete(&key, commit_ts).map_err(|e| {
            veda_core::error::VedaError::Index(format!("mvcc.delete: {e}"))
        })?;
        if let Some(wal) = self.mvcc_wal.as_mut() {
            wal.append(&crate::mvcc_wal::MvccWalEntry::Delete {
                user_key: key,
                commit_ts,
            })
            .map_err(|e| {
                veda_core::error::VedaError::Index(format!("mvcc wal append: {e}"))
            })?;
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
                if let Err(e) = wal.append(&crate::mvcc_wal::MvccWalEntry::SetSafeTs { ts })
                {
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
        let sealed: u64 = self.sealed_segments.iter().map(|s| s.metadata.row_count).sum();
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
            if self
                .sealed_segments
                .iter()
                .any(|s| s.dir == seg_dir)
            {
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

        let readers: Vec<&SegmentReader> = selected.iter().map(|&i| &self.sealed_segments[i].reader).collect();
        let output_dir = self.base_dir.join(format!("segment_{}", Uuid::new_v4()));

        let result = compact_segments(&readers, &self.schema, &output_dir)?;

        // Remove merged segments.
        let merged_ids: std::collections::HashSet<Uuid> = result.merged_segment_ids.iter().copied().collect();
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

        self.sealed_segments.retain(|h| !merged_ids.contains(&h.metadata.id));

        // Add the new merged segment.
        let new_reader = SegmentReader::open(&output_dir)?;
        self.sealed_segments.push(SealedSegmentHandle {
            metadata: result.new_segment,
            reader: new_reader,
            dir: output_dir,
        });

        Ok(removed_count)
    }
}
