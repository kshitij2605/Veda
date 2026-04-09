use std::fs;
use std::path::{Path, PathBuf};

use tracing::instrument;
use uuid::Uuid;
use veda_core::error::{Result, VedaError};
use veda_core::record::Record;
use veda_core::schema::CollectionSchema;
use veda_core::types::SegmentId;

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
        })
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
        let mut loaded = 0;

        let entries = fs::read_dir(&self.base_dir).map_err(|e| {
            VedaError::Storage(format!("failed to read shard dir {:?}: {e}", self.base_dir))
        })?;

        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if !name_str.starts_with("segment_") || !entry.file_type()?.is_dir() {
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

            let reader = SegmentReader::open(&seg_dir)?;
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
}
