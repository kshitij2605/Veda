use std::path::Path;

use uuid::Uuid;
use veda_core::error::Result;
use veda_core::schema::CollectionSchema;

use crate::segment::{SegmentMetadata, SegmentReader, SegmentWriter};

/// Configuration for compaction.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Minimum number of segments to trigger compaction.
    pub min_segments: usize,
    /// Maximum number of segments to merge in one compaction.
    pub max_merge_segments: usize,
    /// Minimum ratio of deleted records to trigger compaction on a single segment.
    pub delete_ratio_threshold: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_segments: 3,
            max_merge_segments: 10,
            delete_ratio_threshold: 0.2,
        }
    }
}

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// Metadata of the new merged segment.
    pub new_segment: SegmentMetadata,
    /// IDs of segments that were merged (can be deleted).
    pub merged_segment_ids: Vec<Uuid>,
    /// Number of records in the merged segment.
    pub record_count: u64,
    /// Number of records that were deleted (purged during compaction).
    pub purged_count: u64,
}

/// Select segments that should be compacted.
///
/// Returns indices into the input slice. Selection strategy:
/// 1. Include any segment with a delete ratio above the threshold.
/// 2. If total segment count >= min_segments, also include the smallest
///    segments (by row_count) until we reach max_merge_segments.
/// 3. Never return more than max_merge_segments indices.
pub fn select_segments_for_compaction(
    segments: &[(SegmentMetadata, &SegmentReader)],
    config: &CompactionConfig,
) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();

    // 1. Select segments with high delete ratios.
    for (i, (meta, reader)) in segments.iter().enumerate() {
        if meta.row_count == 0 {
            continue;
        }
        let deleted = reader.delete_bitmap().deleted_count() as f64;
        let total = meta.row_count as f64;
        if deleted / total >= config.delete_ratio_threshold {
            selected.push(i);
        }
    }

    // 2. If enough segments exist, also select the smallest ones.
    if segments.len() >= config.min_segments {
        // Sort indices by row_count ascending (smallest first).
        let mut by_size: Vec<usize> = (0..segments.len()).collect();
        by_size.sort_by_key(|&i| segments[i].0.row_count);

        for i in by_size {
            if selected.len() >= config.max_merge_segments {
                break;
            }
            if !selected.contains(&i) {
                selected.push(i);
            }
        }
    }

    // 3. Cap at max_merge_segments.
    selected.truncate(config.max_merge_segments);
    selected
}

/// Compact (merge) multiple segments into a single new segment.
///
/// Iterates all input segments, skipping deleted records, and writes
/// non-deleted records into a new sealed segment at `output_dir`.
pub fn compact_segments(
    segments: &[&SegmentReader],
    schema: &CollectionSchema,
    output_dir: &Path,
) -> Result<CompactionResult> {
    let mut writer = SegmentWriter::create(output_dir, schema)?;
    let mut merged_segment_ids: Vec<Uuid> = Vec::new();
    let mut purged_count: u64 = 0;

    for &reader in segments {
        merged_segment_ids.push(reader.metadata().id);
        let row_count = reader.row_count();
        let bitmap = reader.delete_bitmap();

        for doc_id in 0..row_count as u32 {
            if bitmap.is_deleted(doc_id) {
                purged_count += 1;
                continue;
            }
            let record = reader.get_record(doc_id)?;
            writer.insert(&record)?;
        }
    }

    let record_count = writer.count();
    let new_segment = writer.seal()?;

    Ok(CompactionResult {
        new_segment,
        merged_segment_ids,
        record_count,
        purged_count,
    })
}
