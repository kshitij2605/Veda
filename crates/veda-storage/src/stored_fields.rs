//! Reader-side abstraction for a segment's stored scalar / blob data.
//!
//! Two layouts are supported (or planned):
//! - [`JsonlStoredFields`] — the original `records.jsonl` + offset
//!   table + uuid index, mmap'd once at `open()`. This is the default
//!   for every existing segment.
//! - `ColumnarStoredFields` (planned, see
//!   `documentation/columnar-stored-fields.md`) — per-field column
//!   files for projection-friendly reads.
//!
//! Today only the JSONL impl is wired through `SegmentReader`. The
//! module is structured as a single concrete type now so the surface
//! is small; an enum / trait split lands when the columnar variant
//! does.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use serde::Deserialize;
use uuid::Uuid;
use veda_core::error::{Result, VedaError};

/// File name of the JSONL records file inside a segment dir. Mirrors
/// the constant in `segment.rs` — duplicated here so this module
/// doesn't pull a circular dep.
pub const RECORDS_FILE: &str = "records.jsonl";

/// Reader for a segment's per-row scalar/blob data backed by an mmap'd
/// `records.jsonl` file plus a built-at-open-time offset table and
/// UUID index.
///
/// Construction is one linear pass over the mmap that builds both
/// `record_offsets` (so `get_line_bytes(doc_id)` is O(1)) and
/// `uuid_to_doc_id` (so `doc_id_for_uuid` is also O(1)).
pub struct JsonlStoredFields {
    /// Memory map of the JSONL file. Kept alive for the lifetime of the
    /// reader so the byte slices returned by [`get_line_bytes`] stay
    /// valid.
    mmap: Arc<Mmap>,
    /// Per-row start byte offsets. `record_offsets[i] .. record_offsets[i+1]`
    /// covers row `i` including its trailing newline (stripped by
    /// `get_line_bytes`).
    record_offsets: Vec<u64>,
    /// UUID → doc_id for O(1) lookup by record id.
    uuid_to_doc_id: HashMap<Uuid, u32>,
}

impl JsonlStoredFields {
    /// Open a segment's `records.jsonl`, mmap it, and build the offset
    /// table + UUID index in a single linear scan.
    ///
    /// `expected_rows` is the row count from segment metadata; used
    /// only to size the offsets / uuid index up-front. Mismatches
    /// between metadata row count and actual line count are not
    /// rejected here — the segment reader can detect that separately
    /// if it cares.
    pub fn open(dir: &Path, expected_rows: u64) -> Result<Self> {
        let path = dir.join(RECORDS_FILE);
        let file = fs::File::open(&path).map_err(|e| {
            VedaError::Storage(format!("failed to open {path:?}: {e}"))
        })?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offsets: Vec<u64> = Vec::with_capacity(expected_rows as usize + 1);
        let mut uuid_index: HashMap<Uuid, u32> =
            HashMap::with_capacity(expected_rows as usize);

        offsets.push(0);
        let bytes: &[u8] = &mmap[..];
        let mut line_start: usize = 0;
        let mut doc_id: u32 = 0;
        for (i, b) in bytes.iter().enumerate() {
            if *b == b'\n' {
                let line = &bytes[line_start..i];
                if !line.is_empty()
                    && let Ok(id) = extract_record_id(line)
                {
                    uuid_index.insert(id, doc_id);
                }
                doc_id += 1;
                offsets.push((i + 1) as u64);
                line_start = i + 1;
            }
        }
        // Final line without trailing newline.
        if line_start < bytes.len() {
            let line = &bytes[line_start..];
            if let Ok(id) = extract_record_id(line) {
                uuid_index.insert(id, doc_id);
            }
            offsets.push(bytes.len() as u64);
        }

        Ok(Self {
            mmap: Arc::new(mmap),
            record_offsets: offsets,
            uuid_to_doc_id: uuid_index,
        })
    }

    /// Return the JSONL line bytes for `doc_id`, stripped of the
    /// trailing newline. Returns `None` if `doc_id` is past the end of
    /// the offset table.
    pub fn get_line_bytes(&self, doc_id: u32) -> Option<&[u8]> {
        let i = doc_id as usize;
        let start = *self.record_offsets.get(i)? as usize;
        let end = self
            .record_offsets
            .get(i + 1)
            .copied()
            .unwrap_or(self.mmap.len() as u64) as usize;
        let mut slice = &self.mmap[start..end];
        if slice.last() == Some(&b'\n') {
            slice = &slice[..slice.len() - 1];
        }
        Some(slice)
    }

    /// O(1) lookup: doc_id for a given record UUID, or `None`.
    pub fn doc_id_for_uuid(&self, id: &Uuid) -> Option<u32> {
        self.uuid_to_doc_id.get(id).copied()
    }

    /// Iterate (uuid, doc_id) pairs. Used by callers that need to
    /// populate a collection-level reverse index at recovery time.
    pub fn iter_uuid_doc_ids(&self) -> impl Iterator<Item = (Uuid, u32)> + '_ {
        self.uuid_to_doc_id.iter().map(|(u, d)| (*u, *d))
    }
}

/// Parse just the `__record_id__` UUID out of a JSONL line without
/// deserializing the whole record. Records are stored with
/// `FieldValue`'s tagged-enum serde form, so the id field reads as
/// `"__record_id__":{"String":"<uuid>"}` — accept either that or a
/// bare string for forward-compat.
fn extract_record_id(line: &[u8]) -> Result<Uuid> {
    let v: serde_json::Value = serde_json::from_slice(line)
        .map_err(|e| VedaError::Storage(format!("extract_record_id: {e}")))?;
    let id_str = v
        .get("__record_id__")
        .and_then(|tagged| tagged.get("String"))
        .and_then(|s| s.as_str())
        .or_else(|| v.get("__record_id__").and_then(|s| s.as_str()))
        .ok_or_else(|| VedaError::Storage("extract_record_id: __record_id__ missing".into()))?;
    id_str
        .parse()
        .map_err(|e| VedaError::Storage(format!("extract_record_id: invalid uuid: {e}")))
}

/// Tag-only Header struct used to parse just the record-id field. Kept
/// here so unit tests can reach for it without going through the full
/// Record deserializer.
#[allow(dead_code)]
#[derive(Deserialize)]
struct Header {
    #[serde(rename = "__record_id__")]
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_builds_offsets_and_uuid_index() {
        let tmp = tempfile::tempdir().unwrap();
        let seg = tmp.path();
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let content = format!(
            "{{\"__record_id__\":{{\"String\":\"{id_a}\"}},\"x\":1}}\n{{\"__record_id__\":{{\"String\":\"{id_b}\"}},\"x\":2}}\n"
        );
        std::fs::write(seg.join(RECORDS_FILE), &content).unwrap();

        let store = JsonlStoredFields::open(seg, 2).unwrap();
        assert_eq!(store.doc_id_for_uuid(&id_a), Some(0));
        assert_eq!(store.doc_id_for_uuid(&id_b), Some(1));
        let line0 = std::str::from_utf8(store.get_line_bytes(0).unwrap()).unwrap();
        assert!(line0.contains(&id_a.to_string()));
        let line1 = std::str::from_utf8(store.get_line_bytes(1).unwrap()).unwrap();
        assert!(line1.contains(&id_b.to_string()));
    }

    #[test]
    fn missing_trailing_newline_still_indexes_last_row() {
        let tmp = tempfile::tempdir().unwrap();
        let seg = tmp.path();
        let id = Uuid::new_v4();
        let content = format!("{{\"__record_id__\":{{\"String\":\"{id}\"}}}}");
        std::fs::write(seg.join(RECORDS_FILE), content).unwrap();
        let store = JsonlStoredFields::open(seg, 1).unwrap();
        assert_eq!(store.doc_id_for_uuid(&id), Some(0));
    }
}
