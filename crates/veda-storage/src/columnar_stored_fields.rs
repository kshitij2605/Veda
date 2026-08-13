//! Columnar layout for a segment's stored scalar / blob data.
//!
//! Companion to [`crate::stored_fields::JsonlStoredFields`]. Writes one
//! file per scalar field plus a UUID column; reads via mmap. Fixed-
//! width values are contiguous slabs (`<field>.col`); variable-width
//! values get a parallel `<field>.idx` of `u64` offsets. A per-field
//! `<field>.meta` carries the on-disk type tag and an optional null
//! bitmap.
//!
//! See `documentation/columnar-stored-fields.md` for the design and
//! the migration plan that wires this into `SegmentWriter` /
//! `SegmentReader` behind a format marker.
//!
//! Supported value types (Round-4 cut):
//! - `String` (variable-width, UTF-8)
//! - `Int32`, `Int64` (fixed 4 / 8 bytes, little-endian)
//! - `Float32`, `Float64` (fixed 4 / 8 bytes, IEEE 754)
//! - `Bool` (1 byte)
//! - `Timestamp` (8 bytes, alias of Int64)
//! - `Uuid` (16 bytes, big-endian)
//! - `BlobRef` (variable-width UTF-8 — same encoding as String)
//!
//! `Json`, `Array`, vector / sparse / multi-vector field types stay in
//! their existing dedicated storage; they are not part of this module's
//! responsibility. A writer that encounters a record with one of those
//! values for a scalar field will error and the caller should fall
//! back to JSONL.
//!
//! ## Compression
//!
//! When `ColumnarWriter::new_compressed` is used, each column's `.col`
//! payload is zstd-compressed before writing. The `.meta` file carries
//! a `compressed: true` flag so the reader decompresses on open. Old
//! segments with `compressed: false` (or the field absent) load via the
//! original mmap path — backward-compatible with no migration required.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use veda_core::error::{Result, VedaError};
use veda_core::record::FieldValue;

const COLUMNS_DIR: &str = "columns";
const ID_COL: &str = "__record_id__.col";
const FORMAT_MARKER: &str = "storage_format.txt";

/// Default zstd compression level (1–22; 3 is the library default).
const ZSTD_LEVEL: i32 = 3;

/// On-disk type tag stored in each column's `.meta` so the reader
/// knows how to decode bytes without re-checking the segment schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    String,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Uuid,
    BlobRef,
    Timestamp,
}

impl ColumnType {
    /// Byte size of one cell for fixed-width types. `None` for
    /// variable-width types (String, BlobRef).
    fn fixed_width(self) -> Option<usize> {
        match self {
            Self::Int32 | Self::Float32 => Some(4),
            Self::Int64 | Self::Float64 | Self::Timestamp => Some(8),
            Self::Bool => Some(1),
            Self::Uuid => Some(16),
            Self::String | Self::BlobRef => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnMeta {
    column_type: ColumnType,
    row_count: u64,
    /// Bitmap of present (non-null) rows, packed LSB-first. Empty
    /// vector means "all rows present" so trivial columns don't need
    /// to allocate a bitmap.
    #[serde(default)]
    null_bitmap: Vec<u8>,
    /// `true` when the `.col` file is zstd-compressed. Old segments
    /// that predate compression will deserialize this as `false` via
    /// the serde default.
    #[serde(default)]
    compressed: bool,
}

/// Writer for the columnar format. One instance covers all scalar /
/// blob-ref columns plus the UUID column for a segment.
pub struct ColumnarWriter {
    dir: PathBuf,
    columns: HashMap<String, ColumnAccumulator>,
    ids: Vec<Uuid>,
    /// When true, `seal` writes zstd-compressed `.col` files.
    compress: bool,
}

struct ColumnAccumulator {
    column_type: ColumnType,
    /// Fixed-width: cell bytes concatenated. Variable-width: bytes
    /// concatenated; per-row start offsets in `offsets`.
    bytes: Vec<u8>,
    offsets: Vec<u64>,
    null_bits: Vec<bool>,
}

impl ColumnAccumulator {
    fn new(column_type: ColumnType) -> Self {
        Self {
            column_type,
            bytes: Vec::new(),
            offsets: Vec::new(),
            null_bits: Vec::new(),
        }
    }

    fn push(&mut self, value: Option<&FieldValue>) -> Result<()> {
        let value = match value {
            Some(v) => v,
            None => {
                self.push_null();
                return Ok(());
            }
        };
        match (self.column_type, value) {
            (ColumnType::String, FieldValue::String(s))
            | (ColumnType::BlobRef, FieldValue::BlobRef(s)) => {
                self.offsets.push(self.bytes.len() as u64);
                self.bytes.extend_from_slice(s.as_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Int32, FieldValue::Int32(v)) => {
                self.bytes.extend_from_slice(&v.to_le_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Int64, FieldValue::Int64(v))
            | (ColumnType::Timestamp, FieldValue::Timestamp(v))
            | (ColumnType::Int64, FieldValue::Timestamp(v))
            | (ColumnType::Timestamp, FieldValue::Int64(v)) => {
                self.bytes.extend_from_slice(&v.to_le_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Float32, FieldValue::Float32(v)) => {
                self.bytes.extend_from_slice(&v.to_le_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Float64, FieldValue::Float64(v)) => {
                self.bytes.extend_from_slice(&v.to_le_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Bool, FieldValue::Bool(v)) => {
                self.bytes.push(if *v { 1 } else { 0 });
                self.null_bits.push(false);
            }
            (ColumnType::Uuid, FieldValue::Uuid(u)) => {
                self.bytes.extend_from_slice(u.as_bytes());
                self.null_bits.push(false);
            }
            (ColumnType::Uuid, FieldValue::String(s)) => {
                // Tolerate string-form UUIDs (the JSONL path stores
                // them that way for __record_id__).
                let u: Uuid = s
                    .parse()
                    .map_err(|e| VedaError::Storage(format!("columnar: uuid parse failed: {e}")))?;
                self.bytes.extend_from_slice(u.as_bytes());
                self.null_bits.push(false);
            }
            (ct, v) => {
                return Err(VedaError::Storage(format!(
                    "columnar: cannot store {v:?} in column of type {ct:?}"
                )));
            }
        }
        Ok(())
    }

    fn push_null(&mut self) {
        // Variable-width columns still need an offset entry so row N+1
        // computes correctly; the null bitmap is the source of truth.
        if self.column_type.fixed_width().is_none() {
            self.offsets.push(self.bytes.len() as u64);
        } else if let Some(w) = self.column_type.fixed_width() {
            self.bytes.extend(std::iter::repeat_n(0u8, w));
        }
        self.null_bits.push(true);
    }

    fn write(&self, dir: &Path, field: &str, compress: bool) -> Result<()> {
        // <field>.col — payload bytes, optionally zstd-compressed
        let col_path = dir.join(format!("{field}.col"));
        if compress {
            let compressed = zstd::encode_all(self.bytes.as_slice(), ZSTD_LEVEL)
                .map_err(|e| VedaError::Storage(format!("columnar: zstd compress {field}: {e}")))?;
            fs::write(&col_path, &compressed)?;
        } else {
            fs::write(&col_path, &self.bytes)?;
        }
        // <field>.idx — offsets for variable-width (always uncompressed;
        // it's metadata, not payload, and is tiny)
        if self.column_type.fixed_width().is_none() {
            let idx_path = dir.join(format!("{field}.idx"));
            // Append the end-of-bytes offset so reader doesn't need a
            // special case for the last row.
            let mut idx_buf: Vec<u8> = Vec::with_capacity((self.offsets.len() + 1) * 8);
            for off in &self.offsets {
                idx_buf.extend_from_slice(&off.to_le_bytes());
            }
            idx_buf.extend_from_slice(&(self.bytes.len() as u64).to_le_bytes());
            fs::write(&idx_path, &idx_buf)?;
        }
        // <field>.meta — type tag + null bitmap if any null + compression flag
        let any_null = self.null_bits.iter().any(|b| *b);
        let null_bitmap = if any_null {
            pack_bits(&self.null_bits)
        } else {
            Vec::new()
        };
        let meta = ColumnMeta {
            column_type: self.column_type,
            row_count: self.null_bits.len() as u64,
            null_bitmap,
            compressed: compress,
        };
        let meta_path = dir.join(format!("{field}.meta"));
        fs::write(&meta_path, serde_json::to_vec(&meta)?)?;
        Ok(())
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        if *b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn bit_at(bytes: &[u8], i: usize) -> bool {
    let byte = bytes.get(i / 8).copied().unwrap_or(0);
    (byte >> (i % 8)) & 1 == 1
}

impl ColumnarWriter {
    /// Start a new columnar writer for the segment at `segment_dir`.
    /// The writer creates `segment_dir/columns/` lazily on `seal`.
    /// Column data is written uncompressed (legacy behaviour).
    pub fn new(segment_dir: &Path) -> Self {
        Self {
            dir: segment_dir.to_owned(),
            columns: HashMap::new(),
            ids: Vec::new(),
            compress: false,
        }
    }

    /// Like `new`, but `seal` will zstd-compress each column's `.col`
    /// file before writing. The `.meta` sidecar records the flag so
    /// the reader can decompress transparently.
    pub fn new_compressed(segment_dir: &Path) -> Self {
        Self {
            dir: segment_dir.to_owned(),
            columns: HashMap::new(),
            ids: Vec::new(),
            compress: true,
        }
    }

    /// Add row N: the record's UUID + zero or more `(field, value)`
    /// pairs. Rows must be appended in increasing doc_id order.
    /// Fields not provided for this row are recorded as null.
    pub fn push(
        &mut self,
        id: Uuid,
        fields: &HashMap<String, (ColumnType, FieldValue)>,
    ) -> Result<()> {
        self.ids.push(id);
        // Backfill any new fields seen for the first time with leading nulls.
        let row_idx = self.ids.len() - 1;
        for (name, (ct, v)) in fields {
            let col = self
                .columns
                .entry(name.clone())
                .or_insert_with(|| ColumnAccumulator::new(*ct));
            while col.null_bits.len() < row_idx {
                col.push_null();
            }
            col.push(Some(v))?;
        }
        // Any column not touched for this row also gets a null entry.
        for (name, col) in self.columns.iter_mut() {
            if col.null_bits.len() <= row_idx && !fields.contains_key(name) {
                col.push_null();
            }
        }
        Ok(())
    }

    /// Push a row by reading the record's scalar / blob fields against
    /// the provided schema. Vector / sparse / multi-vector / json /
    /// array fields are skipped — they live in their own storage.
    /// Returns the doc_id assigned to this row.
    pub fn push_record(
        &mut self,
        id: uuid::Uuid,
        record: &veda_core::record::Record,
        schema: &veda_core::schema::CollectionSchema,
    ) -> Result<u32> {
        use veda_core::record::{
            EXPIRES_AT_FIELD, FieldValue, GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD,
            GUARDRAIL_SCORE_FIELD, INDEXED_AT_FIELD,
        };
        use veda_core::types::FieldType;

        let mut fields: HashMap<String, (ColumnType, FieldValue)> = HashMap::new();
        for field_def in schema.fields() {
            let name = field_def.name();
            let ft = field_def.field_type();
            let ct = match ft {
                FieldType::String => ColumnType::String,
                FieldType::Int32 => ColumnType::Int32,
                FieldType::Int64 => ColumnType::Int64,
                FieldType::Float32 => ColumnType::Float32,
                FieldType::Float64 => ColumnType::Float64,
                FieldType::Bool => ColumnType::Bool,
                FieldType::Timestamp => ColumnType::Timestamp,
                FieldType::Uuid => ColumnType::Uuid,
                FieldType::Blob => ColumnType::BlobRef,
                // Vector / sparse / multi-vector / json / array live elsewhere.
                _ => continue,
            };
            if let Some(v) = record.get(name) {
                if matches!(v, FieldValue::Null) {
                    continue;
                }
                fields.insert(name.to_owned(), (ct, v.clone()));
            }
        }
        // Persist the reserved TTL / retrieval-guardrail fields as columns
        // even when not declared in the schema.
        if let Some(FieldValue::Int64(ts)) = record.get(EXPIRES_AT_FIELD) {
            fields.insert(
                EXPIRES_AT_FIELD.to_owned(),
                (ColumnType::Int64, FieldValue::Int64(*ts)),
            );
        }
        if let Some(FieldValue::Bool(b)) = record.get(GUARDRAIL_FLAGGED_FIELD) {
            fields.insert(
                GUARDRAIL_FLAGGED_FIELD.to_owned(),
                (ColumnType::Bool, FieldValue::Bool(*b)),
            );
        }
        if let Some(FieldValue::Float64(s)) = record.get(GUARDRAIL_SCORE_FIELD) {
            fields.insert(
                GUARDRAIL_SCORE_FIELD.to_owned(),
                (ColumnType::Float64, FieldValue::Float64(*s)),
            );
        }
        if let Some(FieldValue::String(c)) = record.get(GUARDRAIL_CATEGORY_FIELD) {
            fields.insert(
                GUARDRAIL_CATEGORY_FIELD.to_owned(),
                (ColumnType::String, FieldValue::String(c.clone())),
            );
        }
        if let Some(FieldValue::Int64(ts)) = record.get(INDEXED_AT_FIELD) {
            fields.insert(
                INDEXED_AT_FIELD.to_owned(),
                (ColumnType::Int64, FieldValue::Int64(*ts)),
            );
        }
        let doc_id = self.ids.len() as u32;
        self.push(id, &fields)?;
        Ok(doc_id)
    }

    /// Finalize: write all column files + the UUID column + the format
    /// marker. After this the writer is consumed.
    pub fn seal(self) -> Result<()> {
        let columns_dir = self.dir.join(COLUMNS_DIR);
        fs::create_dir_all(&columns_dir)?;

        // Per-field columns
        for (name, col) in &self.columns {
            col.write(&columns_dir, name, self.compress)?;
        }
        // UUID column: 16 bytes per row, big-endian. Not compressed —
        // it's the hot lookup table for uuid→doc_id and is mmap'd directly.
        let mut id_buf = Vec::with_capacity(self.ids.len() * 16);
        for id in &self.ids {
            id_buf.extend_from_slice(id.as_bytes());
        }
        fs::write(columns_dir.join(ID_COL), &id_buf)?;

        // Marker file at the segment root so SegmentReader knows which
        // backend to construct.
        let mut f = fs::File::create(self.dir.join(FORMAT_MARKER))?;
        f.write_all(b"columnar")?;
        Ok(())
    }
}

/// Column data — either memory-mapped (uncompressed, legacy) or heap-
/// allocated after decompression. Both expose `&[u8]` via `as_slice`.
enum ColData {
    Mmap(Arc<Mmap>),
    Owned(Vec<u8>),
}

impl ColData {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_ref(),
            Self::Owned(v) => v.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

/// Reader for the columnar format. Constructed once per segment open.
pub struct ColumnarStoredFields {
    columns: HashMap<String, OpenColumn>,
    id_mmap: Arc<Mmap>,
    row_count: u64,
    uuid_to_doc_id: HashMap<Uuid, u32>,
}

struct OpenColumn {
    meta: ColumnMeta,
    /// Column payload — mmap'd for uncompressed columns, heap-allocated
    /// after zstd decompression for compressed ones.
    col_data: ColData,
    /// Cached offsets (variable-width). Empty for fixed-width.
    offsets: Vec<u64>,
}

impl ColumnarStoredFields {
    /// Detect whether `segment_dir` is in columnar format.
    pub fn is_columnar(segment_dir: &Path) -> bool {
        match fs::read_to_string(segment_dir.join(FORMAT_MARKER)) {
            Ok(s) => s.trim() == "columnar",
            Err(_) => false,
        }
    }

    pub fn open(segment_dir: &Path) -> Result<Self> {
        let columns_dir = segment_dir.join(COLUMNS_DIR);
        // ID column
        let id_path = columns_dir.join(ID_COL);
        let id_file = fs::File::open(&id_path)
            .map_err(|e| VedaError::Storage(format!("columnar: open id col {id_path:?}: {e}")))?;
        let id_mmap = unsafe { Mmap::map(&id_file)? };
        let row_count = (id_mmap.len() / 16) as u64;
        let mut uuid_to_doc_id = HashMap::with_capacity(row_count as usize);
        for i in 0..row_count as usize {
            let start = i * 16;
            let id = Uuid::from_slice(&id_mmap[start..start + 16])
                .map_err(|e| VedaError::Storage(format!("columnar: id slice: {e}")))?;
            uuid_to_doc_id.insert(id, i as u32);
        }

        // Discover .meta files
        let mut columns = HashMap::new();
        if columns_dir.exists() {
            for entry in fs::read_dir(&columns_dir)? {
                let entry = entry?;
                let path = entry.path();
                let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                let Some(field) = fname.strip_suffix(".meta") else {
                    continue;
                };
                if field == "__record_id__" {
                    continue;
                }
                let meta_bytes = fs::read(&path)?;
                let meta: ColumnMeta = serde_json::from_slice(&meta_bytes)
                    .map_err(|e| VedaError::Storage(format!("columnar: meta parse: {e}")))?;
                let col_path = columns_dir.join(format!("{field}.col"));

                // Load column payload: decompress into heap if compressed,
                // otherwise mmap for zero-copy access.
                let col_data = if meta.compressed {
                    let compressed_bytes = fs::read(&col_path)?;
                    let decompressed =
                        zstd::decode_all(compressed_bytes.as_slice()).map_err(|e| {
                            VedaError::Storage(format!("columnar: zstd decompress {field}: {e}"))
                        })?;
                    ColData::Owned(decompressed)
                } else {
                    let col_file = fs::File::open(&col_path)?;
                    let col_mmap = unsafe { Mmap::map(&col_file)? };
                    ColData::Mmap(Arc::new(col_mmap))
                };

                let mut offsets: Vec<u64> = Vec::new();
                if meta.column_type.fixed_width().is_none() {
                    let idx_path = columns_dir.join(format!("{field}.idx"));
                    let idx_bytes = fs::read(&idx_path)?;
                    let n = idx_bytes.len() / 8;
                    offsets.reserve(n);
                    for i in 0..n {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&idx_bytes[i * 8..(i + 1) * 8]);
                        offsets.push(u64::from_le_bytes(b));
                    }
                }

                columns.insert(
                    field.to_owned(),
                    OpenColumn {
                        meta,
                        col_data,
                        offsets,
                    },
                );
            }
        }

        Ok(Self {
            columns,
            id_mmap: Arc::new(id_mmap),
            row_count,
            uuid_to_doc_id,
        })
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn doc_id_for_uuid(&self, id: &Uuid) -> Option<u32> {
        self.uuid_to_doc_id.get(id).copied()
    }

    pub fn iter_uuid_doc_ids(&self) -> impl Iterator<Item = (Uuid, u32)> + '_ {
        self.uuid_to_doc_id.iter().map(|(u, d)| (*u, *d))
    }

    pub fn record_id(&self, doc_id: u32) -> Option<Uuid> {
        let i = doc_id as usize;
        if (i + 1) * 16 > self.id_mmap.len() {
            return None;
        }
        Uuid::from_slice(&self.id_mmap[i * 16..(i + 1) * 16]).ok()
    }

    /// Read one scalar field for a doc — the projection-friendly path.
    /// Returns `Ok(None)` for null cells or unknown fields.
    pub fn get_field(&self, field: &str, doc_id: u32) -> Result<Option<FieldValue>> {
        let Some(col) = self.columns.get(field) else {
            return Ok(None);
        };
        let i = doc_id as usize;
        if !col.meta.null_bitmap.is_empty() && bit_at(&col.meta.null_bitmap, i) {
            return Ok(None);
        }
        let data = col.col_data.as_slice();
        let v = match col.meta.column_type {
            ColumnType::Int32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[i * 4..(i + 1) * 4]);
                FieldValue::Int32(i32::from_le_bytes(b))
            }
            ColumnType::Int64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[i * 8..(i + 1) * 8]);
                FieldValue::Int64(i64::from_le_bytes(b))
            }
            ColumnType::Timestamp => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[i * 8..(i + 1) * 8]);
                FieldValue::Timestamp(i64::from_le_bytes(b))
            }
            ColumnType::Float32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[i * 4..(i + 1) * 4]);
                FieldValue::Float32(f32::from_le_bytes(b))
            }
            ColumnType::Float64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[i * 8..(i + 1) * 8]);
                FieldValue::Float64(f64::from_le_bytes(b))
            }
            ColumnType::Bool => FieldValue::Bool(data[i] != 0),
            ColumnType::Uuid => {
                let bytes = &data[i * 16..(i + 1) * 16];
                FieldValue::Uuid(
                    Uuid::from_slice(bytes)
                        .map_err(|e| VedaError::Storage(format!("uuid slice: {e}")))?,
                )
            }
            ColumnType::String | ColumnType::BlobRef => {
                let start = col.offsets.get(i).copied().ok_or_else(|| {
                    VedaError::Storage(format!("columnar: doc_id {doc_id} out of range"))
                })? as usize;
                let end = col
                    .offsets
                    .get(i + 1)
                    .copied()
                    .unwrap_or(col.col_data.len() as u64) as usize;
                let s = std::str::from_utf8(&data[start..end])
                    .map_err(|e| VedaError::Storage(format!("columnar utf8: {e}")))?
                    .to_owned();
                if matches!(col.meta.column_type, ColumnType::BlobRef) {
                    FieldValue::BlobRef(s)
                } else {
                    FieldValue::String(s)
                }
            }
        };
        Ok(Some(v))
    }

    /// Materialize the full scalar map for `doc_id`. Skips fields with
    /// null cells. The `__record_id__` is *not* part of the map (it's
    /// available via `record_id(doc_id)`).
    pub fn get_scalar_map(&self, doc_id: u32) -> Result<HashMap<String, FieldValue>> {
        let mut out = HashMap::with_capacity(self.columns.len());
        for name in self.columns.keys() {
            if let Some(v) = self.get_field(name, doc_id)? {
                out.insert(name.clone(), v);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn mk(field: &str, ct: ColumnType, v: FieldValue) -> HashMap<String, (ColumnType, FieldValue)> {
        let mut m = HashMap::new();
        m.insert(field.to_owned(), (ct, v));
        m
    }

    #[test]
    fn roundtrip_fixed_width_columns() {
        let tmp = tempdir().unwrap();
        let seg = tmp.path();
        let mut w = ColumnarWriter::new(seg);

        let id0 = Uuid::new_v4();
        let id1 = Uuid::new_v4();
        let mut row0 = HashMap::new();
        row0.insert(
            "chunk_id".to_owned(),
            (ColumnType::Int64, FieldValue::Int64(42)),
        );
        row0.insert(
            "score".to_owned(),
            (ColumnType::Float32, FieldValue::Float32(0.87)),
        );
        row0.insert(
            "active".to_owned(),
            (ColumnType::Bool, FieldValue::Bool(true)),
        );
        w.push(id0, &row0).unwrap();

        let mut row1 = HashMap::new();
        row1.insert(
            "chunk_id".to_owned(),
            (ColumnType::Int64, FieldValue::Int64(99)),
        );
        row1.insert(
            "score".to_owned(),
            (ColumnType::Float32, FieldValue::Float32(-1.5)),
        );
        row1.insert(
            "active".to_owned(),
            (ColumnType::Bool, FieldValue::Bool(false)),
        );
        w.push(id1, &row1).unwrap();

        w.seal().unwrap();
        assert!(ColumnarStoredFields::is_columnar(seg));

        let r = ColumnarStoredFields::open(seg).unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.doc_id_for_uuid(&id0), Some(0));
        assert_eq!(r.doc_id_for_uuid(&id1), Some(1));
        assert_eq!(r.record_id(0), Some(id0));
        assert_eq!(r.record_id(1), Some(id1));
        assert_eq!(
            r.get_field("chunk_id", 0).unwrap(),
            Some(FieldValue::Int64(42))
        );
        assert_eq!(
            r.get_field("score", 1).unwrap(),
            Some(FieldValue::Float32(-1.5))
        );
        assert_eq!(
            r.get_field("active", 0).unwrap(),
            Some(FieldValue::Bool(true))
        );
    }

    #[test]
    fn roundtrip_variable_width_strings() {
        let tmp = tempdir().unwrap();
        let seg = tmp.path();
        let mut w = ColumnarWriter::new(seg);
        let id = Uuid::new_v4();
        let row = mk(
            "content",
            ColumnType::String,
            FieldValue::String("ランサムウェア対策".to_owned()),
        );
        w.push(id, &row).unwrap();
        let mut row2 = HashMap::new();
        row2.insert(
            "content".to_owned(),
            (
                ColumnType::String,
                FieldValue::String("Adobe Acrobat".to_owned()),
            ),
        );
        w.push(Uuid::new_v4(), &row2).unwrap();
        w.seal().unwrap();

        let r = ColumnarStoredFields::open(seg).unwrap();
        assert_eq!(
            r.get_field("content", 0).unwrap(),
            Some(FieldValue::String("ランサムウェア対策".to_owned()))
        );
        assert_eq!(
            r.get_field("content", 1).unwrap(),
            Some(FieldValue::String("Adobe Acrobat".to_owned()))
        );
    }

    #[test]
    fn null_cells_round_trip() {
        let tmp = tempdir().unwrap();
        let seg = tmp.path();
        let mut w = ColumnarWriter::new(seg);
        // Row 0 has the field, row 1 omits it.
        let mut row0 = HashMap::new();
        row0.insert(
            "tag".to_owned(),
            (ColumnType::String, FieldValue::String("alpha".to_owned())),
        );
        w.push(Uuid::new_v4(), &row0).unwrap();
        w.push(Uuid::new_v4(), &HashMap::new()).unwrap();
        w.seal().unwrap();

        let r = ColumnarStoredFields::open(seg).unwrap();
        assert_eq!(
            r.get_field("tag", 0).unwrap(),
            Some(FieldValue::String("alpha".to_owned()))
        );
        assert_eq!(r.get_field("tag", 1).unwrap(), None);
    }

    #[test]
    fn scalar_map_omits_null_fields() {
        let tmp = tempdir().unwrap();
        let seg = tmp.path();
        let mut w = ColumnarWriter::new(seg);
        let mut row = HashMap::new();
        row.insert("a".to_owned(), (ColumnType::Int32, FieldValue::Int32(7)));
        row.insert(
            "b".to_owned(),
            (ColumnType::String, FieldValue::String("hi".to_owned())),
        );
        w.push(Uuid::new_v4(), &row).unwrap();
        w.push(Uuid::new_v4(), &HashMap::new()).unwrap();
        w.seal().unwrap();
        let r = ColumnarStoredFields::open(seg).unwrap();
        let m0 = r.get_scalar_map(0).unwrap();
        assert_eq!(m0.get("a"), Some(&FieldValue::Int32(7)));
        assert_eq!(m0.get("b"), Some(&FieldValue::String("hi".to_owned())));
        let m1 = r.get_scalar_map(1).unwrap();
        assert!(m1.is_empty());
    }

    /// Compressed columns round-trip: write N records with sizable text
    /// fields using `ColumnarWriter::new_compressed`, read them back —
    /// all values must be byte-identical to what was written.
    #[test]
    fn compressed_columns_roundtrip() {
        let tmp = tempdir().unwrap();
        let seg = tmp.path();
        let mut w = ColumnarWriter::new_compressed(seg);

        let n = 50usize;
        let mut ids = Vec::with_capacity(n);
        let mut expected_texts = Vec::with_capacity(n);
        let mut expected_ints = Vec::with_capacity(n);

        for i in 0..n {
            let id = Uuid::new_v4();
            ids.push(id);
            let text = format!(
                "The quick brown fox jumps over the lazy dog. Record number {i}. \
                 Lorem ipsum dolor sit amet consectetur adipiscing elit."
            );
            expected_texts.push(text.clone());
            expected_ints.push(i as i64);

            let mut fields = HashMap::new();
            fields.insert(
                "body".to_owned(),
                (ColumnType::String, FieldValue::String(text)),
            );
            fields.insert(
                "seq".to_owned(),
                (ColumnType::Int64, FieldValue::Int64(i as i64)),
            );
            w.push(id, &fields).unwrap();
        }
        w.seal().unwrap();

        let r = ColumnarStoredFields::open(seg).unwrap();
        assert_eq!(r.row_count(), n as u64);

        for i in 0..n {
            let doc_id = r.doc_id_for_uuid(&ids[i]).expect("uuid must resolve");
            assert_eq!(
                r.get_field("body", doc_id).unwrap(),
                Some(FieldValue::String(expected_texts[i].clone())),
                "body mismatch at row {i}"
            );
            assert_eq!(
                r.get_field("seq", doc_id).unwrap(),
                Some(FieldValue::Int64(expected_ints[i])),
                "seq mismatch at row {i}"
            );
        }
    }

    /// Compression shrinks repetitive text: the on-disk `.col` file for
    /// a highly-repetitive string column must be smaller than the raw
    /// uncompressed payload.
    #[test]
    fn compression_shrinks_repetitive_text() {
        // Write uncompressed
        let tmp_plain = tempdir().unwrap();
        let seg_plain = tmp_plain.path();
        // Write compressed
        let tmp_comp = tempdir().unwrap();
        let seg_comp = tmp_comp.path();

        let n = 200usize;
        let repetitive_text = "abcdefghijklmnopqrstuvwxyz ".repeat(40); // ~1 080 bytes per record

        let build = |seg: &Path, compress: bool| {
            let mut w = if compress {
                ColumnarWriter::new_compressed(seg)
            } else {
                ColumnarWriter::new(seg)
            };
            for _ in 0..n {
                let id = Uuid::new_v4();
                let mut fields = HashMap::new();
                fields.insert(
                    "body".to_owned(),
                    (
                        ColumnType::String,
                        FieldValue::String(repetitive_text.clone()),
                    ),
                );
                w.push(id, &fields).unwrap();
            }
            w.seal().unwrap();
        };

        build(seg_plain, false);
        build(seg_comp, true);

        let plain_col = seg_plain.join("columns").join("body.col");
        let comp_col = seg_comp.join("columns").join("body.col");

        let plain_size = fs::metadata(&plain_col).unwrap().len();
        let comp_size = fs::metadata(&comp_col).unwrap().len();

        // Expect at least 50% reduction on highly repetitive input.
        assert!(
            comp_size < plain_size / 2,
            "expected compressed ({comp_size} B) < plain/2 ({} B), but it wasn't",
            plain_size / 2
        );

        // Also verify the compressed segment still round-trips correctly.
        let r = ColumnarStoredFields::open(seg_comp).unwrap();
        assert_eq!(r.row_count(), n as u64);
        // Spot-check a few rows
        for doc_id in [0u32, 50, 199] {
            assert_eq!(
                r.get_field("body", doc_id).unwrap(),
                Some(FieldValue::String(repetitive_text.clone())),
                "round-trip failed at doc_id {doc_id}"
            );
        }
    }
}
