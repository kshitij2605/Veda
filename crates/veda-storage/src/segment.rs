use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use veda_core::error::{Result, VedaError};
use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;
use veda_core::types::{FieldType, SegmentId};

use crate::bitmap_index::BitmapIndex;
use crate::delete_bitmap::DeleteBitmap;
use crate::bm25_index::Bm25Index;
use crate::inverted_index::InvertedIndex;
use crate::multivector_storage::{MultiVectorStorageReader, MultiVectorStorageWriter};
use crate::range_index::RangeIndex;
use crate::sparse_storage::{SparseVectorStorageReader, SparseVectorStorageWriter};
use crate::vector_storage::{VectorStorageReader, VectorStorageWriter};
use veda_core::sparse::SparseVec;

// ── Directory / file name conventions ──────────────────────────────────────

const VECTORS_DIR: &str = "vectors";
const SPARSE_DIR: &str = "sparse";
const MULTIVEC_DIR: &str = "multivec";
const SCALARS_DIR: &str = "scalars";
const DELETE_BITMAP_FILE: &str = "delete_bitmap.bin";
const METADATA_FILE: &str = "metadata.json";
const BLOB_REFS_FILE: &str = "blob_refs.json";
const RECORDS_FILE: &str = "records.jsonl";

fn vector_file(name: &str) -> String {
    format!("{name}.vec")
}

/// Parse just the `__record_id__` UUID out of a JSONL line without
/// deserializing the whole record. Used at SegmentReader::open() time to
/// build the uuid → doc_id index on a cold start.
///
/// Records are stored with `FieldValue`'s tagged-enum serde form, so the
/// id field reads as `"__record_id__":{"String":"<uuid>"}`. Use a
/// minimal `serde_json::Value` lookup rather than a full Record parse
/// to keep cold-start cheap.
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

fn sparse_file(name: &str) -> String {
    format!("{name}.spr")
}

fn multivec_file(name: &str) -> String {
    format!("{name}.mvec")
}

fn inverted_file(name: &str) -> String {
    format!("{name}.inv")
}

fn bitmap_file(name: &str) -> String {
    format!("{name}.bmp")
}

fn range_file(name: &str) -> String {
    format!("{name}.rng")
}

fn bm25_file(name: &str) -> String {
    format!("{name}.bm25")
}

// ── SegmentMetadata ────────────────────────────────────────────────────────

/// Storage tier for a sealed segment.
///
/// - `Hot` (default): full ANN index built in memory; lowest-latency search.
/// - `Warm`: vectors are mmap-resident but no ANN index is built — search
///   falls back to brute-force scan over the mmap. Trade a few ms of query
///   latency for the memory of the ANN graph.
/// - `Cold`: data is offloaded to remote object storage and not present
///   locally. Queries return an "in cold tier" sentinel until a background
///   job promotes the segment back. **Cold-tier promotion is not yet
///   wired through the segment manager**; see `documentation/tiered-storage.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SegmentTier {
    #[default]
    Hot,
    Warm,
    Cold,
}

/// Metadata stored alongside a segment, persisted as `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetadata {
    pub id: SegmentId,
    pub row_count: u64,
    pub schema_version: u32,
    pub created_at: i64,
    pub sealed: bool,
    /// Hot/Warm/Cold tier marker. Older segments without this field load
    /// as `Hot` (the serde default).
    #[serde(default)]
    pub tier: SegmentTier,
}

// ── Scalar index type selector ─────────────────────────────────────────────

/// Which scalar index type to use for a given field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarIndexKind {
    Inverted,
    Range,
    Bitmap,
}

fn scalar_index_kind(field_type: &FieldType) -> ScalarIndexKind {
    match field_type {
        FieldType::String | FieldType::Uuid | FieldType::Json => ScalarIndexKind::Inverted,
        FieldType::Bool => ScalarIndexKind::Bitmap,
        FieldType::Int32
        | FieldType::Int64
        | FieldType::Float32
        | FieldType::Float64
        | FieldType::Timestamp => ScalarIndexKind::Range,
        // Fallback for any other types — treat as inverted
        _ => ScalarIndexKind::Inverted,
    }
}

/// Extract an f64 from a FieldValue for range indexing.
fn field_value_to_f64(value: &FieldValue) -> Option<f64> {
    match value {
        FieldValue::Int32(v) => Some(*v as f64),
        FieldValue::Int64(v) => Some(*v as f64),
        FieldValue::Float32(v) => Some(*v as f64),
        FieldValue::Float64(v) => Some(*v),
        FieldValue::Timestamp(v) => Some(*v as f64),
        _ => None,
    }
}

/// Extract a string representation from a FieldValue for inverted/bitmap indexing.
fn field_value_to_string(value: &FieldValue) -> Option<String> {
    match value {
        FieldValue::String(s) => Some(s.clone()),
        FieldValue::Bool(b) => Some(b.to_string()),
        FieldValue::Uuid(u) => Some(u.to_string()),
        FieldValue::Json(j) => Some(j.to_string()),
        _ => None,
    }
}

// ── SegmentWriter ──────────────────────────────────────────────────────────

/// Writes records into a segment directory, building vector storage files and
/// scalar indexes. The segment is finalized by calling [`seal`].
pub struct SegmentWriter {
    dir: PathBuf,
    id: SegmentId,
    schema: CollectionSchema,
    row_count: u64,

    // Vector writers keyed by field name.
    vector_writers: HashMap<String, VectorStorageWriter>,

    // Sparse vector writers keyed by field name.
    sparse_writers: HashMap<String, SparseVectorStorageWriter>,

    // Multi-vector writers keyed by field name.
    multivec_writers: HashMap<String, MultiVectorStorageWriter>,

    // Scalar indexes keyed by field name.
    inverted_indexes: HashMap<String, InvertedIndex>,
    range_indexes: HashMap<String, RangeIndex>,
    bitmap_indexes: HashMap<String, BitmapIndex>,

    // BM25 full-text indexes keyed by field name (only for String fields
    // marked `bm25 = true` in the schema).
    bm25_indexes: HashMap<String, Bm25Index>,

    // JSONL writer for full record scalar data.
    record_writer: BufWriter<fs::File>,

    // Blob refs: doc_id -> { field_name -> blob_ref_uri }
    blob_refs: Vec<HashMap<String, String>>,
}

impl SegmentWriter {
    /// Create a new segment writer for the given schema.
    ///
    /// Creates the directory structure and initializes all storage writers
    /// and index builders based on the schema.
    pub fn create(dir: &Path, schema: &CollectionSchema) -> Result<Self> {
        let id = Uuid::new_v4();

        // Create directory structure.
        let vectors_dir = dir.join(VECTORS_DIR);
        let sparse_dir = dir.join(SPARSE_DIR);
        let scalars_dir = dir.join(SCALARS_DIR);
        fs::create_dir_all(&vectors_dir)?;
        fs::create_dir_all(&sparse_dir)?;
        fs::create_dir_all(&scalars_dir)?;

        // Initialize vector writers.
        let mut vector_writers = HashMap::new();
        for field in schema.vector_fields() {
            if let FieldType::Vector { dimension, .. } = field.field_type() {
                let path = vectors_dir.join(vector_file(field.name()));
                let writer = VectorStorageWriter::create(&path, *dimension)?;
                vector_writers.insert(field.name().to_owned(), writer);
            }
        }

        // Initialize sparse vector writers.
        let mut sparse_writers = HashMap::new();
        for field in schema.sparse_vector_fields() {
            sparse_writers.insert(field.name().to_owned(), SparseVectorStorageWriter::create());
        }

        // Initialize multi-vector writers.
        let mut multivec_writers = HashMap::new();
        for field in schema.multi_vector_fields() {
            multivec_writers
                .insert(field.name().to_owned(), MultiVectorStorageWriter::create());
        }
        let multivec_dir = dir.join(MULTIVEC_DIR);
        fs::create_dir_all(&multivec_dir)?;

        // Initialize scalar indexes.
        let mut inverted_indexes = HashMap::new();
        let mut range_indexes = HashMap::new();
        let mut bitmap_indexes = HashMap::new();
        let mut bm25_indexes = HashMap::new();

        for field in schema.scalar_fields() {
            match scalar_index_kind(field.field_type()) {
                ScalarIndexKind::Inverted => {
                    inverted_indexes.insert(field.name().to_owned(), InvertedIndex::new());
                }
                ScalarIndexKind::Range => {
                    range_indexes.insert(field.name().to_owned(), RangeIndex::new());
                }
                ScalarIndexKind::Bitmap => {
                    bitmap_indexes.insert(field.name().to_owned(), BitmapIndex::new());
                }
            }
            // BM25 is orthogonal to the inverted/range/bitmap routing: a
            // string field may want both a filterable Inverted index and
            // a full-text BM25 index. Track them in a separate map.
            if field.bm25() && matches!(field.field_type(), FieldType::String) {
                bm25_indexes.insert(field.name().to_owned(), Bm25Index::new());
            }
        }

        // Open JSONL file for record data.
        let record_file = fs::File::create(dir.join(RECORDS_FILE))?;
        let record_writer = BufWriter::new(record_file);

        Ok(Self {
            dir: dir.to_owned(),
            id,
            schema: schema.clone(),
            row_count: 0,
            vector_writers,
            sparse_writers,
            multivec_writers,
            inverted_indexes,
            range_indexes,
            bitmap_indexes,
            bm25_indexes,
            record_writer,
            blob_refs: Vec::new(),
        })
    }

    /// Insert a record into the segment. Returns the internal 0-indexed doc_id.
    ///
    /// The record's vector fields are appended to the corresponding vector
    /// storage files, scalar fields are indexed, and the full scalar payload
    /// is written to the JSONL file for later reconstruction.
    pub fn insert(&mut self, record: &Record) -> Result<u32> {
        let doc_id = self.row_count as u32;

        // 1. Write vector fields.
        for field_def in self.schema.vector_fields() {
            let name = field_def.name();
            match record.get(name) {
                Some(FieldValue::Vector(dense)) => {
                    let writer = self.vector_writers.get_mut(name).ok_or_else(|| {
                        VedaError::Storage(format!("no vector writer for field '{name}'"))
                    })?;
                    writer.append(dense.as_slice())?;
                }
                Some(_) => {
                    return Err(VedaError::Storage(format!(
                        "field '{name}' expected Vector, got different type"
                    )));
                }
                None => {
                    return Err(VedaError::Storage(format!(
                        "record missing required vector field '{name}'"
                    )));
                }
            }
        }

        // 1c. Write multi-vector fields. Missing values get an empty list.
        for field_def in self.schema.multi_vector_fields() {
            let name = field_def.name();
            let rows: Vec<Vec<f32>> = match record.get(name) {
                Some(FieldValue::MultiVector(rows)) => rows.clone(),
                Some(_) => {
                    return Err(VedaError::Storage(format!(
                        "field '{name}' expected MultiVector, got different type"
                    )));
                }
                None => Vec::new(),
            };
            if let Some(w) = self.multivec_writers.get_mut(name) {
                w.append(rows);
            }
        }

        // 1b. Write sparse vector fields. Missing values get an empty SparseVec.
        for field_def in self.schema.sparse_vector_fields() {
            let name = field_def.name();
            let sv = match record.get(name) {
                Some(FieldValue::SparseVector(sv)) => sv.clone(),
                Some(_) => {
                    return Err(VedaError::Storage(format!(
                        "field '{name}' expected SparseVector, got different type"
                    )));
                }
                None => SparseVec::new(Vec::new(), Vec::new())
                    .expect("empty sparse vector is always valid"),
            };
            if let Some(w) = self.sparse_writers.get_mut(name) {
                w.append(sv);
            }
        }

        // 2. Index scalar fields.
        for field_def in self.schema.scalar_fields() {
            let name = field_def.name();
            if let Some(value) = record.get(name) {
                // Array fields: index each element individually so equality
                // filters (`field = 'tag-a'`) match if any element equals.
                if let FieldValue::Array(elems) = value {
                    if let Some(idx) = self.inverted_indexes.get_mut(name) {
                        for elem in elems {
                            if let Some(s) = field_value_to_string(elem) {
                                idx.insert(doc_id, &s);
                            }
                        }
                    }
                    continue;
                }
                match scalar_index_kind(field_def.field_type()) {
                    ScalarIndexKind::Inverted => {
                        if let Some(s) = field_value_to_string(value)
                            && let Some(idx) = self.inverted_indexes.get_mut(name)
                        {
                            idx.insert(doc_id, &s);
                        }
                    }
                    ScalarIndexKind::Range => {
                        if let Some(v) = field_value_to_f64(value)
                            && let Some(idx) = self.range_indexes.get_mut(name)
                        {
                            idx.insert(doc_id, v);
                        }
                    }
                    ScalarIndexKind::Bitmap => {
                        if let Some(s) = field_value_to_string(value)
                            && let Some(idx) = self.bitmap_indexes.get_mut(name)
                        {
                            idx.insert(doc_id, &s);
                        }
                    }
                }
                // BM25 — runs alongside whichever scalar index is selected.
                if let Some(bm25) = self.bm25_indexes.get_mut(name)
                    && let FieldValue::String(s) = value
                {
                    bm25.insert(doc_id, s);
                }
            }
        }

        // 3. Collect blob refs.
        let mut doc_blob_refs = HashMap::new();
        for field_def in self.schema.blob_fields() {
            let name = field_def.name();
            if let Some(FieldValue::BlobRef(uri)) = record.get(name) {
                doc_blob_refs.insert(name.to_owned(), uri.clone());
            }
        }
        self.blob_refs.push(doc_blob_refs);

        // 4. Write scalar field data as a JSONL line for record reconstruction.
        //    We store the record id and all non-vector, non-blob fields.
        let mut scalar_data: HashMap<String, FieldValue> = HashMap::new();
        // Store the record id so we can reconstruct it later.
        scalar_data.insert(
            "__record_id__".to_owned(),
            FieldValue::String(record.id().to_string()),
        );
        for field_def in self.schema.scalar_fields() {
            let name = field_def.name();
            if let Some(value) = record.get(name) {
                scalar_data.insert(name.to_owned(), value.clone());
            }
        }
        // Also store blob refs in the JSONL for full reconstruction.
        for field_def in self.schema.blob_fields() {
            let name = field_def.name();
            if let Some(value) = record.get(name) {
                scalar_data.insert(name.to_owned(), value.clone());
            }
        }
        let line = serde_json::to_string(&scalar_data)?;
        writeln!(self.record_writer, "{line}")?;

        self.row_count += 1;
        Ok(doc_id)
    }

    /// Return the number of records inserted so far.
    pub fn count(&self) -> u64 {
        self.row_count
    }

    /// Seal the segment: flush all data, build indexes, and write metadata.
    ///
    /// Consumes the writer. After sealing, the segment directory is immutable
    /// and can be opened with [`SegmentReader`].
    pub fn seal(mut self) -> Result<SegmentMetadata> {
        // 1. Close all vector writers (finalizes headers).
        for (_, writer) in self.vector_writers {
            writer.close()?;
        }

        // 1b. Close all sparse vector writers.
        let sparse_dir = self.dir.join(SPARSE_DIR);
        for (name, writer) in self.sparse_writers {
            writer.close(&sparse_dir.join(sparse_file(&name)))?;
        }

        // 1c. Close all multi-vector writers.
        let multivec_dir = self.dir.join(MULTIVEC_DIR);
        for (name, writer) in self.multivec_writers {
            writer.close(&multivec_dir.join(multivec_file(&name)))?;
        }

        // 2. Flush JSONL writer.
        self.record_writer.flush()?;

        // 3. Build and save range indexes.
        let scalars_dir = self.dir.join(SCALARS_DIR);
        for (name, mut idx) in self.range_indexes {
            idx.build();
            idx.save(&scalars_dir.join(range_file(&name)))?;
        }

        // 4. Save inverted indexes.
        for (name, idx) in &self.inverted_indexes {
            idx.save(&scalars_dir.join(inverted_file(name)))?;
        }

        // 5. Save bitmap indexes.
        for (name, idx) in &self.bitmap_indexes {
            idx.save(&scalars_dir.join(bitmap_file(name)))?;
        }

        // 5b. Save BM25 indexes (parallel to the scalar indexes; a
        //     string field can have both).
        for (name, idx) in &self.bm25_indexes {
            idx.save(&scalars_dir.join(bm25_file(name)))?;
        }

        // 6. Save empty delete bitmap.
        let delete_bitmap = DeleteBitmap::new();
        delete_bitmap.save(&self.dir.join(DELETE_BITMAP_FILE))?;

        // 7. Save blob refs.
        let blob_json = serde_json::to_vec(&self.blob_refs)?;
        fs::write(self.dir.join(BLOB_REFS_FILE), blob_json)?;

        // 8. Write metadata.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let metadata = SegmentMetadata {
            id: self.id,
            row_count: self.row_count,
            schema_version: 1,
            created_at: now,
            sealed: true,
            tier: SegmentTier::Hot,
        };
        let meta_json = serde_json::to_vec_pretty(&metadata)?;
        fs::write(self.dir.join(METADATA_FILE), meta_json)?;

        // 9. Save schema for the reader.
        let schema_json = serde_json::to_vec(&self.schema)?;
        fs::write(self.dir.join("schema.json"), schema_json)?;

        Ok(metadata)
    }
}

// ── SegmentReader ──────────────────────────────────────────────────────────

/// Reads records from a sealed, immutable segment directory.
///
/// Provides access to vector data (via mmap), scalar indexes, the delete
/// bitmap, and can reconstruct full [`Record`] objects from stored data.
pub struct SegmentReader {
    dir: PathBuf,
    metadata: SegmentMetadata,
    schema: CollectionSchema,

    // Vector readers keyed by field name.
    vector_readers: HashMap<String, VectorStorageReader>,

    // Sparse vector readers keyed by field name.
    sparse_readers: HashMap<String, SparseVectorStorageReader>,

    // Multi-vector readers keyed by field name.
    multivec_readers: HashMap<String, MultiVectorStorageReader>,

    // Scalar indexes keyed by field name.
    inverted_indexes: HashMap<String, InvertedIndex>,
    range_indexes: HashMap<String, RangeIndex>,
    bitmap_indexes: HashMap<String, BitmapIndex>,

    // BM25 full-text indexes keyed by field name. Loaded from disk if
    // the corresponding `.bm25` file exists, otherwise the map is empty.
    bm25_indexes: HashMap<String, Bm25Index>,

    // Delete bitmap.
    delete_bitmap: DeleteBitmap,

    // mmap'd records.jsonl + per-row byte offsets so `get_record(doc_id)` is
    // a single slice lookup + JSON parse instead of opening the file and
    // walking `lines().nth(doc_id)` every time. `record_offsets[i]` is the
    // start byte of row i and `record_offsets[i+1]` is the start byte of
    // row i+1 (or the file length for the last row).
    records_mmap: Arc<Mmap>,
    record_offsets: Vec<u64>,

    // UUID → doc_id for O(1) lookup by record id. Built once at open time.
    uuid_to_doc_id: HashMap<Uuid, u32>,
}

impl SegmentReader {
    /// Open a sealed segment directory for reading.
    ///
    /// Loads metadata, schema, all vector readers, scalar indexes, and
    /// the delete bitmap.
    pub fn open(dir: &Path) -> Result<Self> {
        // Load metadata.
        let meta_bytes = fs::read(dir.join(METADATA_FILE))?;
        let metadata: SegmentMetadata = serde_json::from_slice(&meta_bytes)?;

        if !metadata.sealed {
            return Err(VedaError::Storage(
                "cannot open an unsealed segment for reading".into(),
            ));
        }

        // Load schema.
        let schema_bytes = fs::read(dir.join("schema.json"))?;
        let schema: CollectionSchema = serde_json::from_slice(&schema_bytes)?;

        // Open vector readers.
        let vectors_dir = dir.join(VECTORS_DIR);
        let mut vector_readers = HashMap::new();
        for field in schema.vector_fields() {
            let path = vectors_dir.join(vector_file(field.name()));
            let reader = VectorStorageReader::open(&path)?;
            vector_readers.insert(field.name().to_owned(), reader);
        }

        // Open sparse vector readers. Sparse dir may not exist on segments
        // sealed before sparse support was added.
        let sparse_dir = dir.join(SPARSE_DIR);
        let mut sparse_readers = HashMap::new();
        for field in schema.sparse_vector_fields() {
            let path = sparse_dir.join(sparse_file(field.name()));
            if path.exists() {
                let reader = SparseVectorStorageReader::open(&path)?;
                sparse_readers.insert(field.name().to_owned(), reader);
            }
        }

        // Open multi-vector readers.
        let multivec_dir = dir.join(MULTIVEC_DIR);
        let mut multivec_readers = HashMap::new();
        for field in schema.multi_vector_fields() {
            let path = multivec_dir.join(multivec_file(field.name()));
            if path.exists() {
                let reader = MultiVectorStorageReader::open(&path)?;
                multivec_readers.insert(field.name().to_owned(), reader);
            }
        }

        // Load scalar indexes.
        let scalars_dir = dir.join(SCALARS_DIR);
        let mut inverted_indexes = HashMap::new();
        let mut range_indexes = HashMap::new();
        let mut bitmap_indexes = HashMap::new();

        for field in schema.scalar_fields() {
            match scalar_index_kind(field.field_type()) {
                ScalarIndexKind::Inverted => {
                    let path = scalars_dir.join(inverted_file(field.name()));
                    if path.exists() {
                        inverted_indexes
                            .insert(field.name().to_owned(), InvertedIndex::load(&path)?);
                    }
                }
                ScalarIndexKind::Range => {
                    let path = scalars_dir.join(range_file(field.name()));
                    if path.exists() {
                        range_indexes.insert(field.name().to_owned(), RangeIndex::load(&path)?);
                    }
                }
                ScalarIndexKind::Bitmap => {
                    let path = scalars_dir.join(bitmap_file(field.name()));
                    if path.exists() {
                        bitmap_indexes
                            .insert(field.name().to_owned(), BitmapIndex::load(&path)?);
                    }
                }
            }
        }

        // Load BM25 indexes (one per field that has the `bm25 = true`
        // schema flag). Tolerate missing files so segments built before
        // BM25 support load cleanly.
        let mut bm25_indexes = HashMap::new();
        for field in schema.scalar_fields() {
            if field.bm25() && matches!(field.field_type(), FieldType::String) {
                let path = scalars_dir.join(bm25_file(field.name()));
                if path.exists() {
                    bm25_indexes
                        .insert(field.name().to_owned(), Bm25Index::load(&path)?);
                }
            }
        }

        // Load delete bitmap.
        let delete_bitmap_path = dir.join(DELETE_BITMAP_FILE);
        let delete_bitmap = if delete_bitmap_path.exists() {
            DeleteBitmap::load(&delete_bitmap_path)?
        } else {
            DeleteBitmap::new()
        };

        // mmap records.jsonl + build per-row offset table in one pass. This
        // turns get_record(doc_id) from O(N) (open file + lines().nth()) into
        // O(1) (slice lookup + serde parse), and gives us a UUID → doc_id
        // index for free.
        let records_path = dir.join(RECORDS_FILE);
        let records_file = fs::File::open(&records_path).map_err(|e| {
            VedaError::Storage(format!("failed to open {records_path:?}: {e}"))
        })?;
        let records_mmap = unsafe { Mmap::map(&records_file)? };
        let mut record_offsets: Vec<u64> = Vec::with_capacity(metadata.row_count as usize + 1);
        let mut uuid_to_doc_id: HashMap<Uuid, u32> =
            HashMap::with_capacity(metadata.row_count as usize);
        record_offsets.push(0);
        let bytes: &[u8] = &records_mmap[..];
        let mut line_start: usize = 0;
        let mut doc_id: u32 = 0;
        for (i, b) in bytes.iter().enumerate() {
            if *b == b'\n' {
                let line = &bytes[line_start..i];
                if !line.is_empty()
                    && let Ok(id) = extract_record_id(line)
                {
                    uuid_to_doc_id.insert(id, doc_id);
                }
                doc_id += 1;
                record_offsets.push((i + 1) as u64);
                line_start = i + 1;
            }
        }
        // Handle a final line without trailing newline.
        if line_start < bytes.len() {
            let line = &bytes[line_start..];
            if let Ok(id) = extract_record_id(line) {
                uuid_to_doc_id.insert(id, doc_id);
            }
            record_offsets.push(bytes.len() as u64);
        }

        Ok(Self {
            dir: dir.to_owned(),
            metadata,
            schema,
            vector_readers,
            sparse_readers,
            multivec_readers,
            inverted_indexes,
            range_indexes,
            bitmap_indexes,
            bm25_indexes,
            delete_bitmap,
            records_mmap: Arc::new(records_mmap),
            record_offsets,
            uuid_to_doc_id,
        })
    }

    /// Access the BM25 index for a given field name, if one was built.
    pub fn bm25_index(&self, field: &str) -> Option<&Bm25Index> {
        self.bm25_indexes.get(field)
    }

    /// O(1) lookup: doc_id by record UUID, or `None` if not in this segment.
    pub fn doc_id_for_uuid(&self, id: &Uuid) -> Option<u32> {
        self.uuid_to_doc_id.get(id).copied()
    }

    /// Iterate the (uuid, doc_id) pairs in this segment — used for building
    /// per-collection reverse indexes at recovery time.
    pub fn iter_uuid_doc_ids(&self) -> impl Iterator<Item = (Uuid, u32)> + '_ {
        self.uuid_to_doc_id.iter().map(|(u, d)| (*u, *d))
    }

    /// Return the raw bytes of the JSONL line for the given doc_id, or
    /// `None` if `doc_id` is out of range. Strips the trailing newline.
    fn record_line_bytes(&self, doc_id: u32) -> Option<&[u8]> {
        let i = doc_id as usize;
        let start = *self.record_offsets.get(i)? as usize;
        let end = self
            .record_offsets
            .get(i + 1)
            .copied()
            .unwrap_or(self.records_mmap.len() as u64) as usize;
        let mut slice = &self.records_mmap[start..end];
        if slice.last() == Some(&b'\n') {
            slice = &slice[..slice.len() - 1];
        }
        Some(slice)
    }

    /// Get segment metadata.
    pub fn metadata(&self) -> &SegmentMetadata {
        &self.metadata
    }

    /// Get a vector by field name and doc_id.
    pub fn get_vector(&self, field: &str, doc_id: u32) -> Result<&[f32]> {
        let reader = self.vector_readers.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("vector field '{field}' not found in segment"))
        })?;
        reader.get(doc_id)
    }

    /// Get a sparse vector by field name and doc_id.
    pub fn get_sparse_vector(&self, field: &str, doc_id: u32) -> Result<&SparseVec> {
        let reader = self.sparse_readers.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("sparse field '{field}' not found in segment"))
        })?;
        reader.get(doc_id)
    }

    /// Iterate `(doc_id, &SparseVec)` for a sparse field, or `None` if the
    /// field has no sparse reader (e.g., legacy segment).
    pub fn iter_sparse(
        &self,
        field: &str,
    ) -> Option<impl Iterator<Item = (u32, &SparseVec)>> {
        self.sparse_readers.get(field).map(|r| r.iter())
    }

    /// Get a multi-vector by field name and doc_id.
    pub fn get_multi_vector(&self, field: &str, doc_id: u32) -> Result<&Vec<Vec<f32>>> {
        let reader = self.multivec_readers.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("multi-vector field '{field}' not found"))
        })?;
        reader.get(doc_id)
    }

    /// Iterate `(doc_id, &Vec<Vec<f32>>)` for a multi-vector field.
    pub fn iter_multi_vector(
        &self,
        field: &str,
    ) -> Option<impl Iterator<Item = (u32, &Vec<Vec<f32>>)>> {
        self.multivec_readers.get(field).map(|r| r.iter())
    }

    /// Reconstruct a full [`Record`] for the given doc_id from stored JSONL data.
    ///
    /// The record includes scalar fields, blob refs, and vector fields.
    /// O(1) — looks up `doc_id`'s byte range in the offset table built at
    /// `open()` time and parses that slice of the mmap.
    pub fn get_record(&self, doc_id: u32) -> Result<Record> {
        if doc_id as u64 >= self.metadata.row_count {
            return Err(VedaError::Storage(format!(
                "doc_id {doc_id} out of range (row_count={})",
                self.metadata.row_count
            )));
        }

        let line = self.record_line_bytes(doc_id).ok_or_else(|| {
            VedaError::Storage(format!("missing JSONL line for doc_id {doc_id}"))
        })?;

        let scalar_data: HashMap<String, FieldValue> = serde_json::from_slice(line)?;

        // Reconstruct the record.
        let record_id_str = match scalar_data.get("__record_id__") {
            Some(FieldValue::String(s)) => s.clone(),
            _ => {
                return Err(VedaError::Storage(
                    "missing __record_id__ in stored data".into(),
                ))
            }
        };
        let record_id: uuid::Uuid = record_id_str
            .parse()
            .map_err(|e| VedaError::Storage(format!("invalid record id: {e}")))?;

        let mut record = Record::with_id(record_id);

        // Restore scalar and blob fields.
        for (key, value) in &scalar_data {
            if key == "__record_id__" {
                continue;
            }
            record.set(key, value.clone());
        }

        // Restore vector fields.
        for field_def in self.schema.vector_fields() {
            let name = field_def.name();
            if let Some(reader) = self.vector_readers.get(name) {
                let slice = reader.get(doc_id)?;
                let dense = veda_core::vector::DenseVector::new(slice.to_vec());
                record.set(name, FieldValue::Vector(dense));
            }
        }

        // Restore sparse vector fields (skip empty placeholders).
        for field_def in self.schema.sparse_vector_fields() {
            let name = field_def.name();
            if let Some(reader) = self.sparse_readers.get(name)
                && let Ok(sv) = reader.get(doc_id)
                && sv.nnz() > 0
            {
                record.set(name, FieldValue::SparseVector(sv.clone()));
            }
        }

        // Restore multi-vector fields.
        for field_def in self.schema.multi_vector_fields() {
            let name = field_def.name();
            if let Some(reader) = self.multivec_readers.get(name)
                && let Ok(rows) = reader.get(doc_id)
                && !rows.is_empty()
            {
                record.set(name, FieldValue::MultiVector(rows.clone()));
            }
        }

        Ok(record)
    }

    /// Get a reference to the delete bitmap.
    pub fn delete_bitmap(&self) -> &DeleteBitmap {
        &self.delete_bitmap
    }

    /// Get a mutable reference to the delete bitmap.
    pub fn delete_bitmap_mut(&mut self) -> &mut DeleteBitmap {
        &mut self.delete_bitmap
    }

    /// Return the segment directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Get an inverted index by scalar field name.
    pub fn inverted_index(&self, field: &str) -> Option<&InvertedIndex> {
        self.inverted_indexes.get(field)
    }

    /// Get a range index by scalar field name.
    pub fn range_index(&self, field: &str) -> Option<&RangeIndex> {
        self.range_indexes.get(field)
    }

    /// Get a bitmap index by scalar field name.
    pub fn bitmap_index(&self, field: &str) -> Option<&BitmapIndex> {
        self.bitmap_indexes.get(field)
    }

    /// Get the collection schema.
    pub fn schema(&self) -> &CollectionSchema {
        &self.schema
    }

    /// Return the total row count (including deleted rows).
    pub fn row_count(&self) -> u64 {
        self.metadata.row_count
    }

    /// Return the active row count (total minus deleted).
    pub fn active_row_count(&self) -> u64 {
        self.metadata.row_count - self.delete_bitmap.deleted_count() as u64
    }
}
