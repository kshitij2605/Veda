use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use roaring::RoaringBitmap;
use uuid::Uuid;
use veda_core::error::{Result, VedaError};
use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;
use veda_core::simd;
use veda_core::sparse::SparseVec;
use veda_core::types::{DistanceMetric, FieldType, SegmentId};

use crate::delete_bitmap::DeleteBitmap;
use crate::segment::{SegmentMetadata, SegmentWriter};

/// Default seal threshold: 512 MiB.
const DEFAULT_MAX_SIZE_BYTES: usize = 512 * 1024 * 1024;

/// Estimated overhead per record for scalar fields (bytes).
const SCALAR_OVERHEAD_BYTES: usize = 200;

/// The active, mutable segment that receives new writes.
///
/// All data lives in memory. Supports fast appends, brute-force vector search,
/// and sealing to flush to disk as a sealed segment.
pub struct GrowingSegment {
    id: SegmentId,
    schema: CollectionSchema,
    records: Vec<Record>,
    /// Per dense vector field: flattened `Vec<f32>` storage for fast scan.
    vectors: HashMap<String, Vec<Vec<f32>>>,
    /// Per sparse vector field: ordered list of SparseVec per doc_id.
    /// Records that omit a sparse field get an empty SparseVec slot.
    sparse_vectors: HashMap<String, Vec<SparseVec>>,
    /// Per multi-vector field: per doc_id a list of dense vectors.
    multi_vectors: HashMap<String, Vec<Vec<Vec<f32>>>>,
    /// Estimated memory usage in bytes.
    size_bytes: usize,
    /// Seal threshold — when `size_bytes` exceeds this, `should_seal()` returns true.
    max_size_bytes: usize,
    /// Soft-delete tracking.
    delete_bitmap: DeleteBitmap,
    /// UUID → doc_id index. Maintained on insert so lookups are O(1).
    uuid_to_doc_id: HashMap<Uuid, u32>,
    /// In-memory BM25 indexes for opt-in string fields. Maintained on
    /// insert so unsealed records are immediately searchable by BM25 —
    /// previously BM25 only ran against sealed segments, which meant
    /// freshly-ingested batches were invisible to text search until they
    /// hit the seal threshold.
    bm25_indexes: HashMap<String, crate::bm25_index::Bm25Index>,
}

impl GrowingSegment {
    /// Create a new growing segment with the default 512 MiB seal threshold.
    pub fn new(schema: CollectionSchema) -> Self {
        Self::with_max_size(schema, DEFAULT_MAX_SIZE_BYTES)
    }

    /// Create a new growing segment with a custom seal threshold.
    pub fn with_max_size(schema: CollectionSchema, max_size_bytes: usize) -> Self {
        let mut vectors = HashMap::new();
        for field in schema.vector_fields() {
            vectors.insert(field.name().to_owned(), Vec::new());
        }
        let mut sparse_vectors = HashMap::new();
        for field in schema.sparse_vector_fields() {
            sparse_vectors.insert(field.name().to_owned(), Vec::new());
        }
        let mut multi_vectors = HashMap::new();
        for field in schema.multi_vector_fields() {
            multi_vectors.insert(field.name().to_owned(), Vec::new());
        }
        let mut bm25_indexes = HashMap::new();
        for field in schema.scalar_fields() {
            if field.bm25() && matches!(field.field_type(), veda_core::types::FieldType::String) {
                let tok = field
                    .bm25_tokenizer()
                    .and_then(crate::bm25_index::TokenizerKind::from_name)
                    .unwrap_or_default();
                bm25_indexes.insert(
                    field.name().to_owned(),
                    crate::bm25_index::Bm25Index::with_tokenizer(tok),
                );
            }
        }

        Self {
            id: Uuid::new_v4(),
            schema,
            records: Vec::new(),
            vectors,
            sparse_vectors,
            multi_vectors,
            size_bytes: 0,
            max_size_bytes,
            delete_bitmap: DeleteBitmap::new(),
            uuid_to_doc_id: HashMap::new(),
            bm25_indexes,
        }
    }

    /// Access the BM25 index for a given field, if one was configured.
    /// Returns `None` for fields without `bm25 = true` or non-string fields.
    pub fn bm25_index(&self, field: &str) -> Option<&crate::bm25_index::Bm25Index> {
        self.bm25_indexes.get(field)
    }

    /// O(1) lookup: doc_id by record UUID, or `None` if not in this segment.
    pub fn doc_id_for_uuid(&self, id: &Uuid) -> Option<u32> {
        self.uuid_to_doc_id.get(id).copied()
    }

    /// Insert a record. Returns the 0-indexed doc_id.
    ///
    /// Returns `Err` if the segment has already exceeded `max_size_bytes`.
    pub fn insert(&mut self, record: Record) -> Result<u32> {
        if self.size_bytes >= self.max_size_bytes {
            return Err(VedaError::Storage(
                "growing segment is full (exceeds max_size_bytes)".into(),
            ));
        }

        let doc_id = self.records.len() as u32;
        let mut added_bytes = SCALAR_OVERHEAD_BYTES;

        // Extract and store dense vector fields.
        for field_def in self.schema.vector_fields() {
            let name = field_def.name();
            if let FieldType::Vector { dimension, .. } = field_def.field_type() {
                match record.get(name) {
                    Some(FieldValue::Vector(dense)) => {
                        if dense.dimension() != *dimension as usize {
                            return Err(VedaError::DimensionMismatch {
                                expected: *dimension,
                                actual: dense.dimension() as u32,
                            });
                        }
                        let vec_data = dense.as_slice().to_vec();
                        added_bytes += vec_data.len() * std::mem::size_of::<f32>();
                        if let Some(store) = self.vectors.get_mut(name) {
                            store.push(vec_data);
                        }
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
        }

        // Extract and store multi-vector fields. Missing values become an
        // empty list. Validates dimension and max_vectors against schema.
        for field_def in self.schema.multi_vector_fields() {
            let name = field_def.name();
            if let FieldType::MultiVector { dimension, max_vectors, .. } = field_def.field_type() {
                let rows = match record.get(name) {
                    Some(FieldValue::MultiVector(rows)) => {
                        if *max_vectors > 0 && rows.len() as u32 > *max_vectors {
                            return Err(VedaError::Storage(format!(
                                "multi-vector field '{name}' has {} rows but max_vectors={}",
                                rows.len(),
                                max_vectors
                            )));
                        }
                        for r in rows {
                            if r.len() as u32 != *dimension {
                                return Err(VedaError::DimensionMismatch {
                                    expected: *dimension,
                                    actual: r.len() as u32,
                                });
                            }
                        }
                        added_bytes += rows
                            .iter()
                            .map(|r| r.len() * std::mem::size_of::<f32>())
                            .sum::<usize>();
                        rows.clone()
                    }
                    Some(_) => {
                        return Err(VedaError::Storage(format!(
                            "field '{name}' expected MultiVector, got different type"
                        )));
                    }
                    None => Vec::new(),
                };
                if let Some(store) = self.multi_vectors.get_mut(name) {
                    store.push(rows);
                }
            }
        }

        // Extract and store sparse vector fields. Sparse fields are optional —
        // missing values become an empty SparseVec.
        for field_def in self.schema.sparse_vector_fields() {
            let name = field_def.name();
            if let FieldType::SparseVector { dimension, .. } = field_def.field_type() {
                let sv = match record.get(name) {
                    Some(FieldValue::SparseVector(sv)) => {
                        if sv.max_index() > *dimension {
                            return Err(VedaError::Storage(format!(
                                "sparse field '{name}' max index {} exceeds dimension {}",
                                sv.max_index(),
                                dimension
                            )));
                        }
                        added_bytes += sv.nnz()
                            * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>());
                        sv.clone()
                    }
                    Some(_) => {
                        return Err(VedaError::Storage(format!(
                            "field '{name}' expected SparseVector, got different type"
                        )));
                    }
                    None => SparseVec::new(Vec::new(), Vec::new()).expect("empty is valid"),
                };
                if let Some(store) = self.sparse_vectors.get_mut(name) {
                    store.push(sv);
                }
            }
        }

        // Populate BM25 indexes for any opt-in string fields on this record.
        if !self.bm25_indexes.is_empty() {
            for (name, idx) in self.bm25_indexes.iter_mut() {
                if let Some(veda_core::record::FieldValue::String(s)) = record.get(name) {
                    idx.insert(doc_id, s);
                }
            }
        }

        self.uuid_to_doc_id.insert(record.id(), doc_id);
        self.records.push(record);
        self.size_bytes += added_bytes;

        Ok(doc_id)
    }

    /// Check if the segment should be sealed (exceeds `max_size_bytes`).
    pub fn should_seal(&self) -> bool {
        self.size_bytes >= self.max_size_bytes
    }

    /// Number of records (including deleted).
    pub fn count(&self) -> u32 {
        self.records.len() as u32
    }

    /// Estimated memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get a record by doc_id.
    pub fn get_record(&self, doc_id: u32) -> Option<&Record> {
        self.records.get(doc_id as usize)
    }

    /// Get a vector by field name and doc_id.
    pub fn get_vector(&self, field: &str, doc_id: u32) -> Option<&[f32]> {
        self.vectors
            .get(field)
            .and_then(|vecs| vecs.get(doc_id as usize))
            .map(|v| v.as_slice())
    }

    /// Get a sparse vector by field name and doc_id.
    pub fn get_sparse_vector(&self, field: &str, doc_id: u32) -> Option<&SparseVec> {
        self.sparse_vectors
            .get(field)
            .and_then(|vecs| vecs.get(doc_id as usize))
    }

    /// Get a multi-vector by field name and doc_id.
    pub fn get_multi_vector(&self, field: &str, doc_id: u32) -> Option<&Vec<Vec<f32>>> {
        self.multi_vectors
            .get(field)
            .and_then(|vecs| vecs.get(doc_id as usize))
    }

    /// Brute-force MaxSim scan over a multi-vector field.
    pub fn search_multi_vector(
        &self,
        field: &str,
        query_rows: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>> {
        let vecs = self.multi_vectors.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("multi-vector field '{field}' not found"))
        })?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let per_row: fn(&[f32], &[f32]) -> f32 = match metric {
            DistanceMetric::Cosine => simd::cosine_distance,
            DistanceMetric::Euclidean => simd::euclidean_distance,
            DistanceMetric::DotProduct => |a, b| -simd::dot_product(a, b),
            _ => simd::cosine_distance,
        };
        let mut heap: BinaryHeap<OrdF32Entry> = BinaryHeap::new();
        for (doc_id, doc_rows) in vecs.iter().enumerate() {
            let doc_id = doc_id as u32;
            if self.delete_bitmap.is_deleted(doc_id) {
                continue;
            }
            if let Some(bm) = filter
                && !bm.contains(doc_id)
            {
                continue;
            }
            if doc_rows.is_empty() {
                continue;
            }
            let dist = simd::maxsim_distance(query_rows, doc_rows, per_row);
            if heap.len() < k {
                heap.push(OrdF32Entry { dist, doc_id });
            } else if let Some(top) = heap.peek()
                && dist < top.dist
            {
                heap.pop();
                heap.push(OrdF32Entry { dist, doc_id });
            }
        }
        let mut results: Vec<(u32, f32)> =
            heap.into_iter().map(|e| (e.doc_id, e.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Brute-force sparse scan against this growing segment.
    ///
    /// Skips deleted docs and docs not in `filter` (if given). Returns top-k
    /// `(doc_id, distance)` pairs sorted by distance ascending.
    pub fn search_sparse(
        &self,
        field: &str,
        query: &SparseVec,
        k: usize,
        metric: DistanceMetric,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>> {
        let vecs = self.sparse_vectors.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("sparse vector field '{field}' not found"))
        })?;

        if k == 0 {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<OrdF32Entry> = BinaryHeap::new();

        for (doc_id, sv) in vecs.iter().enumerate() {
            let doc_id = doc_id as u32;
            if self.delete_bitmap.is_deleted(doc_id) {
                continue;
            }
            if let Some(bm) = filter
                && !bm.contains(doc_id)
            {
                continue;
            }
            if sv.nnz() == 0 {
                continue;
            }
            let dist = match metric {
                DistanceMetric::Cosine => sv.cosine_distance(query),
                DistanceMetric::DotProduct => -sv.dot(query),
                _ => sv.cosine_distance(query),
            };
            if heap.len() < k {
                heap.push(OrdF32Entry { dist, doc_id });
            } else if let Some(top) = heap.peek()
                && dist < top.dist
            {
                heap.pop();
                heap.push(OrdF32Entry { dist, doc_id });
            }
        }

        let mut results: Vec<(u32, f32)> =
            heap.into_iter().map(|e| (e.doc_id, e.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Brute-force scan with optional Matryoshka truncation. When `truncate`
    /// is `Some(n)`, distances are computed using only the first n components
    /// of both query and stored vectors.
    pub fn search_truncated(
        &self,
        field: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        filter: Option<&RoaringBitmap>,
        truncate: Option<u32>,
    ) -> Result<Vec<(u32, f32)>> {
        let vecs = self.vectors.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("vector field '{field}' not found"))
        })?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let n = truncate.map(|t| t as usize).unwrap_or(query.len()).min(query.len());
        let distance_fn = select_distance_fn(metric);
        let mut heap: BinaryHeap<OrdF32Entry> = BinaryHeap::new();
        for (doc_id, vec) in vecs.iter().enumerate() {
            let doc_id = doc_id as u32;
            if self.delete_bitmap.is_deleted(doc_id) {
                continue;
            }
            if let Some(bm) = filter
                && !bm.contains(doc_id)
            {
                continue;
            }
            let limit = n.min(vec.len());
            let dist = distance_fn(&query[..limit], &vec[..limit]);
            if heap.len() < k {
                heap.push(OrdF32Entry { dist, doc_id });
            } else if let Some(top) = heap.peek()
                && dist < top.dist
            {
                heap.pop();
                heap.push(OrdF32Entry { dist, doc_id });
            }
        }
        let mut results: Vec<(u32, f32)> =
            heap.into_iter().map(|e| (e.doc_id, e.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Brute-force scan: find top-k nearest vectors for a given query on a specific field.
    ///
    /// If `filter` is provided, only doc_ids present in the bitmap are considered.
    /// Returns `Vec<(doc_id, distance)>` sorted by distance ascending.
    pub fn search(
        &self,
        field: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>> {
        let vecs = self.vectors.get(field).ok_or_else(|| {
            VedaError::FieldNotFound(format!("vector field '{field}' not found"))
        })?;

        if k == 0 {
            return Ok(Vec::new());
        }

        let distance_fn = select_distance_fn(metric);

        // Max-heap of (distance, doc_id) — we keep the k closest by evicting
        // the farthest when the heap exceeds size k.
        let mut heap: BinaryHeap<OrdF32Entry> = BinaryHeap::new();

        for (doc_id, vec) in vecs.iter().enumerate() {
            let doc_id = doc_id as u32;

            // Skip deleted.
            if self.delete_bitmap.is_deleted(doc_id) {
                continue;
            }

            // Skip if not in filter.
            if let Some(bm) = filter
                && !bm.contains(doc_id)
            {
                continue;
            }

            let dist = distance_fn(query, vec);

            if heap.len() < k {
                heap.push(OrdF32Entry { dist, doc_id });
            } else if let Some(top) = heap.peek()
                && dist < top.dist
            {
                heap.pop();
                heap.push(OrdF32Entry { dist, doc_id });
            }
        }

        // Drain heap into sorted vec (ascending distance).
        let mut results: Vec<(u32, f32)> =
            heap.into_iter().map(|e| (e.doc_id, e.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Seal: flush this growing segment to disk as a sealed segment.
    ///
    /// Consumes self, writes all non-deleted records to the given directory,
    /// and returns the resulting `SegmentMetadata`.
    pub fn seal(self, dir: &Path) -> Result<SegmentMetadata> {
        // Pick the writer flavour from the schema's stored_format
        // default. Unknown / missing values fall back to JSONL — that
        // preserves behavior for every existing collection.
        let mut writer = match self.schema.stored_format() {
            Some("columnar") => SegmentWriter::create_columnar(dir, &self.schema)?,
            _ => SegmentWriter::create(dir, &self.schema)?,
        };

        for (doc_id, record) in self.records.iter().enumerate() {
            if self.delete_bitmap.is_deleted(doc_id as u32) {
                continue;
            }
            writer.insert(record)?;
        }

        writer.seal()
    }

    /// Segment ID.
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Mark a record as deleted (soft delete).
    pub fn mark_deleted(&mut self, doc_id: u32) {
        self.delete_bitmap.mark_deleted(doc_id);
    }

    /// Check if a record is deleted.
    pub fn is_deleted(&self, doc_id: u32) -> bool {
        self.delete_bitmap.is_deleted(doc_id)
    }
}

// ── Distance function selection ───────────────────────────────────────────

fn select_distance_fn(metric: DistanceMetric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => simd::cosine_distance,
        DistanceMetric::Euclidean => simd::euclidean_distance,
        DistanceMetric::DotProduct => |a, b| {
            // For dot product similarity, higher is better.
            // Convert to a distance: dist = -dot(a, b) so that smaller is better.
            -simd::dot_product(a, b)
        },
        DistanceMetric::Manhattan => simd::manhattan_distance,
        // Hamming and Jaccard are not applicable to dense f32 vectors;
        // fall back to Euclidean.
        DistanceMetric::Hamming | DistanceMetric::Jaccard => simd::euclidean_distance,
    }
}

// ── Ordered f32 entry for BinaryHeap ──────────────────────────────────────

/// A max-heap entry that orders by distance descending (largest on top).
/// This lets us efficiently maintain the top-k nearest neighbors.
#[derive(Debug, Clone)]
struct OrdF32Entry {
    dist: f32,
    doc_id: u32,
}

impl PartialEq for OrdF32Entry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.doc_id == other.doc_id
    }
}

impl Eq for OrdF32Entry {}

impl PartialOrd for OrdF32Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance comes first.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}
