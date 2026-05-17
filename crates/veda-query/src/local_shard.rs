use std::collections::HashMap;

use uuid::Uuid;
use veda_core::error::Result;
use veda_core::types::{DistanceMetric, FieldType, IndexType};
use veda_index::diskann::DiskAnnIndex;
use veda_index::flat::FlatIndex;
use veda_index::hnsw::HnswIndex;
use veda_index::ivfpq::IvfPqIndex;
use veda_index::traits::VectorIndex;
use veda_storage::segment_manager::SegmentManager;

use crate::distributed::ShardSearcher;
use crate::search::{
    ScoredRecord, SearchExecutor, SearchTimings, SparseSearchRequest, VectorSearchRequest,
};

/// Build a vector index of the declared type, populated from `vectors`.
///
/// For unknown / non-vector index types, falls back to `Flat`.
fn build_vector_index(
    index_type: IndexType,
    metric: DistanceMetric,
    dimension: u32,
    vectors: Vec<Vec<f32>>,
) -> Result<Box<dyn VectorIndex>> {
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    match index_type {
        IndexType::Hnsw => {
            let mut idx = HnswIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::DiskAnn => {
            let mut idx = DiskAnnIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::IvfPq => {
            let mut idx = IvfPqIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        // Flat (or anything else that isn't a recognised vector index type).
        _ => {
            let mut idx = FlatIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
    }
}

/// Searches a local shard (in-process, no network).
///
/// Wraps a `SegmentManager` and pre-built vector indexes for sealed segments,
/// executing queries via the existing `SearchExecutor`.
pub struct LocalShardSearcher {
    segment_manager: SegmentManager,
    /// Vector indexes for each sealed segment, keyed by segment_id then field name.
    indexes: HashMap<Uuid, HashMap<String, Box<dyn VectorIndex>>>,
}

impl LocalShardSearcher {
    pub fn new(segment_manager: SegmentManager) -> Self {
        Self {
            segment_manager,
            indexes: HashMap::new(),
        }
    }

    /// Build per-field vector indexes for all sealed segments.
    ///
    /// The index type is taken from the field definition's `index_type`
    /// (HNSW / DiskANN / IVF-PQ / Flat). Unknown types fall back to Flat.
    pub fn build_indexes(&mut self) -> Result<()> {
        for handle in self.segment_manager.sealed_segments() {
            let seg_id = handle.metadata.id;
            if self.indexes.contains_key(&seg_id) {
                continue;
            }

            let reader = &handle.reader;
            let schema = reader.schema();
            let row_count = reader.row_count() as u32;

            let mut field_indexes: HashMap<String, Box<dyn VectorIndex>> = HashMap::new();

            for field_def in schema.vector_fields() {
                if let FieldType::Vector { dimension, metric } = field_def.field_type() {
                    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(row_count as usize);
                    for doc_id in 0..row_count {
                        let vec = reader.get_vector(field_def.name(), doc_id)?;
                        vectors.push(vec.to_vec());
                    }
                    let index = build_vector_index(
                        field_def.index_type(),
                        *metric,
                        *dimension,
                        vectors,
                    )?;
                    field_indexes.insert(field_def.name().to_owned(), index);
                }
            }

            self.indexes.insert(seg_id, field_indexes);
        }

        Ok(())
    }
}

impl ShardSearcher for LocalShardSearcher {
    fn search_shard(
        &self,
        _shard_id: u32,
        request: &VectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        // Collect sealed segments paired with the appropriate vector index
        // for the requested field.
        let mut sealed_pairs: Vec<(&veda_storage::segment::SegmentReader, &dyn VectorIndex)> =
            Vec::new();

        for handle in self.segment_manager.sealed_segments() {
            let seg_id = handle.metadata.id;
            if let Some(field_indexes) = self.indexes.get(&seg_id)
                && let Some(index) = field_indexes.get(&request.field)
            {
                sealed_pairs.push((&handle.reader, index.as_ref()));
            }
        }

        let growing = self.segment_manager.growing_segment();
        let has_growing = growing.count() > 0;

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut agg_timings = SearchTimings::default();

        // Search growing segment.
        if has_growing {
            let (response, timings) = SearchExecutor::search(request, Some(growing), &[])?;
            all_results.extend(response.results);
            agg_timings.filter_ms += timings.filter_ms;
            agg_timings.ann_search_ms += timings.ann_search_ms;
            agg_timings.scoring_ms += timings.scoring_ms;
        }

        // Search each sealed segment individually.
        let sealed_start = std::time::Instant::now();
        for (reader, index) in &sealed_pairs {
            let results = search_sealed_segment_ref(reader, *index, request)?;
            all_results.extend(results);
        }
        agg_timings.ann_search_ms += sealed_start.elapsed().as_secs_f64() * 1000.0;

        // Sort by distance ascending, take top-k.
        let scoring_start = std::time::Instant::now();
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);
        agg_timings.scoring_ms += scoring_start.elapsed().as_secs_f64() * 1000.0;

        Ok((all_results, agg_timings))
    }

    fn search_shard_sparse(
        &self,
        _shard_id: u32,
        request: &SparseSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        let growing = self.segment_manager.growing_segment();
        let growing_opt = if growing.count() > 0 { Some(growing) } else { None };

        let sealed_refs: Vec<&veda_storage::segment::SegmentReader> = self
            .segment_manager
            .sealed_segments()
            .iter()
            .map(|h| &h.reader)
            .collect();

        let (response, timings) =
            SearchExecutor::search_sparse(request, growing_opt, &sealed_refs)?;
        Ok((response.results, timings))
    }
}

/// Search a single sealed segment using an index reference.
///
/// Simplified version of the sealed segment search that works with references
/// rather than requiring owned `Box<dyn VectorIndex>`.
fn search_sealed_segment_ref(
    reader: &veda_storage::segment::SegmentReader,
    index: &dyn VectorIndex,
    request: &VectorSearchRequest,
) -> Result<Vec<ScoredRecord>> {
    use crate::filter::parser::FilterParser;
    use crate::filter::FilterExpr;
    use roaring::RoaringBitmap;

    let row_count = reader.row_count() as u32;
    let segment_id = reader.metadata().id;

    let filter_expr: Option<FilterExpr> = match &request.filter {
        Some(filter_str) => Some(FilterParser::parse(filter_str)?),
        None => None,
    };

    let raw_results = match &filter_expr {
        Some(expr) => {
            use crate::filter::evaluator::FilterContext;

            let mut ctx = FilterContext::new(row_count);

            let mut fields = Vec::new();
            collect_filter_fields(expr, &mut fields);
            for field_name in &fields {
                if let Some(idx) = reader.inverted_index(field_name) {
                    ctx.add_inverted(field_name, idx);
                }
                if let Some(idx) = reader.range_index(field_name) {
                    ctx.add_range(field_name, idx);
                }
                if let Some(idx) = reader.bitmap_index(field_name) {
                    ctx.add_bitmap(field_name, idx);
                }
            }

            let mut allowed = ctx.evaluate(expr)?;
            let delete_bm = reader.delete_bitmap();
            for doc_id in 0..row_count {
                if delete_bm.is_deleted(doc_id) {
                    allowed.remove(doc_id);
                }
            }

            index.search_filtered(&request.query_vector, request.k, &allowed)?
        }
        None => {
            let delete_bm = reader.delete_bitmap();
            if delete_bm.deleted_count() > 0 {
                let mut allowed = RoaringBitmap::from_sorted_iter(0..row_count).unwrap();
                for doc_id in 0..row_count {
                    if delete_bm.is_deleted(doc_id) {
                        allowed.remove(doc_id);
                    }
                }
                index.search_filtered(&request.query_vector, request.k, &allowed)?
            } else {
                index.search(&request.query_vector, request.k)?
            }
        }
    };

    let results: Vec<ScoredRecord> = raw_results
        .into_iter()
        .map(|sr| {
            let record_id = reader
                .get_record(sr.id)
                .ok()
                .map(|r| r.id())
                .unwrap_or_else(Uuid::new_v4);
            ScoredRecord {
                id: record_id,
                segment_id,
                doc_id: sr.id,
                distance: sr.distance,
            }
        })
        .collect();

    Ok(results)
}

fn collect_filter_fields(expr: &crate::filter::FilterExpr, fields: &mut Vec<String>) {
    use crate::filter::FilterExpr;
    match expr {
        FilterExpr::Eq { field, .. }
        | FilterExpr::Ne { field, .. }
        | FilterExpr::Gt { field, .. }
        | FilterExpr::Gte { field, .. }
        | FilterExpr::Lt { field, .. }
        | FilterExpr::Lte { field, .. }
        | FilterExpr::Between { field, .. }
        | FilterExpr::In { field, .. }
        | FilterExpr::Contains { field, .. }
        | FilterExpr::StartsWith { field, .. }
        | FilterExpr::IsNull { field }
        | FilterExpr::IsNotNull { field } => {
            if !fields.contains(field) {
                fields.push(field.clone());
            }
        }
        FilterExpr::And(left, right) | FilterExpr::Or(left, right) => {
            collect_filter_fields(left, fields);
            collect_filter_fields(right, fields);
        }
        FilterExpr::Not(inner) => {
            collect_filter_fields(inner, fields);
        }
    }
}
