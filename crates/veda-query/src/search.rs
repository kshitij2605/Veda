use roaring::RoaringBitmap;
use tracing::instrument;
use uuid::Uuid;
use veda_core::error::Result;
use veda_core::types::DistanceMetric;
use veda_index::traits::{SearchResult, VectorIndex};
use veda_storage::growing_segment::GrowingSegment;
use veda_storage::segment::SegmentReader;

use crate::filter::evaluator::FilterContext;
use crate::filter::parser::FilterParser;
use crate::filter::record_eval::evaluate_record;
use crate::filter::FilterExpr;

/// A search request for a single vector field.
#[derive(Debug, Clone)]
pub struct VectorSearchRequest {
    pub field: String,
    pub query_vector: Vec<f32>,
    pub k: usize,
    pub metric: DistanceMetric,
    /// Optional SQL-like filter expression string.
    pub filter: Option<String>,
}

/// A search response containing scored records.
#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<ScoredRecord>,
    pub total_candidates: usize,
}

/// A result with score and record identifier.
#[derive(Debug, Clone)]
pub struct ScoredRecord {
    /// Record UUID.
    pub id: Uuid,
    /// Which segment this came from.
    pub segment_id: Uuid,
    /// Internal doc_id within the segment.
    pub doc_id: u32,
    /// Distance to the query vector (lower is closer).
    pub distance: f32,
}

/// Executes searches across a collection's segments.
///
/// Stateless -- constructed per query. Ties together filter evaluation,
/// vector indexes, and multi-segment result merging.
pub struct SearchExecutor;

impl SearchExecutor {
    /// Execute a single-vector search across all segments.
    ///
    /// Searches the optional growing segment and all sealed segments,
    /// then merges results and returns the top-k by distance.
    #[instrument(name = "execute_search", skip(request, growing_segment, sealed_segments), fields(field = %request.field, k = request.k, has_filter = request.filter.is_some()))]
    pub fn search(
        request: &VectorSearchRequest,
        growing_segment: Option<&GrowingSegment>,
        sealed_segments: &[(SegmentReader, Box<dyn VectorIndex>)],
    ) -> Result<SearchResponse> {
        // 1. Parse the filter expression (if provided).
        let filter_expr: Option<FilterExpr> = match &request.filter {
            Some(filter_str) => Some(FilterParser::parse(filter_str)?),
            None => None,
        };

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut total_candidates: usize = 0;

        // 2. Search the growing segment (if any).
        if let Some(gs) = growing_segment {
            let (results, candidates) =
                search_growing_segment(gs, request, filter_expr.as_ref())?;
            total_candidates += candidates;
            all_results.extend(results);
        }

        // 3. Search each sealed segment.
        for (reader, index) in sealed_segments {
            let (results, candidates) =
                search_sealed_segment(reader, index.as_ref(), request, filter_expr.as_ref())?;
            total_candidates += candidates;
            all_results.extend(results);
        }

        // 4. Merge: sort by distance ascending, take top-k.
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);

        Ok(SearchResponse {
            results: all_results,
            total_candidates,
        })
    }
}

/// Search a growing segment with optional filter.
fn search_growing_segment(
    gs: &GrowingSegment,
    request: &VectorSearchRequest,
    filter_expr: Option<&FilterExpr>,
) -> Result<(Vec<ScoredRecord>, usize)> {
    let count = gs.count();

    // Build a filter bitmap by scanning records if a filter is provided.
    let filter_bitmap: Option<RoaringBitmap> = match filter_expr {
        Some(expr) => {
            let mut bitmap = RoaringBitmap::new();
            for doc_id in 0..count {
                if let Some(record) = gs.get_record(doc_id)
                    && evaluate_record(expr, record)
                {
                    bitmap.insert(doc_id);
                }
            }
            Some(bitmap)
        }
        None => None,
    };

    let raw_results = gs.search(
        &request.field,
        &request.query_vector,
        request.k,
        request.metric,
        filter_bitmap.as_ref(),
    )?;

    let segment_id = gs.id();
    let results: Vec<ScoredRecord> = raw_results
        .into_iter()
        .map(|(doc_id, distance)| {
            let record_id = gs
                .get_record(doc_id)
                .map(|r| r.id())
                .unwrap_or_else(Uuid::new_v4);
            ScoredRecord {
                id: record_id,
                segment_id,
                doc_id,
                distance,
            }
        })
        .collect();

    Ok((results, count as usize))
}

/// Search a sealed segment with optional filter.
fn search_sealed_segment(
    reader: &SegmentReader,
    index: &dyn VectorIndex,
    request: &VectorSearchRequest,
    filter_expr: Option<&FilterExpr>,
) -> Result<(Vec<ScoredRecord>, usize)> {
    let row_count = reader.row_count() as u32;
    let segment_id = reader.metadata().id;

    let raw_results: Vec<SearchResult> = match filter_expr {
        Some(expr) => {
            // Build a FilterContext from the segment's scalar indexes.
            let mut ctx = FilterContext::new(row_count);

            // Discover which fields the filter references, then register
            // the corresponding indexes from the sealed segment.
            let mut fields = Vec::new();
            collect_filter_fields(expr, &mut fields);
            for field_name in &fields {
                register_field_index(&mut ctx, reader, field_name);
            }

            // Evaluate the filter to get matching doc IDs.
            let mut allowed = ctx.evaluate(expr)?;

            // Exclude deleted documents.
            let delete_bm = reader.delete_bitmap();
            for doc_id in 0..row_count {
                if delete_bm.is_deleted(doc_id) {
                    allowed.remove(doc_id);
                }
            }

            index.search_filtered(&request.query_vector, request.k, &allowed)?
        }
        None => {
            // No filter -- search with delete bitmap exclusion.
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

    Ok((results, row_count as usize))
}

/// Register a specific field's scalar indexes from a SegmentReader into a FilterContext.
fn register_field_index<'a>(
    ctx: &mut FilterContext<'a>,
    reader: &'a SegmentReader,
    field: &str,
) {
    if let Some(idx) = reader.inverted_index(field) {
        ctx.add_inverted(field, idx);
    }
    if let Some(idx) = reader.range_index(field) {
        ctx.add_range(field, idx);
    }
    if let Some(idx) = reader.bitmap_index(field) {
        ctx.add_bitmap(field, idx);
    }
}

/// Collect all field names referenced in a filter expression.
fn collect_filter_fields(expr: &FilterExpr, fields: &mut Vec<String>) {
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
