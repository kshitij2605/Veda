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
    MultiVectorSearchRequest, ScoredRecord, SearchExecutor, SearchTimings, SparseSearchRequest,
    VectorSearchRequest,
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
        IndexType::RaBitQ => {
            // 1-bit-per-dim quantization, brute-force scan over codes.
            // Cheap memory; wins at very large N. See
            // `crates/veda-index/src/quantization/rabitq_vector_index.rs`.
            use veda_index::quantization::rabitq_vector_index::RaBitQIndex;
            let mut idx = RaBitQIndex::new(metric);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::RaBitQRerank => {
            // RaBitQ for coarse candidates, raw vectors for exact rerank.
            // See `crates/veda-index/src/quantization/rabitq_rerank.rs`.
            use veda_index::quantization::rabitq_rerank::RaBitQRerankIndex;
            let mut idx = RaBitQRerankIndex::new(metric);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::HnswRaBitQ => {
            // Deep-quantized HNSW: graph + RaBitQ codes, no raw f32.
            // See `crates/veda-index/src/hnsw/quantized.rs`.
            use veda_index::hnsw::quantized::QuantizedHnswIndex;
            let mut idx = QuantizedHnswIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::Cagra => {
            // NVIDIA cuVS GPU-native graph ANN. Builds + searches
            // on the GPU; falls back to a clear runtime error if
            // the binary was built without `--features veda-index/gpu_cuvs`.
            // See `crates/veda-index/src/gpu_cuvs.rs`.
            use veda_index::gpu_cuvs::CagraIndex;
            let mut idx = CagraIndex::new(metric, dimension);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::RaBitQExtRerank => {
            // 1-bit RaBitQ for coarse candidates, 4-bit
            // Extended-RaBitQ for rerank. ~8× less rerank memory
            // than the raw-float `RaBitQRerank` at similar recall.
            // See `crates/veda-index/src/quantization/rabitq_ext_rerank.rs`.
            use veda_index::quantization::rabitq_ext_rerank::RaBitQExtRerankIndex;
            let mut idx = RaBitQExtRerankIndex::new(metric);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::Sq4 => {
            // 4-bit per-dimension scalar quantization. Brute-force scan.
            // ~D/2 bytes/vec. See crates/veda-index/src/quantization/scalar_index.rs.
            use veda_index::quantization::scalar_index::ScalarQuantIndex;
            let mut idx = ScalarQuantIndex::sq4(metric);
            idx.build(&refs)?;
            Ok(Box::new(idx))
        }
        IndexType::Sq8 => {
            // 8-bit per-dimension scalar quantization. Brute-force scan.
            // ~D bytes/vec; highest-recall quantized option (~0.99).
            use veda_index::quantization::scalar_index::ScalarQuantIndex;
            let mut idx = ScalarQuantIndex::sq8(metric);
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
    /// PLAID candidate-pruning indexes for multi-vector fields, keyed by
    /// segment_id then field name.
    plaid_indexes: HashMap<Uuid, HashMap<String, crate::plaid::PlaidIndex>>,
    /// Matryoshka per-dimension indexes: `seg_id -> field -> dim ->
    /// truncated-vector HNSW`. Built when a vector field declares
    /// `mrl_dims`; queried when search includes `truncate_to`.
    mrl_indexes:
        HashMap<Uuid, HashMap<String, HashMap<u32, Box<dyn VectorIndex>>>>,
    /// One MUVERA encoder per (shard, field). Lazily created on first
    /// build_indexes call for a multi-vector field. The encoder is
    /// deterministic from the field name so all segments share anchors —
    /// a single query encoding then scores against every segment's FDE list.
    muvera_encoders: HashMap<String, crate::muvera::MuveraEncoder>,
    /// Per-doc MUVERA FDEs: `seg_id -> field -> Vec<(doc_id, fde)>`.
    muvera_fdes: HashMap<Uuid, HashMap<String, Vec<(u32, Vec<f32>)>>>,
}

impl LocalShardSearcher {
    pub fn new(segment_manager: SegmentManager) -> Self {
        Self {
            segment_manager,
            indexes: HashMap::new(),
            plaid_indexes: HashMap::new(),
            mrl_indexes: HashMap::new(),
            muvera_encoders: HashMap::new(),
            muvera_fdes: HashMap::new(),
        }
    }

    /// Look up a full record by UUID across this shard's segments.
    ///
    /// Passthrough to the underlying `SegmentManager` so the gRPC fetch
    /// service (`remote_shard::ShardSearchService::fetch_records`) can
    /// resolve top-k ids to full records without exposing the manager.
    pub fn lookup_record(&self, id: Uuid) -> Option<veda_core::record::Record> {
        self.segment_manager.lookup_record(id)
    }

    /// Build per-field vector indexes for all sealed segments.
    ///
    /// The index type is taken from the field definition's `index_type`
    /// (HNSW / DiskANN / IVF-PQ / Flat). Unknown types fall back to Flat.
    /// Warm-tier segments are skipped (search will brute-force scan their
    /// mmap-resident vectors instead).
    pub fn build_indexes(&mut self) -> Result<()> {
        use veda_storage::segment::SegmentTier;
        for handle in self.segment_manager.sealed_segments() {
            let seg_id = handle.metadata.id;
            if self.indexes.contains_key(&seg_id) {
                continue;
            }
            if handle.metadata.tier != SegmentTier::Hot {
                // Warm: keep vector storage mmap'd but skip ANN build.
                // Cold: ditto for now (cold-tier copy-back lands in a
                // follow-up; see documentation/tiered-storage.md).
                continue;
            }

            let reader = &handle.reader;
            let schema = reader.schema();
            let row_count = reader.row_count() as u32;

            let mut field_indexes: HashMap<String, Box<dyn VectorIndex>> = HashMap::new();

            // Per-field MRL indexes for this segment. Built only when the
            // field declared mrl_dims.
            let mut field_mrl: HashMap<String, HashMap<u32, Box<dyn VectorIndex>>> =
                HashMap::new();

            for field_def in schema.vector_fields() {
                if let FieldType::Vector { dimension, metric, mrl_dims, .. } =
                    field_def.field_type()
                {
                    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(row_count as usize);
                    for doc_id in 0..row_count {
                        let vec = reader.get_vector(field_def.name(), doc_id)?;
                        vectors.push(vec.to_vec());
                    }

                    // Full-dim primary index.
                    let index = build_vector_index(
                        field_def.index_type(),
                        *metric,
                        *dimension,
                        vectors.clone(),
                    )?;
                    field_indexes.insert(field_def.name().to_owned(), index);

                    // Truncated indexes for each declared MRL ladder rung.
                    if let Some(dims) = mrl_dims {
                        let mut per_dim: HashMap<u32, Box<dyn VectorIndex>> = HashMap::new();
                        for &d in dims {
                            if d == 0 || d >= *dimension {
                                continue;
                            }
                            let dim_usize = d as usize;
                            let truncated: Vec<Vec<f32>> = vectors
                                .iter()
                                .map(|v| v[..dim_usize.min(v.len())].to_vec())
                                .collect();
                            let idx = build_vector_index(
                                veda_core::types::IndexType::Hnsw,
                                *metric,
                                d,
                                truncated,
                            )?;
                            per_dim.insert(d, idx);
                        }
                        if !per_dim.is_empty() {
                            field_mrl.insert(field_def.name().to_owned(), per_dim);
                        }
                    }
                }
            }

            self.indexes.insert(seg_id, field_indexes);
            if !field_mrl.is_empty() {
                self.mrl_indexes.insert(seg_id, field_mrl);
            }

            // Build PLAID, optional Light-ColPali reduction, and MUVERA FDEs
            // for each multi-vector field in this segment.
            //
            // Env knobs:
            //   VEDA_COLPALI_TOKEN_FRACTION=0.1   → enable Light-ColPali, keep ~10% of tokens
            //   VEDA_MUVERA_ENABLED=1             → build MUVERA FDEs alongside PLAID
            //   VEDA_MUVERA_K_SIM=128             → anchor count (default 128)
            let reduce_fraction: Option<f32> = std::env::var("VEDA_COLPALI_TOKEN_FRACTION")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|f| (0.01..=1.0).contains(f));
            let muvera_enabled = std::env::var("VEDA_MUVERA_ENABLED")
                .ok()
                .map(|s| !s.is_empty() && s != "0" && s.to_ascii_lowercase() != "false")
                .unwrap_or(false);
            let muvera_k_sim: usize = std::env::var("VEDA_MUVERA_K_SIM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(crate::muvera::DEFAULT_K_SIM);

            let mut plaid_field: HashMap<String, crate::plaid::PlaidIndex> = HashMap::new();
            let mut muvera_field: HashMap<String, Vec<(u32, Vec<f32>)>> = HashMap::new();

            for field_def in schema.multi_vector_fields() {
                let Some(iter) = reader.iter_multi_vector(field_def.name()) else {
                    continue;
                };

                // Pull (id, rows) and optionally reduce tokens.
                let mut collected: Vec<(u32, Vec<Vec<f32>>)> = iter
                    .map(|(id, rows)| {
                        let rows = if let Some(frac) = reduce_fraction {
                            crate::light_colpali::reduce_tokens(rows, frac)
                        } else {
                            rows.clone()
                        };
                        (id, rows)
                    })
                    .collect();

                // Build PLAID centroid pruning.
                let plaid = crate::plaid::PlaidIndex::build(
                    collected.iter().map(|(id, rows)| (*id, rows)),
                );
                if plaid.centroid_count() > 0 {
                    plaid_field.insert(field_def.name().to_owned(), plaid);
                }

                // Build MUVERA FDEs (per doc) if enabled. Use a per-field
                // deterministic seed so all segments share anchors and a
                // single query encoding works across them.
                if muvera_enabled
                    && let Some((_, first_rows)) = collected.first()
                    && let Some(first_row) = first_rows.first()
                {
                    let token_dim = first_row.len();
                    let field_name = field_def.name().to_owned();
                    let encoder = self
                        .muvera_encoders
                        .entry(field_name.clone())
                        .or_insert_with(|| {
                            crate::muvera::MuveraEncoder::new(
                                token_dim,
                                muvera_k_sim,
                                seed_from_field(&field_name),
                            )
                        });
                    let fdes: Vec<(u32, Vec<f32>)> = collected
                        .iter_mut()
                        .map(|(id, rows)| {
                            let fde = encoder.encode(rows);
                            (*id, fde)
                        })
                        .collect();
                    if !fdes.is_empty() {
                        muvera_field.insert(field_name, fdes);
                    }
                }
            }

            if !plaid_field.is_empty() {
                self.plaid_indexes.insert(seg_id, plaid_field);
            }
            if !muvera_field.is_empty() {
                self.muvera_fdes.insert(seg_id, muvera_field);
            }
        }

        Ok(())
    }
}

/// Deterministic seed from a field name for MUVERA anchor generation.
/// FNV-1a hash; doesn't need to be cryptographic.
fn seed_from_field(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl ShardSearcher for LocalShardSearcher {
    #[tracing::instrument(
        name = "local_shard_search",
        skip(self, request),
        fields(shard_id = _shard_id, k = request.k)
    )]
    fn search_shard(
        &self,
        _shard_id: u32,
        request: &VectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        let growing = self.segment_manager.growing_segment();
        let has_growing = growing.count() > 0;

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut agg_timings = SearchTimings::default();

        // Phase F: precompute the visible-doc bitmap for the
        // growing segment once. `None` means either no read_ts
        // requested OR the growing segment has no MVCC stamps —
        // in either case every doc is considered visible.
        let growing_visible = match request.read_ts {
            Some(ts) => growing.visible_doc_ids_at(ts),
            None => None,
        };
        let growing_segment_id = growing.id();

        // Search growing segment.
        if has_growing {
            let (response, timings) = SearchExecutor::search(request, Some(growing), &[])?;
            // Filter to visible-at-read_ts when applicable.
            let filtered: Vec<ScoredRecord> = match &growing_visible {
                Some(vis) => response
                    .results
                    .into_iter()
                    .filter(|r| r.segment_id != growing_segment_id || vis.contains(r.doc_id))
                    .collect(),
                None => response.results,
            };
            all_results.extend(filtered);
            agg_timings.filter_ms += timings.filter_ms;
            agg_timings.ann_search_ms += timings.ann_search_ms;
            agg_timings.scoring_ms += timings.scoring_ms;
        }

        // Search each sealed segment individually.
        let sealed_start = std::time::Instant::now();
        for handle in self.segment_manager.sealed_segments() {
            let seg_id = handle.metadata.id;
            let reader = &handle.reader;

            // If a truncate_to is set AND we have a pre-built MRL index at
            // exactly that dim, use it (real ANN search at truncated dim
            // rather than brute-force scan).
            if let Some(trunc) = request.truncate_to
                && let Some(mrl_index) = self
                    .mrl_indexes
                    .get(&seg_id)
                    .and_then(|m| m.get(&request.field))
                    .and_then(|m| m.get(&trunc))
            {
                let truncated_query: Vec<f32> = request
                    .query_vector
                    .iter()
                    .take(trunc as usize)
                    .copied()
                    .collect();
                let trunc_req = VectorSearchRequest {
                    field: request.field.clone(),
                    query_vector: truncated_query,
                    k: request.k,
                    metric: request.metric,
                    filter: request.filter.clone(),
                    filter_expr: request.filter_expr.clone(),
                    truncate_to: None, // already truncated
                    read_ts: request.read_ts,
                };
                let results =
                    search_sealed_segment_ref(reader, mrl_index.as_ref(), &trunc_req)?;
                all_results.extend(results);
                continue;
            }

            // Otherwise: use full-dim primary index. If the caller set
            // truncate_to without a matching MRL index, search_sealed_segment_ref
            // will fall through to brute-force at truncated dim.
            if let Some(field_indexes) = self.indexes.get(&seg_id)
                && let Some(index) = field_indexes.get(&request.field)
            {
                let results = search_sealed_segment_ref(reader, index.as_ref(), request)?;
                all_results.extend(results);
            } else {
                // No ANN index (e.g. Warm tier): brute-force scan vectors
                // directly from the segment reader.
                let row_count = reader.row_count() as u32;
                let segment_id = reader.metadata().id;
                let mut local: Vec<(u32, f32)> = Vec::with_capacity(request.k * 2);
                let delete_bm = reader.delete_bitmap();
                // Phase F: precompute the MVCC visibility
                // bitmap for this sealed segment if read_ts is
                // requested. `None` means either no read_ts OR
                // a pre-MVCC segment (treat every doc visible).
                let mvcc_visible = match request.read_ts {
                    Some(ts) => reader.visible_doc_ids_at(ts),
                    None => None,
                };
                let metric_fn: fn(&[f32], &[f32]) -> f32 = match request.metric {
                    DistanceMetric::Cosine => veda_core::simd::cosine_distance,
                    DistanceMetric::Euclidean => veda_core::simd::euclidean_distance,
                    DistanceMetric::DotProduct => |a, b| -veda_core::simd::dot_product(a, b),
                    DistanceMetric::Manhattan => veda_core::simd::manhattan_distance,
                    _ => veda_core::simd::euclidean_distance,
                };
                for doc_id in 0..row_count {
                    if delete_bm.is_deleted(doc_id) {
                        continue;
                    }
                    if let Some(vis) = mvcc_visible.as_ref()
                        && !vis.contains(doc_id)
                    {
                        continue;
                    }
                    let Ok(stored) = reader.get_vector(&request.field, doc_id) else {
                        continue;
                    };
                    let dist = metric_fn(&request.query_vector, stored);
                    local.push((doc_id, dist));
                    if local.len() > request.k * 4 {
                        local.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        local.truncate(request.k);
                    }
                }
                local.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                local.truncate(request.k);
                for (doc_id, distance) in local {
                    let record_id = reader
                        .get_record(doc_id)
                        .ok()
                        .map(|r| r.id())
                        .unwrap_or_else(Uuid::new_v4);
                    all_results.push(ScoredRecord {
                        id: record_id,
                        segment_id,
                        doc_id,
                        distance,
                    });
                }
            }
        }
        agg_timings.ann_search_ms += sealed_start.elapsed().as_secs_f64() * 1000.0;

        // Phase F sealed-segment visibility used to be a
        // post-filter pass here. It's gone now: the ANN path
        // intersects the bitmap into `allowed` BEFORE
        // index.search_filtered runs (see
        // `search_sealed_segment_ref`), and the brute-force
        // fallback checks `mvcc_visible.contains(doc_id)`
        // inside its scan loop. Both pay zero candidate budget
        // for invisible docs.
        let _ = growing_segment_id; // referenced only by the (now-removed) post-filter

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

        // Phase F: filter hits against the MVCC visibility
        // bitmap at request.read_ts. Both growing and sealed
        // segments now carry commit_ts; segments with no
        // stamps pass through.
        let results = match request.read_ts {
            Some(ts) => {
                use std::collections::HashMap;
                let mut visible_per_seg: HashMap<veda_core::types::SegmentId, roaring::RoaringBitmap> =
                    HashMap::new();
                if let Some(vis) = growing.visible_doc_ids_at(ts) {
                    visible_per_seg.insert(growing.id(), vis);
                }
                for handle in self.segment_manager.sealed_segments() {
                    if let Some(vis) = handle.reader.visible_doc_ids_at(ts) {
                        visible_per_seg.insert(handle.metadata.id, vis);
                    }
                }
                if visible_per_seg.is_empty() {
                    response.results
                } else {
                    response
                        .results
                        .into_iter()
                        .filter(|r| match visible_per_seg.get(&r.segment_id) {
                            Some(vis) => vis.contains(r.doc_id),
                            None => true,
                        })
                        .collect()
                }
            }
            None => response.results,
        };
        Ok((results, timings))
    }

    fn search_shard_multi_vector(
        &self,
        _shard_id: u32,
        request: &MultiVectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        use roaring::RoaringBitmap;
        use std::time::Instant;
        use uuid::Uuid;
        use veda_core::simd;
        use veda_core::types::DistanceMetric;

        // Search filter -- shared across segments.
        let filter_start = Instant::now();
        // Prefer a pre-parsed filter_expr (e.g. injected by RLS at AST level).
        let filter_expr: Option<crate::filter::FilterExpr> = if let Some(expr) = request.filter_expr.clone() {
            Some(expr)
        } else {
            match &request.filter {
                Some(s) => Some(crate::filter::parser::FilterParser::parse(s)?),
                None => None,
            }
        };
        let filter_ms = filter_start.elapsed().as_secs_f64() * 1000.0;

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let ann_start = Instant::now();

        let per_row: fn(&[f32], &[f32]) -> f32 = match request.metric {
            DistanceMetric::Cosine => simd::cosine_distance,
            DistanceMetric::Euclidean => simd::euclidean_distance,
            DistanceMetric::DotProduct => |a, b| -simd::dot_product(a, b),
            _ => simd::cosine_distance,
        };

        // Growing segment — brute force (no PLAID there yet).
        let growing = self.segment_manager.growing_segment();
        if growing.count() > 0 {
            let mut growing_filter: Option<RoaringBitmap> = match filter_expr.as_ref() {
                Some(expr) => {
                    let mut bm = RoaringBitmap::new();
                    for doc_id in 0..growing.count() {
                        if let Some(rec) = growing.get_record(doc_id)
                            && crate::filter::evaluate_record(expr, rec)
                        {
                            bm.insert(doc_id);
                        }
                    }
                    Some(bm)
                }
                None => None,
            };
            // Phase F: intersect the candidate set with the
            // MVCC visibility bitmap BEFORE the multi-vector
            // search — cheaper than post-filter on the merged
            // results list.
            if let Some(ts) = request.read_ts
                && let Some(visible) = growing.visible_doc_ids_at(ts)
            {
                growing_filter = Some(match growing_filter {
                    Some(existing) => existing & visible,
                    None => visible,
                });
            }
            let raw = growing.search_multi_vector(
                &request.field,
                &request.query_rows,
                request.k,
                request.metric,
                growing_filter.as_ref(),
            )?;
            let segment_id = growing.id();
            all_results.extend(raw.into_iter().map(|(doc_id, distance)| {
                let record_id = growing
                    .get_record(doc_id)
                    .map(|r| r.id())
                    .unwrap_or_else(Uuid::new_v4);
                ScoredRecord {
                    id: record_id,
                    segment_id,
                    doc_id,
                    distance,
                }
            }));
        }

        // Sealed segments — use PLAID pruning when available.
        // probe count grows with query size; "2 centroids per query token" is
        // a reasonable default that captures the relevant cluster mass
        // without exploding the candidate set.
        let probe = 2usize;
        for handle in self.segment_manager.sealed_segments() {
            let reader = &handle.reader;
            let row_count = reader.row_count() as u32;

            // Build allowed bitmap from filter + delete tombstones.
            let mut allowed: Option<RoaringBitmap> = match filter_expr.as_ref() {
                Some(expr) => {
                    let mut ctx = crate::filter::evaluator::FilterContext::new(row_count);
                    let mut fields = Vec::new();
                    collect_filter_fields(expr, &mut fields);
                    for f in &fields {
                        if let Some(idx) = reader.inverted_index(f) {
                            ctx.add_inverted(f, idx);
                        }
                        if let Some(idx) = reader.range_index(f) {
                            ctx.add_range(f, idx);
                        }
                        if let Some(idx) = reader.bitmap_index(f) {
                            ctx.add_bitmap(f, idx);
                        }
                    }
                    Some(ctx.evaluate(expr)?)
                }
                None => None,
            };
            let delete_bm = reader.delete_bitmap();
            if let Some(bm) = allowed.as_mut() {
                for doc_id in 0..row_count {
                    if delete_bm.is_deleted(doc_id) {
                        bm.remove(doc_id);
                    }
                }
            }

            // Phase F: intersect with MVCC visibility bitmap
            // when read_ts is set AND this sealed segment carries
            // commit_ts stamps. Pre-MVCC segments pass through.
            if let Some(ts) = request.read_ts
                && let Some(vis) = reader.visible_doc_ids_at(ts)
            {
                allowed = Some(match allowed {
                    Some(existing) => existing & vis,
                    None => vis,
                });
            }

            // PLAID candidate pruning if available.
            let plaid_candidates: Option<RoaringBitmap> = self
                .plaid_indexes
                .get(&handle.metadata.id)
                .and_then(|m| m.get(&request.field))
                .map(|p| p.candidates(&request.query_rows, probe));

            // MUVERA FDE candidate pruning if available — score each doc's
            // FDE by cosine against the query FDE, take top `request.k * 4`.
            let muvera_candidates: Option<RoaringBitmap> = self
                .muvera_encoders
                .get(&request.field)
                .and_then(|enc| {
                    self.muvera_fdes
                        .get(&handle.metadata.id)
                        .and_then(|m| m.get(&request.field))
                        .map(|fdes| (enc, fdes))
                })
                .map(|(enc, fdes)| {
                    let q_fde = enc.encode(&request.query_rows);
                    let q_norm: f32 = q_fde.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let mut scored: Vec<(u32, f32)> = fdes
                        .iter()
                        .map(|(doc_id, fde)| {
                            let dot: f32 = q_fde.iter().zip(fde).map(|(a, b)| a * b).sum();
                            let d_norm: f32 = fde.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let denom = q_norm * d_norm;
                            let sim = if denom < 1e-10 { 0.0 } else { dot / denom };
                            (*doc_id, sim)
                        })
                        .collect();
                    scored.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    scored.truncate((request.k * 4).max(20));
                    let mut bm = RoaringBitmap::new();
                    for (id, _) in scored {
                        bm.insert(id);
                    }
                    bm
                });

            // Final candidate set: intersect filter ∩ plaid ∩ muvera (if any).
            let candidates: Option<RoaringBitmap> = [allowed, plaid_candidates, muvera_candidates]
                .into_iter()
                .flatten()
                .reduce(|acc, x| acc & x);

            // Brute MaxSim over the candidate set.
            let Some(iter) = reader.iter_multi_vector(&request.field) else {
                continue;
            };
            let mut local: Vec<(u32, f32)> = Vec::with_capacity(request.k * 2);
            for (doc_id, doc_rows) in iter {
                if delete_bm.is_deleted(doc_id) {
                    continue;
                }
                if let Some(bm) = candidates.as_ref()
                    && !bm.contains(doc_id)
                {
                    continue;
                }
                if doc_rows.is_empty() {
                    continue;
                }
                let dist = simd::maxsim_distance(&request.query_rows, doc_rows, per_row);
                local.push((doc_id, dist));
            }
            local.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            local.truncate(request.k);

            let segment_id = handle.metadata.id;
            for (doc_id, distance) in local {
                let record_id = reader
                    .get_record(doc_id)
                    .ok()
                    .map(|r| r.id())
                    .unwrap_or_else(Uuid::new_v4);
                all_results.push(ScoredRecord {
                    id: record_id,
                    segment_id,
                    doc_id,
                    distance,
                });
            }
        }

        let ann_ms = ann_start.elapsed().as_secs_f64() * 1000.0;
        let scoring_start = Instant::now();
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);
        let scoring_ms = scoring_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            all_results,
            SearchTimings {
                filter_ms,
                ann_search_ms: ann_ms,
                scoring_ms,
            },
        ))
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

    // Prefer a pre-parsed filter_expr (e.g. injected by RLS at AST level).
    let filter_expr: Option<FilterExpr> = if let Some(expr) = request.filter_expr.clone() {
        Some(expr)
    } else {
        match &request.filter {
            Some(filter_str) => Some(FilterParser::parse(filter_str)?),
            None => None,
        }
    };

    // MRL truncation: brute-force at truncated dim instead of using the
    // full-dim ANN index.
    if let Some(trunc) = request.truncate_to {
        return search_sealed_brute_truncated(
            reader,
            request,
            filter_expr.as_ref(),
            trunc as usize,
            segment_id,
            row_count,
        );
    }

    // Build the base "allowed" bitmap from the filter
    // expression + tombstones. None = no constraint.
    let base_allowed: Option<RoaringBitmap> = match &filter_expr {
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
            Some(allowed)
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
                Some(allowed)
            } else {
                None
            }
        }
    };

    // Phase F: AND in MVCC visibility when read_ts is set AND
    // this sealed segment carries commit_ts stamps. Pre-MVCC
    // segments have no sidecar so `visible_doc_ids_at` returns
    // None and we leave the base bitmap unchanged. Doing the
    // intersection HERE (before the index walk) means HNSW /
    // IVF-PQ / Flat never spends candidate budget on invisible
    // docs — the visibility filter is now native to the search,
    // not a post-step.
    let allowed: Option<RoaringBitmap> = match (request.read_ts, base_allowed) {
        (Some(ts), base) => match reader.visible_doc_ids_at(ts) {
            Some(vis) => Some(match base {
                Some(b) => b & vis,
                None => vis,
            }),
            None => base,
        },
        (None, base) => base,
    };

    let raw_results = match allowed {
        Some(bm) => index.search_filtered(&request.query_vector, request.k, &bm)?,
        None => index.search(&request.query_vector, request.k)?,
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

fn search_sealed_brute_truncated(
    reader: &veda_storage::segment::SegmentReader,
    request: &VectorSearchRequest,
    filter_expr: Option<&crate::filter::FilterExpr>,
    trunc: usize,
    segment_id: Uuid,
    row_count: u32,
) -> Result<Vec<ScoredRecord>> {
    use crate::filter::evaluator::FilterContext;
    use roaring::RoaringBitmap;
    use veda_core::simd;

    let allowed: Option<RoaringBitmap> = match filter_expr {
        Some(expr) => {
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
            let mut bm = ctx.evaluate(expr)?;
            let delete_bm = reader.delete_bitmap();
            for doc_id in 0..row_count {
                if delete_bm.is_deleted(doc_id) {
                    bm.remove(doc_id);
                }
            }
            Some(bm)
        }
        None => None,
    };

    let q = &request.query_vector;
    let qlim = trunc.min(q.len());
    let distance_fn: fn(&[f32], &[f32]) -> f32 = match request.metric {
        DistanceMetric::Cosine => simd::cosine_distance,
        DistanceMetric::Euclidean => simd::euclidean_distance,
        DistanceMetric::DotProduct => |a, b| -simd::dot_product(a, b),
        DistanceMetric::Manhattan => simd::manhattan_distance,
        _ => simd::euclidean_distance,
    };

    let mut local: Vec<(u32, f32)> = Vec::with_capacity(request.k * 2);
    let delete_bm = reader.delete_bitmap();
    for doc_id in 0..row_count {
        if delete_bm.is_deleted(doc_id) {
            continue;
        }
        if let Some(bm) = allowed.as_ref()
            && !bm.contains(doc_id)
        {
            continue;
        }
        let Ok(stored) = reader.get_vector(&request.field, doc_id) else {
            continue;
        };
        let limit = qlim.min(stored.len());
        let dist = distance_fn(&q[..limit], &stored[..limit]);
        local.push((doc_id, dist));
        if local.len() > request.k * 4 {
            local.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            local.truncate(request.k);
        }
    }
    local.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    local.truncate(request.k);

    Ok(local
        .into_iter()
        .map(|(doc_id, distance)| {
            let record_id = reader
                .get_record(doc_id)
                .ok()
                .map(|r| r.id())
                .unwrap_or_else(Uuid::new_v4);
            ScoredRecord {
                id: record_id,
                segment_id,
                doc_id,
                distance,
            }
        })
        .collect())
}
