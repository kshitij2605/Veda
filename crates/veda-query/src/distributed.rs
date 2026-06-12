use std::collections::HashMap;

use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::fusion::{fuse_results, FusedResult, MultiVectorQuery};
use crate::search::{
    MultiVectorSearchRequest, ScoredRecord, SearchResponse, SearchTimings, SparseSearchRequest,
    VectorSearchRequest,
};

/// Trait for executing search on a single shard.
/// Implemented by local shard executor (in-process) or remote gRPC client.
pub trait ShardSearcher: Send + Sync {
    fn search_shard(
        &self,
        shard_id: u32,
        request: &VectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)>;

    /// Sparse-vector search. Default implementation returns an error so new
    /// shard implementations are not forced to support sparse retrieval.
    fn search_shard_sparse(
        &self,
        _shard_id: u32,
        _request: &SparseSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        Err(VedaError::Query(
            "sparse search not supported by this shard searcher".into(),
        ))
    }

    /// Multi-vector (ColPali / MaxSim) search.
    fn search_shard_multi_vector(
        &self,
        _shard_id: u32,
        _request: &MultiVectorSearchRequest,
    ) -> Result<(Vec<ScoredRecord>, SearchTimings)> {
        Err(VedaError::Query(
            "multi-vector search not supported by this shard searcher".into(),
        ))
    }
}

/// Distributed query coordinator.
///
/// Fans out searches to per-shard executors and merges results into a
/// global top-k.
pub struct DistributedQueryExecutor {
    shard_searchers: HashMap<u32, Box<dyn ShardSearcher>>,
}

impl Default for DistributedQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedQueryExecutor {
    pub fn new() -> Self {
        Self {
            shard_searchers: HashMap::new(),
        }
    }

    /// Register a shard searcher for a given shard ID.
    pub fn register_shard(&mut self, shard_id: u32, searcher: Box<dyn ShardSearcher>) {
        self.shard_searchers.insert(shard_id, searcher);
    }

    /// Execute a multi-vector (ColPali) search across all registered shards.
    pub fn search_multi_vector_colpali(
        &self,
        request: &MultiVectorSearchRequest,
    ) -> Result<(SearchResponse, SearchTimings)> {
        if self.shard_searchers.is_empty() {
            return Err(VedaError::Query("no shards registered".into()));
        }
        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut total_candidates: usize = 0;
        let mut agg = SearchTimings::default();
        for (&shard_id, searcher) in &self.shard_searchers {
            let (rs, ts) = searcher.search_shard_multi_vector(shard_id, request)?;
            total_candidates += rs.len();
            all_results.extend(rs);
            agg.filter_ms += ts.filter_ms;
            agg.ann_search_ms += ts.ann_search_ms;
            agg.scoring_ms += ts.scoring_ms;
        }
        let scoring_start = std::time::Instant::now();
        all_results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);
        agg.scoring_ms += scoring_start.elapsed().as_secs_f64() * 1000.0;
        Ok((
            SearchResponse {
                results: all_results,
                total_candidates,
            },
            agg,
        ))
    }

    /// Execute a sparse-vector search across all registered shards.
    pub fn search_sparse(
        &self,
        request: &SparseSearchRequest,
    ) -> Result<(SearchResponse, SearchTimings)> {
        if self.shard_searchers.is_empty() {
            return Err(VedaError::Query("no shards registered".into()));
        }

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut total_candidates: usize = 0;
        let mut agg_timings = SearchTimings::default();

        for (&shard_id, searcher) in &self.shard_searchers {
            let (shard_results, timings) = searcher.search_shard_sparse(shard_id, request)?;
            total_candidates += shard_results.len();
            all_results.extend(shard_results);
            agg_timings.filter_ms += timings.filter_ms;
            agg_timings.ann_search_ms += timings.ann_search_ms;
            agg_timings.scoring_ms += timings.scoring_ms;
        }

        let scoring_start = std::time::Instant::now();
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);
        agg_timings.scoring_ms += scoring_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            SearchResponse {
                results: all_results,
                total_candidates,
            },
            agg_timings,
        ))
    }

    /// Execute a single-vector search across all registered shards.
    ///
    /// Fans out to each shard, merges results, returns global top-k.
    pub fn search(&self, request: &VectorSearchRequest) -> Result<(SearchResponse, SearchTimings)> {
        if self.shard_searchers.is_empty() {
            return Err(VedaError::Query("no shards registered".into()));
        }

        let mut all_results: Vec<ScoredRecord> = Vec::new();
        let mut total_candidates: usize = 0;
        let mut agg_timings = SearchTimings::default();

        for (&shard_id, searcher) in &self.shard_searchers {
            let (shard_results, timings) = searcher.search_shard(shard_id, request)?;
            total_candidates += shard_results.len();
            all_results.extend(shard_results);
            agg_timings.filter_ms += timings.filter_ms;
            agg_timings.ann_search_ms += timings.ann_search_ms;
            agg_timings.scoring_ms += timings.scoring_ms;
        }

        let scoring_start = std::time::Instant::now();
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(request.k);
        agg_timings.scoring_ms += scoring_start.elapsed().as_secs_f64() * 1000.0;

        Ok((
            SearchResponse {
                results: all_results,
                total_candidates,
            },
            agg_timings,
        ))
    }

    /// Execute a multi-vector fusion search across all registered shards.
    ///
    /// For each field query, fans out to shards, collects per-field results,
    /// then fuses across fields.
    pub fn search_multi_vector(
        &self,
        query: &MultiVectorQuery,
        filter: Option<&str>,
        metric: DistanceMetric,
    ) -> Result<Vec<FusedResult>> {
        if self.shard_searchers.is_empty() {
            return Err(VedaError::Query("no shards registered".into()));
        }

        let mut field_results: Vec<(String, Vec<ScoredRecord>)> = Vec::new();
        let weights: Vec<f32> = query.queries.iter().map(|q| q.weight).collect();

        for field_query in &query.queries {
            let request = VectorSearchRequest {
                field: field_query.field.clone(),
                query_vector: field_query.vector.clone(),
                k: query.limit,
                metric,
                filter: filter.map(|s| s.to_owned()),
                filter_expr: None,
                truncate_to: None,
                read_ts: None,
            };

            let mut merged: Vec<ScoredRecord> = Vec::new();
            for (&shard_id, searcher) in &self.shard_searchers {
                let (shard_results, _timings) = searcher.search_shard(shard_id, &request)?;
                merged.extend(shard_results);
            }

            // Sort per-field shard results by distance ascending.
            merged.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            field_results.push((field_query.field.clone(), merged));
        }

        let fused = fuse_results(&field_results, query.fusion, &weights, query.limit);
        Ok(fused)
    }
}
