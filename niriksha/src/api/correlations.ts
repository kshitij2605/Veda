import api from './client'

/**
 * Typed client + DTOs for the slow-query / error correlation analyzer
 * (`GET /v1/admin/correlations`). Mirrors
 * `crates/veda-gateway/src/rest/correlations.rs`. Admin-only; the bearer
 * token is attached by the shared axios instance.
 */

/** Per-bucket rolled-up statistics within a dimension. */
export interface CorrelationBucket {
  /** Bucket label (collection name, "dense", "filter=true", "hnsw", …). */
  key: string
  count: number
  p50_ms: number
  p95_ms: number
  slow_count: number
  error_count: number
  /** Fraction of queries in this bucket that errored, in [0, 1]. */
  error_rate: number
}

/** One correlation dimension and its worst-first ranked buckets. */
export interface CorrelationDimension {
  /** collection | query_type | has_filter | reranked | index_type |
   *  result_count | top_k. */
  dimension: string
  buckets: CorrelationBucket[]
}

export interface CorrelationSummary {
  window_secs: number
  total_count: number
  p50_ms: number
  p95_ms: number
  slow_count: number
  error_count: number
  error_rate: number
}

export interface CorrelationsResponse {
  summary: CorrelationSummary
  dimensions: CorrelationDimension[]
}

export interface CorrelationsParams {
  /** Restrict to a single collection (user-facing name). */
  collection?: string
  /** Analysis window in seconds (default 300). */
  window_secs?: number
  /** Single dimension to analyze. Omitted → all dimensions. */
  dimension?: string
}

export const fetchCorrelations = (params?: CorrelationsParams) =>
  api.get<CorrelationsResponse>('/admin/correlations', { params })
