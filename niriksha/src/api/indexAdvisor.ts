import api from './client'

/**
 * Typed client + DTOs for the ANN index advisor
 * (`GET /v1/admin/index-advisor`). Mirrors
 * `crates/veda-gateway/src/rest/index_advisor.rs`. Admin-only; the bearer
 * token is attached by the shared axios instance.
 *
 * Each row pairs a collection's configured index type with its observed
 * query p95 and its latest PROXY recall@k (sampled queries + capped brute
 * force — directional, not exact), plus the index-advisor rule
 * recommendations and any open index events. The opt-in "Apply reindex"
 * button POSTs the suggested body to `/v1/collections/{name}/reindex`.
 */

export type Severity = 'low' | 'medium' | 'high'

export interface IndexParams {
  m?: number
  ef_construction?: number
  ef_search?: number
}

export interface Recommendation {
  rule_id: string
  severity: Severity
  title: string
  detail: string
  recommendation: string
  runbook?: string
}

export interface IndexOpenEvent {
  id: string
  rule_id: string
  severity: Severity
  title: string
  detail: string
  recommendation: string
  acknowledged?: boolean
}

export interface IndexAdvisorRow {
  collection: string
  index_type: string
  params: IndexParams
  p95_ms?: number
  recall?: number
  /** Whether the recall figure is a proxy measurement (always true when set). */
  is_proxy?: boolean
  /** Suggested body to POST to `/collections/{name}/reindex` (directional). */
  suggested_reindex?: Record<string, unknown>
  recommendations: Recommendation[]
  open_events: IndexOpenEvent[]
}

export interface IndexAdvisorResponse {
  rows: IndexAdvisorRow[]
}

export const fetchIndexAdvisor = () =>
  api.get<IndexAdvisorResponse>('/admin/index-advisor')

/** POST the (opt-in) suggested reindex body for a collection. */
export const applyReindex = (collection: string, body: Record<string, unknown>) =>
  api.post(`/collections/${encodeURIComponent(collection)}/reindex`, body)
