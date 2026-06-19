import api from './client'

/**
 * Typed client + DTOs for the searchable semantic LLM trace store
 * (`GET /v1/admin/llm-traces`, `GET /v1/admin/llm-traces/clusters`).
 * Mirrors `crates/veda-gateway/src/rest/llm_traces.rs`. Admin-only; the
 * bearer token is attached by the shared axios instance.
 */

/** One trace returned from a semantic search. */
export interface TraceHit {
  prompt: string
  response: string
  /** Comma-joined tool ids invoked during the run ("" when none). */
  tool_ids: string
  /** Grounding/faithfulness score in [0,1]; -1 when not scored. */
  faithfulness: number
  /** Estimated cost of the run in USD (0 when the model is unpriced). */
  cost_usd: number
  cache_hit: boolean
  guardrail_flagged: boolean
  latency_ms: number
  ts: number
  /** Cosine similarity of the stored trace to the query, in [-1, 1]. */
  score: number
}

export interface TraceSearchResponse {
  hits: TraceHit[]
}

/** One failure cluster: a group of semantically similar trace prompts. */
export interface TraceCluster {
  representative: string
  size: number
  member_ids: string[]
  /** Mean faithfulness across scored members; -1 when none was scored. */
  avg_faithfulness: number
}

export interface TraceClustersResponse {
  clusters: TraceCluster[]
}

export interface TraceSearchParams {
  /** Natural-language search query. */
  q?: string
  /** Max hits to return (default 20). */
  limit?: number
}

export interface TraceClustersParams {
  /** Max recent traces to consider (default 20). */
  limit?: number
  /** Only cluster runs with faithfulness < this ("failure" view). */
  faithfulness_max?: number
}

export const searchLlmTraces = (params: TraceSearchParams) =>
  api.get<TraceSearchResponse>('/admin/llm-traces', { params })

export const fetchLlmTraceClusters = (params?: TraceClustersParams) =>
  api.get<TraceClustersResponse>('/admin/llm-traces/clusters', { params })
