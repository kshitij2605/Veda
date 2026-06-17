import api from './client'

/**
 * Typed client + DTOs for the self-service dependency map
 * (`GET /v1/admin/depmap`). Mirrors
 * `crates/veda-gateway/src/rest/depmap.rs`. Admin-only; the bearer token
 * is attached by the shared axios instance.
 *
 * The graph is static / declarative (v1): it describes Veda's request
 * topology (gateway → query → shards/index → embedding/llm/rerank →
 * blob/nats), not live per-node health.
 */

/** Coarse node kind, used to group / color nodes in the UI. */
export type DepNodeKind =
  | 'gateway'
  | 'query'
  | 'storage'
  | 'index'
  | 'external'

export interface DepNode {
  id: string
  kind: DepNodeKind
  label: string
}

export interface DepEdge {
  from: string
  to: string
  label?: string
}

export interface DepMapResponse {
  nodes: DepNode[]
  edges: DepEdge[]
}

export const fetchDepMap = () => api.get<DepMapResponse>('/admin/depmap')
