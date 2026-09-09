import api from './client'

/**
 * Shape of one field in a collection schema, as returned by
 * GET /v1/collections/{name}. Mirrors `FieldInfo` on the gateway:
 * keys are `type` (not `field_type`) and `dimension` (not `dimensions`).
 */
export interface FieldSchema {
  name: string
  type: string
  filterable: boolean
  dimension?: number
  metric?: string
  model?: string
}

/**
 * One row in GET /v1/collections (list endpoint) — summary only.
 * The list endpoint does not return field arrays or record counts.
 */
export interface CollectionSummary {
  name: string
  field_count: number
  shard_count: number
}

/**
 * Full schema for one collection, returned by GET /v1/collections/{name}.
 */
/**
 * Read-in-place source on an external (Parquet) collection. Present only
 * when the collection was registered with `external`; such a collection
 * reports an EMPTY `fields` array, because its schema lives in the Parquet
 * footer rather than in Veda's own schema.
 */
export interface ExternalSource {
  format: string
  uri: string
  id_field: string
  vector_field: string
  metric?: string
}

export interface CollectionInfo {
  name: string
  fields: FieldSchema[]
  shard_count: number
  replication_factor: number
  external?: ExternalSource
}

export interface HealthResponse {
  status: string
  version: string
  checks?: Record<string, string>
}

/** Maps to gateway `SingleVectorQuery`. */
export interface VectorQuery {
  field: string
  values: number[]
}

export interface SparseVectorQuery {
  field: string
  indices: number[]
  values: number[]
}

export interface Bm25Query {
  field: string
  query: string
}

export interface HybridQuery {
  /** Dense leg — name MUST be `dense` to match the gateway's
   *  HybridQuery struct (was previously `vector` and the request
   *  422-ed on the server). */
  dense: VectorQuery
  bm25: Bm25Query
  /** Weight on the dense leg, in [0, 1]. The BM25 leg gets
   *  (1 - dense_weight). Default on the gateway is 0.5. */
  dense_weight?: number
  /** "rrf" (default) or "weighted_sum". */
  fusion?: string
}

export interface FusionParams {
  k?: number
}

export interface RerankConfig {
  field: string
  query: string
  top_k?: number
  model?: string
}

/** Full /search body. Mirrors the gateway's `SearchRequest`. */
export interface SearchRequest {
  query: {
    vector?: VectorQuery
    sparse_vector?: SparseVectorQuery
    bm25?: Bm25Query
    hybrid?: HybridQuery
    fusion?: string
    fusion_params?: FusionParams
  }
  filter?: string
  limit: number
  explain?: boolean
  truncate_to?: number
  rerank?: RerankConfig
}

export interface EmbedRequest {
  text?: string
  image_b64?: string
  model?: string
}

export interface EmbedResponse {
  model: string
  modality: string
  dimension: number
  values: number[]
  elapsed_ms: number
}

export const embedQuery = (body: EmbedRequest) =>
  api.post<EmbedResponse>('/embed', body)

export interface RouteResponse {
  strategy: 'Dense' | 'Sparse' | 'BM25' | 'Hybrid' | string
  rerank: boolean
  suggested_k: number
  reasoning: string
}

export const routeQuery = (query: string) =>
  api.post<RouteResponse>('/route', { query })

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}
export interface LlmCompleteResponse {
  model: string
  content: string
  finish_reason?: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  elapsed_ms: number
}
export interface CoverageHit {
  doc_id: string
  count: number
  fraction: number
}

export interface CoverageResponse {
  window_secs: number
  total_queries: number
  total_chunks: number
  retrieved_chunks: number
  orphan_count: number
  top_hits: CoverageHit[]
  orphan_sample: string[]
}

export const fetchCoverage = (collection: string, window: string = '7d') =>
  api.get<CoverageResponse>(
    `/collections/${encodeURIComponent(collection)}/coverage?window=${encodeURIComponent(window)}`,
  )

// ── Admin Nodes (Phase 6.4) ──────────────────────────────────────────────
export interface NodeInfo {
  id: string
  address: string
  role: string
  status: string
  version: string
  uptime_secs: number
  shard_count: number
  collection_count: number
  is_self: boolean
}
export interface ListNodesResponse {
  /** "single" (no Raft) or "raft" (quorum). */
  mode: string
  nodes: NodeInfo[]
}
export const listNodes = () =>
  api.get<ListNodesResponse>('/admin/nodes')

// ── SavedObjects (Phase 6.5 follow-up) ──────────────────────────────────
export interface SavedObjectDTO {
  id: string
  type: string
  name: string
  body: unknown
  owner?: string
  created_at: number
  updated_at: number
  last_used_at?: number
}
export const listSavedObjects = () =>
  api.get<{ objects: SavedObjectDTO[] }>('/saved_objects')
export const putSavedObject = (id: string, obj: SavedObjectDTO) =>
  api.put<SavedObjectDTO>(
    `/saved_objects/${encodeURIComponent(id)}`,
    obj,
  )
export const deleteSavedObject = (id: string) =>
  api.delete<void>(`/saved_objects/${encodeURIComponent(id)}`)

export const llmComplete = (
  messages: ChatMessage[],
  opts?: { temperature?: number; max_tokens?: number; json_mode?: boolean },
) =>
  api.post<LlmCompleteResponse>('/llm/complete', {
    messages,
    temperature: opts?.temperature,
    max_tokens: opts?.max_tokens,
    json_mode: opts?.json_mode,
  })

export interface SearchHit {
  id: string
  distance: number
  fields: Record<string, unknown>
}

export interface ExplainTiming {
  filter_ms: number
  ann_search_ms: number
  scoring_ms: number
  fusion_ms: number
  total_ms: number
}

export interface HybridExplain {
  fusion: string
  dense_weight: number
  bm25_weight: number
  dense_top: string[]
  bm25_top: string[]
  final_top: string[]
}

export interface SearchResponse {
  results: SearchHit[]
  took_ms: number
  explain?: ExplainTiming
  hybrid_explain?: HybridExplain
}

/**
 * Shape of one field when creating a collection. Matches gateway
 * `FieldDef`.
 */
export interface CreateFieldDef {
  name: string
  type: string
  filterable?: boolean
  dimension?: number
  metric?: string
  model?: string
}

export const fetchHealth = () => api.get<HealthResponse>('/admin/health')

export const fetchCollections = () =>
  api.get<{ collections: CollectionSummary[] }>('/collections')

export const fetchCollection = (name: string) =>
  api.get<CollectionInfo>(`/collections/${name}`)

export const createCollection = (body: {
  name: string
  fields: CreateFieldDef[]
  shard_count?: number
  replication_factor?: number
}) => api.post('/collections', body)

export const dropCollection = (name: string) =>
  api.delete(`/collections/${name}`)

export const searchCollection = (name: string, body: SearchRequest) =>
  api.post<SearchResponse>(`/collections/${name}/search`, body)

/**
 * One row in GET /v1/collections/{name}/records — gateway wraps fields
 * under a `fields` key alongside the canonical id.
 */
export interface RecordRow {
  id: string
  fields: Record<string, unknown>
}

export const fetchRecords = (
  name: string,
  params?: { limit?: number; offset?: number },
) =>
  api.get<{ records: RecordRow[]; total: number }>(
    `/collections/${name}/records`,
    { params },
  )

export interface FieldQuality {
  missing: number
  min_length: number | null
  max_length: number | null
  avg_length: number
  null_or_empty: number
}

export interface DuplicatePair {
  id_a: string
  id_b: string
  distance: number
}

export interface EvaluateResponse {
  collection: string
  records_evaluated: number
  duplicate_pairs: DuplicatePair[]
  field_quality: Record<string, FieldQuality>
}

export const evaluateCollection = (
  name: string,
  params?: {
    vector_field?: string
    duplicate_threshold?: number
    sample?: number
  },
) => api.get<EvaluateResponse>(`/collections/${name}/evaluate`, { params })

// ── Index maintenance ────────────────────────────────────────────────────
// The server has exposed these since before Niriksha's audit was written; the
// audit's "IndexManager is read-only … the server doesn't expose those
// mutations either" was stale on both halves.

export interface ShardOptimizeResult {
  shard_id: number
  segments_before: number
  segments_after: number
}

export interface OptimizeResponse {
  collection: string
  per_shard: ShardOptimizeResult[]
}

/**
 * Force-merge every shard's sealed segments down to one, then rebuild the
 * query executors.
 *
 * Safe to run at any time and safe to repeat — a collection that is already
 * consolidated merges nothing. Worth doing on a fragmented collection: a bulk
 * load that seals into many segments makes each per-segment ANN search hand
 * fewer true neighbours to the cross-segment merge, which is the recall
 * collapse `recall_at_scale_many_hnsw_segments` covers.
 */
export const optimizeCollection = (name: string) =>
  api.post<OptimizeResponse>(`/collections/${name}/optimize`, {})

export interface ReindexRequest {
  /** Re-run the embedding model over each chunk. Expensive. */
  re_embed?: boolean
  /** Source field holding chunk text. Server default: "content". */
  text_field?: string
}

export interface ReindexResponse {
  /** New collection created with the same schema, then alias-swapped in. */
  target: string
  status: string
}

/**
 * Rebuild the collection into a fresh one and alias-swap it in.
 *
 * NOT the same weight class as `optimizeCollection`. This copies every
 * record into a new collection and, with `re_embed`, re-runs the embedding
 * model over the whole corpus — minutes to hours and real cost against the
 * embedding endpoint on anything large. The UI gates it behind a typed
 * confirmation for that reason, not because it is destructive: the alias swap
 * is atomic and the prior data is not deleted.
 */
export const reindexCollection = (name: string, body: ReindexRequest = {}) =>
  api.post<ReindexResponse>(`/collections/${name}/reindex`, body)
