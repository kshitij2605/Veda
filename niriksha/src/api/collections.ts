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
export interface CollectionInfo {
  name: string
  fields: FieldSchema[]
  shard_count: number
  replication_factor: number
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
  vector: VectorQuery
  bm25: Bm25Query
  /** Per-leg weights for weighted-sum fusion. */
  vector_weight?: number
  bm25_weight?: number
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

export interface SearchResponse {
  results: SearchHit[]
  took_ms: number
  explain?: ExplainTiming
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
