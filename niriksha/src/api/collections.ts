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

/** Full /search body. Mirrors the gateway's `SearchRequest`. */
export interface SearchRequest {
  query: {
    vector?: VectorQuery
  }
  filter?: string
  limit: number
  explain?: boolean
  truncate_to?: number
}

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

export const fetchRecords = (
  name: string,
  params?: { limit?: number; offset?: number },
) =>
  api.get<{ records: Record<string, unknown>[]; total: number }>(
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
