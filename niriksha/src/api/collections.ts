import api from './client'

export interface FieldSchema {
  name: string
  field_type: string
  dimensions?: number
}

export interface CollectionInfo {
  name: string
  fields: FieldSchema[]
  record_count: number
  created_at?: string
}

export interface HealthResponse {
  status: string
  version: string
}

export interface SearchParams {
  field: string
  vector: number[]
  filter?: string
  limit: number
}

export interface SearchHit {
  id: string
  distance: number
  fields: Record<string, unknown>
}

export interface SearchResponse {
  results: SearchHit[]
  took_ms: number
}

export const fetchHealth = () =>
  api.get<HealthResponse>('/admin/health')

export const fetchCollections = () =>
  api.get<{ collections: CollectionInfo[] }>('/collections')

export const fetchCollection = (name: string) =>
  api.get<CollectionInfo>(`/collections/${name}`)

export const createCollection = (body: { name: string; fields: FieldSchema[] }) =>
  api.post('/collections', body)

export const searchCollection = (name: string, params: SearchParams) =>
  api.post<SearchResponse>(`/collections/${name}/search`, params)

export const fetchRecords = (name: string, params?: { limit?: number; offset?: number }) =>
  api.get<{ records: Record<string, unknown>[]; total: number }>(`/collections/${name}/records`, { params })
