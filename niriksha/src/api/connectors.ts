import api from './client'

/**
 * Typed client + DTOs for the data-source connectors admin API.
 * Mirrors `crates/veda-gateway/src/rest/connectors.rs`. Admin-only and
 * tenant-scoped; the bearer token is attached by the shared axios
 * instance.
 *
 * `kind` is a plain lowercase string on the wire (Rust enum with
 * `#[serde(rename_all = "lowercase")]`): `"web"` or `"websearch"`.
 */
export type ConnectorKind = 'web' | 'websearch'

/** Config for a `web` connector. Other kinds carry an opaque config. */
export interface WebConnectorConfig {
  urls: string[]
  collection: string
  deep_crawl?: boolean
  max_pages?: number
  chunker?: string
  chunk_size?: number
  chunk_overlap?: number
  ttl_secs?: number
}

/** A persisted data-source connector. `id` is server-assigned on create. */
export interface Connector {
  id: string
  kind: ConnectorKind
  name: string
  config: WebConnectorConfig | Record<string, unknown>
  enabled: boolean
  created_at: number
  updated_at: number
}

/** Create body — `id` optional (server assigns `conn-…` when absent). */
export type ConnectorCreate = Omit<Connector, 'id' | 'created_at' | 'updated_at'> & {
  id?: string
}

export interface ConnectorsResponse {
  connectors: Connector[]
}

export interface RunPerUrl {
  url: string
  title: string | null
  chunks_inserted: number
}

export interface RunResponse {
  pages: number
  chunks_inserted: number
  per_url: RunPerUrl[]
}

export const listConnectors = () => api.get<ConnectorsResponse>('/admin/connectors')

export const createConnector = (c: ConnectorCreate) =>
  api.post<Connector>('/admin/connectors', c)

export const getConnector = (id: string) =>
  api.get<Connector>(`/admin/connectors/${encodeURIComponent(id)}`)

export const updateConnector = (id: string, c: ConnectorCreate) =>
  api.put<Connector>(`/admin/connectors/${encodeURIComponent(id)}`, c)

export const deleteConnector = (id: string) =>
  api.delete(`/admin/connectors/${encodeURIComponent(id)}`)

export const runConnector = (id: string) =>
  api.post<RunResponse>(`/admin/connectors/${encodeURIComponent(id)}/run`)

/** Human-readable label for a kind. */
export function connectorKindLabel(kind: ConnectorKind): string {
  switch (kind) {
    case 'web':
      return 'Web crawl'
    case 'websearch':
      return 'Web search'
  }
}

/** Narrow a connector's config to the web shape (null-safe). */
export function asWebConfig(c: Connector): WebConnectorConfig | null {
  if (c.kind !== 'web') return null
  const cfg = c.config as WebConnectorConfig
  return { ...cfg, urls: Array.isArray(cfg?.urls) ? cfg.urls : [] }
}
