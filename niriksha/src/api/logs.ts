import api from './client'

/**
 * Typed client + DTOs for the Log Insights admin API.
 * Mirrors `crates/veda-gateway/src/rest/logs.rs`. All endpoints are
 * admin-only; the bearer token is attached by the shared axios instance.
 */

/** A single captured log record (matches `logs::LogRecord`). */
export interface LogRecord {
  /** Unix milliseconds when the event was captured. */
  ts_ms: number
  /** `ERROR` | `WARN` | `INFO` | `DEBUG` | `TRACE`. */
  level: string
  /** Emitting module path / target. */
  target: string
  message: string
  /** Remaining structured fields on the event. */
  fields: Record<string, unknown>
  /** OTel trace id (32-hex) when a valid trace context was active. */
  trace_id?: string | null
  /** OTel span id (16-hex) when a valid trace context was active. */
  span_id?: string | null
}

export interface LogsResponse {
  logs: LogRecord[]
}

/** A semantic category of recent WARN/ERROR messages (`logs::LogCategory`). */
export interface LogCategory {
  id: string
  representative: string
  count: number
  max_severity: string
  examples: string[]
  first_seen_ms: number
  last_seen_ms: number
}

export interface CategoriesResponse {
  categories: LogCategory[]
}

/** A slow search read from a per-collection query log. */
export interface SlowQuery {
  collection: string
  ts_ms: number
  took_ms: number
  top_k: number
  trace_id?: string | null
}

export interface SlowQueriesResponse {
  slow_queries: SlowQuery[]
}

export interface LogsQuery {
  /** Minimum severity (`ERROR`, `WARN`, ...). Omitted → all levels. */
  level?: string
  /** Case-insensitive substring over message/target. */
  q?: string
  /** Max records (default 200, capped at 1000 server-side). */
  limit?: number
}

export const fetchLogs = (params?: LogsQuery) =>
  api.get<LogsResponse>('/admin/logs', { params })

export const fetchLogCategories = () =>
  api.get<CategoriesResponse>('/admin/logs/categories')

export interface SlowQueriesParams {
  collection?: string
  limit?: number
}

export const fetchSlowQueries = (params?: SlowQueriesParams) =>
  api.get<SlowQueriesResponse>('/admin/slow-queries', { params })
