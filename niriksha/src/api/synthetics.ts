import api from './client'

/**
 * Typed client + DTOs for the synthetic self-check probe admin API.
 * Mirrors `crates/veda-gateway/src/rest/synthetics.rs` and the model in
 * `advisor/synthetic.rs`. Admin-only; the bearer token is attached by
 * the shared axios instance.
 *
 * `ProbeKind` is INTERNALLY tagged on the wire (`#[serde(tag = "kind",
 * rename_all = "snake_case")]`):
 *   { "kind": "health" }
 *   { "kind": "search", "collection": "docs", "query"?: "…" }
 *   { "kind": "embed", "text": "…" }
 */
export type ProbeKind =
  | { kind: 'health' }
  | { kind: 'search'; collection: string; query?: string }
  | { kind: 'embed'; text: string }

export type ProbeKindTag = ProbeKind['kind']

/** A user-defined synthetic probe. `id` is server-assigned on create. */
export interface SyntheticProbe {
  id: string
  name: string
  kind: ProbeKind
  /** Minimum seconds between runs (bounded below by the tick interval). */
  interval_secs: number
  /** Per-probe timeout for the in-process call. */
  timeout_ms: number
}

/** Create body — `id` optional (server assigns when empty/absent). */
export type SyntheticCreate = Omit<SyntheticProbe, 'id'> & { id?: string }

/** The outcome of one probe run. */
export interface ProbeResult {
  ts_ms: number
  ok: boolean
  latency_ms: number
  error?: string | null
}

/** Derived, point-in-time rollup over a probe's result ring. */
export interface SyntheticSummary {
  probe_id: string
  name: string
  consecutive_failures: number
  last_ok: boolean | null
  last_latency_ms: number | null
  uptime_pct: number | null
  last_ts_ms: number | null
}

export interface SyntheticsResponse {
  synthetics: SyntheticProbe[]
}

export interface SyntheticResultsResponse {
  summary: SyntheticSummary
  results: ProbeResult[]
}

export const listSynthetics = () =>
  api.get<SyntheticsResponse>('/admin/synthetics')

export const createSynthetic = (probe: SyntheticCreate) =>
  api.post<SyntheticProbe>('/admin/synthetics', probe)

export const getSynthetic = (id: string) =>
  api.get<SyntheticProbe>(`/admin/synthetics/${encodeURIComponent(id)}`)

export const updateSynthetic = (id: string, probe: SyntheticCreate) =>
  api.put<SyntheticProbe>(`/admin/synthetics/${encodeURIComponent(id)}`, probe)

export const deleteSynthetic = (id: string) =>
  api.delete(`/admin/synthetics/${encodeURIComponent(id)}`)

export const getSyntheticResults = (id: string) =>
  api.get<SyntheticResultsResponse>(
    `/admin/synthetics/${encodeURIComponent(id)}/results`,
  )

/** Human-readable label for a probe kind. */
export function probeKindLabel(kind: ProbeKind): string {
  switch (kind.kind) {
    case 'health':
      return 'Health'
    case 'search':
      return `Search · ${kind.collection}`
    case 'embed':
      return 'Embed'
  }
}
