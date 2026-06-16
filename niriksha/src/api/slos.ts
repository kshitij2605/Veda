import api from './client'

/**
 * Typed client + DTOs for the service-level-objective admin API.
 * Mirrors `crates/veda-gateway/src/rest/slos.rs` and the engine model in
 * `advisor/slo.rs`. Admin-only; the bearer token is attached by the
 * shared axios instance.
 *
 * `SloKind` is INTERNALLY tagged on the wire (`#[serde(tag = "kind",
 * rename_all = "snake_case")]`) — the discriminant lives next to the
 * variant's own fields, e.g. `{ "kind": "latency", "threshold_ms": 50 }`.
 */
export type SloKind =
  | { kind: 'latency'; threshold_ms: number }
  | { kind: 'availability' }
  | { kind: 'error_rate' }
  | { kind: 'synthetic'; probe_id: string }

export type SloKindTag = SloKind['kind']

/** A burn-rate alert window: alert when the burn rate over
 * `window_secs` exceeds `threshold`. */
export interface BurnConfig {
  window_secs: number
  threshold: number
}

/** A user-defined SLO. `id` is server-assigned on create when omitted. */
export interface SloDef {
  id: string
  name: string
  kind: SloKind
  /** Objective: `SLI >= target_pct` (e.g. 99.0). */
  target_pct: number
  /** Compliance window in seconds (e.g. 2_592_000 = 30d). */
  window_secs: number
  fast_burn: BurnConfig
  slow_burn: BurnConfig
}

/** Create body — `id` optional (server assigns when empty/absent). */
export type SloCreate = Omit<SloDef, 'id'> & { id?: string }

/** Point-in-time SLO evaluation. Ratio fields are `null` with no data. */
export interface SloStatus {
  slo_id: string
  sli_pct: number | null
  target_pct: number
  good: number
  total: number
  error_budget_remaining_pct: number | null
  burn_rate_fast: number | null
  burn_rate_slow: number | null
  compliant: boolean
  fast_burn_threshold: number
  slow_burn_threshold: number
  breaching_fast: boolean
  breaching_slow: boolean
  budget_exhausted: boolean
}

export interface SlosResponse {
  slos: SloDef[]
}

/** Sensible burn-window defaults (match the Rust `default_*_burn`). */
export const DEFAULT_FAST_BURN: BurnConfig = { window_secs: 300, threshold: 14.4 }
export const DEFAULT_SLOW_BURN: BurnConfig = { window_secs: 3600, threshold: 6.0 }

export const listSlos = () => api.get<SlosResponse>('/admin/slos')

export const createSlo = (def: SloCreate) => api.post<SloDef>('/admin/slos', def)

export const getSlo = (id: string) =>
  api.get<SloDef>(`/admin/slos/${encodeURIComponent(id)}`)

export const updateSlo = (id: string, def: SloCreate) =>
  api.put<SloDef>(`/admin/slos/${encodeURIComponent(id)}`, def)

export const deleteSlo = (id: string) =>
  api.delete(`/admin/slos/${encodeURIComponent(id)}`)

export const getSloStatus = (id: string) =>
  api.get<SloStatus>(`/admin/slos/${encodeURIComponent(id)}/status`)

/** Human-readable label for a kind. */
export function sloKindLabel(kind: SloKind): string {
  switch (kind.kind) {
    case 'latency':
      return `Latency < ${kind.threshold_ms} ms`
    case 'availability':
      return 'Availability'
    case 'error_rate':
      return 'Error rate'
    case 'synthetic':
      return `Synthetic · ${kind.probe_id}`
  }
}
