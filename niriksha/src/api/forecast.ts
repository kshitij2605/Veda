import api from './client'

/**
 * Typed client + DTOs for the capacity-forecast advisor
 * (`GET /v1/admin/forecast`). Mirrors
 * `crates/veda-gateway/src/rest/forecast.rs`. Admin-only; the bearer token
 * is attached by the shared axios instance.
 *
 * Each forecast pairs a metric's current value with a least-squares trend
 * (slope + R² confidence) and the projected epoch-ms crossing of a
 * threshold (`eta_ms`, null when the trend is flat or moving away).
 * Forecasts are directional projections — advisory only.
 */

export interface Forecast {
  /** Metric key, e.g. `disk_used_pct`, `mem_available_bytes`, `recall`. */
  metric: string
  /** Scope target (e.g. a collection name for recall); absent otherwise. */
  scope?: string
  current: number
  slope_per_sec: number
  /** Coefficient of determination (fit confidence) in [0, 1]. */
  r2: number
  threshold: number
  /** Projected epoch-ms at which `current` crosses `threshold`; null when not trending toward it. */
  eta_ms: number | null
}

export interface ForecastResponse {
  forecasts: Forecast[]
}

export const fetchForecast = () => api.get<ForecastResponse>('/admin/forecast')
