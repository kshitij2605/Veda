import api from './client'

/**
 * Typed client + DTOs for the node-resources admin API.
 * Mirrors `crates/veda-gateway/src/rest/resources.rs`. Admin-only; the
 * bearer token is attached by the shared axios instance. Every signal
 * field is nullable (null on non-Linux hosts).
 */

export interface DiskView {
  total_bytes: number | null
  free_bytes: number | null
  used_pct: number | null
}

export interface LoadView {
  load1: number | null
  load5: number | null
  load15: number | null
}

export interface FdView {
  open: number | null
  limit: number | null
}

export interface MemoryView {
  total_bytes: number | null
  available_bytes: number | null
}

export interface ResourcesView {
  disk: DiskView
  /** Busy-CPU %; null with fewer than two advisor snapshots. */
  cpu_pct: number | null
  load: LoadView
  num_cpus: number
  fds: FdView
  memory: MemoryView
  rss_bytes: number | null
}

export const getResources = () =>
  api.get<ResourcesView>('/admin/resources')
