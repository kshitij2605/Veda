import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { Badge } from '../components/Badge'
import api from '../api/client'
import { listNodes, type ListNodesResponse } from '../api/collections'
import { getResources, type ResourcesView } from '../api/resources'
import { fetchForecast, type Forecast } from '../api/forecast'
import { SyntheticsPanel } from './SyntheticsPanel'

interface MetricEntry {
  name: string
  value: string
}

function parsePrometheusText(text: string): MetricEntry[] {
  const entries: MetricEntry[] = []
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const spaceIdx = trimmed.lastIndexOf(' ')
    if (spaceIdx === -1) continue
    entries.push({
      name: trimmed.slice(0, spaceIdx),
      value: trimmed.slice(spaceIdx + 1),
    })
  }
  return entries
}

export function SystemHealth() {
  const { collections, health, loading, fetchCollections, fetchHealth } = useCollectionStore()
  const [metrics, setMetrics] = useState<MetricEntry[]>([])
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState<string | null>(null)
  const [nodes, setNodes] = useState<ListNodesResponse | null>(null)
  const [nodesError, setNodesError] = useState<string | null>(null)
  const [resources, setResources] = useState<ResourcesView | null>(null)
  const [resourcesError, setResourcesError] = useState<string | null>(null)

  const loadResources = useCallback(async () => {
    setResourcesError(null)
    try {
      const { data } = await getResources()
      setResources(data)
    } catch (err) {
      setResourcesError(err instanceof Error ? err.message : 'Failed to fetch resources')
    }
  }, [])

  const loadNodes = useCallback(async () => {
    setNodesError(null)
    try {
      const { data } = await listNodes()
      setNodes(data)
    } catch (err) {
      setNodesError(err instanceof Error ? err.message : 'Failed to fetch nodes')
    }
  }, [])

  const loadMetrics = useCallback(async () => {
    setMetricsLoading(true)
    setMetricsError(null)
    try {
      const { data } = await api.get('/admin/metrics', { responseType: 'text' })
      setMetrics(parsePrometheusText(typeof data === 'string' ? data : String(data)))
    } catch (err) {
      setMetricsError(err instanceof Error ? err.message : 'Failed to fetch metrics')
    } finally {
      setMetricsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    fetchCollections()
    loadMetrics()
    loadNodes()
    loadResources()
  }, [fetchHealth, fetchCollections, loadMetrics, loadNodes, loadResources])

  const statusVariant = health?.status === 'healthy' ? 'success' as const : 'error' as const
  const statusLabel = health?.status === 'healthy' ? 'Healthy' : health ? 'Unhealthy' : 'Unknown'

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="System Health"
        actions={<Badge label={statusLabel} variant={statusVariant} />}
      />

      {loading && <LoadingSpinner label="Loading system info..." />}

      {/* Server Info */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">Server Info</h2>
        <div className="rounded-lg border border-border bg-bg-secondary p-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <InfoCell label="Version" value={health?.version ?? '--'} />
            <InfoCell label="Status" value={statusLabel} />
            <InfoCell
              label="Uptime"
              value={
                nodes?.nodes[0]?.uptime_secs !== undefined
                  ? formatUptime(nodes.nodes[0].uptime_secs)
                  : '--'
              }
            />
            <InfoCell label="API" value="/v1" />
          </div>
        </div>
      </div>

      {/* Cluster nodes — Phase 6.4. Single-node today; Raft-ready
          shape so adding peers later doesn't change the UI. */}
      {nodes && (
        <div className="mb-6">
          <div className="flex items-baseline gap-3 mb-3">
            <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
              Cluster Nodes
            </h2>
            <span className="text-[10.5px] uppercase tracking-[0.08em] text-text-muted font-mono">
              mode: {nodes.mode}
            </span>
            {nodes.mode === 'single' && (
              <span className="text-[10.5px] text-text-muted italic">
                Raft consensus not yet wired — this is the lone leader.
              </span>
            )}
          </div>
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary text-text-muted text-left">
                  <th className="px-4 py-3 font-medium">Node</th>
                  <th className="px-4 py-3 font-medium">Role</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Address</th>
                  <th className="px-4 py-3 font-medium">Version</th>
                  <th className="px-4 py-3 font-medium">Uptime</th>
                  <th className="px-4 py-3 font-medium text-right">
                    Collections
                  </th>
                  <th className="px-4 py-3 font-medium text-right">Shards</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {nodes.nodes.map((n) => (
                  <tr
                    key={n.id}
                    className="hover:bg-bg-tertiary transition-colors"
                  >
                    <td className="px-4 py-3 font-mono text-xs text-text-primary">
                      {n.id}
                      {n.is_self && (
                        <span className="ml-2 text-[9.5px] uppercase tracking-[0.08em] text-text-muted">
                          this node
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <Badge
                        label={n.role}
                        variant={n.role === 'leader' ? 'accent' : 'default'}
                      />
                    </td>
                    <td className="px-4 py-3">
                      <Badge
                        label={n.status}
                        variant={
                          n.status === 'healthy'
                            ? 'success'
                            : n.status === 'degraded'
                              ? 'warning'
                              : 'error'
                        }
                      />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-text-secondary">
                      {n.address}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-text-secondary">
                      v{n.version}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-text-secondary tabular-nums">
                      {formatUptime(n.uptime_secs)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-text-secondary">
                      {n.collection_count}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-text-secondary">
                      {n.shard_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {nodesError && (
        <div className="mb-6 rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          Nodes lookup failed: {nodesError}
        </div>
      )}

      {/* Node resources — disk / CPU / load / fds / memory / RSS, from
          the ops-advisor's latest snapshot (GET /admin/resources).
          Signal fields are null on non-Linux hosts → rendered as "—". */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
            Resources
          </h2>
          <button
            onClick={loadResources}
            className="text-xs text-accent hover:text-accent-hover transition-colors"
          >
            Refresh
          </button>
        </div>
        {resourcesError ? (
          <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
            Resources lookup failed: {resourcesError}
          </div>
        ) : resources ? (
          <div className="rounded-lg border border-border bg-bg-secondary p-5">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-5 text-sm">
              <ResourceCell
                label="Disk used"
                value={fmtPct(resources.disk.used_pct)}
                sub={`${fmtGiB(resources.disk.free_bytes)} free`}
                warn={(resources.disk.used_pct ?? 0) >= 85}
                danger={(resources.disk.used_pct ?? 0) >= 90}
              />
              <ResourceCell label="CPU" value={fmtPct(resources.cpu_pct)} />
              <ResourceCell
                label="Load (1/5/15m)"
                value={`${fmtNum(resources.load.load1)} / ${fmtNum(resources.load.load5)} / ${fmtNum(resources.load.load15)}`}
                sub={`${resources.num_cpus} CPU${resources.num_cpus === 1 ? '' : 's'}`}
              />
              <ResourceCell
                label="Open FDs"
                value={
                  resources.fds.open != null
                    ? `${resources.fds.open}${resources.fds.limit != null ? ` / ${resources.fds.limit}` : ''}`
                    : '—'
                }
              />
              <ResourceCell
                label="System memory"
                value={`${fmtGiB(resources.memory.available_bytes)} avail`}
                sub={`of ${fmtGiB(resources.memory.total_bytes)}`}
              />
              <ResourceCell label="Process RSS" value={fmtGiB(resources.rss_bytes)} />
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
            Loading resources...
          </div>
        )}
      </div>

      {/* Capacity forecasts — predictive ETAs (disk-full / memory-exhaust /
          recall-decay) from the ops-advisor's snapshot-history trend fits
          (GET /admin/forecast). Warning-colored when an ETA is near. */}
      <ForecastPanel />

      {/* Synthetic self-check probes — uptime / last result / failures,
          from the ops-advisor's probe rings (GET /admin/synthetics). */}
      <SyntheticsPanel />

      {/* Subsystem checks from /admin/health */}
      {health?.checks && Object.keys(health.checks).length > 0 && (
        <div className="mb-6">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">
            Subsystem Checks
          </h2>
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary text-text-muted text-left">
                  <th className="px-4 py-3 font-medium">Subsystem</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {Object.entries(health.checks).map(([sub, st]) => (
                  <tr key={sub} className="hover:bg-bg-tertiary transition-colors">
                    <td className="px-4 py-3 font-mono text-xs text-text-primary">
                      {sub}
                    </td>
                    <td className="px-4 py-3">
                      <Badge
                        label={st}
                        variant={st === 'healthy' ? 'success' : 'error'}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Collections summary */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">
          Collections Overview
        </h2>
        {collections.length === 0 && !loading ? (
          <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
            No collections found.
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary text-text-muted text-left">
                  <th className="px-4 py-3 font-medium">Name</th>
                  <th className="px-4 py-3 font-medium">Fields</th>
                  <th className="px-4 py-3 font-medium">Shards</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {collections.map((col) => (
                  <tr
                    key={col.name}
                    className="hover:bg-bg-tertiary transition-colors"
                  >
                    <td className="px-4 py-3 font-medium text-text-primary">
                      {col.name}
                    </td>
                    <td className="px-4 py-3 text-text-secondary">
                      {col.field_count}
                    </td>
                    <td className="px-4 py-3 text-text-secondary">
                      {col.shard_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Metrics */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">Metrics</h2>
          <button
            onClick={loadMetrics}
            disabled={metricsLoading}
            className="text-xs text-accent hover:text-accent-hover transition-colors disabled:opacity-50"
          >
            {metricsLoading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        {metricsLoading && metrics.length === 0 && <LoadingSpinner label="Loading metrics..." />}

        {metricsError && (
          <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
            {metricsError}
          </div>
        )}

        {!metricsLoading && !metricsError && metrics.length === 0 && (
          <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
            No metrics available. The metrics endpoint may not be enabled.
          </div>
        )}

        {metrics.length > 0 && (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary text-text-muted text-left">
                  <th className="px-4 py-3 font-medium">Metric</th>
                  <th className="px-4 py-3 font-medium">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {metrics.map((m, i) => (
                  <tr key={i} className="hover:bg-bg-tertiary transition-colors">
                    <td className="px-4 py-3 font-mono text-xs text-text-primary">{m.name}</td>
                    <td className="px-4 py-3 font-mono text-xs text-text-secondary">{m.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  )
}

/** Human label + a metric's current value for a forecast row. */
function forecastLabel(f: Forecast): { title: string; current: string } {
  switch (f.metric) {
    case 'disk_used_pct':
      return { title: 'Disk full', current: `${f.current.toFixed(1)}% used` }
    case 'mem_available_bytes':
      return {
        title: 'Memory exhaustion',
        current: `${(f.current / 1024 ** 3).toFixed(1)} GiB available`,
      }
    case 'recall':
      return {
        title: `Recall decay${f.scope ? ` · ${f.scope}` : ''}`,
        current: `recall ${f.current.toFixed(3)}`,
      }
    default:
      return { title: f.metric, current: f.current.toFixed(2) }
  }
}

/** ETA epoch-ms → "in ~N days" / "in ~N hours"; null → "no trend". */
function formatEta(etaMs: number | null): { text: string; days: number | null } {
  if (etaMs == null) return { text: 'stable — no projected crossing', days: null }
  const dtMs = Math.max(0, etaMs - Date.now())
  const days = dtMs / 86_400_000
  if (days >= 1) return { text: `in ~${days.toFixed(1)} days`, days }
  const hours = dtMs / 3_600_000
  return { text: `in ~${hours.toFixed(1)} hours`, days }
}

/**
 * Capacity-forecast panel. Reads `GET /admin/forecast` (admin-only) and
 * renders one row per metric with its current value, projected ETA, and
 * fit confidence (R²). Rows whose ETA is within 7 days are warning-colored;
 * within 2 days, danger-colored.
 */
function ForecastPanel() {
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  const load = useCallback(async () => {
    setError(null)
    try {
      const { data } = await fetchForecast()
      setForecasts(data.forecasts)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch forecasts')
    } finally {
      setLoaded(true)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
          Capacity Forecasts
        </h2>
        <button
          onClick={load}
          className="text-xs text-accent hover:text-accent-hover transition-colors"
        >
          Refresh
        </button>
      </div>
      {error ? (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          Forecast lookup failed: {error}
        </div>
      ) : forecasts.length === 0 ? (
        <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
          {loaded
            ? 'No forecasts yet — the advisor needs enough snapshot history to fit a trend.'
            : 'Loading forecasts...'}
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-bg-secondary text-text-muted text-left">
                <th className="px-4 py-3 font-medium">Metric</th>
                <th className="px-4 py-3 font-medium">Current</th>
                <th className="px-4 py-3 font-medium">Projected</th>
                <th className="px-4 py-3 font-medium text-right">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {forecasts.map((f, i) => {
                const { title, current } = forecastLabel(f)
                const { text, days } = formatEta(f.eta_ms)
                const etaColor =
                  days == null
                    ? 'text-text-muted'
                    : days <= 2
                      ? 'text-error'
                      : days <= 7
                        ? 'text-warning'
                        : 'text-text-secondary'
                return (
                  <tr key={i} className="hover:bg-bg-tertiary transition-colors">
                    <td className="px-4 py-3 text-text-primary font-medium">{title}</td>
                    <td className="px-4 py-3 font-mono text-xs text-text-secondary tabular-nums">
                      {current}
                    </td>
                    <td className={`px-4 py-3 tabular-nums ${etaColor}`}>{text}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-text-muted tabular-nums">
                      R²={f.r2.toFixed(2)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-text-muted mb-0.5">{label}</div>
      <div className="text-text-primary font-medium">{value}</div>
    </div>
  )
}

function ResourceCell({
  label,
  value,
  sub,
  warn = false,
  danger = false,
}: {
  label: string
  value: string
  sub?: string
  warn?: boolean
  danger?: boolean
}) {
  const valueColor = danger
    ? 'text-error'
    : warn
      ? 'text-warning'
      : 'text-text-primary'
  return (
    <div>
      <div className="text-xs text-text-muted mb-0.5">{label}</div>
      <div className={`font-medium tabular-nums ${valueColor}`}>{value}</div>
      {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
    </div>
  )
}

/** Bytes → "X.X GiB"; null/undefined → "—". */
function fmtGiB(bytes: number | null): string {
  if (bytes == null) return '—'
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`
}

/** Percentage → "X.X%"; null → "—". */
function fmtPct(pct: number | null): string {
  if (pct == null) return '—'
  return `${pct.toFixed(1)}%`
}

/** Float → 2-decimal string; null → "—". */
function fmtNum(n: number | null): string {
  if (n == null) return '—'
  return n.toFixed(2)
}

function formatUptime(secs: number): string {
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  if (m < 60) return `${m}m ${secs % 60}s`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ${m % 60}m`
  const d = Math.floor(h / 24)
  return `${d}d ${h % 24}h`
}
