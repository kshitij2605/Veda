import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { Badge } from '../components/Badge'
import api from '../api/client'

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
  }, [fetchHealth, fetchCollections, loadMetrics])

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
            <InfoCell label="Uptime" value="--" />
            <InfoCell label="API" value="/v1" />
          </div>
        </div>
      </div>

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

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-text-muted mb-0.5">{label}</div>
      <div className="text-text-primary font-medium">{value}</div>
    </div>
  )
}
