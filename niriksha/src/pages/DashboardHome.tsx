import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'
import { Badge } from '../components/Badge'

export function DashboardHome() {
  const { collections, health, loading, error, fetchCollections, fetchHealth } =
    useCollectionStore()
  const navigate = useNavigate()

  useEffect(() => {
    fetchHealth()
    fetchCollections()
  }, [fetchHealth, fetchCollections])

  const totalFields = collections.reduce((sum, c) => sum + c.field_count, 0)
  const totalShards = collections.reduce((sum, c) => sum + c.shard_count, 0)

  const statusVariant =
    health?.status === 'healthy' ? ('success' as const) : ('error' as const)
  const statusLabel =
    health?.status === 'healthy' ? 'Healthy' : health ? 'Unhealthy' : 'Unknown'

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Veda Dashboard"
        actions={<Badge label={statusLabel} variant={statusVariant} />}
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Collections"
          value={String(collections.length)}
          gradient="from-accent/10 to-accent/5"
        />
        <MetricCard
          label="Total Fields"
          value={String(totalFields)}
          gradient="from-success/10 to-success/5"
        />
        <MetricCard
          label="Total Shards"
          value={String(totalShards)}
          gradient="from-warning/10 to-warning/5"
        />
        <MetricCard
          label="Version"
          value={health?.version ?? '--'}
          gradient="from-accent/10 to-accent/5"
        />
      </div>

      <div>
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-4">
          Collections
        </h2>

        {loading && <LoadingSpinner label="Loading collections..." />}

        {error && (
          <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
            {error}
          </div>
        )}

        {!loading && !error && collections.length === 0 && (
          <EmptyState
            title="No collections yet"
            description="Create your first collection to start storing vectors."
          />
        )}

        {!loading && !error && collections.length > 0 && (
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
                    onClick={() => navigate(`/collections/${col.name}`)}
                    className="hover:bg-bg-tertiary cursor-pointer transition-colors"
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
    </motion.div>
  )
}

function MetricCard({
  label,
  value,
  gradient,
}: {
  label: string
  value: string
  gradient: string
}) {
  return (
    <div
      className={`rounded-lg bg-gradient-to-br ${gradient} border border-border p-5`}
    >
      <div className="text-2xl font-semibold text-text-primary">{value}</div>
      <div className="text-xs text-text-muted mt-1">{label}</div>
    </div>
  )
}
