import { type ReactNode, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Boxes, Database, Hash, Layers, Plus } from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import api from '../api/client'
import { sumPromMetric } from '../lib/metrics'
import { deriveModality, MODALITY_LABEL, modalityColorVar } from '../lib/modality'
import { deriveModalityMix } from '../lib/modalityMix'
import { Card, ConnectionPill, EmptyState, ModalityChip, StatTile } from '../components/primitives'
import { SpotlightCard } from '../components/reactbits/SpotlightCard'
import { CountUp } from '../components/reactbits/CountUp'
import { DonutChart, type DonutDatum } from '../components/charts/DonutChart'

export function DashboardHome() {
  const {
    collections,
    details,
    health,
    loading,
    error,
    fetchCollections,
    fetchHealth,
    fetchAllDetails,
  } = useCollectionStore()
  const navigate = useNavigate()
  const [vectors, setVectors] = useState<number | null>(null)

  useEffect(() => {
    fetchHealth()
    fetchCollections().then(() => fetchAllDetails())
  }, [fetchHealth, fetchCollections, fetchAllDetails])

  useEffect(() => {
    let alive = true
    api
      .get<string>('/admin/metrics', { responseType: 'text' })
      .then((r) => {
        if (alive) setVectors(sumPromMetric(r.data, 'veda_records_total'))
      })
      .catch(() => {
        if (alive) setVectors(null)
      })
    return () => {
      alive = false
    }
  }, [collections.length])

  const totalFields = collections.reduce((s, c) => s + c.field_count, 0)
  const totalShards = collections.reduce((s, c) => s + c.shard_count, 0)
  const healthy = health?.status === 'healthy'

  const modalityByName = new Map(details.map((d) => [d.name, deriveModality(d.fields)]))
  const mix = deriveModalityMix(details)
  const donut: DonutDatum[] = mix.map((s) => ({
    label: MODALITY_LABEL[s.modality],
    value: s.count,
    color: modalityColorVar(s.modality),
  }))

  return (
    <motion.div
      className="p-8 space-y-6"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      {/* gated hero glow */}
      <div
        aria-hidden
        className="pointer-events-none fixed left-[220px] right-0 top-11 h-64"
        style={{ background: 'var(--glow-accent)' }}
      />

      {/* header */}
      <div className="relative flex items-start justify-between">
        <div>
          <h1 className="text-[24px] font-semibold tracking-tight text-[var(--color-text)]">
            Overview
          </h1>
          <p className="mt-1 text-[13px] text-[var(--color-text-muted)]">
            Your Veda cluster at a glance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <ConnectionPill ok={!!healthy} label={healthy ? 'healthy' : health ? 'degraded' : 'offline'} />
          <Link
            to="/collections"
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-[var(--radius-sm)] text-[12px] font-medium text-[var(--color-text-on-accent)] bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] transition-colors"
          >
            <Plus size={14} strokeWidth={2.5} />
            New collection
          </Link>
        </div>
      </div>

      {/* KPI band */}
      <div className="relative grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <Kpi label="Collections" icon={<Boxes size={13} />} value={<CountUp to={collections.length} />} />
        <Kpi label="Vectors" icon={<Hash size={13} />} value={vectors == null ? '—' : <CountUp to={vectors} />} />
        <Kpi label="Fields" icon={<Layers size={13} />} value={<CountUp to={totalFields} />} />
        <Kpi label="Shards" icon={<Database size={13} />} value={<CountUp to={totalShards} />} />
        <Kpi label="Version" value={health?.version ?? '—'} />
      </div>

      {/* modality + collections */}
      <div className="relative grid gap-4 lg:grid-cols-3">
        <Card className="p-4">
          <h3 className="text-[11px] uppercase tracking-[0.06em] text-[var(--color-text-muted)]">
            Modality mix
          </h3>
          {donut.length > 0 ? (
            <>
              <div className="mt-2">
                <DonutChart data={donut} />
              </div>
              <ul className="mt-3 space-y-1.5">
                {mix.map((s) => (
                  <li key={s.modality} className="flex items-center justify-between text-[12px]">
                    <ModalityChip modality={s.modality} />
                    <span className="tabular-nums text-[var(--color-text-muted)]">{s.count}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <div className="mt-2 h-[168px] animate-pulse rounded-[var(--radius-md)] bg-[var(--color-bg-hover)]" />
          )}
        </Card>

        <Card className="lg:col-span-2 overflow-hidden">
          <div className="px-4 py-3 border-b border-[var(--color-border)] text-[11px] uppercase tracking-[0.06em] text-[var(--color-text-muted)]">
            All collections
          </div>

          {loading && (
            <div className="p-4 space-y-2">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-8 animate-pulse rounded bg-[var(--color-bg-hover)]" />
              ))}
            </div>
          )}

          {error && (
            <div
              className="m-4 rounded-[var(--radius-md)] px-3 py-2 text-[12px]"
              style={{
                color: 'var(--color-error)',
                border: '1px solid color-mix(in oklch, var(--color-error) 30%, transparent)',
                background: 'color-mix(in oklch, var(--color-error) 12%, transparent)',
              }}
            >
              {error}
            </div>
          )}

          {!loading && !error && collections.length === 0 && (
            <div className="p-4">
              <EmptyState
                title="No collections yet"
                description="Create your first collection to start storing multimodal vectors."
                action={
                  <Link to="/collections" className="text-[12px] text-[var(--color-accent)]">
                    Create collection →
                  </Link>
                }
              />
            </div>
          )}

          {!loading && !error && collections.length > 0 && (
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-[var(--color-text-subtle)]">
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 font-medium">Modality</th>
                  <th className="px-4 py-2 font-medium text-right">Fields</th>
                  <th className="px-4 py-2 font-medium text-right">Shards</th>
                </tr>
              </thead>
              <tbody>
                {collections.map((c) => {
                  const m = modalityByName.get(c.name)
                  return (
                    <tr
                      key={c.name}
                      onClick={() => navigate(`/collections/${c.name}`)}
                      className="border-t border-[var(--color-border)] hover:bg-[var(--color-bg-hover)] cursor-pointer transition-colors"
                    >
                      <td className="px-4 py-2.5 font-medium text-[var(--color-text)]">{c.name}</td>
                      <td className="px-4 py-2.5">
                        {m ? (
                          <ModalityChip modality={m} />
                        ) : (
                          <span className="text-[var(--color-text-subtle)]">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right tabular-nums text-[var(--color-text-muted)]">
                        {c.field_count}
                      </td>
                      <td className="px-4 py-2.5 text-right tabular-nums text-[var(--color-text-muted)]">
                        {c.shard_count}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </motion.div>
  )
}

function Kpi({ label, value, icon }: { label: string; value: ReactNode; icon?: ReactNode }) {
  return (
    <SpotlightCard>
      <StatTile label={label} value={value} icon={icon} />
    </SpotlightCard>
  )
}
