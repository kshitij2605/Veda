import { type ReactNode, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Boxes, Database, Hash, Layers, Plus } from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import api from '../api/client'
import { sumPromMetric } from '../lib/metrics'
import { deriveModality, MODALITY_LABEL, modalityColorVar } from '../lib/modality'
import { deriveModalityMix } from '../lib/modalityMix'
import { Button, Card, ConnectionPill, EmptyState, ModalityChip, StatTile } from '../components/primitives'
import { SpotlightCard } from '../components/reactbits/SpotlightCard'
import { CountUp } from '../components/reactbits/CountUp'
import { DonutChart, type DonutDatum } from '../components/charts/DonutChart'
import { greetingFor } from '../lib/greeting'
import { useReveal } from '../lib/useReveal'

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

  const kpiRef = useReveal<HTMLDivElement>()
  const bentoRef = useReveal<HTMLDivElement>()

  return (
    <div className="mx-auto max-w-[1280px] px-8 py-10">
      {/* ── Hero ─────────────────────────────────────────────── */}
      <header className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute -top-28 -left-28 h-80 w-[44rem]"
          style={{
            background:
              'radial-gradient(45% 60% at 30% 30%, color-mix(in oklch, var(--color-accent) 13%, transparent), transparent 70%)',
          }}
        />
        <div className="relative flex flex-wrap items-start justify-between gap-5">
          <div>
            <p className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
              Niriksha · Veda
            </p>
            <h1 className="mt-2 text-[46px] leading-[1.04] [font-family:var(--font-display)] font-semibold tracking-[-0.02em] text-[var(--color-text)]">
              {greetingFor(new Date())}
            </h1>
            <p className="mt-2.5 max-w-xl text-[16px] leading-relaxed text-[var(--color-text-muted)]">
              Your multimodal vector cluster at a glance — {collections.length}{' '}
              collection{collections.length === 1 ? '' : 's'}
              {mix.length > 0
                ? ` across ${mix.length} ${mix.length === 1 ? 'modality' : 'modalities'}`
                : ''}
              .
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <ConnectionPill
              ok={!!healthy}
              label={healthy ? 'healthy' : health ? 'degraded' : 'offline'}
            />
            <Button
              variant="primary"
              size="lg"
              leading={<Plus size={16} strokeWidth={2.5} />}
              onClick={() => navigate('/collections')}
            >
              New collection
            </Button>
          </div>
        </div>
      </header>

      {/* ── KPI band ─────────────────────────────────────────── */}
      <div
        ref={kpiRef}
        className="mt-9 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5"
      >
        <Kpi label="Collections" icon={<Boxes size={13} />} value={<CountUp to={collections.length} />} />
        <Kpi label="Vectors" icon={<Hash size={13} />} value={vectors == null ? '—' : <CountUp to={vectors} />} />
        <Kpi label="Fields" icon={<Layers size={13} />} value={<CountUp to={totalFields} />} />
        <Kpi label="Shards" icon={<Database size={13} />} value={<CountUp to={totalShards} />} />
        <Kpi label="Version" value={health?.version ?? '—'} />
      </div>

      {/* ── Bento: modality mix + collections ────────────────── */}
      <div ref={bentoRef} className="mt-6 grid gap-5 lg:grid-cols-3">
        <Card className="p-5">
          <SectionLabel>Modality mix</SectionLabel>
          {donut.length > 0 ? (
            <>
              <div className="mt-3">
                <DonutChart data={donut} />
              </div>
              <ul className="mt-4 space-y-2">
                {mix.map((s) => (
                  <li key={s.modality} className="flex items-center justify-between text-[13px]">
                    <ModalityChip modality={s.modality} />
                    <span className="tabular-nums text-[var(--color-text-muted)]">{s.count}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <div className="mt-3 h-[168px] animate-pulse rounded-[var(--radius-md)] bg-[var(--color-bg-hover)]" />
          )}
        </Card>

        <Card className="overflow-hidden lg:col-span-2">
          <div className="border-b border-[var(--color-border)] px-5 py-4">
            <SectionLabel>All collections</SectionLabel>
          </div>

          {loading && (
            <div className="space-y-2 p-5">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-9 animate-pulse rounded bg-[var(--color-bg-hover)]" />
              ))}
            </div>
          )}

          {error && (
            <div
              className="m-5 rounded-[var(--radius-md)] px-3 py-2 text-[13px]"
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
            <div className="p-5">
              <EmptyState
                title="No collections yet"
                description="Create your first collection to start storing multimodal vectors."
                action={
                  <Button variant="primary" size="md" leading={<Plus size={14} strokeWidth={2.5} />} onClick={() => navigate('/collections')}>
                    Create collection
                  </Button>
                }
              />
            </div>
          )}

          {!loading && !error && collections.length > 0 && (
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
                  <th className="px-5 py-3 font-semibold">Name</th>
                  <th className="px-5 py-3 font-semibold">Modality</th>
                  <th className="px-5 py-3 text-right font-semibold">Fields</th>
                  <th className="px-5 py-3 text-right font-semibold">Shards</th>
                </tr>
              </thead>
              <tbody>
                {collections.map((c) => {
                  const m = modalityByName.get(c.name)
                  return (
                    <tr
                      key={c.name}
                      onClick={() => navigate(`/collections/${c.name}`)}
                      className="cursor-pointer border-t border-[var(--color-border)] transition-colors hover:bg-[var(--color-bg-hover)]"
                    >
                      <td className="px-5 py-3 font-medium text-[var(--color-text)]">{c.name}</td>
                      <td className="px-5 py-3">
                        {m ? (
                          <ModalityChip modality={m} />
                        ) : (
                          <span className="text-[var(--color-text-subtle)]">—</span>
                        )}
                      </td>
                      <td className="px-5 py-3 text-right tabular-nums text-[var(--color-text-muted)]">
                        {c.field_count}
                      </td>
                      <td className="px-5 py-3 text-right tabular-nums text-[var(--color-text-muted)]">
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
    </div>
  )
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
      {children}
    </h2>
  )
}

function Kpi({ label, value, icon }: { label: string; value: ReactNode; icon?: ReactNode }) {
  return (
    <SpotlightCard>
      <StatTile label={label} value={value} icon={icon} />
    </SpotlightCard>
  )
}
