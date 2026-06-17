import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  BarChart3,
  Clock,
  RotateCw,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import api from '../api/client'
import { fetchSlowQueries, type SlowQuery } from '../api/logs'
import {
  fetchCorrelations,
  type CorrelationDimension,
} from '../api/correlations'
import {
  fetchDepMap,
  type DepNode,
  type DepEdge,
  type DepMapResponse,
} from '../api/depmap'
import {
  fetchIndexAdvisor,
  applyReindex,
  type IndexAdvisorRow,
} from '../api/indexAdvisor'
import { Button, CopyButton, Skeleton } from '../components/primitives'
import {
  useSavedObjects,
  type SavedObject,
} from '../stores/savedObjectStore'
import { Sparkline } from '../components/primitives/Sparkline'
import { cn } from '../lib/cn'
import type { EvalRunBody } from './quality/types'

interface MetricEntry {
  name: string
  labels: Record<string, string>
  value: number
}

/**
 * Observability surface — parses /v1/admin/metrics (Prometheus text)
 * and renders a table-with-sparklines + slow-query log + eval-drift
 * trail. Frontend-only for v1: no in-process time-series store, so
 * sparklines come from successive polls (every 5 s while the page is
 * open). The slow-query section is server-backed via
 * `GET /v1/admin/slow-queries`; the full log live-tail lives on the
 * dedicated Logs page.
 */
export function Observability() {
  const [snapshots, setSnapshots] = useState<MetricEntry[][]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [lastFetched, setLastFetched] = useState<number | null>(null)

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await api.get<string>('/admin/metrics', {
        responseType: 'text',
        transformResponse: (d) => d,
      })
      const parsed = parsePrometheus(
        typeof data === 'string' ? data : String(data),
      )
      setSnapshots((prev) => [...prev.slice(-29), parsed])
      setLastFetched(Date.now())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch metrics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
    const id = setInterval(fetchMetrics, 5_000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const latest = snapshots[snapshots.length - 1] ?? []

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <BarChart3
          size={16}
          strokeWidth={2}
          className="text-[var(--color-accent)]"
        />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
          Observability
        </h1>
        {lastFetched && (
          <span className="text-[11px] font-mono text-[var(--color-text-subtle)]">
            polled {formatRelative(lastFetched)} · live
          </span>
        )}
        <Button
          size="xs"
          variant="ghost"
          leading={
            <RotateCw
              size={11}
              strokeWidth={2.25}
              className={loading ? 'animate-spin' : ''}
            />
          }
          onClick={fetchMetrics}
          disabled={loading}
          className="ml-auto"
        >
          Refresh
        </Button>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-4">
        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
            <AlertCircle size={12} strokeWidth={2.25} />
            {error}
          </div>
        )}
        {snapshots.length === 0 && !error && (
          <SkeletonRows />
        )}
        {snapshots.length > 0 && (
          <>
            <TopMetrics snapshots={snapshots} />
            <CollectionTable snapshots={snapshots} />
            <RawMetricsTable latest={latest} />
            <EvalDriftPanel />
            <CorrelationsPanel />
            <IndexAdvisorPanel />
            <DependenciesPanel />
            <RagTraceLegend />
            <ServerSlowQueries />
          </>
        )}
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Top KPIs — uptime, QPS, active collections, recent error rate
   ──────────────────────────────────────────────────────────────────── */

function TopMetrics({ snapshots }: { snapshots: MetricEntry[][] }) {
  const latest = snapshots[snapshots.length - 1] ?? []
  const series = (name: string) =>
    snapshots.map((s) => sumByName(s, name)).filter((v) => Number.isFinite(v))

  // The metrics-rs Prometheus exporter ships per-collector summaries as
  // {name}_count + {name}_sum (no histogram buckets). Mean latency is the
  // best we can compute client-side; surface that with a "mean" label
  // rather than misleadingly calling it p95.
  const searchCount = sumByName(latest, 'veda_search_duration_seconds_count')
  const searchSum = sumByName(latest, 'veda_search_duration_seconds_sum')
  const meanSearchMs =
    searchCount > 0 ? (searchSum / searchCount) * 1000 : 0
  const embedCount = sumByName(latest, 'veda_embedding_duration_seconds_count')
  const embedSum = sumByName(latest, 'veda_embedding_duration_seconds_sum')
  const meanEmbedMs = embedCount > 0 ? (embedSum / embedCount) * 1000 : 0

  const tiles: {
    label: string
    value: string
    series: number[]
    icon: React.ReactNode
  }[] = [
    {
      label: 'Requests',
      value: humanFormat(sumByName(latest, 'veda_requests_total') || 0),
      series: series('veda_requests_total'),
      icon: <BarChart3 size={11} strokeWidth={2.25} />,
    },
    {
      label: 'Errors',
      value: humanFormat(sumByName(latest, 'veda_errors_total') || 0),
      series: series('veda_errors_total'),
      icon: <Activity size={11} strokeWidth={2.25} />,
    },
    {
      label: 'Mean search latency',
      value: `${meanSearchMs.toFixed(1)}ms`,
      series: snapshots.map((s) => {
        const c = sumByName(s, 'veda_search_duration_seconds_count')
        const sum = sumByName(s, 'veda_search_duration_seconds_sum')
        return c > 0 ? (sum / c) * 1000 : 0
      }),
      icon: <Clock size={11} strokeWidth={2.25} />,
    },
    {
      label: 'Mean embed latency',
      value: `${meanEmbedMs.toFixed(1)}ms`,
      series: snapshots.map((s) => {
        const c = sumByName(s, 'veda_embedding_duration_seconds_count')
        const sum = sumByName(s, 'veda_embedding_duration_seconds_sum')
        return c > 0 ? (sum / c) * 1000 : 0
      }),
      icon: <TrendingUp size={11} strokeWidth={2.25} />,
    },
  ]

  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {tiles.map((t) => (
        <KpiCard key={t.label} {...t} />
      ))}
    </section>
  )
}

function KpiCard({
  label,
  value,
  series,
  icon,
}: {
  label: string
  value: string
  series: number[]
  icon: React.ReactNode
}) {
  const trend = series.length >= 2
    ? series[series.length - 1]! - series[0]!
    : 0
  const TrendIcon = trend > 0 ? TrendingUp : trend < 0 ? TrendingDown : null
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
      <div className="flex items-center justify-between text-[10.5px] text-[var(--color-text-subtle)] uppercase tracking-wider mb-1">
        <span className="inline-flex items-center gap-1">
          {icon}
          {label}
        </span>
        {TrendIcon && (
          <span
            className={cn(
              'inline-flex items-center gap-0.5',
              trend > 0 ? 'text-[var(--color-success)]' : 'text-[var(--color-warning)]',
            )}
          >
            <TrendIcon size={10} strokeWidth={2.25} />
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-2">
        <div className="text-[18px] font-mono tabular-nums text-[var(--color-text)] font-semibold">
          {value}
        </div>
        {series.length > 1 && (
          <Sparkline values={series} width={70} height={18} />
        )}
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Per-collection table — table-with-sparklines (Stripe-style)
   ──────────────────────────────────────────────────────────────────── */

function CollectionTable({ snapshots }: { snapshots: MetricEntry[][] }) {
  const collections = useMemo(() => {
    const set = new Set<string>()
    for (const s of snapshots) {
      for (const e of s) {
        const c = e.labels.collection
        if (c) set.add(c)
      }
    }
    return Array.from(set).sort()
  }, [snapshots])
  if (collections.length === 0) return null

  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Per-collection rollup
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
        <table className="w-full text-[12px] tabular-nums">
          <thead className="bg-[var(--color-bg-elevated)] text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Collection</th>
              <th className="px-3 py-2 text-right font-medium">Requests</th>
              <th className="px-3 py-2 text-right font-medium">Mean</th>
              <th className="px-3 py-2 text-right font-medium">Searches</th>
              <th className="px-3 py-2 text-right font-medium">Errors</th>
              <th className="px-3 py-2 text-left font-medium">Trend</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {collections.map((c) => (
              <CollectionRow
                key={c}
                collection={c}
                snapshots={snapshots}
              />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function CollectionRow({
  collection,
  snapshots,
}: {
  collection: string
  snapshots: MetricEntry[][]
}) {
  const series = snapshots.map((s) =>
    sumWhere(s, 'veda_requests_total', { collection }),
  )
  const latest = snapshots[snapshots.length - 1] ?? []
  const errors = sumWhere(latest, 'veda_errors_total', { collection })
  const searchCount = sumWhere(
    latest,
    'veda_search_duration_seconds_count',
    { collection },
  )
  const searchSum = sumWhere(
    latest,
    'veda_search_duration_seconds_sum',
    { collection },
  )
  const meanMs = searchCount > 0 ? (searchSum / searchCount) * 1000 : 0
  return (
    <tr className="hover:bg-[var(--color-bg-hover)]">
      <td className="px-3 py-1.5 font-mono text-[var(--color-text)]">
        {collection}
      </td>
      <td className="px-3 py-1.5 text-right text-[var(--color-text-muted)]">
        {humanFormat(series[series.length - 1] ?? 0)}
      </td>
      <td className="px-3 py-1.5 text-right text-[var(--color-text-muted)]">
        {meanMs > 0 ? `${meanMs.toFixed(1)}ms` : '—'}
      </td>
      <td className="px-3 py-1.5 text-right text-[var(--color-text-muted)] tabular-nums">
        {humanFormat(searchCount)}
      </td>
      <td
        className={cn(
          'px-3 py-1.5 text-right',
          errors > 0 ? 'text-[var(--color-error)]' : 'text-[var(--color-text-muted)]',
        )}
      >
        {errors > 0 ? humanFormat(errors) : '—'}
      </td>
      <td className="px-3 py-1.5">
        {series.length > 1 && <Sparkline values={series} width={120} height={16} />}
      </td>
    </tr>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Eval drift — small mini-chart of mean recall per run over time
   ──────────────────────────────────────────────────────────────────── */

function EvalDriftPanel() {
  const objectsMap = useSavedObjects((s) => s.objects)
  const runs = useMemo(
    () =>
      Object.values(objectsMap)
        .filter(
          (o): o is SavedObject<EvalRunBody> => o.type === 'eval_run',
        )
        .sort((a, b) => a.body.startedAt - b.body.startedAt),
    [objectsMap],
  )
  // Bucket by evalSetId, render one row per set.
  const bySet = useMemo(() => {
    const m = new Map<string, SavedObject<EvalRunBody>[]>()
    for (const r of runs) {
      const k = r.body.evalSetId
      if (!m.has(k)) m.set(k, [])
      m.get(k)!.push(r)
    }
    return Array.from(m.entries())
      .filter(([, rs]) => rs.length >= 2)
      .sort(
        (a, b) =>
          b[1][b[1].length - 1]!.body.startedAt -
          a[1][a[1].length - 1]!.body.startedAt,
      )
  }, [runs])

  if (bySet.length === 0) return null

  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Eval-score drift
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
        <table className="w-full text-[12px] tabular-nums">
          <thead className="bg-[var(--color-bg-elevated)] text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Eval set</th>
              <th className="px-3 py-2 text-right font-medium">Runs</th>
              <th className="px-3 py-2 text-right font-medium">First MRR</th>
              <th className="px-3 py-2 text-right font-medium">Last MRR</th>
              <th className="px-3 py-2 text-right font-medium">Δ</th>
              <th className="px-3 py-2 text-left font-medium">Trend</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {bySet.map(([setId, rs]) => {
              const series = rs.map((r) => r.body.summary.mrr)
              const first = series[0]!
              const last = series[series.length - 1]!
              const delta = last - first
              return (
                <tr key={setId} className="hover:bg-[var(--color-bg-hover)]">
                  <td className="px-3 py-1.5 text-[var(--color-text)] truncate max-w-[280px]">
                    {rs[0]!.body.evalSetName}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--color-text-muted)]">
                    {rs.length}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--color-text-muted)] font-mono">
                    {first.toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--color-text)] font-mono">
                    {last.toFixed(3)}
                  </td>
                  <td
                    className={cn(
                      'px-3 py-1.5 text-right font-mono',
                      delta > 0 && 'text-[var(--color-success)]',
                      delta < 0 && 'text-[var(--color-error)]',
                    )}
                  >
                    {delta > 0 ? '+' : ''}
                    {delta.toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5">
                    <Sparkline
                      values={series}
                      width={120}
                      height={16}
                      stroke={
                        delta < 0
                          ? 'var(--color-error)'
                          : 'var(--color-data-5)'
                      }
                    />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Correlations — slow-query / error drivers from GET /v1/admin/correlations.
   Per-dimension table ranking the buckets (collection, query type, filter,
   rerank, index type, result-count, top_k) that drive latency / errors.
   ──────────────────────────────────────────────────────────────────── */

/** A query slower than this (ms) is "slow" — matches the backend's
 *  SLOW_QUERY_MS threshold; used to warning-color p95 cells. */
const SLOW_QUERY_MS = 1000

const DIMENSION_LABELS: Record<string, string> = {
  collection: 'Collection',
  query_type: 'Query type',
  has_filter: 'Filter',
  reranked: 'Rerank',
  index_type: 'Index type',
  result_count: 'Result count',
  top_k: 'top_k',
}

function CorrelationsPanel() {
  const [dimensions, setDimensions] = useState<CorrelationDimension[] | null>(
    null,
  )
  useEffect(() => {
    let alive = true
    fetchCorrelations()
      .then(({ data }) => {
        if (alive) setDimensions(data?.dimensions ?? [])
      })
      .catch(() => {
        if (alive) setDimensions([])
      })
    return () => {
      alive = false
    }
  }, [])

  // Only show dimensions that have data.
  const withData = (dimensions ?? []).filter((d) => d.buckets.length > 0)
  if (dimensions === null || withData.length === 0) return null

  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Correlations
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (slow-query / error drivers · last 5m)
          </span>
        </h2>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {withData.map((d) => (
          <DimensionCard key={d.dimension} dim={d} />
        ))}
      </div>
    </section>
  )
}

function DimensionCard({ dim }: { dim: CorrelationDimension }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
      <div className="bg-[var(--color-bg-elevated)] px-3 py-1.5 text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)] border-b border-[var(--color-border)]">
        {DIMENSION_LABELS[dim.dimension] ?? dim.dimension}
      </div>
      <table className="w-full text-[11.5px] tabular-nums">
        <thead className="text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
          <tr>
            <th className="px-3 py-1.5 text-left font-medium">Bucket</th>
            <th className="px-3 py-1.5 text-right font-medium">Count</th>
            <th className="px-3 py-1.5 text-right font-medium">p95</th>
            <th className="px-3 py-1.5 text-right font-medium">Slow%</th>
            <th className="px-3 py-1.5 text-right font-medium">Err%</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--color-border)]">
          {dim.buckets.map((b) => {
            const slowPct = b.count > 0 ? (b.slow_count / b.count) * 100 : 0
            const errPct = (b.error_rate ?? 0) * 100
            return (
              <tr key={b.key} className="hover:bg-[var(--color-bg-hover)]">
                <td className="px-3 py-1 font-mono text-[var(--color-text)] truncate max-w-[180px]">
                  {b.key}
                </td>
                <td className="px-3 py-1 text-right text-[var(--color-text-muted)]">
                  {humanFormat(b.count)}
                </td>
                <td
                  className={cn(
                    'px-3 py-1 text-right font-mono',
                    b.p95_ms >= SLOW_QUERY_MS
                      ? 'text-[var(--color-warning)]'
                      : 'text-[var(--color-text-muted)]',
                  )}
                >
                  {b.p95_ms.toFixed(0)}ms
                </td>
                <td
                  className={cn(
                    'px-3 py-1 text-right',
                    slowPct > 0
                      ? 'text-[var(--color-warning)]'
                      : 'text-[var(--color-text-muted)]',
                  )}
                >
                  {slowPct > 0 ? `${slowPct.toFixed(0)}%` : '—'}
                </td>
                <td
                  className={cn(
                    'px-3 py-1 text-right',
                    errPct > 0
                      ? 'text-[var(--color-error)]'
                      : 'text-[var(--color-text-muted)]',
                  )}
                >
                  {errPct > 0 ? `${errPct.toFixed(0)}%` : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Index advisor — per-collection recall-vs-latency from
   GET /v1/admin/index-advisor, with advisory recommendations. Recall is a
   PROXY measurement (labelled as such); rows warn when recall is below the
   target or p95 is high. An opt-in "Apply reindex" button POSTs the
   suggested /reindex config (with a confirm).
   ──────────────────────────────────────────────────────────────────── */

/** Recall target floor + p95 ceiling for warning coloring (match the
 *  advisor rule defaults: recall_min 0.90, p95_ms 50). */
const RECALL_TARGET = 0.9
const P95_WARN_MS = 50

function IndexAdvisorPanel() {
  const [rows, setRows] = useState<IndexAdvisorRow[] | null>(null)
  const [applying, setApplying] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const load = () => {
    let alive = true
    fetchIndexAdvisor()
      .then(({ data }) => {
        if (alive) setRows(data?.rows ?? [])
      })
      .catch(() => {
        if (alive) setRows([])
      })
    return () => {
      alive = false
    }
  }
  useEffect(load, [])

  const onApply = async (row: IndexAdvisorRow) => {
    const body = row.suggested_reindex
    if (!body) return
    const ok = window.confirm(
      `Apply a reindex to "${row.collection}" with config ${JSON.stringify(
        body,
      )}? This rebuilds the collection's index and may take a while.`,
    )
    if (!ok) return
    setApplying(row.collection)
    setNote(null)
    try {
      await applyReindex(row.collection, body)
      setNote(`Reindex submitted for "${row.collection}".`)
    } catch {
      setNote(`Reindex failed for "${row.collection}".`)
    } finally {
      setApplying(null)
    }
  }

  if (rows === null || rows.length === 0) return null

  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Index advisor
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (recall vs p95 per collection · recall is a proxy)
          </span>
        </h2>
      </header>
      {note && (
        <div className="mb-2 text-[11.5px] text-[var(--color-text-muted)]">
          {note}
        </div>
      )}
      <div className="space-y-2">
        {rows.map((row) => {
          const recall = row.recall ?? null
          const p95 = row.p95_ms ?? null
          const recallLow = recall !== null && recall < RECALL_TARGET
          const p95High = p95 !== null && p95 > P95_WARN_MS
          const warn = recallLow || p95High
          return (
            <div
              key={row.collection}
              className={cn(
                'rounded-[var(--radius-md)] border bg-[var(--color-bg-elevated)] p-3',
                warn
                  ? 'border-[var(--color-warning)]'
                  : 'border-[var(--color-border)]',
              )}
            >
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2 text-[12px]">
                  <span className="text-[var(--color-text)] font-medium">
                    {row.collection}
                  </span>
                  <span className="inline-flex items-center rounded-[var(--radius-xs)] border border-[var(--color-border)] bg-[var(--color-bg)] px-1.5 py-0.5 text-[10px] font-mono text-[var(--color-text-muted)]">
                    {row.index_type}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[11.5px] tabular-nums font-mono">
                  <span
                    className={cn(
                      recallLow
                        ? 'text-[var(--color-warning)]'
                        : 'text-[var(--color-text-muted)]',
                    )}
                  >
                    recall{' '}
                    {recall !== null ? recall.toFixed(2) : '—'}
                    {row.is_proxy && (
                      <span className="ml-1 text-[9.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
                        proxy
                      </span>
                    )}
                  </span>
                  <span
                    className={cn(
                      p95High
                        ? 'text-[var(--color-warning)]'
                        : 'text-[var(--color-text-muted)]',
                    )}
                  >
                    p95 {p95 !== null ? `${p95.toFixed(0)}ms` : '—'}
                  </span>
                </div>
              </div>
              {row.recommendations.length > 0 && (
                <ul className="mt-2 space-y-1.5 border-t border-[var(--color-border)] pt-2">
                  {row.recommendations.map((rec) => (
                    <li key={rec.rule_id} className="text-[11.5px]">
                      <span className="text-[var(--color-text)]">
                        {rec.title}
                      </span>
                      <span className="text-[var(--color-text-muted)]">
                        {' — '}
                        {rec.recommendation}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {row.suggested_reindex && (
                <div className="mt-2">
                  <Button
                    variant="secondary"
                    onClick={() => onApply(row)}
                    disabled={applying === row.collection}
                  >
                    {applying === row.collection
                      ? 'Applying…'
                      : 'Apply reindex'}
                  </Button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Dependencies — static service / dependency graph from
   GET /v1/admin/depmap. Rendered as nodes grouped by kind + an edge list
   (no heavy graph lib; reuses tokens). Shows what a request fans out to.
   ──────────────────────────────────────────────────────────────────── */

const DEP_KIND_LABELS: Record<string, string> = {
  gateway: 'Gateway',
  query: 'Query',
  index: 'Index',
  storage: 'Storage',
  external: 'External services',
}

/** Stable display order for the kind groups. */
const DEP_KIND_ORDER = ['gateway', 'query', 'index', 'storage', 'external']

function DependenciesPanel() {
  const [graph, setGraph] = useState<DepMapResponse | null>(null)
  useEffect(() => {
    let alive = true
    fetchDepMap()
      .then(({ data }) => {
        if (alive) setGraph(data ?? null)
      })
      .catch(() => {
        if (alive) setGraph(null)
      })
    return () => {
      alive = false
    }
  }, [])

  const nodes = graph?.nodes ?? []
  const edges = graph?.edges ?? []
  if (nodes.length === 0) return null

  // Group nodes by kind, preserving the declared display order.
  const byKind = new Map<string, DepNode[]>()
  for (const n of nodes) {
    if (!byKind.has(n.kind)) byKind.set(n.kind, [])
    byKind.get(n.kind)!.push(n)
  }
  const kinds = DEP_KIND_ORDER.filter((k) => byKind.has(k)).concat(
    Array.from(byKind.keys()).filter((k) => !DEP_KIND_ORDER.includes(k)),
  )
  const labelOf = (id: string) =>
    nodes.find((n) => n.id === id)?.label ?? id

  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Dependencies
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (request topology · what a search fans out to)
          </span>
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {kinds.map((k) => (
            <div
              key={k}
              className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg)] p-2"
            >
              <div className="text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)] mb-1.5">
                {DEP_KIND_LABELS[k] ?? k}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {byKind.get(k)!.map((n) => (
                  <span
                    key={n.id}
                    className="inline-flex items-center rounded-[var(--radius-xs)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-2 py-0.5 text-[11px] text-[var(--color-text)]"
                  >
                    {n.label}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        {edges.length > 0 && (
          <div className="border-t border-[var(--color-border)] pt-2">
            <div className="text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)] mb-1.5">
              Edges
            </div>
            <ul className="flex flex-col gap-1">
              {edges.map((e: DepEdge, i) => (
                <li
                  key={i}
                  className="flex items-center gap-1.5 text-[11.5px] text-[var(--color-text-muted)] font-mono"
                >
                  <span className="text-[var(--color-text)]">
                    {labelOf(e.from)}
                  </span>
                  <span className="text-[var(--color-text-subtle)]">→</span>
                  <span className="text-[var(--color-text)]">
                    {labelOf(e.to)}
                  </span>
                  {e.label && (
                    <span className="text-[var(--color-text-subtle)]">
                      ({e.label})
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   RAG trace legend — explains the retrieval span breakdown
   (filter / ANN / scoring / fusion / total) attached to the `search`
   span as `retrieval.*` attributes (APM Phase 5.1). Static reference so
   operators reading a Veda RAG trace know what each phase means.
   ──────────────────────────────────────────────────────────────────── */

const RAG_PHASES: { key: string; label: string; desc: string }[] = [
  {
    key: 'retrieval.filter_ms',
    label: 'Filter',
    desc: 'Applying the scalar / RLS filter before the vector walk.',
  },
  {
    key: 'retrieval.ann_search_ms',
    label: 'ANN search',
    desc: 'Walking the vector index (HNSW / IVF-PQ / …) for candidates.',
  },
  {
    key: 'retrieval.scoring_ms',
    label: 'Scoring',
    desc: 'Exact distance / similarity scoring of candidates.',
  },
  {
    key: 'retrieval.fusion_ms',
    label: 'Fusion',
    desc: 'Fusing multi-leg (hybrid) / multi-vector results.',
  },
  {
    key: 'retrieval.total_ms',
    label: 'Total',
    desc: 'Sum of the retrieval phases above.',
  },
]

function RagTraceLegend() {
  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          RAG trace
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (search span retrieval breakdown)
          </span>
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
        <p className="text-[11.5px] text-[var(--color-text-muted)] mb-2">
          Each <code className="font-mono text-[var(--color-text)]">search</code>{' '}
          span carries{' '}
          <code className="font-mono text-[var(--color-text)]">db.system</code>,{' '}
          <code className="font-mono text-[var(--color-text)]">db.collection</code>{' '}
          and{' '}
          <code className="font-mono text-[var(--color-text)]">retrieval.*</code>{' '}
          attributes. When the query is run with{' '}
          <code className="font-mono text-[var(--color-text)]">explain</code>, the
          per-phase timing breakdown is also recorded so a trace shows whether
          retrieval timing — not the LLM — is the bottleneck.
        </p>
        <ul className="flex flex-col gap-1">
          {RAG_PHASES.map((p) => (
            <li key={p.key} className="flex items-baseline gap-2 text-[11.5px]">
              <code className="font-mono text-[var(--color-accent)] whitespace-nowrap">
                {p.key}
              </code>
              <span className="text-[var(--color-text-subtle)]">·</span>
              <span className="text-[var(--color-text)]">{p.label}</span>
              <span className="text-[var(--color-text-muted)]">— {p.desc}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Slow queries — server-backed from the per-collection query log
   (GET /v1/admin/slow-queries). Replaces the old browser-local view.
   The full live-tail + categories live on the dedicated Logs page.
   ──────────────────────────────────────────────────────────────────── */

function ServerSlowQueries() {
  const [items, setItems] = useState<SlowQuery[] | null>(null)
  useEffect(() => {
    let alive = true
    fetchSlowQueries({ limit: 20 })
      .then(({ data }) => {
        if (alive) setItems(data.slow_queries)
      })
      .catch(() => {
        if (alive) setItems([])
      })
    return () => {
      alive = false
    }
  }, [])
  if (items === null || items.length === 0) return null
  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Slowest queries
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (server query log · last 24h)
          </span>
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
        <table className="w-full text-[11.5px] tabular-nums">
          <thead className="bg-[var(--color-bg-elevated)] text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
            <tr>
              <th className="px-3 py-1.5 text-left font-medium">Collection</th>
              <th className="px-3 py-1.5 text-right font-medium">Duration</th>
              <th className="px-3 py-1.5 text-right font-medium">top_k</th>
              <th className="px-3 py-1.5 text-left font-medium w-[150px]">Trace</th>
              <th className="px-3 py-1.5 text-right font-medium">When</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {items.map((q, i) => (
              <tr key={i} className="hover:bg-[var(--color-bg-hover)]">
                <td className="px-3 py-1 font-mono text-[var(--color-text)] truncate max-w-[280px]">
                  {q.collection}
                </td>
                <td className="px-3 py-1 text-right text-[var(--color-text)] font-mono">
                  {q.took_ms}ms
                </td>
                <td className="px-3 py-1 text-right font-mono text-[var(--color-text-muted)]">
                  {q.top_k}
                </td>
                <td className="px-3 py-1">
                  {q.trace_id ? (
                    <span className="inline-flex items-center gap-1">
                      <code className="font-mono text-[10.5px] text-[var(--color-text-subtle)]">
                        {q.trace_id.slice(0, 12)}…
                      </code>
                      <CopyButton value={q.trace_id} />
                    </span>
                  ) : (
                    <span className="text-[var(--color-text-disabled)]">—</span>
                  )}
                </td>
                <td className="px-3 py-1 text-right text-[var(--color-text-subtle)]">
                  {formatRelative(q.ts_ms)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Raw metrics fallback table
   ──────────────────────────────────────────────────────────────────── */

function RawMetricsTable({ latest }: { latest: MetricEntry[] }) {
  const [show, setShow] = useState(false)
  const [filter, setFilter] = useState('')
  const filtered = latest
    .filter((m) => m.name.startsWith('veda_'))
    .filter((m) => !filter || m.name.toLowerCase().includes(filter.toLowerCase()))
    .slice(0, 200)
  return (
    <section>
      <button
        onClick={() => setShow((v) => !v)}
        className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] hover:text-[var(--color-text-muted)] flex items-center gap-1.5"
      >
        Raw metrics
        <span className="font-mono normal-case tracking-normal text-[var(--color-text-disabled)]">
          {show ? '−' : '+'}
        </span>
      </button>
      {show && (
        <div className="mt-2">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter veda_ metrics…"
            className="w-full h-7 px-2 mb-2 rounded-[var(--radius-xs)] bg-[var(--color-base-1)] border border-[var(--color-border)] font-mono text-[11.5px] focus:outline-none focus:border-[var(--color-accent)]"
          />
          <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden max-h-72 overflow-y-auto">
            <table className="w-full text-[11px] tabular-nums">
              <tbody>
                {filtered.map((m, i) => (
                  <tr
                    key={i}
                    className="hover:bg-[var(--color-bg-hover)] border-b border-[var(--color-border)]"
                  >
                    <td className="px-3 py-0.5 font-mono text-[var(--color-text)] truncate max-w-[600px]">
                      {m.name}
                      {Object.keys(m.labels).length > 0 && (
                        <span className="text-[var(--color-text-subtle)]">
                          {'{' +
                            Object.entries(m.labels)
                              .map(([k, v]) => `${k}="${v}"`)
                              .join(', ') +
                            '}'}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-0.5 text-right font-mono text-[var(--color-text-muted)] w-32">
                      {m.value.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}

function SkeletonRows() {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-20 w-full" />
        ))}
      </div>
      <Skeleton className="h-48 w-full" />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Prometheus parse + helpers
   ──────────────────────────────────────────────────────────────────── */

function parsePrometheus(text: string): MetricEntry[] {
  const out: MetricEntry[] = []
  for (const line of text.split('\n')) {
    const t = line.trim()
    if (!t || t.startsWith('#')) continue
    const m = t.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([\d.eE+-]+)/)
    if (!m) continue
    const name = m[1]!
    const labelStr = m[2] ?? ''
    const value = parseFloat(m[3]!)
    const labels: Record<string, string> = {}
    if (labelStr) {
      const inner = labelStr.slice(1, -1)
      for (const part of splitLabels(inner)) {
        const eq = part.indexOf('=')
        if (eq < 0) continue
        const k = part.slice(0, eq).trim()
        let v = part.slice(eq + 1).trim()
        if (v.startsWith('"') && v.endsWith('"')) v = v.slice(1, -1)
        labels[k] = v
      }
    }
    out.push({ name, labels, value })
  }
  return out
}

function splitLabels(s: string): string[] {
  // Labels may contain commas inside quoted values; split with state.
  const out: string[] = []
  let cur = ''
  let inQ = false
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (c === '"' && s[i - 1] !== '\\') inQ = !inQ
    if (c === ',' && !inQ) {
      out.push(cur)
      cur = ''
    } else {
      cur += c
    }
  }
  if (cur) out.push(cur)
  return out
}

function sumByName(arr: MetricEntry[], name: string): number {
  let s = 0
  for (const e of arr) if (e.name === name) s += e.value
  return s
}

function sumWhere(
  arr: MetricEntry[],
  name: string,
  labels: Record<string, string>,
): number {
  let s = 0
  for (const e of arr) {
    if (e.name !== name) continue
    let ok = true
    for (const [k, v] of Object.entries(labels)) {
      if (e.labels[k] !== v) {
        ok = false
        break
      }
    }
    if (ok) s += e.value
  }
  return s
}

function humanFormat(n: number): string {
  if (!Number.isFinite(n)) return '—'
  if (n === 0) return '0'
  const abs = Math.abs(n)
  if (abs >= 1e9) return (n / 1e9).toFixed(1) + 'B'
  if (abs >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (abs >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  if (Number.isInteger(n)) return n.toLocaleString()
  return n.toFixed(2)
}

function formatRelative(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}
