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
import { Button, Skeleton } from '../components/primitives'
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
 * open). When the Phase 5.7 query log lands the slow-query log moves
 * to a server-backed view.
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
          size={14}
          strokeWidth={2}
          className="text-[var(--color-accent-9)]"
        />
        <h1 className="text-[13.5px] font-medium text-[var(--color-text)]">
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
            <ConsoleSlowQueries />
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
   Slow queries — from local console history (until Phase 5.7 server log)
   ──────────────────────────────────────────────────────────────────── */

function ConsoleSlowQueries() {
  const [items, setItems] = useState<
    {
      method: string
      path: string
      durationMs: number
      ts: number
      status?: number
      ok: boolean
    }[]
  >([])
  useEffect(() => {
    try {
      const raw = localStorage.getItem('niriksha.console.history.v1')
      if (!raw) return
      const arr = JSON.parse(raw) as typeof items
      setItems(
        arr
          .filter((x) => x.path?.includes('/search') || x.path?.includes('/embed'))
          .sort((a, b) => b.durationMs - a.durationMs)
          .slice(0, 20),
      )
    } catch {
      /* ignore */
    }
  }, [])
  if (items.length === 0) return null
  return (
    <section>
      <header className="mb-2">
        <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Slowest local requests
          <span className="ml-2 font-mono normal-case tracking-normal text-[var(--color-text-muted)]">
            (browser-local until server query log ships)
          </span>
        </h2>
      </header>
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
        <table className="w-full text-[11.5px] tabular-nums">
          <thead className="bg-[var(--color-bg-elevated)] text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
            <tr>
              <th className="px-3 py-1.5 text-left font-medium">Method</th>
              <th className="px-3 py-1.5 text-left font-medium">Path</th>
              <th className="px-3 py-1.5 text-right font-medium">Duration</th>
              <th className="px-3 py-1.5 text-right font-medium">Status</th>
              <th className="px-3 py-1.5 text-right font-medium">When</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {items.map((it, i) => (
              <tr key={i} className="hover:bg-[var(--color-bg-hover)]">
                <td className="px-3 py-1 font-mono text-[var(--color-text-muted)]">
                  {it.method}
                </td>
                <td className="px-3 py-1 font-mono text-[var(--color-text)] truncate max-w-[400px]">
                  {it.path}
                </td>
                <td className="px-3 py-1 text-right text-[var(--color-text)] font-mono">
                  {it.durationMs.toFixed(0)}ms
                </td>
                <td className="px-3 py-1 text-right font-mono">
                  <span
                    className={cn(
                      it.ok ? 'text-[var(--color-success)]' : 'text-[var(--color-error)]',
                    )}
                  >
                    {it.status ?? '—'}
                  </span>
                </td>
                <td className="px-3 py-1 text-right text-[var(--color-text-subtle)]">
                  {formatRelative(it.ts)}
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
