import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, GitBranch, Loader, Sparkles } from 'lucide-react'
import { routeQuery, type RouteResponse } from '../../api/collections'
import { Badge, Button } from '../../components/primitives'
import type { SavedObject } from '../../stores/savedObjectStore'
import { cn } from '../../lib/cn'
import type { EvalRunBody, EvalSetBody, PerQueryResult } from './types'

interface RouterExplorerProps {
  set: SavedObject<EvalSetBody>
  /** All runs that share this set. Used to overlay observed best-RR per query. */
  runs: SavedObject<EvalRunBody>[]
  onClose: () => void
}

interface RouterRow {
  query: string
  expected: number
  route?: RouteResponse
  error?: string
  /** Per-mode best RR observed in the available runs. */
  perMode: Map<string, PerQueryResult>
}

const STRATEGY_COLORS: Record<string, string> = {
  Dense: 'var(--color-accent-7)',
  Sparse: 'var(--color-data-5)',
  BM25: 'var(--color-warning)',
  Hybrid: 'var(--color-success)',
}

/**
 * Adaptive-router decision explorer. For every query in the eval set,
 * ask /v1/route which retrieval strategy it would pick + why. When
 * eval runs are present, overlay the per-query RR for the chosen mode
 * so the user can spot misfires (router said Dense, Hybrid actually
 * scored 0.42 higher).
 *
 * Pure frontend on the existing endpoint — when Phase 5.7 ships a
 * query-log retention layer, the same UI gains real production-traffic
 * coverage instead of just an eval-set replay.
 */
export function RouterExplorer({ set, runs, onClose }: RouterExplorerProps) {
  const [rows, setRows] = useState<RouterRow[]>([])
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [running, setRunning] = useState(false)
  const [strategyFilter, setStrategyFilter] = useState<string | null>(null)

  const run = async () => {
    setRunning(true)
    setRows([])
    setProgress({ done: 0, total: set.body.items.length })
    const out: RouterRow[] = []
    for (let i = 0; i < set.body.items.length; i++) {
      const item = set.body.items[i]!
      const perMode = new Map<string, PerQueryResult>()
      for (const runObj of runs) {
        const mode = (runObj.body.searchConfig as { mode?: string })?.mode ?? 'dense'
        const r = runObj.body.perQuery.find((q) => q.query === item.query)
        if (r) {
          const prev = perMode.get(mode)
          if (!prev || r.rr > prev.rr) perMode.set(mode, r)
        }
      }
      try {
        const { data } = await routeQuery(item.query)
        out.push({
          query: item.query,
          expected: item.expected_doc_ids?.length ?? 0,
          route: data,
          perMode,
        })
      } catch (e) {
        out.push({
          query: item.query,
          expected: item.expected_doc_ids?.length ?? 0,
          error: e instanceof Error ? e.message : 'Route call failed',
          perMode,
        })
      }
      setProgress({ done: i + 1, total: set.body.items.length })
      setRows([...out])
    }
    setRunning(false)
  }

  // Auto-run on open.
  useEffect(() => {
    run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [set.id])

  const histogram = useMemo(() => {
    const h = new Map<string, number>()
    for (const r of rows) {
      const s = r.route?.strategy ?? 'error'
      h.set(s, (h.get(s) ?? 0) + 1)
    }
    return Array.from(h.entries()).sort((a, b) => b[1] - a[1])
  }, [rows])

  const filteredRows = useMemo(
    () =>
      strategyFilter
        ? rows.filter((r) => r.route?.strategy === strategyFilter)
        : rows,
    [rows, strategyFilter],
  )

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <button
          onClick={onClose}
          className="inline-flex items-center gap-1 text-[11.5px] text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        >
          <ArrowLeft size={11} strokeWidth={2.25} />
          Back
        </button>
        <span className="text-[var(--color-text-subtle)]">·</span>
        <GitBranch
          size={13}
          strokeWidth={2}
          className="text-[var(--color-accent-9)]"
        />
        <span className="text-[13.5px] font-medium text-[var(--color-text)]">
          Router decisions · {set.name}
        </span>
        <Badge label={`${set.body.items.length} q`} variant="ghost" />
        {running && (
          <span className="inline-flex items-center gap-1 ml-2 text-[11px] text-[var(--color-accent)]">
            <Loader size={11} strokeWidth={2.25} className="animate-spin" />
            {progress.done} / {progress.total}
          </span>
        )}
        <Button
          variant="ghost"
          size="xs"
          leading={<Sparkles size={11} strokeWidth={2.25} />}
          onClick={run}
          disabled={running}
          className="ml-auto"
        >
          Re-run
        </Button>
      </header>

      <section className="px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] flex-none">
        <div className="flex items-baseline gap-2 mb-2">
          <h3 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
            Strategy distribution
          </h3>
          <span className="text-[10.5px] font-mono text-[var(--color-text-muted)] tabular-nums">
            {rows.length} of {set.body.items.length} routed
          </span>
        </div>
        <div className="flex items-stretch gap-2">
          {histogram.length === 0 && (
            <span className="text-[11.5px] text-[var(--color-text-subtle)] italic">
              Awaiting first responses…
            </span>
          )}
          {histogram.map(([s, n]) => (
            <button
              key={s}
              onClick={() => setStrategyFilter(strategyFilter === s ? null : s)}
              className={cn(
                'flex-1 max-w-[200px] rounded-[var(--radius-sm)] border px-2.5 py-1.5',
                'transition-colors text-left',
                strategyFilter === s
                  ? 'border-[var(--color-accent)] bg-[var(--color-accent-2)]'
                  : 'border-[var(--color-border)] bg-[var(--color-bg)] hover:border-[var(--color-border-strong)]',
              )}
            >
              <div className="flex items-baseline justify-between mb-1">
                <span
                  className="text-[12px] font-medium"
                  style={{ color: STRATEGY_COLORS[s] ?? 'var(--color-text)' }}
                >
                  {s}
                </span>
                <span className="text-[10.5px] font-mono tabular-nums text-[var(--color-text-muted)]">
                  {n}
                </span>
              </div>
              <div className="h-1 bg-[var(--color-base-2)] rounded-full overflow-hidden">
                <div
                  className="h-full"
                  style={{
                    width: `${(n / rows.length) * 100}%`,
                    background: STRATEGY_COLORS[s] ?? 'var(--color-accent-7)',
                  }}
                />
              </div>
            </button>
          ))}
        </div>
      </section>

      <div className="flex-1 overflow-auto min-h-0">
        <table className="w-full text-[12px] tabular-nums">
          <thead className="sticky top-0 bg-[var(--color-bg)] border-b border-[var(--color-border)]">
            <tr className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
              <th className="w-9 px-2 py-1.5 text-left font-medium">#</th>
              <th className="w-24 px-2 py-1.5 text-left font-medium">
                Strategy
              </th>
              <th className="px-3 py-1.5 text-left font-medium">Query</th>
              <th className="px-3 py-1.5 text-left font-medium">Reasoning</th>
              <th className="w-14 px-2 py-1.5 text-right font-medium">k</th>
              <th className="w-16 px-2 py-1.5 text-right font-medium">Observed RR</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {filteredRows.map((r, i) => (
              <Row key={i} row={r} idx={i} />
            ))}
          </tbody>
        </table>
        {filteredRows.length === 0 && !running && (
          <div className="px-6 py-8 text-center text-[12px] text-[var(--color-text-subtle)] italic">
            {strategyFilter
              ? `No queries routed to ${strategyFilter}.`
              : 'No queries routed yet — click Re-run.'}
          </div>
        )}
      </div>
    </div>
  )
}

function Row({ row, idx }: { row: RouterRow; idx: number }) {
  const observedRr = useMemo(() => {
    if (row.perMode.size === 0) return null
    let best: { mode: string; rr: number } | null = null
    for (const [mode, r] of row.perMode) {
      if (!best || r.rr > best.rr) best = { mode, rr: r.rr }
    }
    return best
  }, [row.perMode])

  const routerSuggested = row.route?.strategy?.toLowerCase()
  const observedMode = observedRr?.mode?.toLowerCase()
  const disagreed =
    observedRr != null &&
    routerSuggested != null &&
    routerSuggested !== observedMode &&
    observedRr.rr >= 0.1

  return (
    <tr
      className={cn(
        'hover:bg-[var(--color-bg-hover)]',
        disagreed && 'bg-[color-mix(in_oklch,var(--color-warning)_6%,transparent)]',
      )}
    >
      <td className="px-2 py-1.5 text-[var(--color-text-subtle)] font-mono text-[10.5px]">
        {(idx + 1).toString().padStart(3, '0')}
      </td>
      <td className="px-2 py-1.5">
        {row.route ? (
          <span
            className="inline-flex items-center px-1.5 h-5 rounded-[var(--radius-xs)] text-[10.5px] font-medium"
            style={{
              color: STRATEGY_COLORS[row.route.strategy] ?? 'var(--color-text)',
              background: `color-mix(in oklch, ${
                STRATEGY_COLORS[row.route.strategy] ?? 'var(--color-text)'
              } 14%, transparent)`,
              borderColor: `color-mix(in oklch, ${
                STRATEGY_COLORS[row.route.strategy] ?? 'var(--color-text)'
              } 28%, transparent)`,
              borderWidth: 1,
              borderStyle: 'solid',
            }}
          >
            {row.route.strategy}
          </span>
        ) : row.error ? (
          <Badge label="error" variant="error" dot />
        ) : (
          <span className="text-[var(--color-text-subtle)] font-mono text-[10.5px]">…</span>
        )}
      </td>
      <td className="px-3 py-1.5 text-[var(--color-text)] truncate max-w-[280px]">
        {row.query}
      </td>
      <td className="px-3 py-1.5 text-[10.5px] text-[var(--color-text-muted)] italic truncate max-w-[360px]">
        {row.error ?? row.route?.reasoning ?? '…'}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-[10.5px] text-[var(--color-text-muted)]">
        {row.route?.suggested_k ?? '—'}
      </td>
      <td className="px-2 py-1.5 text-right font-mono text-[10.5px]">
        {observedRr ? (
          <span
            className={cn(
              'inline-flex items-center gap-1 justify-end',
              disagreed && 'text-[var(--color-warning)]',
            )}
            title={
              disagreed
                ? `Router picked ${row.route?.strategy}; ${observedRr.mode} scored RR ${observedRr.rr.toFixed(3)}`
                : `Observed under ${observedRr.mode}`
            }
          >
            <span className="text-[var(--color-text-subtle)] text-[9.5px] uppercase">
              {observedRr.mode}
            </span>
            {observedRr.rr.toFixed(3)}
          </span>
        ) : (
          <span className="text-[var(--color-text-subtle)]">—</span>
        )}
      </td>
    </tr>
  )
}
