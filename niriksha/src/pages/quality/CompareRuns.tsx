import { useMemo, useState } from 'react'
import { ArrowLeft, ArrowRight, X } from 'lucide-react'
import { Badge, Button } from '../../components/primitives'
import { cn } from '../../lib/cn'
import type { SavedObject } from '../../stores/savedObjectStore'
import type { EvalRunBody, PerQueryResult } from './types'

interface CompareRunsProps {
  a: SavedObject<EvalRunBody>
  b: SavedObject<EvalRunBody>
  onClose: () => void
}

type DiffFilter = 'all' | 'a-better' | 'b-better' | 'tied' | 'either-failed'

interface DiffRow {
  query: string
  a?: PerQueryResult
  b?: PerQueryResult
  /** B.rr − A.rr — positive means B did better. */
  delta: number
}

/**
 * Side-by-side run diff. LangSmith / OpenSearch Workbench pattern:
 * top metrics card showing A vs B at-a-glance, then a per-query
 * table sortable by absolute delta, click to expand and see both
 * retrieved chunk lists.
 */
export function CompareRuns({ a, b, onClose }: CompareRunsProps) {
  const rows = useMemo(() => joinRuns(a, b), [a, b])
  const [filter, setFilter] = useState<DiffFilter>('all')
  const [expanded, setExpanded] = useState<string | null>(null)
  const filtered = useMemo(() => applyFilter(rows, filter), [rows, filter])

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
        <span className="text-[13.5px] font-medium text-[var(--color-text)]">
          A/B compare
        </span>
        <Badge label={`${rows.length} queries`} variant="ghost" />
        <Button
          variant="ghost"
          size="xs"
          leading={<X size={11} strokeWidth={2.25} />}
          onClick={onClose}
          className="ml-auto"
        >
          Close
        </Button>
      </header>

      <SummaryRow a={a} b={b} />

      <FilterBar
        filter={filter}
        onChange={setFilter}
        rows={rows}
      />

      <div className="flex-1 overflow-auto min-h-0">
        <table className="w-full text-[12px] tabular-nums">
          <thead className="sticky top-0 bg-[var(--color-bg)] border-b border-[var(--color-border)]">
            <tr className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
              <th className="w-9 px-2 py-1.5 text-left font-medium">#</th>
              <th className="px-3 py-1.5 text-left font-medium">Query</th>
              <th className="w-20 px-2 py-1.5 text-right font-medium">
                A · rr
              </th>
              <th className="w-20 px-2 py-1.5 text-right font-medium">
                B · rr
              </th>
              <th className="w-20 px-2 py-1.5 text-right font-medium">Δ rr</th>
              <th className="w-20 px-2 py-1.5 text-right font-medium">A took</th>
              <th className="w-20 px-2 py-1.5 text-right font-medium">B took</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--color-border)]">
            {filtered.map((r, i) => (
              <DiffRowView
                key={r.query}
                row={r}
                idx={i}
                expanded={expanded === r.query}
                onToggle={() =>
                  setExpanded(expanded === r.query ? null : r.query)
                }
              />
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="px-6 py-8 text-center text-[12px] text-[var(--color-text-subtle)] italic">
            No queries match this filter.
          </div>
        )}
      </div>
    </div>
  )
}

function SummaryRow({
  a,
  b,
}: {
  a: SavedObject<EvalRunBody>
  b: SavedObject<EvalRunBody>
}) {
  return (
    <div className="grid grid-cols-2 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] flex-none">
      <RunSummaryCard run={a} side="A" />
      <RunSummaryCard run={b} side="B" companion={a} />
    </div>
  )
}

function RunSummaryCard({
  run,
  side,
  companion,
}: {
  run: SavedObject<EvalRunBody>
  side: 'A' | 'B'
  /** When set, deltas vs `companion` are rendered next to each metric. */
  companion?: SavedObject<EvalRunBody>
}) {
  const s = run.body.summary
  const c = companion?.body.summary
  return (
    <div
      className={cn(
        'px-4 py-3',
        side === 'B' && 'border-l border-[var(--color-border)]',
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        <Badge
          label={side}
          variant={side === 'A' ? 'accent' : 'success'}
        />
        <span className="text-[12px] font-medium text-[var(--color-text)] truncate">
          {run.name}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-3 text-[11.5px]">
        <SummaryStat
          label="MRR"
          value={s.mrr.toFixed(3)}
          delta={c ? s.mrr - c.mrr : undefined}
          highlight
        />
        <SummaryStat
          label="Recall"
          value={s.recall_mean.toFixed(3)}
          delta={c ? s.recall_mean - c.recall_mean : undefined}
        />
        <SummaryStat
          label="Hit@1"
          value={`${(s.hit_at_1_pct * 100).toFixed(0)}%`}
          delta={c ? (s.hit_at_1_pct - c.hit_at_1_pct) * 100 : undefined}
          deltaSuffix="pp"
        />
        <SummaryStat
          label="p95"
          value={`${s.p95_ms.toFixed(0)}ms`}
          delta={c ? s.p95_ms - c.p95_ms : undefined}
          deltaSuffix="ms"
          /** Latency: lower is better, so flip the sign for color. */
          deltaLowerIsBetter
        />
      </div>
    </div>
  )
}

function SummaryStat({
  label,
  value,
  delta,
  deltaSuffix,
  deltaLowerIsBetter,
  highlight,
}: {
  label: string
  value: string
  delta?: number
  deltaSuffix?: string
  deltaLowerIsBetter?: boolean
  highlight?: boolean
}) {
  const good = delta == null
    ? null
    : deltaLowerIsBetter
      ? delta < 0
      : delta > 0
  return (
    <div>
      <div className="text-[9.5px] uppercase tracking-wider text-[var(--color-text-subtle)]">
        {label}
      </div>
      <div
        className={cn(
          'font-mono tabular-nums',
          highlight
            ? 'text-[14px] text-[var(--color-text)] font-semibold'
            : 'text-[12px] text-[var(--color-text)]',
        )}
      >
        {value}
      </div>
      {delta != null && Math.abs(delta) > 1e-6 && (
        <div
          className={cn(
            'text-[10px] font-mono tabular-nums',
            good && 'text-[var(--color-success)]',
            !good && 'text-[var(--color-error)]',
          )}
        >
          {delta > 0 ? '+' : ''}
          {Math.abs(delta) >= 10
            ? delta.toFixed(0)
            : delta.toFixed(3)}
          {deltaSuffix ?? ''}
        </div>
      )}
    </div>
  )
}

function FilterBar({
  filter,
  onChange,
  rows,
}: {
  filter: DiffFilter
  onChange: (f: DiffFilter) => void
  rows: DiffRow[]
}) {
  const counts: Record<DiffFilter, number> = {
    all: rows.length,
    'a-better': rows.filter((r) => r.delta < -1e-6).length,
    'b-better': rows.filter((r) => r.delta > 1e-6).length,
    tied: rows.filter((r) => Math.abs(r.delta) <= 1e-6).length,
    'either-failed': rows.filter((r) => r.a?.error || r.b?.error).length,
  }
  const opts: { id: DiffFilter; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'a-better', label: 'A wins' },
    { id: 'b-better', label: 'B wins' },
    { id: 'tied', label: 'Tied' },
    { id: 'either-failed', label: 'Failed' },
  ]
  return (
    <div className="flex items-center gap-1 px-4 h-9 border-b border-[var(--color-border)] flex-none">
      {opts.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          className={cn(
            'inline-flex items-center gap-1 px-2 h-6 rounded-[var(--radius-xs)]',
            'text-[11.5px] transition-colors',
            filter === o.id
              ? 'bg-[var(--color-accent-2)] text-[var(--color-accent-11)]'
              : 'text-[var(--color-text-muted)] hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text)]',
          )}
        >
          {o.label}
          <span className="text-[9.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
            {counts[o.id]}
          </span>
        </button>
      ))}
    </div>
  )
}

function DiffRowView({
  row,
  idx,
  expanded,
  onToggle,
}: {
  row: DiffRow
  idx: number
  expanded: boolean
  onToggle: () => void
}) {
  const better = row.delta > 1e-6 ? 'B' : row.delta < -1e-6 ? 'A' : 'tie'
  return (
    <>
      <tr
        onClick={onToggle}
        className={cn(
          'hover:bg-[var(--color-bg-hover)] cursor-pointer',
          expanded && 'bg-[var(--color-bg-hover)]',
        )}
      >
        <td className="px-2 py-1.5 text-[var(--color-text-subtle)] font-mono text-[10.5px]">
          {(idx + 1).toString().padStart(3, '0')}
        </td>
        <td className="px-3 py-1.5 text-[var(--color-text)] truncate max-w-[420px]">
          {row.query}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[10.5px]">
          {row.a ? row.a.rr.toFixed(3) : '—'}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[10.5px]">
          {row.b ? row.b.rr.toFixed(3) : '—'}
        </td>
        <td
          className={cn(
            'px-2 py-1.5 text-right font-mono text-[10.5px] tabular-nums',
            better === 'B' && 'text-[var(--color-success)]',
            better === 'A' && 'text-[var(--color-warning)]',
            better === 'tie' && 'text-[var(--color-text-subtle)]',
          )}
        >
          {row.delta > 0 ? '+' : ''}
          {row.delta.toFixed(3)}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[10.5px] text-[var(--color-text-muted)]">
          {row.a ? `${row.a.took_ms.toFixed(0)}ms` : '—'}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[10.5px] text-[var(--color-text-muted)]">
          {row.b ? `${row.b.took_ms.toFixed(0)}ms` : '—'}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-[var(--color-bg-elevated)]">
          <td colSpan={7} className="px-4 py-3">
            <div className="grid grid-cols-2 gap-4">
              <RetrievedList side="A" r={row.a} />
              <RetrievedList side="B" r={row.b} />
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function RetrievedList({
  side,
  r,
}: {
  side: 'A' | 'B'
  r: PerQueryResult | undefined
}) {
  const expected = new Set(r?.expected ?? [])
  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <Badge label={side} variant={side === 'A' ? 'accent' : 'success'} />
        <span className="text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)]">
          Retrieved · top {r?.retrieved_ids.length ?? 0}
        </span>
        {r?.error && (
          <Badge label={`error`} variant="error" dot />
        )}
      </div>
      {r?.error ? (
        <pre className="text-[10.5px] font-mono text-[var(--color-error)] bg-[var(--color-bg)] rounded p-2 whitespace-pre-wrap break-words">
          {r.error}
        </pre>
      ) : (
        <ol className="text-[10.5px] font-mono space-y-0.5">
          {(r?.retrieved_ids ?? []).map((id, i) => {
            const hit = expected.has(id)
            return (
              <li
                key={id + i}
                className={cn(
                  'flex items-center gap-1.5',
                  hit && 'text-[var(--color-success)]',
                  !hit && 'text-[var(--color-text-muted)]',
                )}
              >
                <span className="w-5 text-right text-[var(--color-text-subtle)] tabular-nums">
                  {(i + 1).toString().padStart(2, '0')}
                </span>
                {hit && <ArrowRight size={9} strokeWidth={2.25} />}
                <span className="truncate">{id}</span>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────────────── */

function joinRuns(
  a: SavedObject<EvalRunBody>,
  b: SavedObject<EvalRunBody>,
): DiffRow[] {
  const m = new Map<string, DiffRow>()
  for (const r of a.body.perQuery) {
    m.set(r.query, { query: r.query, a: r, delta: 0 })
  }
  for (const r of b.body.perQuery) {
    const existing = m.get(r.query)
    if (existing) {
      existing.b = r
      existing.delta = r.rr - (existing.a?.rr ?? 0)
    } else {
      m.set(r.query, { query: r.query, b: r, delta: r.rr })
    }
  }
  return Array.from(m.values()).sort(
    (x, y) => Math.abs(y.delta) - Math.abs(x.delta),
  )
}

function applyFilter(rows: DiffRow[], filter: DiffFilter): DiffRow[] {
  switch (filter) {
    case 'all':
      return rows
    case 'a-better':
      return rows.filter((r) => r.delta < -1e-6)
    case 'b-better':
      return rows.filter((r) => r.delta > 1e-6)
    case 'tied':
      return rows.filter((r) => Math.abs(r.delta) <= 1e-6)
    case 'either-failed':
      return rows.filter((r) => r.a?.error || r.b?.error)
  }
}
