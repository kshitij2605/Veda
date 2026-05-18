import { useMemo } from 'react'
import type { FieldSchema, RecordRow } from '../../api/collections'
import { isVectorField, isBlobField } from '../collections/types'
import { cn } from '../../lib/cn'

interface BubbleUpProps {
  fields: FieldSchema[]
  /** All records that are plotted in the atlas (sample). */
  allRecords: RecordRow[]
  /** Subset selected by lasso. */
  selectedRecords: RecordRow[]
}

interface FieldDelta {
  field: string
  topValues: {
    value: string
    inLasso: number
    inLassoPct: number
    inCorpus: number
    inCorpusPct: number
    delta: number
  }[]
  /** Sum of absolute deltas — used to rank fields. */
  signal: number
}

/**
 * BubbleUp — Honeycomb-style "what's different about this region?"
 * For every scalar field, compute the value distribution inside the
 * lassoed sample vs the whole sample, sort by the largest delta-share
 * and render the top contributors. Pure client-side.
 */
export function BubbleUp({ fields, allRecords, selectedRecords }: BubbleUpProps) {
  const deltas = useMemo(
    () => computeDeltas(fields, allRecords, selectedRecords),
    [fields, allRecords, selectedRecords],
  )

  if (selectedRecords.length === 0) {
    return (
      <div className="p-3 text-[11.5px] text-[var(--color-text-subtle)] italic">
        Lasso a region to compare field distributions inside the selection
        vs the whole sample.
      </div>
    )
  }

  if (deltas.length === 0) {
    return (
      <div className="p-3 text-[11.5px] text-[var(--color-text-subtle)] italic">
        No scalar fields to compare.
      </div>
    )
  }

  return (
    <div className="overflow-y-auto h-full">
      <header className="px-3 py-2 border-b border-[var(--color-border)] sticky top-0 bg-[var(--color-bg)]">
        <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          BubbleUp
        </div>
        <div className="text-[11.5px] font-mono text-[var(--color-text-muted)] tabular-nums mt-0.5">
          {selectedRecords.length.toLocaleString()} of{' '}
          {allRecords.length.toLocaleString()} in lasso
        </div>
      </header>
      <ul className="divide-y divide-[var(--color-border)]">
        {deltas.map((d) => (
          <FieldRow key={d.field} delta={d} />
        ))}
      </ul>
    </div>
  )
}

function FieldRow({ delta }: { delta: FieldDelta }) {
  return (
    <li className="px-3 py-2.5">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="font-mono text-[12px] text-[var(--color-text)] truncate">
          {delta.field}
        </span>
        <span className="text-[10px] font-mono text-[var(--color-text-subtle)] tabular-nums">
          signal {delta.signal.toFixed(2)}
        </span>
      </div>
      <ul className="space-y-1">
        {delta.topValues.slice(0, 5).map((v) => (
          <ValueRow key={v.value} v={v} />
        ))}
      </ul>
    </li>
  )
}

function ValueRow({
  v,
}: {
  v: FieldDelta['topValues'][number]
}) {
  const positive = v.delta >= 0
  return (
    <li className="grid grid-cols-[1fr_60px_24px] items-center gap-2 text-[11px]">
      <span className="truncate font-mono text-[var(--color-text)]">
        {v.value || <em className="text-[var(--color-text-subtle)]">(empty)</em>}
      </span>
      <div className="relative h-1.5 bg-[var(--color-base-2)] rounded-full overflow-hidden">
        {/* corpus bar (faint) */}
        <div
          className="absolute inset-y-0 left-0 bg-[var(--color-base-5)]"
          style={{ width: `${v.inCorpusPct}%` }}
        />
        {/* lasso bar (accent / data, on top) */}
        <div
          className={cn(
            'absolute inset-y-0 left-0',
            positive ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-warning)]',
          )}
          style={{ width: `${v.inLassoPct}%`, opacity: 0.9 }}
        />
      </div>
      <span
        className={cn(
          'text-[10px] font-mono tabular-nums text-right',
          positive
            ? 'text-[var(--color-accent-9)]'
            : 'text-[var(--color-warning)]',
        )}
      >
        {positive ? '+' : ''}
        {(v.delta * 100).toFixed(0)}%
      </span>
    </li>
  )
}

function computeDeltas(
  fields: FieldSchema[],
  all: RecordRow[],
  sel: RecordRow[],
): FieldDelta[] {
  if (sel.length === 0) return []
  const allN = all.length || 1
  const selN = sel.length || 1
  const out: FieldDelta[] = []
  for (const f of fields) {
    if (isVectorField(f) || isBlobField(f)) continue
    if (f.type === 'json' || f.type === 'sparse_vector') continue

    const corpusCounts = new Map<string, number>()
    const lassoCounts = new Map<string, number>()
    for (const r of all) {
      const v = r.fields[f.name]
      if (v == null || (typeof v === 'string' && v === '')) continue
      const key = String(v)
      corpusCounts.set(key, (corpusCounts.get(key) ?? 0) + 1)
    }
    for (const r of sel) {
      const v = r.fields[f.name]
      if (v == null || (typeof v === 'string' && v === '')) continue
      const key = String(v)
      lassoCounts.set(key, (lassoCounts.get(key) ?? 0) + 1)
    }

    // Skip if the field is constant across the whole sample (boring).
    if (corpusCounts.size === 0 || corpusCounts.size === 1) continue

    const values = new Set<string>([
      ...corpusCounts.keys(),
      ...lassoCounts.keys(),
    ])
    const rows: FieldDelta['topValues'] = []
    let signal = 0
    for (const key of values) {
      const inLasso = lassoCounts.get(key) ?? 0
      const inCorpus = corpusCounts.get(key) ?? 0
      const inLassoPct = (inLasso / selN) * 100
      const inCorpusPct = (inCorpus / allN) * 100
      const delta = inLasso / selN - inCorpus / allN
      signal += Math.abs(delta)
      rows.push({
        value: key,
        inLasso,
        inLassoPct,
        inCorpus,
        inCorpusPct,
        delta,
      })
    }
    rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    if (signal > 0.001) {
      out.push({ field: f.name, topValues: rows, signal })
    }
  }
  out.sort((a, b) => b.signal - a.signal)
  return out
}
