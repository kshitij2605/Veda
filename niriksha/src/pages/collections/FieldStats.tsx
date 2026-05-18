import { useEffect, useMemo, useState } from 'react'
import { Loader, RotateCw } from 'lucide-react'
import { fetchRecords } from '../../api/collections'
import type {
  CollectionInfo,
  FieldSchema,
  RecordRow,
} from '../../api/collections'
import { Badge } from '../../components/primitives/Badge'
import { Button } from '../../components/primitives/Button'
import { Sparkline } from '../../components/primitives/Sparkline'
import { isBlobField, isVectorField } from './types'

interface FieldStatsProps {
  collection: CollectionInfo
}

interface ScalarStats {
  present: number
  missing: number
  /** Cardinality of unique values across the sample. */
  unique: number
  /** Top values (capped). */
  top: { value: unknown; count: number }[]
  /** For numerics: min/max/avg + 10-bin histogram. */
  numeric?: { min: number; max: number; avg: number; bins: number[] }
  /** For strings: length stats. */
  string?: { minLen: number; maxLen: number; avgLen: number }
}

interface VectorStats {
  present: number
  missing: number
  /** Sample of mean values for the first ~64 dims (for the visual sparkline). */
  centroidPreview: number[]
  /** Norms — captures clipping / dead vectors visually. */
  norms: number[]
}

const SAMPLE_SIZE = 1000

/**
 * FieldStats — peer tab to Records. Streams a 1k-row sample, computes
 * cardinality + missing + top-N + min/max/avg per field client-side.
 *
 * Server-side rollup is a Phase 1.5 follow-up; this client pass is good
 * to ~5k records before it gets sluggish.
 */
export function FieldStats({ collection }: FieldStatsProps) {
  const [rows, setRows] = useState<RecordRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)
  const [sampleSize, setSampleSize] = useState(SAMPLE_SIZE)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchRecords(collection.name, { limit: sampleSize })
      .then(({ data }) => {
        if (cancelled) return
        setRows(data.records)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to load sample')
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [collection.name, reloadKey, sampleSize])

  const stats = useMemo(() => computeAllStats(collection.fields, rows), [
    collection.fields,
    rows,
  ])

  return (
    <div className="px-6 py-5 overflow-auto h-full">
      <header className="flex items-center gap-3 mb-4">
        <div className="flex-1">
          <h2 className="text-[14px] font-medium text-[var(--color-text)]">
            Field statistics
          </h2>
          <p className="text-[11.5px] text-[var(--color-text-muted)]">
            Sampled from the first {rows.length.toLocaleString()} of{' '}
            {collection.shard_count > 1
              ? `${collection.shard_count} shards`
              : 'this collection'}
            . Numbers reflect the sample, not the whole corpus.
          </p>
        </div>
        <select
          value={sampleSize}
          onChange={(e) => setSampleSize(parseInt(e.target.value, 10))}
          className="h-7 px-2 rounded-[var(--radius-xs)] bg-[var(--color-base-1)] border border-[var(--color-border)] text-[11.5px] text-[var(--color-text)] font-mono tabular-nums focus:outline-none focus:border-[var(--color-accent)]"
        >
          {[200, 500, 1000, 2000, 5000].map((n) => (
            <option key={n} value={n}>
              {n} rows
            </option>
          ))}
        </select>
        <Button
          size="xs"
          variant="ghost"
          leading={
            loading ? (
              <Loader size={11} strokeWidth={2.25} className="animate-spin" />
            ) : (
              <RotateCw size={11} strokeWidth={2.25} />
            )
          }
          onClick={() => setReloadKey((k) => k + 1)}
          disabled={loading}
        >
          Re-sample
        </Button>
      </header>

      {error && (
        <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] mb-4">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {collection.fields.map((f) => (
          <FieldCard
            key={f.name}
            field={f}
            stats={stats.get(f.name)!}
            totalRows={rows.length}
          />
        ))}
      </div>
    </div>
  )
}

function FieldCard({
  field,
  stats,
  totalRows,
}: {
  field: FieldSchema
  stats: ScalarStats | VectorStats | undefined
  totalRows: number
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3.5">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono text-[12.5px] font-medium text-[var(--color-text)] truncate flex-1">
          {field.name}
        </span>
        <Badge label={field.type} variant={isVectorField(field) ? 'accent' : 'default'} />
      </div>

      <PresenceBar
        present={stats?.present ?? 0}
        total={totalRows}
      />

      {isBlobField(field) && (
        <div className="text-[11px] text-[var(--color-text-subtle)] italic mt-3">
          Blob field — no aggregate stats. Preview via the Records drawer.
        </div>
      )}
      {isVectorField(field) && stats && 'centroidPreview' in stats && (
        <VectorPanel s={stats} />
      )}
      {!isVectorField(field) && !isBlobField(field) && stats && 'top' in stats && (
        <ScalarPanel field={field} s={stats} />
      )}
    </div>
  )
}

function PresenceBar({ present, total }: { present: number; total: number }) {
  const pct = total > 0 ? (present / total) * 100 : 0
  return (
    <div className="space-y-1 mb-3">
      <div className="flex items-baseline justify-between text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
        <span>{present.toLocaleString()} present</span>
        <span>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-1 bg-[var(--color-base-2)] rounded-full overflow-hidden">
        <div
          className="h-full bg-[var(--color-data-5)] transition-[width] duration-[var(--duration-base)]"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function ScalarPanel({ field, s }: { field: FieldSchema; s: ScalarStats }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3 text-[11px]">
        <Stat label="Unique" value={s.unique.toLocaleString()} />
        {s.numeric && (
          <>
            <Stat label="Min" value={fmtNum(s.numeric.min)} />
            <Stat label="Max" value={fmtNum(s.numeric.max)} />
            <Stat label="Avg" value={fmtNum(s.numeric.avg)} />
          </>
        )}
        {s.string && (
          <>
            <Stat label="Min len" value={s.string.minLen.toLocaleString()} />
            <Stat label="Max len" value={s.string.maxLen.toLocaleString()} />
            <Stat label="Avg len" value={s.string.avgLen.toFixed(0)} />
          </>
        )}
      </div>

      {s.numeric && s.numeric.bins.some((b) => b > 0) && (
        <Histogram bins={s.numeric.bins} min={s.numeric.min} max={s.numeric.max} />
      )}

      {s.top.length > 0 && !isStringLong(field, s) && (
        <TopValues top={s.top.slice(0, 6)} totalPresent={s.present} />
      )}
    </div>
  )
}

function VectorPanel({ s }: { s: VectorStats }) {
  return (
    <div className="space-y-3">
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
          Sample centroid (first {s.centroidPreview.length}d)
        </div>
        <Sparkline values={s.centroidPreview} width={300} height={18} />
      </div>
      {s.norms.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
            Vector norms ({s.norms.length} sample)
          </div>
          <NormHistogram norms={s.norms} />
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
        {label}
      </span>
      <span className="text-[var(--color-text)] font-mono tabular-nums truncate">
        {value}
      </span>
    </div>
  )
}

function Histogram({
  bins,
  min,
  max,
}: {
  bins: number[]
  min: number
  max: number
}) {
  const maxCount = Math.max(...bins, 1)
  return (
    <div className="space-y-1">
      <div className="flex items-end gap-px h-8">
        {bins.map((c, i) => {
          const pct = (c / maxCount) * 100
          return (
            <div
              key={i}
              className="flex-1 bg-[var(--color-data-5)] rounded-t-[1px] transition-[height] duration-[var(--duration-base)]"
              style={{ height: `${pct}%`, minHeight: c > 0 ? 1 : 0 }}
              title={`${c.toLocaleString()} rows`}
            />
          )
        })}
      </div>
      <div className="flex justify-between text-[10px] font-mono text-[var(--color-text-subtle)] tabular-nums">
        <span>{fmtNum(min)}</span>
        <span>{fmtNum(max)}</span>
      </div>
    </div>
  )
}

function NormHistogram({ norms }: { norms: number[] }) {
  const bins = useMemo(() => binValues(norms, 24), [norms])
  if (norms.length === 0) return null
  return <Histogram bins={bins.counts} min={bins.min} max={bins.max} />
}

function TopValues({
  top,
  totalPresent,
}: {
  top: { value: unknown; count: number }[]
  totalPresent: number
}) {
  return (
    <ul className="space-y-1">
      {top.map((t, i) => {
        const pct = (t.count / totalPresent) * 100
        return (
          <li key={i} className="flex items-center gap-2 text-[11.5px]">
            <span className="flex-1 truncate font-mono text-[var(--color-text)]">
              {String(t.value)}
            </span>
            <span className="w-14 h-1 rounded-full bg-[var(--color-base-2)] overflow-hidden">
              <span
                className="block h-full bg-[var(--color-accent)]"
                style={{ width: `${pct}%` }}
              />
            </span>
            <span className="w-10 text-right font-mono text-[10.5px] text-[var(--color-text-subtle)] tabular-nums">
              {t.count}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function computeAllStats(
  fields: FieldSchema[],
  rows: RecordRow[],
): Map<string, ScalarStats | VectorStats> {
  const out = new Map<string, ScalarStats | VectorStats>()
  for (const f of fields) {
    if (isVectorField(f)) {
      out.set(f.name, computeVectorStats(f, rows))
    } else {
      out.set(f.name, computeScalarStats(f, rows))
    }
  }
  return out
}

function computeScalarStats(field: FieldSchema, rows: RecordRow[]): ScalarStats {
  let present = 0
  const counts = new Map<string, { value: unknown; count: number }>()
  const nums: number[] = []
  const strLens: number[] = []
  for (const r of rows) {
    const v = r.fields[field.name]
    if (v == null || (typeof v === 'string' && v === '')) continue
    present += 1
    if (typeof v === 'object') continue
    const key = String(v)
    const e = counts.get(key)
    if (e) e.count += 1
    else counts.set(key, { value: v, count: 1 })
    if (typeof v === 'number' && Number.isFinite(v)) nums.push(v)
    if (typeof v === 'string') strLens.push(v.length)
  }
  const top = Array.from(counts.values()).sort((a, b) => b.count - a.count)
  const numeric = nums.length > 0 ? makeNumericStats(nums) : undefined
  const string =
    strLens.length > 0
      ? {
          minLen: Math.min(...strLens),
          maxLen: Math.max(...strLens),
          avgLen: strLens.reduce((a, b) => a + b, 0) / strLens.length,
        }
      : undefined
  return {
    present,
    missing: rows.length - present,
    unique: counts.size,
    top,
    numeric,
    string,
  }
}

function makeNumericStats(nums: number[]): {
  min: number
  max: number
  avg: number
  bins: number[]
} {
  const min = Math.min(...nums)
  const max = Math.max(...nums)
  const avg = nums.reduce((a, b) => a + b, 0) / nums.length
  const { counts } = binValues(nums, 20)
  return { min, max, avg, bins: counts }
}

function binValues(
  values: number[],
  binCount: number,
): { counts: number[]; min: number; max: number } {
  if (values.length === 0) return { counts: [], min: 0, max: 0 }
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const range = max - min || 1
  const counts = new Array(binCount).fill(0)
  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.floor(((v - min) / range) * binCount))
    counts[idx] += 1
  }
  return { counts, min, max }
}

function computeVectorStats(field: FieldSchema, rows: RecordRow[]): VectorStats {
  let present = 0
  const dim = field.dimension ?? 0
  const previewDims = Math.min(dim, 64) || 64
  const centroidSum = new Array(previewDims).fill(0)
  const norms: number[] = []
  for (const r of rows) {
    const v = r.fields[field.name]
    if (!Array.isArray(v) || v.length === 0) continue
    present += 1
    for (let i = 0; i < previewDims && i < v.length; i++) {
      centroidSum[i] += v[i] as number
    }
    let s = 0
    for (let i = 0; i < v.length; i++) {
      const x = v[i] as number
      s += x * x
    }
    norms.push(Math.sqrt(s))
  }
  return {
    present,
    missing: rows.length - present,
    centroidPreview: present > 0 ? centroidSum.map((s) => s / present) : [],
    norms,
  }
}

function fmtNum(n: number): string {
  if (!Number.isFinite(n)) return '—'
  if (Math.abs(n) >= 1e9 || (Math.abs(n) > 0 && Math.abs(n) < 0.001))
    return n.toExponential(2)
  if (Number.isInteger(n)) return n.toLocaleString()
  return n.toFixed(3)
}

function isStringLong(f: FieldSchema, s: ScalarStats): boolean {
  return f.type === 'string' && (s.string?.avgLen ?? 0) > 80
}
