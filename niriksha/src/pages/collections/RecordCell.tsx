import { type ReactNode } from 'react'
import { Sparkline } from '../../components/primitives/Sparkline'
import { cn } from '../../lib/cn'
import type { FieldSchema } from '../../api/collections'
import { isVectorField } from './types'

interface RecordCellProps {
  field: FieldSchema
  value: unknown
  className?: string
}

/**
 * RecordCell — renders one cell in the records grid. Picks the right
 * presentation per type:
 *   - vector: sparkline + dims badge (never raw float dump)
 *   - blob: "blob ✱"
 *   - object/array: truncated JSON snippet
 *   - long string: ellipsis with hover-full-value
 *   - everything else: stringified, tabular-nums for numerics
 */
export function RecordCell({ field, value, className }: RecordCellProps) {
  if (value == null) {
    return (
      <span className={cn('text-[var(--color-text-subtle)]', className)}>
        —
      </span>
    )
  }

  if (isVectorField(field) && Array.isArray(value)) {
    const v = value as number[]
    const preview = v.length > 64 ? v.slice(0, 64) : v
    return (
      <span className={cn('inline-flex items-center gap-1.5 align-middle', className)}>
        <Sparkline values={preview} width={56} height={14} />
        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
          {v.length}d
        </span>
      </span>
    )
  }

  if (field.type === 'blob') {
    return <DotBadge label="blob" className={className} />
  }

  if (field.type === 'sparse_vector') {
    const sv = value as { indices?: number[] }
    const nz = sv.indices?.length ?? 0
    return (
      <span className={cn('inline-flex items-center gap-1.5', className)}>
        <DotBadge label="sparse" />
        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
          {nz}nz
        </span>
      </span>
    )
  }

  if (typeof value === 'object') {
    const text = JSON.stringify(value)
    return (
      <span
        className={cn(
          'font-mono text-[var(--color-text-muted)] truncate',
          className,
        )}
        title={text}
      >
        {text.length > 80 ? text.slice(0, 80) + '…' : text}
      </span>
    )
  }

  if (typeof value === 'string') {
    return (
      <span
        className={cn('truncate text-[var(--color-text)]', className)}
        title={value.length > 200 ? value : undefined}
      >
        {value}
      </span>
    )
  }

  // numeric, bool, etc.
  return (
    <span className={cn('tabular-nums text-[var(--color-text)]', className)}>
      {String(value)}
    </span>
  )
}

function DotBadge({ label, className }: { label: string; className?: string }): ReactNode {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-1 h-4 rounded-[var(--radius-xs)]',
        'text-[10px] font-mono uppercase tracking-wider',
        'bg-[var(--color-base-2)] text-[var(--color-text-muted)]',
        className,
      )}
    >
      <span className="w-1 h-1 rounded-full bg-[var(--color-data-5)]" aria-hidden />
      {label}
    </span>
  )
}
