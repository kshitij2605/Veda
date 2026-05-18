import { useVirtualizer } from '@tanstack/react-virtual'
import { useRef } from 'react'
import type { FieldSchema, RecordRow } from '../../api/collections'
import { RecordCell } from './RecordCell'
import { Skeleton } from '../../components/primitives/Skeleton'
import { cn } from '../../lib/cn'
import { DENSITY_DIMS, type Density } from './types'

export type Row = RecordRow

interface RecordsTableProps {
  rows: Row[]
  fields: FieldSchema[]
  columns: string[]
  density: Density
  loading?: boolean
  onRowClick: (row: Row) => void
  selectedId?: string
}

export function RecordsTable({
  rows,
  fields,
  columns,
  density,
  loading,
  onRowClick,
  selectedId,
}: RecordsTableProps) {
  const parentRef = useRef<HTMLDivElement>(null)
  const dims = DENSITY_DIMS[density]

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => dims.rowH,
    overscan: 12,
  })

  const colDefs = columnDefs(columns, fields)

  return (
    <div className="flex flex-col h-full min-h-0">
      <div
        className="flex border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] sticky top-0 z-10"
        style={{ height: dims.rowH }}
      >
        <HeaderCell width={88} className={cn(dims.cellPx)}>
          _id
        </HeaderCell>
        {colDefs.map((c) => (
          <HeaderCell
            key={c.name}
            width={c.width}
            className={cn(dims.cellPx)}
            type={c.field.type}
          >
            {c.name}
          </HeaderCell>
        ))}
      </div>
      <div ref={parentRef} className="flex-1 overflow-auto min-h-0">
        {loading && rows.length === 0 ? (
          <SkeletonGrid density={density} colCount={colDefs.length + 1} />
        ) : rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6">
            <div className="text-[12.5px] text-[var(--color-text-muted)] mb-1">
              No records match the current filter
            </div>
            <div className="text-[11px] text-[var(--color-text-subtle)]">
              Try removing a chip, or clear all filters from the bar above.
            </div>
          </div>
        ) : (
          <div
            style={{ height: virtualizer.getTotalSize(), position: 'relative' }}
          >
            {virtualizer.getVirtualItems().map((vr) => {
              const row = rows[vr.index]!
              const selected = selectedId === row.id
              return (
                <div
                  key={row.id}
                  onClick={() => onRowClick(row)}
                  className={cn(
                    'flex items-center cursor-pointer border-b border-[var(--color-border)]',
                    'hover:bg-[var(--color-bg-hover)] transition-colors duration-[var(--duration-fast)]',
                    selected && 'bg-[var(--color-bg-selected)]',
                  )}
                  style={{
                    position: 'absolute',
                    top: vr.start,
                    left: 0,
                    right: 0,
                    height: vr.size,
                  }}
                >
                  <div
                    className={cn(
                      'font-mono text-[var(--color-text-subtle)] tabular-nums flex-none',
                      dims.cellPx,
                      dims.textPx,
                    )}
                    style={{ width: 88 }}
                  >
                    {row.id.slice(0, 8)}
                  </div>
                  {colDefs.map((c) => (
                    <div
                      key={c.name}
                      className={cn(
                        'flex items-center min-w-0 flex-none',
                        dims.cellPx,
                        dims.textPx,
                      )}
                      style={{ width: c.width }}
                    >
                      <RecordCell field={c.field} value={row.fields[c.name]} />
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

interface ColDef {
  name: string
  width: number
  field: FieldSchema
}

function columnDefs(columns: string[], fields: FieldSchema[]): ColDef[] {
  return columns
    .map((name) => {
      const field = fields.find((f) => f.name === name)
      if (!field) return null
      return { name, field, width: widthForField(field) }
    })
    .filter((c): c is ColDef => !!c)
}

function widthForField(f: FieldSchema): number {
  if (f.type === 'vector' || f.type === 'sparse_vector' || f.type === 'multi_vector')
    return 120
  if (f.type === 'string') return 260
  if (f.type === 'blob') return 80
  if (f.type === 'int32' || f.type === 'int64' || f.type === 'float32' || f.type === 'float64')
    return 88
  if (f.type === 'bool') return 64
  return 160
}

function HeaderCell({
  width,
  children,
  className,
  type,
}: {
  width: number
  children: React.ReactNode
  className?: string
  type?: string
}) {
  return (
    <div
      className={cn(
        'flex items-center gap-1 min-w-0 flex-none border-r border-[var(--color-border)] last:border-r-0',
        'text-[10.5px] font-medium uppercase tracking-[0.06em] text-[var(--color-text-subtle)]',
        className,
      )}
      style={{ width }}
    >
      <span className="truncate">{children}</span>
      {type && (
        <span className="text-[var(--color-text-disabled)] font-mono normal-case tracking-normal">
          {abbrevType(type)}
        </span>
      )}
    </div>
  )
}

function abbrevType(t: string): string {
  switch (t) {
    case 'vector':
      return 'vec'
    case 'sparse_vector':
      return 'spv'
    case 'multi_vector':
      return 'mvec'
    case 'int32':
      return 'i32'
    case 'int64':
      return 'i64'
    case 'float32':
      return 'f32'
    case 'float64':
      return 'f64'
    case 'string':
      return 'str'
    case 'bool':
      return 'b'
    case 'blob':
      return 'blob'
    case 'timestamp':
      return 'ts'
    case 'json':
      return 'json'
    case 'uuid':
      return 'uuid'
    default:
      return t
  }
}

function SkeletonGrid({ density, colCount }: { density: Density; colCount: number }) {
  const dims = DENSITY_DIMS[density]
  return (
    <div>
      {Array.from({ length: 14 }).map((_, i) => (
        <div
          key={i}
          className={cn('flex items-center border-b border-[var(--color-border)]', dims.cellPx)}
          style={{ height: dims.rowH }}
        >
          {Array.from({ length: colCount }).map((_, j) => (
            <Skeleton
              key={j}
              className="h-3 mr-3"
              style={{ width: j === 0 ? 64 : j === colCount - 1 ? 80 : 120 }}
            />
          ))}
        </div>
      ))}
    </div>
  )
}
