import { useMemo, useState } from 'react'
import * as Popover from '@radix-ui/react-popover'
import {
  ChevronDown,
  ChevronRight,
  Database,
  Eye,
  EyeOff,
  Hash,
  ListFilter,
  Minus,
  Plus,
  Search,
  Type,
} from 'lucide-react'
import type { FieldSchema, RecordRow } from '../../api/collections'
import { cn } from '../../lib/cn'
import { isVectorField, isBlobField } from './types'

interface FieldSidebarProps {
  fields: FieldSchema[]
  records: RecordRow[]
  selectedColumns: string[]
  onColumnToggle: (name: string) => void
  onAddFilter: (field: string, op: '=' | '!=', value: unknown) => void
}

interface FieldStats {
  present: number
  missing: number
  topValues: { value: unknown; count: number }[]
}

const COLLAPSED_KEY = 'niriksha.fieldSidebar.collapsed.v1'

/**
 * FieldSidebar — Discover-style left column showing Selected / Available
 * / Empty fields. Hover a field to see its top-10 sampled values with
 * one-click filter add (+/-) and drag-to-column.
 */
export function FieldSidebar({
  fields,
  records,
  selectedColumns,
  onColumnToggle,
  onAddFilter,
}: FieldSidebarProps) {
  const stats = useMemo(() => computeFieldStats(fields, records), [fields, records])
  const [query, setQuery] = useState('')
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>(
    () => readCollapsed(),
  )

  const selected = useMemo(() => {
    return fields
      .filter((f) => selectedColumns.includes(f.name))
      .filter((f) => f.name.toLowerCase().includes(query.toLowerCase()))
  }, [fields, selectedColumns, query])

  const available = useMemo(() => {
    return fields
      .filter((f) => !selectedColumns.includes(f.name))
      .filter((f) => (stats.get(f.name)?.present ?? 0) > 0)
      .filter((f) => f.name.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [fields, selectedColumns, stats, query])

  const empty = useMemo(() => {
    return fields
      .filter((f) => (stats.get(f.name)?.present ?? 0) === 0)
      .filter((f) => f.name.toLowerCase().includes(query.toLowerCase()))
  }, [fields, stats, query])

  const toggleGroup = (g: string) => {
    setCollapsedGroups((prev) => {
      const next = { ...prev, [g]: !prev[g] }
      writeCollapsed(next)
      return next
    })
  }

  return (
    <aside className="flex flex-col w-[240px] flex-none border-r border-[var(--color-border)] bg-[var(--color-bg)]">
      <div className="px-2.5 py-2 border-b border-[var(--color-border)]">
        <div className="relative">
          <Search
            size={11}
            strokeWidth={2.25}
            className="absolute left-2 top-1/2 -translate-y-1/2 text-[var(--color-text-subtle)]"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search ${fields.length} fields`}
            className="w-full pl-6 pr-2 h-6 rounded-[var(--radius-xs)] bg-[var(--color-base-1)] border border-[var(--color-border)] text-[11.5px] text-[var(--color-text)] placeholder:text-[var(--color-text-subtle)] focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto no-scrollbar">
        <Group
          label="Selected"
          count={selected.length}
          collapsed={collapsedGroups.selected}
          onToggle={() => toggleGroup('selected')}
        >
          {selected.length === 0 && (
            <div className="px-3 py-1.5 text-[11px] text-[var(--color-text-subtle)] italic">
              No columns yet
            </div>
          )}
          {selected.map((f) => (
            <Row
              key={f.name}
              field={f}
              stats={stats.get(f.name)}
              selected
              onColumnToggle={onColumnToggle}
              onAddFilter={onAddFilter}
            />
          ))}
        </Group>

        <Group
          label="Available"
          count={available.length}
          collapsed={collapsedGroups.available}
          onToggle={() => toggleGroup('available')}
        >
          {available.map((f) => (
            <Row
              key={f.name}
              field={f}
              stats={stats.get(f.name)}
              onColumnToggle={onColumnToggle}
              onAddFilter={onAddFilter}
            />
          ))}
        </Group>

        {empty.length > 0 && (
          <Group
            label="Empty"
            count={empty.length}
            collapsed={collapsedGroups.empty ?? true}
            onToggle={() => toggleGroup('empty')}
          >
            {empty.map((f) => (
              <Row
                key={f.name}
                field={f}
                stats={stats.get(f.name)}
                onColumnToggle={onColumnToggle}
                onAddFilter={onAddFilter}
              />
            ))}
          </Group>
        )}
      </div>
    </aside>
  )
}

interface RowProps {
  field: FieldSchema
  stats?: FieldStats
  selected?: boolean
  onColumnToggle: (name: string) => void
  onAddFilter: (field: string, op: '=' | '!=', value: unknown) => void
}

function Row({ field, stats, selected, onColumnToggle, onAddFilter }: RowProps) {
  const [open, setOpen] = useState(false)
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={cn(
            'group w-full flex items-center gap-1.5 pl-3 pr-2 h-6 text-left',
            'hover:bg-[var(--color-bg-hover)] transition-colors duration-[var(--duration-fast)]',
            'cursor-default',
            selected && 'text-[var(--color-text)]',
            !selected && 'text-[var(--color-text-muted)]',
          )}
        >
          <FieldIcon field={field} />
          <span className="flex-1 truncate text-[11.5px]">{field.name}</span>
          <span className="text-[10px] font-mono text-[var(--color-text-subtle)] opacity-0 group-hover:opacity-100">
            {stats?.present ?? 0}
          </span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onColumnToggle(field.name)
            }}
            className="p-0.5 rounded-[2px] text-[var(--color-text-subtle)] opacity-0 group-hover:opacity-100 hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)]"
            aria-label={selected ? 'Hide column' : 'Show as column'}
          >
            {selected ? (
              <EyeOff size={11} strokeWidth={2.25} />
            ) : (
              <Eye size={11} strokeWidth={2.25} />
            )}
          </button>
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="right"
          align="start"
          sideOffset={4}
          className={cn(
            'z-30 w-[260px] rounded-[var(--radius-md)] border border-[var(--color-border-strong)]',
            'bg-[var(--color-bg-overlay)] shadow-[var(--shadow-md)] overflow-hidden',
            'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-left-1',
          )}
          style={{ animationDuration: '120ms' }}
        >
          <FieldPopover
            field={field}
            stats={stats}
            onAddFilter={(op, v) => {
              onAddFilter(field.name, op, v)
              setOpen(false)
            }}
            onColumnToggle={() => {
              onColumnToggle(field.name)
              setOpen(false)
            }}
            isSelected={!!selected}
          />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}

function FieldPopover({
  field,
  stats,
  onAddFilter,
  onColumnToggle,
  isSelected,
}: {
  field: FieldSchema
  stats?: FieldStats
  onAddFilter: (op: '=' | '!=', v: unknown) => void
  onColumnToggle: () => void
  isSelected: boolean
}) {
  const showTopValues =
    stats && stats.topValues.length > 0 && !isVectorField(field) && !isBlobField(field)

  return (
    <div className="text-[var(--color-text)]">
      <header className="px-3 py-2 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-2">
          <FieldIcon field={field} />
          <span className="font-mono text-[12px] truncate">{field.name}</span>
          <span className="ml-auto text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
            {field.type}
            {field.dimension ? `·${field.dimension}d` : ''}
          </span>
        </div>
        <div className="mt-1 flex items-center gap-2 text-[10.5px] text-[var(--color-text-subtle)] font-mono tabular-nums">
          <span>{stats?.present ?? 0} present</span>
          {(stats?.missing ?? 0) > 0 && <span>· {stats!.missing} missing</span>}
          {field.metric && <span>· {field.metric}</span>}
          {field.filterable && (
            <span className="text-[var(--color-success)]">· filterable</span>
          )}
        </div>
      </header>

      {showTopValues && (
        <ul className="max-h-[200px] overflow-y-auto py-1">
          {stats!.topValues.slice(0, 10).map(({ value, count }, i) => {
            const display = displayValue(value)
            return (
              <li
                key={i}
                className="group flex items-center gap-1 px-2 h-6 text-[11.5px] hover:bg-[var(--color-bg-hover)]"
              >
                <span className="flex-1 truncate font-mono">{display}</span>
                <span className="text-[10px] text-[var(--color-text-subtle)] font-mono tabular-nums">
                  {count}
                </span>
                <span className="flex opacity-0 group-hover:opacity-100">
                  <button
                    type="button"
                    aria-label="Filter for this value"
                    onClick={() => onAddFilter('=', value)}
                    className="p-0.5 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-success)] hover:bg-[var(--color-bg-hover)]"
                  >
                    <Plus size={11} strokeWidth={2.5} />
                  </button>
                  <button
                    type="button"
                    aria-label="Filter out this value"
                    onClick={() => onAddFilter('!=', value)}
                    className="p-0.5 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-error)] hover:bg-[var(--color-bg-hover)]"
                  >
                    <Minus size={11} strokeWidth={2.5} />
                  </button>
                </span>
              </li>
            )
          })}
        </ul>
      )}

      {!showTopValues && (
        <div className="px-3 py-3 text-[11px] text-[var(--color-text-subtle)] italic">
          {isVectorField(field)
            ? 'Vector field — top-values not applicable. Click on a row to inspect.'
            : isBlobField(field)
            ? 'Blob field — preview in the row drawer.'
            : 'No sampled values in the current page.'}
        </div>
      )}

      <footer className="px-2 py-1.5 border-t border-[var(--color-border)] flex items-center gap-1">
        <button
          type="button"
          onClick={onColumnToggle}
          className="flex-1 inline-flex items-center justify-center gap-1.5 h-6 rounded-[var(--radius-xs)] text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)]"
        >
          {isSelected ? (
            <>
              <EyeOff size={11} strokeWidth={2.25} /> Hide column
            </>
          ) : (
            <>
              <Eye size={11} strokeWidth={2.25} /> Show as column
            </>
          )}
        </button>
      </footer>
    </div>
  )
}

function FieldIcon({ field }: { field: FieldSchema }) {
  const cls = 'flex-none text-[var(--color-text-subtle)]'
  switch (field.type) {
    case 'vector':
    case 'sparse_vector':
    case 'multi_vector':
      return <Database size={11} strokeWidth={2.25} className={cls} />
    case 'string':
      return <Type size={11} strokeWidth={2.25} className={cls} />
    case 'int32':
    case 'int64':
    case 'float32':
    case 'float64':
      return <Hash size={11} strokeWidth={2.25} className={cls} />
    case 'blob':
      return <ListFilter size={11} strokeWidth={2.25} className={cls} />
    default:
      return <Type size={11} strokeWidth={2.25} className={cls} />
  }
}

function Group({
  label,
  count,
  collapsed,
  onToggle,
  children,
}: {
  label: string
  count: number
  collapsed?: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <section>
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-1 px-2 pt-2 pb-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] hover:text-[var(--color-text-muted)]"
      >
        {collapsed ? (
          <ChevronRight size={10} strokeWidth={2.5} />
        ) : (
          <ChevronDown size={10} strokeWidth={2.5} />
        )}
        <span className="font-medium">{label}</span>
        <span className="ml-auto font-mono tabular-nums">{count}</span>
      </button>
      {!collapsed && <div className="pb-1">{children}</div>}
    </section>
  )
}

function readCollapsed(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(COLLAPSED_KEY)
    return raw ? JSON.parse(raw) : { empty: true }
  } catch {
    return { empty: true }
  }
}
function writeCollapsed(s: Record<string, boolean>) {
  try {
    localStorage.setItem(COLLAPSED_KEY, JSON.stringify(s))
  } catch {
    /* ignore */
  }
}

function computeFieldStats(
  fields: FieldSchema[],
  records: RecordRow[],
): Map<string, FieldStats> {
  const out = new Map<string, FieldStats>()
  for (const f of fields) {
    let present = 0
    const counts = new Map<string, { value: unknown; count: number }>()
    for (const rec of records) {
      const v = rec.fields[f.name]
      if (v == null) continue
      if (typeof v === 'string' && v === '') continue
      present += 1
      if (
        isVectorField(f) ||
        isBlobField(f) ||
        typeof v === 'object'
      ) {
        continue
      }
      const key = String(v)
      const e = counts.get(key)
      if (e) e.count += 1
      else counts.set(key, { value: v, count: 1 })
    }
    const topValues = Array.from(counts.values()).sort(
      (a, b) => b.count - a.count,
    )
    out.set(f.name, { present, missing: records.length - present, topValues })
  }
  return out
}

function displayValue(v: unknown): string {
  if (v == null) return '(null)'
  if (typeof v === 'string') return v.length > 32 ? v.slice(0, 32) + '…' : v
  return String(v)
}
