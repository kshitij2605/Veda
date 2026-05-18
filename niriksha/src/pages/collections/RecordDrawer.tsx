import * as Tabs from '@radix-ui/react-tabs'
import { useMemo, useState } from 'react'
import { Minus, Plus, Maximize2 } from 'lucide-react'
import { Drawer } from '../../components/primitives/Drawer'
import { CopyButton } from '../../components/primitives/CopyButton'
import { Sparkline } from '../../components/primitives/Sparkline'
import { Badge } from '../../components/primitives/Badge'
import type { FieldSchema } from '../../api/collections'
import { cn } from '../../lib/cn'
import { isVectorField } from './types'

export interface RecordDrawerProps {
  open: boolean
  onClose: () => void
  record: { id: string; fields: Record<string, unknown> } | null
  fields: FieldSchema[]
  onAddFilter: (field: string, op: '=' | '!=', value: unknown) => void
}

export function RecordDrawer({
  open,
  onClose,
  record,
  fields,
  onAddFilter,
}: RecordDrawerProps) {
  const json = useMemo(() => (record ? JSON.stringify(record, null, 2) : ''), [
    record,
  ])

  if (!record) {
    return <Drawer open={open} onClose={onClose} title="Record" width={560} children={null} />
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      width={560}
      title={
        <span className="inline-flex items-center gap-2">
          <Badge label="record" variant="accent" />
          <span className="font-mono text-[12px] truncate">{record.id}</span>
        </span>
      }
      actions={<CopyButton value={record.id} label="id" />}
      bleed
    >
      <Tabs.Root defaultValue="table" className="flex flex-col h-full">
        <Tabs.List className="flex items-center gap-1 px-2 h-9 border-b border-[var(--color-border)] flex-none">
          <TabTrigger value="table">Table</TabTrigger>
          <TabTrigger value="json">JSON</TabTrigger>
          <div className="ml-auto flex items-center gap-1">
            <CopyButton value={json} label="json" />
          </div>
        </Tabs.List>
        <Tabs.Content value="table" className="flex-1 overflow-auto">
          <TableView record={record} fields={fields} onAddFilter={onAddFilter} />
        </Tabs.Content>
        <Tabs.Content value="json" className="flex-1 overflow-auto p-3">
          <pre className="font-mono text-[11.5px] leading-relaxed text-[var(--color-text)] whitespace-pre-wrap break-words">
            {json}
          </pre>
        </Tabs.Content>
      </Tabs.Root>
    </Drawer>
  )
}

function TabTrigger({
  value,
  children,
}: {
  value: string
  children: React.ReactNode
}) {
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        'inline-flex items-center px-2.5 h-7 rounded-[var(--radius-xs)]',
        'text-[12px] text-[var(--color-text-muted)]',
        'data-[state=active]:bg-[var(--color-bg-hover)] data-[state=active]:text-[var(--color-text)]',
        'hover:text-[var(--color-text)]',
      )}
    >
      {children}
    </Tabs.Trigger>
  )
}

function TableView({
  record,
  fields,
  onAddFilter,
}: {
  record: { id: string; fields: Record<string, unknown> }
  fields: FieldSchema[]
  onAddFilter: (field: string, op: '=' | '!=', value: unknown) => void
}) {
  return (
    <table className="w-full text-[12px]">
      <tbody>
        <tr>
          <td className="px-3 py-1.5 align-top text-[var(--color-text-subtle)] font-mono w-32 border-b border-[var(--color-border)]">
            _id
          </td>
          <td className="px-2 py-1.5 align-top font-mono break-all border-b border-[var(--color-border)]">
            {record.id}
            <CopyButton value={record.id} className="ml-1 align-middle" />
          </td>
        </tr>
        {fields.map((f) => (
          <FieldRow
            key={f.name}
            field={f}
            value={record.fields[f.name]}
            onAddFilter={onAddFilter}
          />
        ))}
      </tbody>
    </table>
  )
}

function FieldRow({
  field,
  value,
  onAddFilter,
}: {
  field: FieldSchema
  value: unknown
  onAddFilter: (field: string, op: '=' | '!=', value: unknown) => void
}) {
  return (
    <tr className="group">
      <td className="px-3 py-1.5 align-top text-[var(--color-text-subtle)] font-mono w-32 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-1">
          <span className="truncate">{field.name}</span>
          <span className="text-[9.5px] uppercase tracking-wider text-[var(--color-text-disabled)]">
            {field.type}
          </span>
        </div>
        {field.filterable && value != null && (
          <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 mt-0.5">
            <button
              onClick={() => onAddFilter(field.name, '=', value)}
              title="Filter for this value"
              className="p-0.5 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-success)] hover:bg-[var(--color-bg-hover)]"
            >
              <Plus size={10} strokeWidth={2.5} />
            </button>
            <button
              onClick={() => onAddFilter(field.name, '!=', value)}
              title="Filter out this value"
              className="p-0.5 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-error)] hover:bg-[var(--color-bg-hover)]"
            >
              <Minus size={10} strokeWidth={2.5} />
            </button>
          </div>
        )}
      </td>
      <td className="px-2 py-1.5 align-top border-b border-[var(--color-border)] min-w-0">
        <ValueCell field={field} value={value} />
      </td>
    </tr>
  )
}

function ValueCell({ field, value }: { field: FieldSchema; value: unknown }) {
  const [expanded, setExpanded] = useState(false)

  if (value == null) {
    return <span className="text-[var(--color-text-subtle)] italic">null</span>
  }

  if (isVectorField(field) && Array.isArray(value)) {
    const v = value as number[]
    return (
      <div className="flex flex-col gap-1.5">
        <div className="inline-flex items-center gap-2">
          <Sparkline values={v.slice(0, 128)} width={140} height={18} />
          <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)]">
            {v.length}d
          </span>
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="text-[var(--color-text-subtle)] hover:text-[var(--color-text)]"
            title="Show raw values"
          >
            <Maximize2 size={11} strokeWidth={2.25} />
          </button>
          <CopyButton value={JSON.stringify(v)} label="vector" />
        </div>
        {expanded && (
          <pre className="font-mono text-[10.5px] text-[var(--color-text-muted)] bg-[var(--color-base-1)] rounded-[var(--radius-xs)] p-2 max-h-32 overflow-auto">
            [{v.slice(0, 8).map((n) => n.toFixed(5)).join(', ')}
            {v.length > 8 ? ', …+' + (v.length - 8) + ' more' : ''}]
          </pre>
        )}
      </div>
    )
  }

  if (typeof value === 'string') {
    const long = value.length > 240
    return (
      <div className={cn('flex flex-col gap-1.5')}>
        <div
          className={cn(
            'font-sans text-[12.5px] text-[var(--color-text)] whitespace-pre-wrap break-words',
            !expanded && long && 'max-h-48 overflow-hidden mask-fade-bottom',
          )}
          style={
            !expanded && long
              ? {
                  WebkitMaskImage:
                    'linear-gradient(to bottom, black 70%, transparent 100%)',
                  maskImage:
                    'linear-gradient(to bottom, black 70%, transparent 100%)',
                }
              : undefined
          }
        >
          {value}
        </div>
        {long && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="self-start text-[10.5px] text-[var(--color-accent)] hover:underline"
          >
            {expanded ? 'Collapse' : `Show full (${value.length.toLocaleString()} chars)`}
          </button>
        )}
        <CopyButton value={value} />
      </div>
    )
  }

  if (typeof value === 'object') {
    return (
      <pre className="font-mono text-[11px] text-[var(--color-text-muted)] bg-[var(--color-base-1)] rounded-[var(--radius-xs)] p-2 overflow-auto max-h-48">
        {JSON.stringify(value, null, 2)}
      </pre>
    )
  }

  return (
    <span className="font-mono tabular-nums text-[var(--color-text)]">
      {String(value)}
    </span>
  )
}
