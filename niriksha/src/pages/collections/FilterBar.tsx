import { useEffect, useMemo, useRef, useState } from 'react'
import { Code2, Filter, X } from 'lucide-react'
import type { FieldSchema, RecordRow } from '../../api/collections'
import { Button } from '../../components/primitives/Button'
import { FilterChip } from '../../components/primitives/FilterChip'
import { cn } from '../../lib/cn'

/**
 * Filter — one term. Operator subset is what Veda's filter DSL accepts.
 */
export interface Filter {
  field: string
  op: '=' | '!=' | '>' | '<' | '>=' | '<=' | '~' | 'IN'
  value: string | number | boolean | (string | number)[]
}

const OPS: Filter['op'][] = ['=', '!=', '>', '<', '>=', '<=', '~', 'IN']

interface FilterBarProps {
  fields: FieldSchema[]
  filters: Filter[]
  onChange: (filters: Filter[]) => void
  /** Optional record sample so we can suggest values from the loaded page. */
  records?: RecordRow[]
}

/**
 * FilterBar — Discover-style. Top: a search input with 3-stage
 * autocomplete (field → operator → value). Pills row below shows
 * applied filters (Stripe pattern). "Edit as query" reveals the raw DSL
 * for power users.
 */
export function FilterBar({ fields, filters, onChange, records }: FilterBarProps) {
  const [mode, setMode] = useState<'builder' | 'raw'>('builder')
  const [draft, setDraft] = useState('')
  const [rawText, setRawText] = useState(() => filtersToDsl(filters))
  const [open, setOpen] = useState(false)
  const [stage, setStage] = useState<'field' | 'op' | 'value'>('field')
  const [pending, setPending] = useState<{ field?: string; op?: Filter['op'] }>(
    {},
  )
  const inputRef = useRef<HTMLInputElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setRawText(filtersToDsl(filters))
  }, [filters])

  // Field options for the current draft prefix
  const suggestions = useMemo<string[]>(() => {
    const q = draft.toLowerCase().trim()
    if (stage === 'field') {
      const allowed = fields.filter((f) => f.filterable || f.type === 'string')
      return allowed
        .map((f) => f.name)
        .filter((n) => n.toLowerCase().includes(q))
        .slice(0, 8)
    }
    if (stage === 'op') {
      return OPS.filter((op) => op.includes(q))
    }
    // value stage — sample top-N values for the chosen field
    if (!pending.field || !records) return []
    const seen = new Set<string>()
    const out: string[] = []
    for (const r of records) {
      const v = r.fields[pending.field]
      if (v == null) continue
      const s = String(v)
      if (s.toLowerCase().includes(q) && !seen.has(s)) {
        seen.add(s)
        out.push(s)
        if (out.length >= 8) break
      }
    }
    return out
  }, [draft, stage, fields, pending, records])

  // Close popover on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    window.addEventListener('mousedown', handler)
    return () => window.removeEventListener('mousedown', handler)
  }, [open])

  const commit = (rawValue?: string) => {
    if (!pending.field || !pending.op) return
    const v = rawValue ?? draft
    if (v === '') return
    onChange([
      ...filters,
      {
        field: pending.field,
        op: pending.op,
        value: coerce(v, fieldOf(fields, pending.field)),
      },
    ])
    setPending({})
    setStage('field')
    setDraft('')
    setOpen(false)
    inputRef.current?.focus()
  }

  const pickSuggestion = (sug: string) => {
    if (stage === 'field') {
      setPending({ field: sug })
      setStage('op')
      setDraft('')
    } else if (stage === 'op') {
      setPending((p) => ({ ...p, op: sug as Filter['op'] }))
      setStage('value')
      setDraft('')
    } else {
      commit(sug)
    }
  }

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      setOpen(false)
      return
    }
    if (e.key === 'Backspace' && draft === '') {
      if (stage === 'value') {
        setStage('op')
        setPending((p) => ({ ...p, op: undefined }))
      } else if (stage === 'op') {
        setStage('field')
        setPending({})
      } else if (filters.length > 0) {
        onChange(filters.slice(0, -1))
      }
    }
    if (e.key === 'Enter') {
      if (suggestions.length > 0) {
        pickSuggestion(suggestions[0]!)
      } else if (stage === 'value' && draft) {
        commit()
      }
      e.preventDefault()
    }
  }

  if (mode === 'raw') {
    return (
      <div className="flex items-center gap-2 px-3 h-10 border-b border-[var(--color-border)] bg-[var(--color-bg)]">
        <Code2 size={12} strokeWidth={2.25} className="text-[var(--color-text-subtle)]" />
        <input
          value={rawText}
          onChange={(e) => setRawText(e.target.value)}
          onBlur={() => {
            const parsed = dslToFilters(rawText, fields)
            if (parsed) onChange(parsed)
          }}
          placeholder="Veda filter DSL — e.g. parent_doc_id = 'doc-123' AND page_number > 5"
          className="flex-1 bg-transparent border-none outline-none text-[12.5px] font-mono text-[var(--color-text)] placeholder:text-[var(--color-text-subtle)]"
        />
        <Button
          size="xs"
          variant="ghost"
          onClick={() => setMode('builder')}
        >
          Builder
        </Button>
      </div>
    )
  }

  return (
    <div className="border-b border-[var(--color-border)] bg-[var(--color-bg)]">
      <div className="flex items-center gap-2 px-3 h-10">
        <Filter size={12} strokeWidth={2.25} className="text-[var(--color-text-subtle)]" />
        <div ref={wrapperRef} className="relative flex-1">
          <div className="flex items-center gap-1.5">
            {pending.field && (
              <Pending label={pending.field} />
            )}
            {pending.op && (
              <Pending label={pending.op} variant="op" />
            )}
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value)
                setOpen(true)
              }}
              onFocus={() => setOpen(true)}
              onKeyDown={onKey}
              placeholder={
                stage === 'field'
                  ? filters.length === 0
                    ? 'Filter — type a field name'
                    : 'Add filter…'
                  : stage === 'op'
                  ? 'operator (=, !=, >, ~, IN, …)'
                  : `value for ${pending.field}`
              }
              className="flex-1 min-w-[180px] bg-transparent border-none outline-none text-[12.5px] font-mono text-[var(--color-text)] placeholder:text-[var(--color-text-subtle)]"
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          {open && suggestions.length > 0 && (
            <ul
              role="listbox"
              className={cn(
                'absolute left-0 top-full mt-1 z-30 w-[280px]',
                'rounded-[var(--radius-md)] border border-[var(--color-border-strong)]',
                'bg-[var(--color-bg-overlay)] shadow-[var(--shadow-md)] overflow-hidden py-1',
              )}
            >
              {suggestions.map((s, i) => (
                <li
                  key={i}
                  role="option"
                  aria-selected={i === 0}
                  className={cn(
                    'flex items-center gap-2 px-2 py-1 mx-0.5 rounded-[var(--radius-xs)] cursor-pointer',
                    'text-[12px] font-mono text-[var(--color-text)]',
                    'hover:bg-[var(--color-bg-hover)]',
                    i === 0 && 'bg-[var(--color-bg-hover)]',
                  )}
                  onMouseDown={(e) => {
                    e.preventDefault()
                    pickSuggestion(s)
                  }}
                >
                  <span className="flex-1 truncate">{s}</span>
                  {stage === 'field' && (
                    <span className="text-[10px] text-[var(--color-text-subtle)]">
                      {fieldOf(fields, s)?.type ?? ''}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
        <Button
          size="xs"
          variant="ghost"
          leading={<Code2 size={11} strokeWidth={2.25} />}
          onClick={() => setMode('raw')}
          title="Edit raw DSL"
        >
          DSL
        </Button>
        {filters.length > 0 && (
          <Button
            size="xs"
            variant="ghost"
            leading={<X size={11} strokeWidth={2.25} />}
            onClick={() => onChange([])}
            title="Clear all filters"
          >
            Clear
          </Button>
        )}
      </div>

      {filters.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 px-3 pb-2">
          {filters.map((f, i) => (
            <FilterChip
              key={i}
              onRemove={() => onChange(filters.filter((_, j) => j !== i))}
              onClick={() => {
                // edit: drop the chip, repopulate the input
                setPending({ field: f.field, op: f.op })
                setStage('value')
                setDraft(formatValue(f.value))
                onChange(filters.filter((_, j) => j !== i))
                inputRef.current?.focus()
              }}
            >
              <span className="text-[var(--color-text-muted)]">{f.field}</span>
              <span className="mx-1 text-[var(--color-text-subtle)]">{f.op}</span>
              <span className="text-[var(--color-text)]">{formatValue(f.value)}</span>
            </FilterChip>
          ))}
        </div>
      )}
    </div>
  )
}

function Pending({
  label,
  variant,
}: {
  label: string
  variant?: 'field' | 'op'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-1.5 h-5 rounded-[var(--radius-xs)] text-[11.5px] font-mono whitespace-nowrap',
        variant === 'op'
          ? 'bg-[var(--color-base-3)] text-[var(--color-text-muted)]'
          : 'bg-[var(--color-accent-2)] text-[var(--color-accent-9)]',
      )}
    >
      {label}
    </span>
  )
}

function fieldOf(fields: FieldSchema[], name: string): FieldSchema | undefined {
  return fields.find((f) => f.name === name)
}

function coerce(v: string, f?: FieldSchema): Filter['value'] {
  if (!f) return v
  if (f.type === 'int32' || f.type === 'int64') {
    const n = parseInt(v, 10)
    return Number.isFinite(n) ? n : v
  }
  if (f.type === 'float32' || f.type === 'float64') {
    const n = parseFloat(v)
    return Number.isFinite(n) ? n : v
  }
  if (f.type === 'bool') {
    if (v === 'true' || v === '1') return true
    if (v === 'false' || v === '0') return false
  }
  return v
}

function formatValue(v: Filter['value']): string {
  if (Array.isArray(v)) return '[' + v.map(formatValue).join(', ') + ']'
  if (typeof v === 'string') return `"${v}"`
  return String(v)
}

/** Render the filter array as Veda DSL, AND-joined. */
export function filtersToDsl(filters: Filter[]): string {
  return filters
    .map((f) => `${f.field} ${f.op} ${formatValue(f.value)}`)
    .join(' AND ')
}

/** Naive parser — handles ` field op value AND field op value`. */
function dslToFilters(text: string, fields: FieldSchema[]): Filter[] | null {
  if (!text.trim()) return []
  try {
    return text.split(/\s+AND\s+/i).map((clause) => {
      const m = clause.match(
        /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|!=|>=|<=|>|<|~|IN)\s*(.+)\s*$/i,
      )
      if (!m) throw new Error('bad clause: ' + clause)
      const [, field, op, vRaw] = m as unknown as [string, string, string, string]
      let value: Filter['value'] = vRaw.trim()
      if (
        (value as string).startsWith('"') &&
        (value as string).endsWith('"')
      ) {
        value = (value as string).slice(1, -1)
      } else if ((value as string).startsWith('[')) {
        // ignore IN arrays for now — keep as raw string
      } else {
        value = coerce(value as string, fieldOf(fields, field))
      }
      return { field, op: op as Filter['op'], value }
    })
  } catch {
    return null
  }
}

/** Client-side evaluator — used until /v1/.../records gains server filter. */
export function evalFilter(filters: Filter[], rec: RecordRow): boolean {
  if (filters.length === 0) return true
  for (const f of filters) {
    const v = rec.fields[f.field]
    const target = f.value
    if (!matches(v, f.op, target)) return false
  }
  return true
}

function matches(value: unknown, op: Filter['op'], target: unknown): boolean {
  if (value == null) return op === '!=' && target != null
  switch (op) {
    case '=':
      return String(value) === String(target)
    case '!=':
      return String(value) !== String(target)
    case '>':
      return Number(value) > Number(target)
    case '<':
      return Number(value) < Number(target)
    case '>=':
      return Number(value) >= Number(target)
    case '<=':
      return Number(value) <= Number(target)
    case '~':
      return String(value).toLowerCase().includes(String(target).toLowerCase())
    case 'IN':
      return Array.isArray(target) && target.some((t) => String(value) === String(t))
  }
}
