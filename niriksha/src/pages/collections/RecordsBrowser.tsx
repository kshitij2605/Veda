import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  ChevronLeft,
  ChevronRight,
  RotateCw,
  Save,
  Sparkle,
} from 'lucide-react'
import { fetchRecords } from '../../api/collections'
import type { CollectionInfo, RecordRow } from '../../api/collections'
import { Button } from '../../components/primitives/Button'
import { Select } from '../../components/primitives/Select'
import { useUrlState } from '../../hooks/useUrlState'
import { useSavedObjects } from '../../stores/savedObjectStore'
import { toast } from '../../components/primitives/Toast'
import { FieldSidebar } from './FieldSidebar'
import { FilterBar, evalFilter, filtersToDsl, type Filter } from './FilterBar'
import { RecordDrawer } from './RecordDrawer'
import { RecordsTable } from './RecordsTable'
import { isScalarField, isVectorField, type Density } from './types'

interface SavedSearchBody {
  collection: string
  filters: Filter[]
  columns: string[]
  density: Density
}

interface RecordsBrowserProps {
  collection: CollectionInfo
}

const PAGE_LIMIT = 200

export function RecordsBrowser({ collection }: RecordsBrowserProps) {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const savedId = searchParams.get('saved')

  const [page, setPage] = useUrlState<number>('p', 0)
  const [density, setDensity] = useUrlState<Density>('d', 'compact')
  const [filters, setFilters] = useUrlState<Filter[]>('f', [])
  const [columns, setColumns] = useUrlState<string[]>(
    'cols',
    defaultColumns(collection),
  )
  const [expandedId, setExpandedId] = useUrlState<string | null>('rec', null)

  const [rows, setRows] = useState<RecordRow[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)

  const upsertSaved = useSavedObjects((s) => s.upsert)
  const getSaved = useSavedObjects((s) => s.get)
  const touchSaved = useSavedObjects((s) => s.touch)

  // Restore from saved object on mount when ?saved=<id> is present.
  useEffect(() => {
    if (!savedId) return
    const obj = getSaved<SavedSearchBody>(savedId)
    if (!obj || obj.type !== 'search') return
    if (obj.body.collection !== collection.name) {
      navigate(`/collections/${obj.body.collection}?saved=${savedId}`, {
        replace: true,
      })
      return
    }
    setFilters(obj.body.filters)
    setColumns(obj.body.columns)
    setDensity(obj.body.density)
    touchSaved(savedId)
    // Strip the ?saved param now that state is hydrated.
    const next = new URLSearchParams(searchParams)
    next.delete('saved')
    setSearchParams(next, { replace: true })
    toast.success(`Loaded view "${obj.name}"`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedId])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchRecords(collection.name, {
      limit: PAGE_LIMIT,
      offset: page * PAGE_LIMIT,
    })
      .then(({ data }) => {
        if (cancelled) return
        setRows(data.records)
        setTotal(data.total)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to load records')
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [collection.name, page, reloadKey])

  // Client-side filter pass. Eventually the gateway gains server filter
  // and this collapses to a passthrough.
  const filteredRows = useMemo(
    () => rows.filter((r) => evalFilter(filters, r)),
    [rows, filters],
  )

  const expandedRow = useMemo(
    () => filteredRows.find((r) => r.id === expandedId) ?? null,
    [filteredRows, expandedId],
  )

  // Auto-add field as column when user clicks the eye in the sidebar
  const toggleColumn = (name: string) =>
    setColumns(
      columns.includes(name)
        ? columns.filter((c) => c !== name)
        : [...columns, name],
    )

  const addFilter = (field: string, op: '=' | '!=', value: unknown) => {
    setFilters([
      ...filters,
      { field, op, value: value as Filter['value'] },
    ])
  }

  const handleSave = () => {
    const name = window.prompt('Name this view')
    if (!name) return
    upsertSaved({
      id: '',
      type: 'search',
      name,
      body: {
        collection: collection.name,
        filters,
        columns,
        density,
      },
    })
    toast.success(`Saved view "${name}"`)
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--color-bg)]">
      <FieldSidebar
        fields={collection.fields}
        records={rows}
        selectedColumns={columns}
        onColumnToggle={toggleColumn}
        onAddFilter={addFilter}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <FilterBar
          fields={collection.fields}
          filters={filters}
          onChange={setFilters}
          records={rows}
        />
        <Toolbar
          total={total}
          shown={filteredRows.length}
          page={page}
          pageSize={PAGE_LIMIT}
          onPrev={() => setPage(Math.max(0, page - 1))}
          onNext={() => setPage(page + 1)}
          density={density}
          onDensity={setDensity}
          onReload={() => setReloadKey((k) => k + 1)}
          onSave={handleSave}
          dsl={filtersToDsl(filters)}
        />
        {error ? (
          <div className="m-3 rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
            {error}
          </div>
        ) : (
          <RecordsTable
            rows={filteredRows}
            fields={collection.fields}
            columns={columns}
            density={density}
            loading={loading}
            onRowClick={(r) => setExpandedId(r.id)}
            selectedId={expandedId ?? undefined}
          />
        )}
      </div>
      <RecordDrawer
        open={!!expandedRow}
        onClose={() => setExpandedId(null)}
        record={expandedRow}
        fields={collection.fields}
        onAddFilter={addFilter}
      />
    </div>
  )
}

function Toolbar({
  total,
  shown,
  page,
  pageSize,
  onPrev,
  onNext,
  density,
  onDensity,
  onReload,
  onSave,
  dsl,
}: {
  total: number
  shown: number
  page: number
  pageSize: number
  onPrev: () => void
  onNext: () => void
  density: Density
  onDensity: (d: Density) => void
  onReload: () => void
  onSave: () => void
  dsl: string
}) {
  const from = page * pageSize + 1
  const to = Math.min((page + 1) * pageSize, total)
  return (
    <div className="flex items-center gap-2 px-3 h-9 border-b border-[var(--color-border)] bg-[var(--color-bg)] text-[11.5px] text-[var(--color-text-muted)]">
      <div className="flex items-center gap-1 font-mono tabular-nums">
        <Button
          size="xs"
          variant="ghost"
          onClick={onPrev}
          disabled={page === 0}
          leading={<ChevronLeft size={12} strokeWidth={2.25} />}
        >
          Prev
        </Button>
        <Button
          size="xs"
          variant="ghost"
          onClick={onNext}
          disabled={to >= total}
          trailing={<ChevronRight size={12} strokeWidth={2.25} />}
        >
          Next
        </Button>
        <span className="ml-2">
          {from.toLocaleString()}–{to.toLocaleString()} of{' '}
          {total.toLocaleString()}
        </span>
        {dsl && shown !== to - from + 1 && (
          <span className="ml-2 text-[var(--color-text-subtle)]">
            ({shown.toLocaleString()} after filter)
          </span>
        )}
      </div>
      <div className="ml-auto flex items-center gap-1">
        <Select
          size="sm"
          value={density}
          onValueChange={(v) => onDensity(v as Density)}
          options={[
            { value: 'comfortable', label: 'Comfortable' },
            { value: 'compact', label: 'Compact' },
            { value: 'dense', label: 'Dense' },
          ]}
        />
        <Button
          size="xs"
          variant="ghost"
          leading={<RotateCw size={11} strokeWidth={2.25} />}
          onClick={onReload}
          title="Reload"
        >
          Reload
        </Button>
        <Button
          size="xs"
          variant="ghost"
          leading={<Save size={11} strokeWidth={2.25} />}
          onClick={onSave}
          title="Save current view"
        >
          Save view
        </Button>
        <Button
          size="xs"
          variant="primary"
          leading={<Sparkle size={11} strokeWidth={2.25} />}
          onClick={() => toast.message('Search wiring lands in Phase 2')}
        >
          Search
        </Button>
      </div>
    </div>
  )
}

function defaultColumns(collection: CollectionInfo): string[] {
  // Show up to 4 columns: prefer string fields first (likely chunk text),
  // then scalars, vectors last (as sparkline).
  const strings = collection.fields
    .filter((f) => f.type === 'string')
    .map((f) => f.name)
  const scalars = collection.fields
    .filter((f) => isScalarField(f) && f.type !== 'string')
    .map((f) => f.name)
  const vectors = collection.fields.filter(isVectorField).map((f) => f.name)
  return [...strings, ...scalars, ...vectors].slice(0, 4)
}
