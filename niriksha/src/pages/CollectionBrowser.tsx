import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import * as Tabs from '@radix-ui/react-tabs'
import { Database, Plus, Search, Trash2 } from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import {
  createCollection,
  dropCollection,
  fetchCollection,
} from '../api/collections'
import type {
  CollectionInfo,
  CollectionSummary,
  CreateFieldDef,
  FieldSchema,
} from '../api/collections'
import {
  Badge,
  Button,
  Input,
  Modal,
  Select,
  Skeleton,
  toast,
} from '../components/primitives'
import { FieldStats } from './collections/FieldStats'
import { RecordsBrowser } from './collections/RecordsBrowser'
import { isScalarField, isVectorField } from './collections/types'
import { cn } from '../lib/cn'

export function CollectionBrowser() {
  const { name } = useParams<{ name: string }>()
  return name ? <CollectionDetail name={name} /> : <CollectionList />
}

/* ────────────────────────────────────────────────────────────────────────
   List mode
   ──────────────────────────────────────────────────────────────────── */

function CollectionList() {
  const { collections, loading, error, fetchCollections } = useCollectionStore()
  const navigate = useNavigate()
  const [showCreate, setShowCreate] = useState(false)
  const [query, setQuery] = useState('')

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  const filtered = collections.filter((c) =>
    c.name.toLowerCase().includes(query.toLowerCase()),
  )

  return (
    <div className="px-6 py-5">
      <header className="flex items-center gap-3 mb-5">
        <div className="flex-1">
          <h1 className="text-[20px] font-semibold tracking-tight text-[var(--color-text)] mb-0.5">
            Collections
          </h1>
          <p className="text-[12.5px] text-[var(--color-text-muted)]">
            {collections.length} collection
            {collections.length !== 1 ? 's' : ''} on this node
          </p>
        </div>
        <Input
          leading={<Search size={11} strokeWidth={2.25} />}
          placeholder="Filter…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-56"
        />
        <Button
          variant="primary"
          leading={<Plus size={12} strokeWidth={2.25} />}
          onClick={() => setShowCreate(true)}
        >
          New collection
        </Button>
      </header>

      {error && (
        <div className="mb-4 rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
          {error}
        </div>
      )}

      {loading && collections.length === 0 ? (
        <SkeletonCards />
      ) : filtered.length === 0 ? (
        <EmptyState onCreate={() => setShowCreate(true)} hasQuery={!!query} />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map((col) => (
            <CollectionCard
              key={col.name}
              collection={col}
              onClick={() => navigate(`/collections/${col.name}`)}
            />
          ))}
        </div>
      )}

      <CreateCollectionModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={(name) => {
          setShowCreate(false)
          fetchCollections()
          navigate(`/collections/${name}`)
        }}
      />
    </div>
  )
}

function CollectionCard({
  collection,
  onClick,
}: {
  collection: CollectionSummary
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'group relative flex flex-col gap-2 text-left p-4',
        'rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]',
        'hover:border-[var(--color-border-strong)] hover:bg-[var(--color-bg-hover)]',
        'transition-[border-color,background] duration-[var(--duration-base)]',
      )}
    >
      <div className="flex items-center gap-2">
        <Database size={13} strokeWidth={2} className="text-[var(--color-accent-9)]" />
        <span className="font-mono text-[13px] font-medium text-[var(--color-text)] truncate flex-1">
          {collection.name}
        </span>
        <span className="opacity-0 group-hover:opacity-100 text-[10px] text-[var(--color-text-subtle)]">
          open →
        </span>
      </div>
      <div className="flex items-center gap-3 text-[11px] font-mono text-[var(--color-text-muted)] tabular-nums">
        <span>
          {collection.field_count}
          <span className="text-[var(--color-text-subtle)]"> fields</span>
        </span>
        <span className="text-[var(--color-border-strong)]">·</span>
        <span>
          {collection.shard_count}
          <span className="text-[var(--color-text-subtle)]"> shard{collection.shard_count !== 1 ? 's' : ''}</span>
        </span>
      </div>
    </button>
  )
}

function SkeletonCards() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="p-4 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] flex flex-col gap-3"
        >
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-24" />
        </div>
      ))}
    </div>
  )
}

function EmptyState({
  onCreate,
  hasQuery,
}: {
  onCreate: () => void
  hasQuery: boolean
}) {
  return (
    <div className="relative h-[420px] rounded-[var(--radius-md)] border border-dashed border-[var(--color-border-strong)] flex items-center justify-center">
      <div className="dot-grid absolute inset-0 opacity-30 rounded-[var(--radius-md)]" aria-hidden />
      <div className="relative text-center max-w-sm">
        <Database size={28} strokeWidth={1.5} className="text-[var(--color-text-subtle)] mx-auto mb-3" />
        <h2 className="text-[14px] font-medium text-[var(--color-text)] mb-1">
          {hasQuery ? 'No collections match your filter' : 'No collections yet'}
        </h2>
        <p className="text-[12px] text-[var(--color-text-muted)] mb-4">
          {hasQuery
            ? 'Try a different query or create a new collection with this name.'
            : 'Create one to start ingesting embeddings, then use Atlas, Quality Lab, and the Search Playground.'}
        </p>
        <Button variant="primary" onClick={onCreate} leading={<Plus size={12} strokeWidth={2.25} />}>
          Create collection
        </Button>
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Detail mode — object-with-views
   ──────────────────────────────────────────────────────────────────── */

type DetailTab = 'records' | 'schema' | 'stats' | 'settings'

function CollectionDetail({ name }: { name: string }) {
  const navigate = useNavigate()
  const { fetchCollections } = useCollectionStore()
  const [collection, setCollection] = useState<CollectionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<DetailTab>('records')
  const [confirmDrop, setConfirmDrop] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchCollection(name)
      .then(({ data }) => setCollection(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : 'Failed to load collection'),
      )
      .finally(() => setLoading(false))
  }, [name])

  if (loading) {
    return (
      <div className="px-6 py-5">
        <Skeleton className="h-5 w-48 mb-4" />
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    )
  }
  if (error || !collection) {
    return (
      <div className="px-6 py-5">
        <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
          {error ?? 'Collection not found'}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <DetailHeader
        collection={collection}
        tab={tab}
        onTabChange={setTab}
        onSearch={() => navigate(`/search/${name}`)}
        onDrop={() => setConfirmDrop(true)}
      />
      <div className="flex-1 min-h-0">
        {tab === 'records' && <RecordsBrowser collection={collection} />}
        {tab === 'schema' && <SchemaView fields={collection.fields} />}
        {tab === 'stats' && <FieldStats collection={collection} />}
        {tab === 'settings' && <SettingsView collection={collection} />}
      </div>
      <Modal
        open={confirmDrop}
        onClose={() => setConfirmDrop(false)}
        title="Drop collection?"
        description={
          <>
            <span className="font-mono text-[var(--color-text)]">{collection.name}</span>{' '}
            and every record + embedding in it will be permanently removed.
          </>
        }
      >
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setConfirmDrop(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={async () => {
              try {
                await dropCollection(collection.name)
                fetchCollections()
                navigate('/collections')
                toast.success(`Dropped ${collection.name}`)
              } catch (e) {
                toast.error(
                  e instanceof Error ? e.message : 'Failed to drop collection',
                )
              }
            }}
            leading={<Trash2 size={11} strokeWidth={2.5} />}
          >
            Drop collection
          </Button>
        </div>
      </Modal>
    </div>
  )
}

function DetailHeader({
  collection,
  tab,
  onTabChange,
  onSearch,
  onDrop,
}: {
  collection: CollectionInfo
  tab: DetailTab
  onTabChange: (t: DetailTab) => void
  onSearch: () => void
  onDrop: () => void
}) {
  const vCount = collection.fields.filter(isVectorField).length
  return (
    <header className="border-b border-[var(--color-border)] bg-[var(--color-bg)] px-4 pt-3 flex-none">
      <div className="flex items-start gap-3 mb-2">
        <Database size={18} strokeWidth={1.75} className="text-[var(--color-accent-9)] mt-0.5" />
        <div className="flex-1 min-w-0">
          <h1 className="font-mono text-[15px] font-semibold text-[var(--color-text)] truncate">
            {collection.name}
          </h1>
          <div className="flex items-center gap-2.5 mt-0.5 text-[11.5px] font-mono text-[var(--color-text-muted)] tabular-nums">
            <span>{collection.fields.length} fields</span>
            <span className="text-[var(--color-border-strong)]">·</span>
            <span>
              {vCount} vector
            </span>
            <span className="text-[var(--color-border-strong)]">·</span>
            <span>{collection.shard_count} shards</span>
            <span className="text-[var(--color-border-strong)]">·</span>
            <span>rf={collection.replication_factor}</span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            variant="secondary"
            size="sm"
            leading={<Search size={11} strokeWidth={2.25} />}
            onClick={onSearch}
          >
            Search
          </Button>
          <Button
            variant="ghost"
            size="sm"
            leading={<Trash2 size={11} strokeWidth={2.5} />}
            onClick={onDrop}
            className="text-[var(--color-text-subtle)] hover:text-[var(--color-error)]"
          >
            Drop
          </Button>
        </div>
      </div>
      <Tabs.Root value={tab} onValueChange={(v) => onTabChange(v as DetailTab)}>
        <Tabs.List className="flex items-center gap-0.5 -mb-[1px]">
          <TabBtn value="records" current={tab}>Records</TabBtn>
          <TabBtn value="schema" current={tab}>Schema</TabBtn>
          <TabBtn value="stats" current={tab}>Statistics</TabBtn>
          <TabBtn value="settings" current={tab}>Settings</TabBtn>
        </Tabs.List>
      </Tabs.Root>
    </header>
  )
}

function TabBtn({
  value,
  current,
  children,
}: {
  value: DetailTab
  current: DetailTab
  children: React.ReactNode
}) {
  const active = value === current
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        'relative px-3 h-8 text-[12.5px] rounded-t-[var(--radius-xs)]',
        'transition-colors duration-[var(--duration-fast)]',
        active
          ? 'text-[var(--color-text)] font-medium'
          : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
      )}
    >
      {children}
      {active && (
        <span
          aria-hidden
          className="absolute left-1.5 right-1.5 -bottom-px h-[2px] bg-[var(--color-accent)] rounded-t-full"
        />
      )}
    </Tabs.Trigger>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Schema view (in-place rewrite of the prior page)
   ──────────────────────────────────────────────────────────────────── */

function SchemaView({ fields }: { fields: FieldSchema[] }) {
  const vec = fields.filter(isVectorField)
  const scalar = fields.filter(isScalarField)
  const blob = fields.filter((f) => f.type === 'blob')

  return (
    <div className="px-6 py-5 space-y-6 overflow-auto h-full">
      {vec.length > 0 && (
        <Section title="Vector fields" count={vec.length}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
            {vec.map((f) => (
              <VectorFieldCard key={f.name} field={f} />
            ))}
          </div>
        </Section>
      )}
      {scalar.length > 0 && (
        <Section title="Scalar fields" count={scalar.length}>
          <FieldTable fields={scalar} kind="scalar" />
        </Section>
      )}
      {blob.length > 0 && (
        <Section title="Blob fields" count={blob.length}>
          <FieldTable fields={blob} kind="blob" />
        </Section>
      )}
    </div>
  )
}

function Section({
  title,
  count,
  children,
}: {
  title: string
  count: number
  children: React.ReactNode
}) {
  return (
    <section>
      <h2 className="flex items-baseline gap-2 mb-2.5">
        <span className="text-[11.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] font-medium">
          {title}
        </span>
        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
          {count}
        </span>
      </h2>
      {children}
    </section>
  )
}

function VectorFieldCard({ field }: { field: FieldSchema }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3.5">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono text-[13px] font-medium text-[var(--color-text)] truncate flex-1">
          {field.name}
        </span>
        <Badge label={field.type} variant="accent" />
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11.5px]">
        <Field label="Dimension" value={field.dimension ? `${field.dimension}d` : '—'} mono />
        <Field label="Metric" value={field.metric ?? '—'} />
        <Field label="Model" value={field.model ?? '—'} mono small />
        <Field label="Filterable" value={field.filterable ? 'yes' : '—'} />
      </dl>
    </div>
  )
}

function Field({
  label,
  value,
  mono,
  small,
}: {
  label: string
  value: string
  mono?: boolean
  small?: boolean
}) {
  return (
    <div className="flex flex-col gap-0.5 min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
        {label}
      </span>
      <span
        className={cn(
          'text-[var(--color-text)] truncate',
          mono && 'font-mono',
          small && 'text-[10.5px]',
        )}
        title={value}
      >
        {value}
      </span>
    </div>
  )
}

function FieldTable({
  fields,
  kind,
}: {
  fields: FieldSchema[]
  kind: 'scalar' | 'blob'
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
      <table className="w-full text-[12px]">
        <thead className="bg-[var(--color-bg-elevated)] text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Field</th>
            <th className="px-3 py-2 text-left font-medium">Type</th>
            {kind === 'scalar' && (
              <th className="px-3 py-2 text-left font-medium">Filterable</th>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--color-border)]">
          {fields.map((f) => (
            <tr
              key={f.name}
              className="hover:bg-[var(--color-bg-hover)] transition-colors"
            >
              <td className="px-3 py-2 font-mono text-[var(--color-text)]">
                {f.name}
              </td>
              <td className="px-3 py-2">
                <Badge label={f.type} variant={kind === 'blob' ? 'warning' : 'default'} />
              </td>
              {kind === 'scalar' && (
                <td className="px-3 py-2 text-[var(--color-text-muted)]">
                  {f.filterable ? 'yes' : '—'}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SettingsView({ collection }: { collection: CollectionInfo }) {
  return (
    <div className="px-6 py-5 max-w-2xl space-y-4">
      <Section title="Schema" count={collection.fields.length}>
        <pre className="font-mono text-[11.5px] leading-relaxed text-[var(--color-text-muted)] bg-[var(--color-bg-elevated)] rounded-[var(--radius-md)] border border-[var(--color-border)] p-3 overflow-auto max-h-72">
          {JSON.stringify(collection, null, 2)}
        </pre>
      </Section>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Create modal
   ──────────────────────────────────────────────────────────────────── */

const FIELD_TYPE_OPTIONS = [
  'string', 'int32', 'int64', 'float32', 'float64', 'bool',
  'timestamp', 'json', 'uuid', 'vector', 'sparse_vector',
  'multi_vector', 'blob',
] as const

const METRIC_OPTIONS = ['cosine', 'euclidean', 'dot_product', 'manhattan']

function CreateCollectionModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean
  onClose: () => void
  onCreated: (name: string) => void
}) {
  const [name, setName] = useState('')
  const [fields, setFields] = useState<CreateFieldDef[]>([
    { name: '', type: 'string' },
  ])
  const [shards, setShards] = useState(1)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reset = () => {
    setName('')
    setFields([{ name: '', type: 'string' }])
    setShards(1)
    setError(null)
  }

  const submit = async () => {
    if (!name.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      const clean = fields
        .filter((f) => f.name.trim())
        .map((f) => {
          const out: CreateFieldDef = { name: f.name.trim(), type: f.type }
          if (isVectorishType(f.type)) {
            out.dimension = f.dimension ?? 768
            if (f.metric) out.metric = f.metric
            if (f.model) out.model = f.model
          }
          if (f.filterable) out.filterable = true
          return out
        })
      await createCollection({ name: name.trim(), fields: clean, shard_count: shards })
      const created = name.trim()
      reset()
      onCreated(created)
      toast.success(`Created ${created}`)
    } catch (e) {
      const msg =
        (e as { response?: { data?: { error?: string } } })?.response?.data?.error ??
        (e instanceof Error ? e.message : 'Failed to create collection')
      setError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={() => {
        reset()
        onClose()
      }}
      title="New collection"
      description="Define the schema. Vector fields take a dimension + metric; scalars can be marked filterable."
      width={520}
    >
      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          <div className="col-span-2">
            <label className="block text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
              Name
            </label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_collection"
              mono
            />
          </div>
          <div>
            <label className="block text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
              Shards
            </label>
            <Input
              type="number"
              min={1}
              max={64}
              value={shards}
              onChange={(e) => setShards(Math.max(1, parseInt(e.target.value) || 1))}
              mono
            />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)]">
              Fields
            </label>
            <Button
              size="xs"
              variant="ghost"
              leading={<Plus size={11} strokeWidth={2.25} />}
              onClick={() => setFields([...fields, { name: '', type: 'string' }])}
            >
              Add field
            </Button>
          </div>
          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {fields.map((f, i) => (
              <FieldRowEditor
                key={i}
                field={f}
                onChange={(patch) =>
                  setFields(fields.map((x, j) => (j === i ? { ...x, ...patch } : x)))
                }
                onRemove={fields.length > 1 ? () => setFields(fields.filter((_, j) => j !== i)) : undefined}
              />
            ))}
          </div>
        </div>

        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] whitespace-pre-wrap break-words font-mono">
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={submit}
            disabled={submitting || !name.trim()}
          >
            {submitting ? 'Creating…' : 'Create'}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function FieldRowEditor({
  field,
  onChange,
  onRemove,
}: {
  field: CreateFieldDef
  onChange: (patch: Partial<CreateFieldDef>) => void
  onRemove?: () => void
}) {
  const vec = isVectorishType(field.type)
  return (
    <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <Input
          value={field.name}
          onChange={(e) => onChange({ name: e.target.value })}
          placeholder="field_name"
          mono
          className="flex-1"
        />
        <Select
          value={field.type}
          onValueChange={(v) => {
            const patch: Partial<CreateFieldDef> = { type: v }
            if (isVectorishType(v)) {
              patch.dimension = field.dimension ?? 768
              patch.metric = field.metric ?? 'cosine'
            } else {
              patch.dimension = undefined
              patch.metric = undefined
              patch.model = undefined
            }
            onChange(patch)
          }}
          options={FIELD_TYPE_OPTIONS.map((t) => ({ value: t, label: t }))}
        />
        {onRemove && (
          <Button variant="ghost" size="xs" onClick={onRemove} title="Remove field">
            ✕
          </Button>
        )}
      </div>
      {vec && (
        <div className="grid grid-cols-3 gap-2 pl-1">
          <FieldEditor
            label="Dimension"
            value={String(field.dimension ?? 768)}
            type="number"
            onChange={(v) => onChange({ dimension: parseInt(v, 10) || 768 })}
          />
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-0.5">
              Metric
            </label>
            <Select
              size="sm"
              value={field.metric ?? 'cosine'}
              onValueChange={(v) => onChange({ metric: v })}
              options={METRIC_OPTIONS.map((m) => ({ value: m, label: m }))}
            />
          </div>
          <FieldEditor
            label="Model"
            value={field.model ?? ''}
            placeholder="BAAI/bge-m3"
            mono
            onChange={(v) => onChange({ model: v || undefined })}
          />
        </div>
      )}
      {!vec && field.type !== 'blob' && (
        <label className="flex items-center gap-2 pl-1 text-[11.5px] text-[var(--color-text-muted)]">
          <input
            type="checkbox"
            checked={!!field.filterable}
            onChange={(e) => onChange({ filterable: e.target.checked })}
          />
          Filterable
        </label>
      )}
    </div>
  )
}

function FieldEditor({
  label,
  value,
  placeholder,
  type,
  mono,
  onChange,
}: {
  label: string
  value: string
  placeholder?: string
  type?: string
  mono?: boolean
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-0.5">
        {label}
      </label>
      <Input
        value={value}
        type={type}
        mono={mono}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  )
}

function isVectorishType(t: string): boolean {
  return t === 'vector' || t === 'sparse_vector' || t === 'multi_vector'
}
