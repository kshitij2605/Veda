import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { fetchCollection, fetchRecords, createCollection } from '../api/collections'
import type { CollectionInfo, FieldSchema } from '../api/collections'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'
import { Badge } from '../components/Badge'
import { Modal } from '../components/Modal'

export function CollectionBrowser() {
  const { name } = useParams<{ name: string }>()

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      {name ? <CollectionDetail name={name} /> : <CollectionList />}
    </motion.div>
  )
}

/* ── List Mode ──────────────────────────────────────────────── */

function CollectionList() {
  const { collections, loading, error, fetchCollections } = useCollectionStore()
  const navigate = useNavigate()
  const [showCreate, setShowCreate] = useState(false)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  return (
    <>
      <PageHeader
        title="Collections"
        subtitle="Browse and manage your vector collections."
        actions={
          <button
            onClick={() => setShowCreate(true)}
            className="px-4 py-2 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg transition-colors"
          >
            + Create
          </button>
        }
      />

      {loading && <LoadingSpinner label="Loading collections..." />}

      {error && (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {!loading && !error && collections.length === 0 && (
        <EmptyState
          title="No collections yet"
          description="Create your first collection to get started."
        />
      )}

      {!loading && !error && collections.length > 0 && (
        <div className="grid gap-3">
          {collections.map((col) => (
            <CollectionCard key={col.name} collection={col} onClick={() => navigate(`/collections/${col.name}`)} />
          ))}
        </div>
      )}

      <CreateCollectionModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={() => {
          setShowCreate(false)
          fetchCollections()
        }}
      />
    </>
  )
}

function CollectionCard({ collection, onClick }: { collection: CollectionInfo; onClick: () => void }) {
  const vectorFields = collection.fields.filter((f) => f.field_type === 'vector').length
  const blobFields = collection.fields.filter((f) => f.field_type === 'blob').length

  return (
    <button
      onClick={onClick}
      className="w-full text-left rounded-lg border border-border bg-bg-secondary hover:bg-bg-tertiary p-4 transition-colors"
    >
      <div className="font-medium text-text-primary">{collection.name}</div>
      <div className="flex items-center gap-3 mt-2 text-xs text-text-muted">
        <span>{collection.fields.length} field{collection.fields.length !== 1 ? 's' : ''}</span>
        <span className="text-border">|</span>
        <span>{vectorFields} vector</span>
        {blobFields > 0 && (
          <>
            <span className="text-border">|</span>
            <span>{blobFields} blob</span>
          </>
        )}
      </div>
      {collection.created_at && (
        <div className="text-xs text-text-muted mt-1">Created: {collection.created_at}</div>
      )}
    </button>
  )
}

/* ── Detail Mode ────────────────────────────────────────────── */

type DetailTab = 'schema' | 'data'

function CollectionDetail({ name }: { name: string }) {
  const navigate = useNavigate()
  const [tab, setTab] = useState<DetailTab>('schema')
  const [collection, setCollection] = useState<CollectionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchCollection(name)
      .then(({ data }) => setCollection(data))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load collection'))
      .finally(() => setLoading(false))
  }, [name])

  if (loading) return <LoadingSpinner label={`Loading ${name}...`} />
  if (error) {
    return (
      <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">{error}</div>
    )
  }
  if (!collection) return null

  return (
    <>
      <PageHeader
        title={collection.name}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigate('/collections')}
              className="px-3 py-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
            >
              &larr; Back
            </button>
            <TabButton active={tab === 'schema'} onClick={() => setTab('schema')}>Schema</TabButton>
            <TabButton active={tab === 'data'} onClick={() => setTab('data')}>Data</TabButton>
          </div>
        }
      />

      <AnimatePresence mode="wait">
        {tab === 'schema' ? (
          <motion.div key="schema" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <SchemaView fields={collection.fields} />
          </motion.div>
        ) : (
          <motion.div key="data" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <RecordsTable collectionName={name} fields={collection.fields} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-2 text-sm rounded-lg transition-colors ${
        active ? 'bg-accent/10 text-accent' : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
      }`}
    >
      {children}
    </button>
  )
}

/* ── Schema View ────────────────────────────────────────────── */

function SchemaView({ fields }: { fields: FieldSchema[] }) {
  const typeVariant = (ft: string) => {
    if (ft === 'vector') return 'accent' as const
    if (ft === 'blob') return 'warning' as const
    return 'default' as const
  }

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-bg-secondary text-text-muted text-left">
            <th className="px-4 py-3 font-medium">Field Name</th>
            <th className="px-4 py-3 font-medium">Type</th>
            <th className="px-4 py-3 font-medium">Dimensions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {fields.map((f) => (
            <tr key={f.name}>
              <td className="px-4 py-3 font-medium text-text-primary">{f.name}</td>
              <td className="px-4 py-3"><Badge label={f.field_type} variant={typeVariant(f.field_type)} /></td>
              <td className="px-4 py-3 text-text-secondary">{f.dimensions ?? '--'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ── Records Table ──────────────────────────────────────────── */

function RecordsTable({ collectionName, fields }: { collectionName: string; fields: FieldSchema[] }) {
  const [records, setRecords] = useState<Record<string, unknown>[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const limit = 20

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchRecords(collectionName, { limit, offset: page * limit })
      .then(({ data }) => {
        setRecords(data.records)
        setTotal(data.total)
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load records'))
      .finally(() => setLoading(false))
  }, [collectionName, page])

  const totalPages = Math.ceil(total / limit)
  const fieldNames = fields.map((f) => f.name)
  const vectorFieldNames = new Set(fields.filter((f) => f.field_type === 'vector').map((f) => f.name))

  if (loading) return <LoadingSpinner label="Loading records..." />
  if (error) {
    return (
      <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">{error}</div>
    )
  }
  if (records.length === 0) {
    return <EmptyState title="No records" description="This collection is empty." />
  }

  return (
    <div>
      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg-secondary text-text-muted text-left">
              {fieldNames.map((fn) => (
                <th key={fn} className="px-4 py-3 font-medium whitespace-nowrap">{fn}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {records.map((rec, i) => (
              <tr
                key={i}
                onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                className="hover:bg-bg-tertiary cursor-pointer transition-colors"
              >
                {fieldNames.map((fn) => (
                  <td key={fn} className="px-4 py-3 text-text-secondary whitespace-nowrap max-w-[200px] truncate">
                    {formatCellValue(rec[fn], vectorFieldNames.has(fn))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Expanded row detail */}
      <AnimatePresence>
        {expandedRow !== null && records[expandedRow] && (
          <motion.div
            className="mt-3 rounded-lg border border-border bg-bg-secondary p-4"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">Record Detail</h4>
            <pre className="text-xs text-text-secondary overflow-x-auto">
              {JSON.stringify(records[expandedRow], null, 2)}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1.5 text-sm rounded-lg border border-border text-text-secondary hover:bg-bg-tertiary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            &lsaquo;
          </button>
          {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
            const pageNum = totalPages <= 5 ? i : Math.min(Math.max(page - 2, 0), totalPages - 5) + i
            return (
              <button
                key={pageNum}
                onClick={() => setPage(pageNum)}
                className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                  page === pageNum
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-border text-text-secondary hover:bg-bg-tertiary'
                }`}
              >
                {pageNum + 1}
              </button>
            )
          })}
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1.5 text-sm rounded-lg border border-border text-text-secondary hover:bg-bg-tertiary disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            &rsaquo;
          </button>
        </div>
      )}
    </div>
  )
}

/* ── Create Collection Modal ────────────────────────────────── */

function CreateCollectionModal({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState('')
  const [fields, setFields] = useState<FieldSchema[]>([{ name: '', field_type: 'string' }])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const addField = () => setFields([...fields, { name: '', field_type: 'string' }])

  const updateField = (i: number, patch: Partial<FieldSchema>) => {
    setFields(fields.map((f, idx) => (idx === i ? { ...f, ...patch } : f)))
  }

  const removeField = (i: number) => {
    if (fields.length > 1) setFields(fields.filter((_, idx) => idx !== i))
  }

  const handleSubmit = async () => {
    if (!name.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      await createCollection({ name: name.trim(), fields })
      setName('')
      setFields([{ name: '', field_type: 'string' }])
      onCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create collection')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Create Collection">
      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-text-muted mb-1">Collection Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my_collection"
            className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-text-muted">Fields</label>
            <button onClick={addField} className="text-xs text-accent hover:text-accent-hover transition-colors">
              + Add field
            </button>
          </div>
          <div className="space-y-2">
            {fields.map((f, i) => (
              <div key={i} className="flex items-center gap-2">
                <input
                  value={f.name}
                  onChange={(e) => updateField(i, { name: e.target.value })}
                  placeholder="field_name"
                  className="flex-1 px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
                />
                <select
                  value={f.field_type}
                  onChange={(e) => updateField(i, { field_type: e.target.value, dimensions: e.target.value === 'vector' ? 768 : undefined })}
                  className="px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
                >
                  <option value="string">string</option>
                  <option value="integer">integer</option>
                  <option value="float">float</option>
                  <option value="boolean">boolean</option>
                  <option value="vector">vector</option>
                  <option value="blob">blob</option>
                </select>
                {f.field_type === 'vector' && (
                  <input
                    type="number"
                    value={f.dimensions ?? 768}
                    onChange={(e) => updateField(i, { dimensions: parseInt(e.target.value) || 768 })}
                    className="w-20 px-2 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
                    placeholder="dims"
                  />
                )}
                <button
                  onClick={() => removeField(i)}
                  className="text-text-muted hover:text-error transition-colors p-1"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>

        {error && (
          <div className="rounded-lg bg-error/10 border border-error/20 px-3 py-2 text-sm text-error">{error}</div>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || !name.trim()}
            className="px-4 py-2 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? 'Creating...' : 'Create'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

/* ── Utilities ──────────────────────────────────────────────── */

function formatCellValue(value: unknown, isVector: boolean): string {
  if (value == null) return '--'
  if (isVector && Array.isArray(value)) return `[${value.length} dims]`
  if (typeof value === 'string') return value.length > 50 ? value.slice(0, 50) + '...' : value
  if (typeof value === 'object') return JSON.stringify(value).slice(0, 50) + '...'
  return String(value)
}
