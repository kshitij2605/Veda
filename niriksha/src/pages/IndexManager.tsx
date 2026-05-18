import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { fetchCollection } from '../api/collections'
import type { CollectionInfo, FieldSchema } from '../api/collections'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'
import { Badge } from '../components/Badge'

const VECTOR_TYPES = new Set(['vector', 'sparse_vector', 'multi_vector'])

export function IndexManager() {
  const { collections, fetchCollections } = useCollectionStore()
  const [selectedCollection, setSelectedCollection] = useState('')
  const [schema, setSchema] = useState<CollectionInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (!selectedCollection) {
      setSchema(null)
      return
    }
    setLoading(true)
    setError(null)
    fetchCollection(selectedCollection)
      .then(({ data }) => setSchema(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : 'Failed to load collection'),
      )
      .finally(() => setLoading(false))
  }, [selectedCollection])

  const vectorFields = schema?.fields.filter((f) => VECTOR_TYPES.has(f.type)) ?? []
  const scalarFields =
    schema?.fields.filter(
      (f) => !VECTOR_TYPES.has(f.type) && f.type !== 'blob',
    ) ?? []
  const blobFields = schema?.fields.filter((f) => f.type === 'blob') ?? []

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Index Manager"
        actions={
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
          >
            <option value="">Select collection...</option>
            {collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        }
      />

      {!selectedCollection && (
        <EmptyState
          title="Select a collection"
          description="Choose a collection to view its index configuration."
        />
      )}

      {loading && <LoadingSpinner label="Loading schema..." />}

      {error && (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {schema && !loading && !error && (
        <div className="space-y-6">
          <div className="rounded-lg border border-border bg-bg-secondary p-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <Stat label="Shards" value={String(schema.shard_count)} />
            <Stat
              label="Replication"
              value={String(schema.replication_factor)}
            />
            <Stat label="Total Fields" value={String(schema.fields.length)} />
            <Stat
              label="Vector Fields"
              value={String(vectorFields.length)}
            />
          </div>

          {vectorFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">
                Vector Fields
              </h2>
              <div className="space-y-3">
                {vectorFields.map((f) => (
                  <VectorFieldCard key={f.name} field={f} />
                ))}
              </div>
            </div>
          )}

          {scalarFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">
                Scalar Fields
              </h2>
              <div className="rounded-lg border border-border overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-bg-secondary text-text-muted text-left">
                      <th className="px-4 py-3 font-medium">Field</th>
                      <th className="px-4 py-3 font-medium">Type</th>
                      <th className="px-4 py-3 font-medium">Filterable</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {scalarFields.map((f) => (
                      <tr
                        key={f.name}
                        className="hover:bg-bg-tertiary transition-colors"
                      >
                        <td className="px-4 py-3 font-medium text-text-primary">
                          {f.name}
                        </td>
                        <td className="px-4 py-3">
                          <Badge label={f.type} variant="default" />
                        </td>
                        <td className="px-4 py-3 text-text-secondary">
                          {f.filterable ? 'yes' : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {blobFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">
                Blob Fields
              </h2>
              <div className="rounded-lg border border-border overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-bg-secondary text-text-muted text-left">
                      <th className="px-4 py-3 font-medium">Field</th>
                      <th className="px-4 py-3 font-medium">Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {blobFields.map((f) => (
                      <tr
                        key={f.name}
                        className="hover:bg-bg-tertiary transition-colors"
                      >
                        <td className="px-4 py-3 font-medium text-text-primary">
                          {f.name}
                        </td>
                        <td className="px-4 py-3">
                          <Badge label="blob" variant="warning" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {vectorFields.length === 0 &&
            scalarFields.length === 0 &&
            blobFields.length === 0 && (
              <EmptyState
                title="No fields"
                description="This collection has no fields defined."
              />
            )}
        </div>
      )}
    </motion.div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-text-muted uppercase tracking-wider">
        {label}
      </div>
      <div className="text-text-primary font-semibold mt-0.5">{value}</div>
    </div>
  )
}

function VectorFieldCard({ field }: { field: FieldSchema }) {
  return (
    <div className="rounded-lg border border-border bg-bg-secondary p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-text-primary">{field.name}</h3>
        <Badge label={field.type} variant="accent" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
        <Cell
          label="Dimension"
          value={field.dimension ? String(field.dimension) : '—'}
        />
        <Cell label="Metric" value={field.metric ?? '—'} />
        <Cell label="Model" value={field.model ?? '—'} mono={!!field.model} />
        <Cell label="Filterable" value={field.filterable ? 'yes' : '—'} />
      </div>
    </div>
  )
}

function Cell({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <span className="text-xs text-text-muted">{label}</span>
      <div
        className={`text-text-primary ${
          mono ? 'font-mono text-xs break-all' : ''
        }`}
      >
        {value}
      </div>
    </div>
  )
}
