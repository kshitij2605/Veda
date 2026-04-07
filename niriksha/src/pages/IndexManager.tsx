import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { fetchCollection } from '../api/collections'
import type { CollectionInfo, FieldSchema } from '../api/collections'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'
import { Badge } from '../components/Badge'

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
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load collection'))
      .finally(() => setLoading(false))
  }, [selectedCollection])

  const vectorFields = schema?.fields.filter((f) => f.field_type === 'vector') ?? []
  const scalarFields = schema?.fields.filter((f) => f.field_type !== 'vector' && f.field_type !== 'blob') ?? []
  const blobFields = schema?.fields.filter((f) => f.field_type === 'blob') ?? []

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
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
        }
      />

      {!selectedCollection && (
        <EmptyState title="Select a collection" description="Choose a collection to view its index configuration." />
      )}

      {loading && <LoadingSpinner label="Loading schema..." />}

      {error && (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {schema && !loading && !error && (
        <div className="space-y-6">
          {/* Vector Fields */}
          {vectorFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">Vector Fields</h2>
              <div className="space-y-3">
                {vectorFields.map((f) => (
                  <VectorFieldCard key={f.name} field={f} />
                ))}
              </div>
            </div>
          )}

          {/* Scalar Fields */}
          {scalarFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">Scalar Fields</h2>
              <div className="rounded-lg border border-border overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-bg-secondary text-text-muted text-left">
                      <th className="px-4 py-3 font-medium">Field</th>
                      <th className="px-4 py-3 font-medium">Type</th>
                      <th className="px-4 py-3 font-medium">Index</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {scalarFields.map((f) => (
                      <tr key={f.name} className="hover:bg-bg-tertiary transition-colors">
                        <td className="px-4 py-3 font-medium text-text-primary">{f.name}</td>
                        <td className="px-4 py-3">
                          <Badge label={f.field_type} variant="default" />
                        </td>
                        <td className="px-4 py-3 text-text-secondary">
                          {inferScalarIndex(f.field_type)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Blob Fields */}
          {blobFields.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-3">Blob Fields</h2>
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
                      <tr key={f.name} className="hover:bg-bg-tertiary transition-colors">
                        <td className="px-4 py-3 font-medium text-text-primary">{f.name}</td>
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

          {vectorFields.length === 0 && scalarFields.length === 0 && blobFields.length === 0 && (
            <EmptyState title="No fields" description="This collection has no fields defined." />
          )}
        </div>
      )}
    </motion.div>
  )
}

function VectorFieldCard({ field }: { field: FieldSchema }) {
  const dims = field.dimensions ?? 0
  // Infer reasonable defaults for display
  const metric = 'Cosine'
  const indexType = dims > 512 ? 'HNSW' : 'DiskANN'
  const quantization = dims > 512 ? 'RaBitQ 4-bit' : 'SQ8'

  return (
    <div className="rounded-lg border border-border bg-bg-secondary p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-text-primary">{field.name}</h3>
        <Badge label="Ready" variant="success" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
        <div>
          <span className="text-xs text-text-muted">Type</span>
          <div className="text-text-primary">
            <Badge label={`Vector(${dims})`} variant="accent" />
          </div>
        </div>
        <div>
          <span className="text-xs text-text-muted">Metric</span>
          <div className="text-text-primary">{metric}</div>
        </div>
        <div>
          <span className="text-xs text-text-muted">Index</span>
          <div className="text-text-primary">{indexType}</div>
        </div>
        <div>
          <span className="text-xs text-text-muted">Quantization</span>
          <div className="text-text-primary">{quantization}</div>
        </div>
      </div>
    </div>
  )
}

function inferScalarIndex(fieldType: string): string {
  switch (fieldType) {
    case 'string': return 'Inverted Index'
    case 'integer':
    case 'float': return 'Range Index'
    case 'boolean': return 'Bitmap Index'
    default: return 'None'
  }
}
