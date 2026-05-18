import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { useSearchStore } from '../stores/searchStore'
import { fetchCollection } from '../api/collections'
import type { CollectionInfo } from '../api/collections'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'

export function SearchPlayground() {
  const { collection: urlCollection } = useParams<{ collection: string }>()
  const { collections, fetchCollections } = useCollectionStore()
  const { results, loading, error, tookMs, explain, search, clearResults } =
    useSearchStore()

  const [selectedCollection, setSelectedCollection] = useState(
    urlCollection ?? '',
  )
  const [collectionSchema, setCollectionSchema] =
    useState<CollectionInfo | null>(null)
  const [vectorField, setVectorField] = useState('')
  const [vectorInput, setVectorInput] = useState('')
  const [filterInput, setFilterInput] = useState('')
  const [limit, setLimit] = useState(10)
  const [explainOn, setExplainOn] = useState(false)
  const [truncateTo, setTruncateTo] = useState<number | ''>('')
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const [parseError, setParseError] = useState<string | null>(null)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (urlCollection) setSelectedCollection(urlCollection)
  }, [urlCollection])

  useEffect(() => {
    if (!selectedCollection) {
      setCollectionSchema(null)
      setVectorField('')
      return
    }
    fetchCollection(selectedCollection)
      .then(({ data }) => {
        setCollectionSchema(data)
        const firstVector = data.fields.find((f) => f.type === 'vector')
        if (firstVector) setVectorField(firstVector.name)
      })
      .catch(() => setCollectionSchema(null))
  }, [selectedCollection])

  const vectorFields =
    collectionSchema?.fields.filter((f) => f.type === 'vector') ?? []
  const selectedFieldSchema = vectorFields.find((f) => f.name === vectorField)

  const handleSearch = () => {
    if (!selectedCollection || !vectorField || !vectorInput.trim()) return
    setParseError(null)
    setExpandedRow(null)

    let values: number[]
    try {
      values = JSON.parse(vectorInput.trim())
      if (!Array.isArray(values) || !values.every((v) => typeof v === 'number')) {
        throw new Error('Must be an array of numbers')
      }
    } catch (e) {
      setParseError(e instanceof Error ? e.message : 'Invalid JSON')
      return
    }

    if (selectedFieldSchema?.dimension && values.length !== selectedFieldSchema.dimension) {
      setParseError(
        `Vector has ${values.length} dims, field expects ${selectedFieldSchema.dimension}`,
      )
      return
    }

    search(selectedCollection, {
      query: { vector: { field: vectorField, values } },
      filter: filterInput.trim() || undefined,
      limit,
      explain: explainOn || undefined,
      truncate_to: typeof truncateTo === 'number' ? truncateTo : undefined,
    })
  }

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Search Playground"
        actions={
          <select
            value={selectedCollection}
            onChange={(e) => {
              setSelectedCollection(e.target.value)
              clearResults()
            }}
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

      <div className="rounded-lg border border-border bg-bg-secondary p-5 mb-6">
        <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-4">
          Query
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              Vector Field
            </label>
            <select
              value={vectorField}
              onChange={(e) => setVectorField(e.target.value)}
              disabled={vectorFields.length === 0}
              className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent disabled:opacity-50"
            >
              {vectorFields.length === 0 && (
                <option value="">No vector fields</option>
              )}
              {vectorFields.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} ({f.dimension ?? '?'} dims, {f.metric ?? '?'})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              Limit
            </label>
            <input
              type="number"
              value={limit}
              onChange={(e) =>
                setLimit(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))
              }
              min={1}
              max={100}
              className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
            />
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted mb-1">
            Vector (JSON array)
            {selectedFieldSchema?.dimension && (
              <span className="ml-2 text-text-muted">
                expected {selectedFieldSchema.dimension} dims
              </span>
            )}
          </label>
          <textarea
            value={vectorInput}
            onChange={(e) => {
              setVectorInput(e.target.value)
              setParseError(null)
            }}
            placeholder="[0.1, 0.2, 0.3, ...]"
            rows={3}
            className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent font-mono resize-none"
          />
          {parseError && (
            <p className="mt-1 text-xs text-error">{parseError}</p>
          )}
        </div>

        <div className="mb-4">
          <label className="block text-xs font-medium text-text-muted mb-1">
            Filter (optional)
          </label>
          <input
            value={filterInput}
            onChange={(e) => setFilterInput(e.target.value)}
            placeholder="category = 'electronics' AND price < 500"
            className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            <input
              type="checkbox"
              checked={explainOn}
              onChange={(e) => setExplainOn(e.target.checked)}
            />
            Request per-phase timing
          </label>
          <div>
            <label className="block text-xs font-medium text-text-muted mb-1">
              Truncate to (MRL, optional)
            </label>
            <input
              type="number"
              min={1}
              value={truncateTo}
              onChange={(e) => {
                const v = e.target.value
                setTruncateTo(v === '' ? '' : Math.max(1, parseInt(v) || 1))
              }}
              placeholder={selectedFieldSchema?.dimension?.toString() ?? ''}
              className="w-full px-3 py-2 bg-bg-primary border border-border rounded-lg text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
            />
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSearch}
            disabled={
              loading ||
              !selectedCollection ||
              !vectorField ||
              !vectorInput.trim()
            }
            className="px-5 py-2 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {loading && <LoadingSpinner label="Searching..." />}

      {error && (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error mb-4">
          {error}
        </div>
      )}

      {explain && (
        <div className="rounded-lg border border-border bg-bg-secondary p-4 mb-4 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <ExplainCell label="Filter" ms={explain.filter_ms} />
          <ExplainCell label="ANN" ms={explain.ann_search_ms} />
          <ExplainCell label="Scoring" ms={explain.scoring_ms} />
          <ExplainCell label="Fusion" ms={explain.fusion_ms} />
          <ExplainCell label="Total" ms={explain.total_ms} highlight />
        </div>
      )}

      {!loading && !error && results.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-3">
            <h3 className="text-sm font-medium text-text-secondary">
              Results ({results.length} found, {tookMs.toFixed(1)}ms)
            </h3>
          </div>

          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary text-text-muted text-left">
                  <th className="px-4 py-3 font-medium w-16">Rank</th>
                  <th className="px-4 py-3 font-medium">ID</th>
                  <th className="px-4 py-3 font-medium w-28">Distance</th>
                  <th className="px-4 py-3 font-medium">Fields</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {results.map((hit, i) => (
                  <tr
                    key={hit.id}
                    onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                    className="hover:bg-bg-tertiary cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3 text-text-muted">{i + 1}</td>
                    <td className="px-4 py-3 font-mono text-text-primary text-xs">
                      {hit.id.length > 12 ? hit.id.slice(0, 12) + '...' : hit.id}
                    </td>
                    <td className="px-4 py-3">
                      <DistanceBadge distance={hit.distance} />
                    </td>
                    <td className="px-4 py-3 text-text-secondary text-xs truncate max-w-xs">
                      {formatFieldsSummary(hit.fields)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <AnimatePresence>
            {expandedRow !== null && results[expandedRow] && (
              <motion.div
                className="mt-3 rounded-lg border border-border bg-bg-secondary p-4"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
                  Result #{expandedRow + 1} - {results[expandedRow].id}
                </h4>
                <pre className="text-xs text-text-secondary overflow-x-auto">
                  {JSON.stringify(results[expandedRow], null, 2)}
                </pre>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {!loading && !error && results.length === 0 && selectedCollection && (
        <EmptyState
          title="No results yet"
          description="Enter a vector and click Search to find similar records."
        />
      )}

      {!selectedCollection && (
        <EmptyState
          title="Select a collection"
          description="Choose a collection from the dropdown to begin searching."
        />
      )}
    </motion.div>
  )
}

function ExplainCell({
  label,
  ms,
  highlight,
}: {
  label: string
  ms: number
  highlight?: boolean
}) {
  return (
    <div>
      <div className="text-text-muted uppercase tracking-wider text-[10px]">
        {label}
      </div>
      <div
        className={`font-mono ${
          highlight ? 'text-accent font-semibold' : 'text-text-primary'
        }`}
      >
        {ms.toFixed(2)}ms
      </div>
    </div>
  )
}

function DistanceBadge({ distance }: { distance: number }) {
  let colorClass: string
  if (distance < 0.3) colorClass = 'text-success bg-success/15'
  else if (distance < 0.7) colorClass = 'text-warning bg-warning/15'
  else colorClass = 'text-error bg-error/15'

  return (
    <span
      className={`inline-flex px-2 py-0.5 rounded text-xs font-mono font-medium ${colorClass}`}
    >
      {distance.toFixed(4)}
    </span>
  )
}

function formatFieldsSummary(fields: Record<string, unknown>): string {
  return Object.entries(fields)
    .filter(([, v]) => !Array.isArray(v))
    .slice(0, 3)
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join(', ')
}
