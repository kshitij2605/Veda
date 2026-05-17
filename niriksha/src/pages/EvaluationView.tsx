import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { evaluateCollection, fetchCollection } from '../api/collections'
import type {
  CollectionInfo,
  EvaluateResponse,
  FieldQuality,
} from '../api/collections'

const COMING_SOON = [
  { title: 'RAG Usability Scoring', description: 'Score how well your data supports retrieval-augmented generation. Requires an LLM judge endpoint.' },
  { title: 'FAQ Usability Scoring', description: 'Assess data suitability for FAQ-style QA. Requires an LLM judge endpoint.' },
  { title: 'Cross-modal Alignment Detection', description: 'Detect misalignment between text, image, and audio embeddings for the same content.' },
]

export function EvaluationView() {
  const { collections, fetchCollections } = useCollectionStore()
  const [selectedCollection, setSelectedCollection] = useState('')
  const [schema, setSchema] = useState<CollectionInfo | null>(null)
  const [vectorField, setVectorField] = useState('')
  const [sample, setSample] = useState(200)
  const [threshold, setThreshold] = useState(0.01)
  const [result, setResult] = useState<EvaluateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (!selectedCollection) {
      setSchema(null)
      setVectorField('')
      return
    }
    fetchCollection(selectedCollection)
      .then(({ data }) => {
        setSchema(data)
        const firstVec = data.fields.find((f) => f.field_type === 'vector')
        setVectorField(firstVec?.name ?? '')
      })
      .catch(() => setSchema(null))
  }, [selectedCollection])

  const runEvaluation = async () => {
    if (!selectedCollection) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const params: Record<string, string | number> = { sample }
      if (vectorField) {
        params.vector_field = vectorField
        params.duplicate_threshold = threshold
      }
      const { data } = await evaluateCollection(selectedCollection, params)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Data Quality Evaluation"
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

      <div className="max-w-4xl space-y-6">
        {/* Controls */}
        <div className="rounded-lg border border-border bg-bg-secondary p-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-text-muted mb-1">Vector field (optional, enables duplicate detection)</label>
              <select
                value={vectorField}
                onChange={(e) => setVectorField(e.target.value)}
                disabled={!schema}
                className="w-full px-3 py-2 bg-bg-primary border border-border rounded text-sm"
              >
                <option value="">— none —</option>
                {schema?.fields.filter((f) => f.field_type === 'vector').map((f) => (
                  <option key={f.name} value={f.name}>{f.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">Sample size</label>
              <input
                type="number"
                value={sample}
                min={10}
                max={5000}
                onChange={(e) => setSample(Number(e.target.value))}
                className="w-full px-3 py-2 bg-bg-primary border border-border rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">Duplicate threshold</label>
              <input
                type="number"
                value={threshold}
                step="0.001"
                min={0}
                max={1}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full px-3 py-2 bg-bg-primary border border-border rounded text-sm"
              />
            </div>
          </div>
          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={runEvaluation}
              disabled={!selectedCollection || loading}
              className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {loading ? 'Running…' : 'Run evaluation'}
            </button>
            {error && <span className="text-xs text-error">{error}</span>}
          </div>
        </div>

        {loading && <LoadingSpinner label="Evaluating sample..." />}

        {result && (
          <>
            {/* Duplicates */}
            {result.duplicate_pairs.length > 0 && (
              <div className="rounded-lg border border-border bg-bg-secondary p-5">
                <h3 className="text-sm font-semibold text-text-primary mb-3">
                  Duplicate candidates ({result.duplicate_pairs.length})
                </h3>
                <div className="space-y-1 max-h-72 overflow-auto font-mono text-xs">
                  {result.duplicate_pairs.slice(0, 100).map((p, i) => (
                    <div key={i} className="flex justify-between gap-3 py-0.5">
                      <span className="text-text-secondary truncate">{p.id_a}</span>
                      <span className="text-text-muted">≈</span>
                      <span className="text-text-secondary truncate">{p.id_b}</span>
                      <span className="text-accent w-16 text-right">{p.distance.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Field quality */}
            <div className="rounded-lg border border-border bg-bg-secondary p-5">
              <h3 className="text-sm font-semibold text-text-primary mb-3">
                Field quality ({result.records_evaluated} records sampled)
              </h3>
              <table className="w-full text-xs">
                <thead className="text-text-muted">
                  <tr>
                    <th className="text-left py-1 font-medium">Field</th>
                    <th className="text-right py-1 font-medium">Missing</th>
                    <th className="text-right py-1 font-medium">Empty</th>
                    <th className="text-right py-1 font-medium">Min len</th>
                    <th className="text-right py-1 font-medium">Avg len</th>
                    <th className="text-right py-1 font-medium">Max len</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {Object.entries(result.field_quality).map(([field, q]) => (
                    <FieldRow key={field} field={field} q={q} />
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* Pending features */}
        <div className="rounded-lg border border-border border-dashed bg-bg-tertiary/50 p-5">
          <div className="text-xs font-medium text-text-muted mb-3 uppercase tracking-wider">
            Not yet implemented
          </div>
          <div className="space-y-2">
            {COMING_SOON.map((f) => (
              <div key={f.title} className="flex gap-3">
                <div className="mt-1 w-1.5 h-1.5 rounded-full bg-text-muted shrink-0" />
                <div>
                  <div className="text-sm font-medium text-text-primary">{f.title}</div>
                  <div className="text-xs text-text-muted mt-0.5">{f.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function FieldRow({ field, q }: { field: string; q: FieldQuality }) {
  return (
    <tr>
      <td className="py-1 text-text-primary font-mono">{field}</td>
      <td className="py-1 text-right text-text-secondary">{q.missing}</td>
      <td className="py-1 text-right text-text-secondary">{q.null_or_empty}</td>
      <td className="py-1 text-right text-text-secondary">{q.min_length ?? '—'}</td>
      <td className="py-1 text-right text-text-secondary">{q.avg_length.toFixed(1)}</td>
      <td className="py-1 text-right text-text-secondary">{q.max_length ?? '—'}</td>
    </tr>
  )
}
