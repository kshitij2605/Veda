import { useEffect, useState, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { fetchCollection, fetchRecords } from '../api/collections'
import type { CollectionInfo, FieldSchema } from '../api/collections'
import { PageHeader } from '../components/PageHeader'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { EmptyState } from '../components/EmptyState'
import { ScatterPlot } from '../components/ScatterPlot'
import type { ScatterPoint } from '../components/ScatterPlot'

const COLORS = [
  '#7c6af0', '#4ade80', '#fbbf24', '#f87171', '#38bdf8',
  '#fb923c', '#a78bfa', '#34d399', '#f472b6', '#22d3ee',
]

export function VectorAtlas() {
  const { collections, fetchCollections } = useCollectionStore()

  const [selectedCollection, setSelectedCollection] = useState('')
  const [schema, setSchema] = useState<CollectionInfo | null>(null)
  const [vectorField, setVectorField] = useState('')
  const [colorField, setColorField] = useState('')
  const [sampleSize, setSampleSize] = useState(5000)
  const [points, setPoints] = useState<ScatterPoint[]>([])
  const [selectedPoint, setSelectedPoint] = useState<ScatterPoint | null>(null)
  const [hoveredPoint, setHoveredPoint] = useState<ScatterPoint | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ w: 600, h: 500 })

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  // Resize observer for the canvas container
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        setCanvasSize({
          w: Math.floor(entry.contentRect.width),
          h: Math.floor(entry.contentRect.height),
        })
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Load schema when collection changes
  useEffect(() => {
    if (!selectedCollection) {
      setSchema(null)
      setVectorField('')
      setColorField('')
      return
    }
    fetchCollection(selectedCollection)
      .then(({ data }) => {
        setSchema(data)
        const firstVector = data.fields.find((f: FieldSchema) => f.field_type === 'vector')
        if (firstVector) setVectorField(firstVector.name)
        const firstScalar = data.fields.find((f: FieldSchema) => f.field_type !== 'vector' && f.field_type !== 'blob')
        if (firstScalar) setColorField(firstScalar.name)
      })
      .catch(() => setSchema(null))
  }, [selectedCollection])

  const vectorFields = schema?.fields.filter((f) => f.field_type === 'vector') ?? []
  const scalarFields = schema?.fields.filter((f) => f.field_type !== 'vector' && f.field_type !== 'blob') ?? []

  const handleGenerate = useCallback(async () => {
    if (!selectedCollection || !vectorField) return
    setLoading(true)
    setError(null)
    setSelectedPoint(null)

    try {
      const { data } = await fetchRecords(selectedCollection, { limit: sampleSize })
      const records = data.records

      // Build a color map from the colorField values
      const uniqueValues = new Map<string, number>()
      for (const rec of records) {
        const val = String(rec[colorField] ?? 'unknown')
        if (!uniqueValues.has(val)) uniqueValues.set(val, uniqueValues.size)
      }

      // Gather vectors that are valid arrays of the same dimension.
      const vectors: number[][] = []
      const validRecords: Record<string, unknown>[] = []
      for (const rec of records) {
        const vec = rec[vectorField]
        if (Array.isArray(vec) && vec.length >= 2) {
          vectors.push(vec as number[])
          validRecords.push(rec)
        }
      }

      // PCA via two rounds of power iteration. Reduces vectors to 2D coords
      // that maximise variance — far more useful for visualisation than the
      // first two raw dimensions, and works for any embedding dimensionality.
      const coords = pcaTwoDim(vectors)

      const newPoints: ScatterPoint[] = []
      for (let i = 0; i < validRecords.length; i++) {
        const rec = validRecords[i]
        const [x, y] = coords[i]
        const colorVal = String(rec[colorField] ?? 'unknown')
        const colorIdx = uniqueValues.get(colorVal) ?? 0

        newPoints.push({
          id: String(rec['id'] ?? rec['_id'] ?? newPoints.length),
          x,
          y,
          color: COLORS[colorIdx % COLORS.length],
          label: colorVal,
          data: rec,
        })
      }

      setPoints(newPoints)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load records')
    } finally {
      setLoading(false)
    }
  }, [selectedCollection, vectorField, colorField, sampleSize])

  const displayPoint = selectedPoint ?? hoveredPoint

  return (
    <motion.div
      className="p-8 h-full flex flex-col"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Vector Atlas"
        actions={
          <div className="flex items-center gap-2">
            <select
              value={selectedCollection}
              onChange={(e) => {
                setSelectedCollection(e.target.value)
                setPoints([])
                setSelectedPoint(null)
              }}
              className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
            >
              <option value="">Select collection...</option>
              {collections.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            {vectorFields.length > 1 && (
              <select
                value={vectorField}
                onChange={(e) => setVectorField(e.target.value)}
                className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
              >
                {vectorFields.map((f) => (
                  <option key={f.name} value={f.name}>{f.name}</option>
                ))}
              </select>
            )}
          </div>
        }
      />

      {!selectedCollection && (
        <EmptyState title="Select a collection" description="Choose a collection to visualize its embedding space." />
      )}

      {selectedCollection && schema && (
        <div className="flex gap-6 flex-1 min-h-0">
          {/* Left: controls + canvas */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Controls bar */}
            <div className="flex items-end gap-4 mb-4 flex-wrap">
              <div>
                <label className="block text-xs font-medium text-text-muted mb-1">Method</label>
                <select
                  disabled
                  className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none opacity-60"
                >
                  <option>First 2 dims</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-text-muted mb-1">Sample size</label>
                <input
                  type="number"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Math.max(100, Math.min(10000, parseInt(e.target.value) || 5000)))}
                  min={100}
                  max={10000}
                  className="w-24 px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
                />
              </div>
              {scalarFields.length > 0 && (
                <div>
                  <label className="block text-xs font-medium text-text-muted mb-1">Color by</label>
                  <select
                    value={colorField}
                    onChange={(e) => setColorField(e.target.value)}
                    className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
                  >
                    {scalarFields.map((f) => (
                      <option key={f.name} value={f.name}>{f.name}</option>
                    ))}
                  </select>
                </div>
              )}
              <button
                onClick={handleGenerate}
                disabled={loading || !vectorField}
                className="px-5 py-2 bg-accent hover:bg-accent-hover text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Loading...' : 'Generate'}
              </button>
            </div>

            {/* Canvas area */}
            <div ref={containerRef} className="flex-1 min-h-0 border border-border rounded-lg overflow-hidden">
              {loading && <LoadingSpinner label="Fetching records..." />}
              {error && (
                <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error m-4">
                  {error}
                </div>
              )}
              {!loading && !error && points.length === 0 && (
                <div className="flex items-center justify-center h-full">
                  <p className="text-sm text-text-muted">Click Generate to plot embeddings</p>
                </div>
              )}
              {!loading && !error && points.length > 0 && canvasSize.w > 0 && (
                <ScatterPlot
                  points={points}
                  onSelect={setSelectedPoint}
                  onHover={setHoveredPoint}
                  width={canvasSize.w}
                  height={canvasSize.h}
                />
              )}
            </div>

            {points.length > 0 && (
              <div className="mt-2 text-xs text-text-muted">
                {points.length} points plotted. Scroll to zoom, drag to pan, click to select.
              </div>
            )}
          </div>

          {/* Right: detail panel */}
          <div className="w-72 shrink-0 flex flex-col gap-4">
            <div className="rounded-lg border border-border bg-bg-secondary p-4">
              <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">Point Details</h3>
              {displayPoint ? (
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-text-muted">ID: </span>
                    <span className="text-text-primary font-mono text-xs">{displayPoint.id}</span>
                  </div>
                  {Object.entries(displayPoint.data)
                    .filter(([, v]) => !Array.isArray(v))
                    .slice(0, 8)
                    .map(([k, v]) => (
                      <div key={k}>
                        <span className="text-text-muted">{k}: </span>
                        <span className="text-text-primary">{String(v)}</span>
                      </div>
                    ))}
                  {displayPoint.label && (
                    <div className="mt-2 pt-2 border-t border-border">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium" style={{ backgroundColor: displayPoint.color + '30', color: displayPoint.color }}>
                        {displayPoint.label}
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-text-muted">Hover or click a point to view details.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

/* ── PCA via power iteration ───────────────────────────────────────────────
 *
 * Reduces an N×D matrix of vectors to N 2-d points by computing the top-2
 * principal components. Power iteration on the centred data with a few
 * passes converges quickly enough for visualisation (~10 iter is plenty).
 * Returns coordinates in the same order as the input.
 */
function pcaTwoDim(vectors: number[][]): [number, number][] {
  const n = vectors.length
  if (n === 0) return []
  const d = vectors[0].length
  if (d < 2) return vectors.map(() => [0, 0])

  // 1. Compute per-dimension mean and centre.
  const mean = new Float64Array(d)
  for (const v of vectors) {
    for (let i = 0; i < d; i++) mean[i] += v[i]
  }
  for (let i = 0; i < d; i++) mean[i] /= n
  const centered: Float64Array[] = vectors.map(v => {
    const out = new Float64Array(d)
    for (let i = 0; i < d; i++) out[i] = v[i] - mean[i]
    return out
  })

  // 2. Power iteration for the top eigenvector of X^T X (without forming it).
  //    Each step: v' = X^T (X v); ~10 iterations suffices for visualisation.
  const v1 = randomUnit(d)
  powerIterate(v1, centered, d, 12)
  const proj1 = centered.map(row => dot(row, v1))

  // 3. Second component: orthogonalise against v1, then power iterate.
  //    To capture variance orthogonal to v1, we deflate by subtracting the
  //    v1 projection at each step.
  const v2 = randomUnit(d)
  orthogonalize(v2, v1, d)
  powerIterateDeflated(v2, centered, v1, d, 12)
  const proj2 = centered.map(row => dot(row, v2))

  return proj1.map((x, i) => [x, proj2[i]])
}

function randomUnit(d: number): Float64Array {
  const v = new Float64Array(d)
  let norm = 0
  for (let i = 0; i < d; i++) {
    // deterministic-ish seed so re-renders don't shuffle the layout
    v[i] = Math.sin(i * 12.9898 + 78.233)
    norm += v[i] * v[i]
  }
  norm = Math.sqrt(norm)
  if (norm > 0) for (let i = 0; i < d; i++) v[i] /= norm
  return v
}

function dot(a: Float64Array | number[], b: Float64Array | number[]): number {
  let s = 0
  const n = a.length
  for (let i = 0; i < n; i++) s += (a[i] as number) * (b[i] as number)
  return s
}

function normalise(v: Float64Array, d: number) {
  let n = 0
  for (let i = 0; i < d; i++) n += v[i] * v[i]
  n = Math.sqrt(n)
  if (n > 0) for (let i = 0; i < d; i++) v[i] /= n
}

function orthogonalize(v: Float64Array, ref: Float64Array, d: number) {
  const p = dot(v, ref)
  for (let i = 0; i < d; i++) v[i] -= p * ref[i]
  normalise(v, d)
}

function powerIterate(v: Float64Array, rows: Float64Array[], d: number, iters: number) {
  const next = new Float64Array(d)
  for (let it = 0; it < iters; it++) {
    next.fill(0)
    for (const row of rows) {
      const proj = dot(row, v)
      for (let i = 0; i < d; i++) next[i] += proj * row[i]
    }
    for (let i = 0; i < d; i++) v[i] = next[i]
    normalise(v, d)
  }
}

function powerIterateDeflated(
  v: Float64Array,
  rows: Float64Array[],
  v1: Float64Array,
  d: number,
  iters: number,
) {
  const next = new Float64Array(d)
  for (let it = 0; it < iters; it++) {
    next.fill(0)
    for (const row of rows) {
      const proj = dot(row, v)
      for (let i = 0; i < d; i++) next[i] += proj * row[i]
    }
    // Remove the v1 component each step to keep v in the orthogonal subspace.
    for (let i = 0; i < d; i++) v[i] = next[i]
    orthogonalize(v, v1, d)
  }
}
