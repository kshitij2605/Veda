import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Atom,
  Hand,
  Maximize,
  Play,
  Lasso,
  Sparkles,
} from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import {
  fetchCollection,
  fetchRecords,
  type CollectionInfo,
  type RecordRow,
} from '../../api/collections'
import {
  Badge,
  Button,
  Select,
  Skeleton,
  Tooltip,
} from '../../components/primitives'
import { useUrlState } from '../../hooks/useUrlState'
import { cn } from '../../lib/cn'
import { AtlasCanvas, type AtlasPoint } from './AtlasCanvas'
import { BubbleUp } from './BubbleUp'
import { pcaTwoDim } from './pca'
import { RecordDrawer } from '../collections/RecordDrawer'

const COLORS = [
  '#7c6af0', '#4ade80', '#fbbf24', '#f87171', '#38bdf8',
  '#fb923c', '#a78bfa', '#34d399', '#f472b6', '#22d3ee',
]

interface AtlasState {
  collection: string
  vectorField: string
  colorField: string
  sample: number
}

const DEFAULT_STATE: AtlasState = {
  collection: '',
  vectorField: '',
  colorField: '',
  sample: 2000,
}

export function AtlasPage() {
  const { collections, fetchCollections } = useCollectionStore()
  const [state, setState] = useUrlState<AtlasState>('a', DEFAULT_STATE)
  const setS = <K extends keyof AtlasState>(k: K, v: AtlasState[K]) =>
    setState({ ...state, [k]: v })

  const [schema, setSchema] = useState<CollectionInfo | null>(null)
  const [records, setRecords] = useState<RecordRow[]>([])
  const [points, setPoints] = useState<AtlasPoint[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [lassoMode, setLassoMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [hoverPoint, setHoverPoint] = useState<AtlasPoint | null>(null)
  const [activePoint, setActivePoint] = useState<AtlasPoint | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ w: 600, h: 500 })

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (!state.collection && collections.length > 0) {
      setS('collection', collections[0]!.name)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collections])

  // Resize observer
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      const e = entries[0]
      if (e) {
        setCanvasSize({
          w: Math.floor(e.contentRect.width),
          h: Math.floor(e.contentRect.height),
        })
      }
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Load schema when collection changes
  useEffect(() => {
    if (!state.collection) {
      setSchema(null)
      return
    }
    fetchCollection(state.collection)
      .then(({ data }) => {
        setSchema(data)
        const firstVec = data.fields.find((f) => f.type === 'vector')
        const firstScalar = data.fields.find(
          (f) =>
            f.type !== 'vector' &&
            f.type !== 'blob' &&
            f.type !== 'multi_vector' &&
            f.type !== 'sparse_vector' &&
            f.type !== 'json',
        )
        setState((prev) => ({
          ...prev,
          vectorField: prev.vectorField || firstVec?.name || '',
          colorField: prev.colorField || firstScalar?.name || '',
        }))
      })
      .catch(() => setSchema(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.collection])

  const vectorFields = useMemo(
    () => schema?.fields.filter((f) => f.type === 'vector') ?? [],
    [schema],
  )
  const scalarFields = useMemo(
    () =>
      schema?.fields.filter(
        (f) =>
          f.type !== 'vector' &&
          f.type !== 'blob' &&
          f.type !== 'multi_vector' &&
          f.type !== 'sparse_vector' &&
          f.type !== 'json',
      ) ?? [],
    [schema],
  )

  const handleGenerate = useCallback(async () => {
    if (!state.collection || !state.vectorField) return
    setLoading(true)
    setError(null)
    setPoints([])
    setRecords([])
    setSelectedIds(new Set())
    try {
      const { data } = await fetchRecords(state.collection, { limit: state.sample })
      const rows = data.records
      const vectors: number[][] = []
      const validRecs: RecordRow[] = []
      for (const r of rows) {
        const v = r.fields[state.vectorField]
        if (Array.isArray(v) && v.length >= 2) {
          vectors.push(v as number[])
          validRecs.push(r)
        }
      }
      if (vectors.length === 0) {
        throw new Error(
          'No vector values found in the sample. Pick a populated vector field.',
        )
      }
      const coords = pcaTwoDim(vectors)
      const colorMap = new Map<string, number>()
      for (const r of validRecs) {
        const k = String(r.fields[state.colorField] ?? 'unknown')
        if (!colorMap.has(k)) colorMap.set(k, colorMap.size)
      }
      const pts: AtlasPoint[] = validRecs.map((r, i) => ({
        id: r.id,
        x: coords[i]![0],
        y: coords[i]![1],
        color:
          COLORS[
            (colorMap.get(String(r.fields[state.colorField] ?? 'unknown')) ?? 0) %
              COLORS.length
          ]!,
        label: String(r.fields[state.colorField] ?? ''),
      }))
      setPoints(pts)
      setRecords(validRecs)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate atlas')
    } finally {
      setLoading(false)
    }
  }, [state.collection, state.vectorField, state.colorField, state.sample])

  const selectedRecords = useMemo(
    () => records.filter((r) => selectedIds.has(r.id)),
    [records, selectedIds],
  )

  const activeRecord = useMemo(
    () => (activePoint ? records.find((r) => r.id === activePoint.id) ?? null : null),
    [activePoint, records],
  )

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <Header
        collection={state.collection}
        collections={collections.map((c) => c.name)}
        onCollection={(c) => {
          setState({ ...DEFAULT_STATE, collection: c })
          setPoints([])
          setRecords([])
          setSelectedIds(new Set())
        }}
        vectorFields={vectorFields.map((f) => f.name)}
        vectorField={state.vectorField}
        onVectorField={(v) => setS('vectorField', v)}
        colorField={state.colorField}
        onColorField={(v) => setS('colorField', v)}
        colorFields={scalarFields.map((f) => f.name)}
        sample={state.sample}
        onSample={(n) => setS('sample', n)}
        onGenerate={handleGenerate}
        loading={loading}
        canGenerate={!!state.vectorField && !!state.collection}
        lassoMode={lassoMode}
        onToggleLasso={() => setLassoMode((v) => !v)}
      />
      <div className="flex flex-1 min-h-0">
        <div ref={containerRef} className="relative flex-1 min-w-0">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-bg)]/70 backdrop-blur-[1px]">
              <div className="flex flex-col items-center gap-2 text-[var(--color-text-muted)]">
                <Sparkles size={20} className="text-[var(--color-accent)] animate-pulse" />
                <span className="text-[12px]">Running PCA…</span>
              </div>
            </div>
          )}
          {error && !loading && (
            <div className="absolute inset-x-4 top-4 rounded-[var(--radius-sm)] bg-[color-mix(in_oklch,var(--color-error)_10%,transparent)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
              {error}
            </div>
          )}
          {!loading && points.length === 0 && !error && (
            <EmptyAtlas onGenerate={handleGenerate} canGenerate={!!state.vectorField && !!state.collection} />
          )}
          {!loading && points.length > 0 && canvasSize.w > 0 && (
            <AtlasCanvas
              points={points}
              width={canvasSize.w}
              height={canvasSize.h}
              onHover={setHoverPoint}
              onSelect={(p) => {
                setActivePoint(p)
                if (p) setDrawerOpen(true)
              }}
              lasso={lassoMode}
              onLasso={(ids) => {
                setSelectedIds(ids)
                setLassoMode(false)
              }}
              selectedIds={selectedIds.size > 0 ? selectedIds : undefined}
            />
          )}
          {/* Hover tooltip */}
          {hoverPoint && (
            <div
              className="absolute top-2 left-2 px-2 py-1 rounded-[var(--radius-xs)] bg-[var(--color-bg-overlay)] border border-[var(--color-border-strong)] text-[10.5px] font-mono text-[var(--color-text)] pointer-events-none"
            >
              <span className="text-[var(--color-text-subtle)]">id </span>
              {hoverPoint.id.slice(0, 12)}…
              {hoverPoint.label && (
                <span className="ml-2 text-[var(--color-text-muted)]">
                  · {hoverPoint.label}
                </span>
              )}
            </div>
          )}
        </div>
        <aside className="w-[300px] flex-none border-l border-[var(--color-border)] bg-[var(--color-bg)] flex flex-col min-h-0">
          {schema && (
            <BubbleUp
              fields={schema.fields}
              allRecords={records}
              selectedRecords={selectedRecords}
            />
          )}
        </aside>
      </div>
      <RecordDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        record={activeRecord}
        fields={schema?.fields ?? []}
        onAddFilter={() => {
          /* navigation back into records is a follow-up; the drawer
             already exposes the row's filter buttons */
        }}
      />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Header (collection picker + vector + color + sample + lasso toggle)
   ──────────────────────────────────────────────────────────────────── */

function Header({
  collection,
  collections,
  onCollection,
  vectorField,
  onVectorField,
  vectorFields,
  colorField,
  onColorField,
  colorFields,
  sample,
  onSample,
  onGenerate,
  loading,
  canGenerate,
  lassoMode,
  onToggleLasso,
}: {
  collection: string
  collections: string[]
  onCollection: (c: string) => void
  vectorField: string
  onVectorField: (v: string) => void
  vectorFields: string[]
  colorField: string
  onColorField: (v: string) => void
  colorFields: string[]
  sample: number
  onSample: (n: number) => void
  onGenerate: () => void
  loading: boolean
  canGenerate: boolean
  lassoMode: boolean
  onToggleLasso: () => void
}) {
  return (
    <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none bg-[var(--color-bg)] overflow-x-auto">
      <Atom size={14} strokeWidth={2} className="text-[var(--color-accent-9)]" />
      <h1 className="text-[14px] font-medium text-[var(--color-text)]">Atlas</h1>
      <span className="text-[var(--color-text-subtle)]">·</span>
      <Select
        value={collection}
        onValueChange={onCollection}
        placeholder="collection"
        options={collections.map((c) => ({ value: c, label: c }))}
      />
      <Select
        value={vectorField}
        onValueChange={onVectorField}
        placeholder="vector field"
        options={vectorFields.map((v) => ({ value: v, label: v }))}
      />
      <Select
        value={colorField || '__none__'}
        onValueChange={(v) => onColorField(v === '__none__' ? '' : v)}
        placeholder="color by"
        options={[
          { value: '__none__', label: 'none' },
          ...colorFields.map((v) => ({ value: v, label: v })),
        ]}
      />
      <Select
        size="sm"
        value={String(sample)}
        onValueChange={(v) => onSample(parseInt(v, 10))}
        options={[500, 1000, 2000, 5000].map((n) => ({
          value: String(n),
          label: `${n} pts`,
        }))}
      />
      <Tooltip label="PCA (2-D) — UMAP / t-SNE in a follow-up">
        <Badge label="PCA · 2D" variant="ghost" />
      </Tooltip>
      <div className="ml-auto flex items-center gap-1.5">
        <Tooltip label={lassoMode ? 'Switch back to pan / zoom' : 'Lasso — drag to enclose points'}>
          <button
            onClick={onToggleLasso}
            className={cn(
              'inline-flex items-center gap-1 px-2 h-7 rounded-[var(--radius-sm)] border',
              'transition-colors duration-[var(--duration-fast)] text-[12px]',
              lassoMode
                ? 'border-[var(--color-accent)] bg-[var(--color-accent-2)] text-[var(--color-accent-11)]'
                : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text)]',
            )}
          >
            {lassoMode ? (
              <Lasso size={12} strokeWidth={2.25} />
            ) : (
              <Hand size={12} strokeWidth={2.25} />
            )}
            {lassoMode ? 'Lasso' : 'Pan'}
          </button>
        </Tooltip>
        <Button
          variant="primary"
          size="sm"
          leading={
            loading ? (
              <Sparkles size={11} strokeWidth={2.25} className="animate-pulse" />
            ) : (
              <Play size={11} strokeWidth={2.5} />
            )
          }
          onClick={onGenerate}
          disabled={loading || !canGenerate}
        >
          {loading ? 'Running' : 'Generate'}
        </Button>
      </div>
    </header>
  )
}

function EmptyAtlas({
  onGenerate,
  canGenerate,
}: {
  onGenerate: () => void
  canGenerate: boolean
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="dot-grid absolute inset-0 opacity-30" aria-hidden />
      <div className="relative text-center max-w-md px-6">
        <Maximize
          size={28}
          strokeWidth={1.5}
          className="text-[var(--color-text-subtle)] mx-auto mb-3"
        />
        <h2 className="text-[14px] font-medium text-[var(--color-text)] mb-1">
          Vector Atlas
        </h2>
        <p className="text-[12px] text-[var(--color-text-muted)] mb-4 leading-relaxed">
          Project a sample of this collection's embeddings to 2-D with
          PCA. Pan, zoom, hover a point for its id, click for the full
          record. Lasso a region and BubbleUp tells you which fields are
          over- or under-represented inside the selection vs the whole
          sample.
        </p>
        <Button
          variant="primary"
          onClick={onGenerate}
          disabled={!canGenerate}
          leading={<Play size={12} strokeWidth={2.5} />}
        >
          Generate Atlas
        </Button>
        {!canGenerate && (
          <p className="mt-2 text-[10.5px] text-[var(--color-text-subtle)]">
            Pick a collection + vector field in the header first.
          </p>
        )}
      </div>
    </div>
  )
}

// Re-export Skeleton so tree-shaking sees it referenced if other phases
// want to import it via this module.
export { Skeleton }
