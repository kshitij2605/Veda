import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import * as Tabs from '@radix-ui/react-tabs'
import {
  AlertCircle,
  Atom,
  Copy,
  Play,
  Save,
  Sparkles,
  Type,
  Zap,
} from 'lucide-react'
import {
  embedQuery,
  fetchCollection,
  type CollectionInfo,
  type EmbedResponse,
  type ExplainTiming,
  type FieldSchema,
  type SearchHit,
  type SearchRequest,
} from '../api/collections'
import { useCollectionStore } from '../stores/collectionStore'
import { useSavedObjects } from '../stores/savedObjectStore'
import { useSearchStore } from '../stores/searchStore'
import { HybridSankey } from './search/HybridSankey'
import { useUrlState } from '../hooks/useUrlState'
import {
  Badge,
  Button,
  CopyButton,
  Input,
  Select,
  Skeleton,
  toast,
  Tooltip,
} from '../components/primitives'
import { Textarea } from '../components/primitives/Input'
import { cn } from '../lib/cn'

type QueryMode = 'dense' | 'sparse' | 'bm25' | 'hybrid'

interface PlaygroundState {
  collection: string
  mode: QueryMode
  text: string
  vectorField: string
  bm25Field: string
  filter: string
  topK: number
  truncateTo: number | null
  fusion: 'rrf' | 'weighted'
  rrfK: number
  vectorWeight: number
  bm25Weight: number
  rerankOn: boolean
  rerankField: string
  explainOn: boolean
}

const DEFAULT_STATE = (initialCollection: string): PlaygroundState => ({
  collection: initialCollection,
  mode: 'dense',
  text: '',
  vectorField: '',
  bm25Field: '',
  filter: '',
  topK: 10,
  truncateTo: null,
  fusion: 'rrf',
  rrfK: 60,
  vectorWeight: 0.5,
  bm25Weight: 0.5,
  rerankOn: false,
  rerankField: '',
  explainOn: true,
})

export function SearchPlayground() {
  const { collection: urlCollection } = useParams<{ collection: string }>()
  const { collections, fetchCollections } = useCollectionStore()
  const {
    results,
    loading,
    error,
    tookMs,
    explain,
    hybridExplain,
    search,
    clearResults,
  } = useSearchStore()

  const [state, setState] = useUrlState<PlaygroundState>(
    's',
    DEFAULT_STATE(urlCollection ?? ''),
  )
  const setS = <K extends keyof PlaygroundState>(
    k: K,
    v: PlaygroundState[K],
  ) => setState({ ...state, [k]: v })

  const [schema, setSchema] = useState<CollectionInfo | null>(null)
  const [embedInfo, setEmbedInfo] = useState<EmbedResponse | null>(null)
  const [embedding, setEmbedding] = useState(false)
  const [embedError, setEmbedError] = useState<string | null>(null)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (urlCollection && urlCollection !== state.collection) {
      setState({ ...DEFAULT_STATE(urlCollection) })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlCollection])

  // Load schema when collection changes; pick reasonable defaults for fields.
  useEffect(() => {
    if (!state.collection) {
      setSchema(null)
      return
    }
    fetchCollection(state.collection)
      .then(({ data }) => {
        setSchema(data)
        const firstVec = data.fields.find((f) => f.type === 'vector')
        const firstStr = data.fields.find((f) => f.type === 'string')
        // Fall back to the external source's vector column (see `vectorFields`).
        const vecName = firstVec?.name || data.external?.vector_field || ''
        setState((prev) => ({
          ...prev,
          vectorField: prev.vectorField || vecName,
          bm25Field: prev.bm25Field || firstStr?.name || '',
          rerankField: prev.rerankField || firstStr?.name || '',
        }))
      })
      .catch(() => setSchema(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.collection])

  // An external (read-in-place Parquet) collection reports an empty `fields`
  // array — its schema lives in the Parquet footer, and only the vector
  // column is named, under `external`. Without this the field picker stays
  // empty and the collection cannot be queried from the UI at all.
  const vectorFields = useMemo(() => {
    const declared = schema?.fields.filter((f) => f.type === 'vector') ?? []
    if (declared.length > 0 || !schema?.external?.vector_field) return declared
    return [
      {
        name: schema.external.vector_field,
        type: 'vector',
        filterable: false,
        metric: schema.external.metric?.toLowerCase(),
      } as FieldSchema,
    ]
  }, [schema])
  const stringFields = useMemo(
    () => schema?.fields.filter((f) => f.type === 'string') ?? [],
    [schema],
  )
  const activeVectorField = vectorFields.find((f) => f.name === state.vectorField)

  const handleEmbedAndSearch = async () => {
    if (!state.collection || !state.text.trim()) return
    setEmbedError(null)
    setEmbedInfo(null)
    let values: number[] | null = null

    // For dense + hybrid we need a query vector. BM25/sparse can skip embedding.
    if (state.mode === 'dense' || state.mode === 'hybrid') {
      if (!activeVectorField) {
        setEmbedError('Pick a vector field for this query mode.')
        return
      }
      setEmbedding(true)
      try {
        const { data } = await embedQuery({
          text: state.text,
          model: activeVectorField.model,
        })
        if (
          activeVectorField.dimension &&
          data.dimension !== activeVectorField.dimension
        ) {
          setEmbedError(
            `Embedding dim ${data.dimension} does not match field dim ${activeVectorField.dimension}`,
          )
          setEmbedding(false)
          return
        }
        values = data.values
        setEmbedInfo(data)
      } catch (e) {
        setEmbedError(
          (e as { response?: { data?: { error?: string } } })?.response?.data
            ?.error ??
            (e instanceof Error ? e.message : 'Embedding failed'),
        )
        setEmbedding(false)
        return
      }
      setEmbedding(false)
    }

    const body = buildSearchRequest(state, values)
    if (!body) return
    search(state.collection, body)
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--color-bg)]">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Composer
          collection={state.collection}
          collections={collections.map((c) => c.name)}
          onCollectionChange={(v) => {
            setState({ ...DEFAULT_STATE(v) })
            clearResults()
          }}
          mode={state.mode}
          onModeChange={(m) => setS('mode', m)}
          text={state.text}
          onTextChange={(t) => setS('text', t)}
          onRun={handleEmbedAndSearch}
          loading={loading || embedding}
        />

        <div className="flex-1 overflow-auto">
          {embedError && (
            <Banner kind="error" icon={<AlertCircle size={12} strokeWidth={2.5} />}>
              {embedError}
            </Banner>
          )}
          {error && (
            <Banner kind="error" icon={<AlertCircle size={12} strokeWidth={2.5} />}>
              {error}
            </Banner>
          )}
          {embedInfo && (
            <EmbedInfoBar info={embedInfo} />
          )}
          {explain && state.explainOn && <ExplainBar explain={explain} tookMs={tookMs} />}
          {hybridExplain && state.explainOn && (
            <div className="px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-bg)]">
              <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-2">
                Hybrid retrieval · {hybridExplain.fusion} fusion
              </div>
              <HybridSankey data={hybridExplain} />
            </div>
          )}
          <ResultsArea
            loading={loading || embedding}
            results={results}
            schema={schema}
            mode={state.mode}
            text={state.text}
          />
        </div>
      </div>
      <KnobPanel
        state={state}
        setS={setS}
        schema={schema}
        vectorFields={vectorFields}
        stringFields={stringFields}
        onCopyCurl={() => copyCurl(state, embedInfo)}
        onSave={() => saveCurrentView(state)}
      />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Composer — text/image input + query-type tabs + run button
   ──────────────────────────────────────────────────────────────────── */

function Composer({
  collection,
  collections,
  onCollectionChange,
  mode,
  onModeChange,
  text,
  onTextChange,
  onRun,
  loading,
}: {
  collection: string
  collections: string[]
  onCollectionChange: (v: string) => void
  mode: QueryMode
  onModeChange: (m: QueryMode) => void
  text: string
  onTextChange: (t: string) => void
  onRun: () => void
  loading: boolean
}) {
  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      onRun()
    }
  }
  return (
    <div className="border-b border-[var(--color-border)] bg-[var(--color-bg)]">
      <div className="flex items-center gap-2 px-4 pt-3 pb-2">
        <Atom size={16} strokeWidth={2} className="text-[var(--color-accent)]" />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">Search</h1>
        <span className="text-[var(--color-text-subtle)]">·</span>
        <Select
          value={collection}
          onValueChange={onCollectionChange}
          placeholder="Select collection"
          options={collections.map((c) => ({ value: c, label: c }))}
        />
        <div className="ml-auto" />
        <ModeTabs value={mode} onChange={onModeChange} />
      </div>
      <div className="px-4 pb-3 flex items-start gap-2">
        <div className="flex-1">
          <Textarea
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={onKey}
            placeholder={placeholderFor(mode)}
            rows={2}
            className="resize-none min-h-[44px]"
          />
        </div>
        <Button
          variant="primary"
          size="md"
          onClick={onRun}
          disabled={loading || !collection || !text.trim()}
          leading={
            loading ? (
              <Sparkles size={12} strokeWidth={2.25} className="animate-pulse" />
            ) : (
              <Play size={11} strokeWidth={2.5} />
            )
          }
          trailing={
            <span className="text-[10px] font-mono opacity-70">⌘↵</span>
          }
        >
          {loading ? 'Running' : 'Run'}
        </Button>
      </div>
    </div>
  )
}

function ModeTabs({
  value,
  onChange,
}: {
  value: QueryMode
  onChange: (m: QueryMode) => void
}) {
  const modes: { id: QueryMode; label: string; icon: React.ReactNode; hint: string }[] = [
    { id: 'dense', label: 'Dense', icon: <Atom size={11} strokeWidth={2.25} />, hint: 'kNN over a vector field' },
    { id: 'sparse', label: 'Sparse', icon: <Zap size={11} strokeWidth={2.5} />, hint: 'SPLADE-style sparse retrieval' },
    { id: 'bm25', label: 'BM25', icon: <Type size={11} strokeWidth={2.25} />, hint: 'Lexical full-text' },
    { id: 'hybrid', label: 'Hybrid', icon: <Sparkles size={11} strokeWidth={2.25} />, hint: 'Dense + BM25 fused' },
  ]
  return (
    <Tabs.Root value={value} onValueChange={(v) => onChange(v as QueryMode)}>
      <Tabs.List className="flex items-center gap-0.5 p-0.5 rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)]">
        {modes.map((m) => (
          <Tooltip key={m.id} label={m.hint} side="bottom">
            <Tabs.Trigger
              value={m.id}
              className={cn(
                'inline-flex items-center gap-1 px-2 h-6 rounded-[var(--radius-xs)]',
                'text-[11.5px]',
                'data-[state=active]:bg-[var(--color-accent)] data-[state=active]:text-[var(--color-text-on-accent)]',
                'data-[state=inactive]:text-[var(--color-text-muted)] hover:data-[state=inactive]:text-[var(--color-text)]',
              )}
            >
              {m.icon}
              {m.label}
            </Tabs.Trigger>
          </Tooltip>
        ))}
      </Tabs.List>
    </Tabs.Root>
  )
}

function placeholderFor(m: QueryMode): string {
  switch (m) {
    case 'dense':
      return 'Type a natural-language query — Veda embeds it server-side.'
    case 'sparse':
      return 'Type a query — SPLADE inference produces the sparse vector.'
    case 'bm25':
      return 'Type lexical terms — BM25 ranks against the indexed string field.'
    case 'hybrid':
      return 'Type a query — dense vector + BM25 fused with RRF or weighted-sum.'
  }
}

/* ────────────────────────────────────────────────────────────────────────
   Knob panel — right rail
   ──────────────────────────────────────────────────────────────────── */

function KnobPanel({
  state,
  setS,
  schema,
  vectorFields,
  stringFields,
  onCopyCurl,
  onSave,
}: {
  state: PlaygroundState
  setS: <K extends keyof PlaygroundState>(k: K, v: PlaygroundState[K]) => void
  schema: CollectionInfo | null
  vectorFields: FieldSchema[]
  stringFields: FieldSchema[]
  onCopyCurl: () => void
  onSave: () => void
}) {
  const vf = vectorFields.find((f) => f.name === state.vectorField)
  const mrlOptions = vf?.dimension
    ? [vf.dimension, 512, 256, 128, 64].filter(
        (n, i, arr) =>
          n <= vf.dimension! && arr.indexOf(n) === i,
      )
    : []
  return (
    <aside className="flex flex-col w-[280px] flex-none border-l border-[var(--color-border)] bg-[var(--color-bg)] overflow-y-auto">
      <header className="px-3 py-2 border-b border-[var(--color-border)]">
        <h3 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Query knobs
        </h3>
      </header>

      <Section title="Fields">
        {(state.mode === 'dense' || state.mode === 'hybrid') && (
          <Knob label="Vector field">
            <Select
              size="sm"
              value={state.vectorField}
              onValueChange={(v) => setS('vectorField', v)}
              options={vectorFields.map((f) => ({
                value: f.name,
                label: f.name,
                description: `${f.dimension ?? '?'}d · ${f.metric ?? '?'}`,
              }))}
              placeholder="Pick a vector field"
            />
          </Knob>
        )}
        {(state.mode === 'bm25' || state.mode === 'hybrid') && (
          <Knob label="BM25 field">
            <Select
              size="sm"
              value={state.bm25Field}
              onValueChange={(v) => setS('bm25Field', v)}
              options={stringFields.map((f) => ({ value: f.name, label: f.name }))}
              placeholder="Pick a string field"
            />
          </Knob>
        )}
        {state.mode === 'sparse' && (
          <Knob label="Sparse field">
            <Select
              size="sm"
              value={state.vectorField}
              onValueChange={(v) => setS('vectorField', v)}
              options={
                schema?.fields
                  .filter((f) => f.type === 'sparse_vector')
                  .map((f) => ({ value: f.name, label: f.name })) ?? []
              }
              placeholder="Pick a sparse vector field"
            />
          </Knob>
        )}
      </Section>

      <Section title="Retrieval">
        <Knob label={`Top-K · ${state.topK}`}>
          <input
            type="range"
            min={1}
            max={100}
            value={state.topK}
            onChange={(e) => setS('topK', parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-accent)]"
          />
        </Knob>
        {(state.mode === 'dense' || state.mode === 'hybrid') && vf?.dimension && (
          <Knob label="Truncate to (MRL)">
            <Select
              size="sm"
              value={state.truncateTo == null ? 'full' : String(state.truncateTo)}
              onValueChange={(v) =>
                setS('truncateTo', v === 'full' ? null : parseInt(v, 10))
              }
              options={[
                { value: 'full', label: `Full · ${vf.dimension}d` },
                ...mrlOptions
                  .filter((d) => d !== vf.dimension)
                  .map((d) => ({ value: String(d), label: `${d}d` })),
              ]}
            />
          </Knob>
        )}
        <Knob label="Filter (Veda DSL)">
          <Input
            mono
            value={state.filter}
            onChange={(e) => setS('filter', e.target.value)}
            placeholder='field = "value" AND …'
          />
        </Knob>
      </Section>

      {state.mode === 'hybrid' && (
        <Section title="Fusion">
          <Knob label="Strategy">
            <Select
              size="sm"
              value={state.fusion}
              onValueChange={(v) => setS('fusion', v as 'rrf' | 'weighted')}
              options={[
                { value: 'rrf', label: 'RRF (reciprocal rank fusion)' },
                { value: 'weighted', label: 'Weighted sum' },
              ]}
            />
          </Knob>
          {state.fusion === 'rrf' && (
            <Knob label={`k · ${state.rrfK}`}>
              <input
                type="range"
                min={1}
                max={200}
                value={state.rrfK}
                onChange={(e) => setS('rrfK', parseInt(e.target.value, 10))}
                className="w-full accent-[var(--color-accent)]"
              />
            </Knob>
          )}
          {state.fusion === 'weighted' && (
            <>
              <Knob label={`Vector weight · ${state.vectorWeight.toFixed(2)}`}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={state.vectorWeight}
                  onChange={(e) => setS('vectorWeight', parseFloat(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </Knob>
              <Knob label={`BM25 weight · ${state.bm25Weight.toFixed(2)}`}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={state.bm25Weight}
                  onChange={(e) => setS('bm25Weight', parseFloat(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </Knob>
            </>
          )}
        </Section>
      )}

      <Section title="Rerank">
        <label className="flex items-center gap-2 text-[11.5px] text-[var(--color-text-muted)] mb-1.5">
          <input
            type="checkbox"
            checked={state.rerankOn}
            onChange={(e) => setS('rerankOn', e.target.checked)}
          />
          Re-rank top-K with a cross-encoder
        </label>
        {state.rerankOn && (
          <Knob label="Rerank field">
            <Select
              size="sm"
              value={state.rerankField}
              onValueChange={(v) => setS('rerankField', v)}
              options={stringFields.map((f) => ({ value: f.name, label: f.name }))}
              placeholder="Pick a string field"
            />
          </Knob>
        )}
      </Section>

      <Section title="Debug">
        <label className="flex items-center gap-2 text-[11.5px] text-[var(--color-text-muted)]">
          <input
            type="checkbox"
            checked={state.explainOn}
            onChange={(e) => setS('explainOn', e.target.checked)}
          />
          Show per-phase timing (explain)
        </label>
      </Section>

      <div className="mt-auto px-3 py-3 border-t border-[var(--color-border)] flex flex-col gap-1.5">
        <Button
          variant="secondary"
          size="sm"
          leading={<Save size={11} strokeWidth={2.25} />}
          onClick={onSave}
        >
          Save view
        </Button>
        <Button
          variant="ghost"
          size="sm"
          leading={<Copy size={11} strokeWidth={2.25} />}
          onClick={onCopyCurl}
        >
          Copy as cURL
        </Button>
      </div>
    </aside>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="px-3 py-2.5 border-b border-[var(--color-border)]">
      <h4 className="text-[9.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-2 font-medium">
        {title}
      </h4>
      <div className="space-y-2.5">{children}</div>
    </section>
  )
}

function Knob({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10.5px] text-[var(--color-text-muted)] mb-1">
        {label}
      </label>
      {children}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Banner / Explain / Embed-info
   ──────────────────────────────────────────────────────────────────── */

function Banner({
  kind,
  icon,
  children,
}: {
  kind: 'error' | 'info'
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        'mx-4 mt-3 flex items-center gap-2 px-3 py-2 rounded-[var(--radius-sm)]',
        'text-[12px]',
        kind === 'error' &&
          'bg-[color-mix(in_oklch,var(--color-error)_10%,transparent)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] text-[var(--color-error)]',
        kind === 'info' &&
          'bg-[var(--color-bg-elevated)] border border-[var(--color-border)] text-[var(--color-text-muted)]',
      )}
    >
      {icon}
      <span className="flex-1">{children}</span>
    </div>
  )
}

function EmbedInfoBar({ info }: { info: EmbedResponse }) {
  return (
    <div className="mx-4 mt-3 flex flex-wrap items-center gap-2 text-[10.5px] font-mono text-[var(--color-text-subtle)]">
      <Badge label="embed" variant="ghost" dot />
      <span>{info.model}</span>
      <span className="text-[var(--color-border-strong)]">·</span>
      <span>{info.dimension}d</span>
      <span className="text-[var(--color-border-strong)]">·</span>
      <span className="text-[var(--color-data-7)]">
        {info.elapsed_ms.toFixed(1)}ms
      </span>
    </div>
  )
}

function ExplainBar({
  explain,
  tookMs,
}: {
  explain: ExplainTiming
  tookMs: number
}) {
  const cells: { label: string; ms: number; highlight?: boolean }[] = [
    { label: 'Filter', ms: explain.filter_ms },
    { label: 'ANN', ms: explain.ann_search_ms },
    { label: 'Score', ms: explain.scoring_ms },
    { label: 'Fuse', ms: explain.fusion_ms },
    { label: 'Total', ms: explain.total_ms || tookMs, highlight: true },
  ]
  return (
    <div className="mx-4 mt-3 flex items-stretch rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] overflow-hidden divide-x divide-[var(--color-border)]">
      {cells.map((c) => (
        <div key={c.label} className="px-3 py-1.5 flex-1">
          <div className="text-[9.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-0.5">
            {c.label}
          </div>
          <div
            className={cn(
              'text-[12px] font-mono tabular-nums',
              c.highlight ? 'text-[var(--color-accent-9)] font-semibold' : 'text-[var(--color-text)]',
            )}
          >
            {c.ms.toFixed(2)}ms
          </div>
        </div>
      ))}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Results
   ──────────────────────────────────────────────────────────────────── */

function ResultsArea({
  loading,
  results,
  schema,
  mode,
  text,
}: {
  loading: boolean
  results: SearchHit[]
  schema: CollectionInfo | null
  mode: QueryMode
  text: string
}) {
  if (loading) {
    return (
      <div className="px-4 py-4 space-y-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    )
  }
  if (results.length === 0) {
    return <EmptyResults />
  }
  return (
    <div className="px-4 py-3 space-y-1.5">
      {results.map((hit, i) => (
        <ResultRow
          key={hit.id + i}
          hit={hit}
          rank={i + 1}
          schema={schema}
          mode={mode}
          queryText={text}
        />
      ))}
    </div>
  )
}

function EmptyResults() {
  return (
    <div className="relative h-[400px] m-4 rounded-[var(--radius-md)] border border-dashed border-[var(--color-border-strong)] flex items-center justify-center">
      <div className="dot-grid absolute inset-0 opacity-30 rounded-[var(--radius-md)]" aria-hidden />
      <div className="relative text-center max-w-md px-4">
        <h2 className="text-[14px] font-medium text-[var(--color-text)] mb-1">
          No results yet
        </h2>
        <p className="text-[12px] text-[var(--color-text-muted)]">
          Type a query above and press <span className="font-mono text-[11px]">⌘↵</span>{' '}
          to run. Each query type calls the right Veda endpoint —
          dense → kNN, sparse → SPLADE, BM25 → lexical, hybrid → RRF
          fusion.
        </p>
      </div>
    </div>
  )
}

function ResultRow({
  hit,
  rank,
  schema,
  mode,
  queryText,
}: {
  hit: SearchHit
  rank: number
  schema: CollectionInfo | null
  mode: QueryMode
  queryText: string
}) {
  const [open, setOpen] = useState(false)
  const stringField = schema?.fields.find(
    (f) => f.type === 'string' && hit.fields[f.name],
  )
  const snippet = stringField
    ? String(hit.fields[stringField.name] ?? '').slice(0, 280)
    : Object.entries(hit.fields)
        .filter(([, v]) => typeof v !== 'object')
        .slice(0, 3)
        .map(([k, v]) => `${k}: ${v}`)
        .join('  ·  ')

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => setOpen((v) => !v)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          setOpen((v) => !v)
        }
      }}
      className={cn(
        'group w-full text-left rounded-[var(--radius-sm)] border border-[var(--color-border)]',
        'bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-strong)]',
        'transition-[border-color,background] duration-[var(--duration-fast)]',
        'p-3 cursor-pointer',
      )}
    >
      <div className="flex items-start gap-3">
        <span className="flex-none w-7 text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums pt-0.5">
          {rank.toString().padStart(2, '0')}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 mb-1">
            <span className="font-mono text-[11px] text-[var(--color-text-muted)] truncate">
              {hit.id}
            </span>
            <DistanceBadge distance={hit.distance} />
            <span
              onClick={(e) => e.stopPropagation()}
              className="ml-auto opacity-0 group-hover:opacity-100"
            >
              <CopyButton value={hit.id} />
            </span>
          </div>
          <div
            className="text-[12.5px] text-[var(--color-text)] leading-relaxed line-clamp-2"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{
              __html: highlightSnippet(snippet, queryText, mode),
            }}
          />
        </div>
      </div>
      {open && (
        <pre className="mt-2 ml-10 font-mono text-[11px] text-[var(--color-text-muted)] whitespace-pre-wrap break-words bg-[var(--color-base-1)] rounded-[var(--radius-xs)] p-2 border border-[var(--color-border)]">
          {JSON.stringify(hit, null, 2)}
        </pre>
      )}
    </div>
  )
}

function DistanceBadge({ distance }: { distance: number }) {
  let variant: 'success' | 'warning' | 'error' = 'success'
  if (distance >= 0.7) variant = 'error'
  else if (distance >= 0.3) variant = 'warning'
  return <Badge label={distance.toFixed(4)} variant={variant} mono />
}

/* ────────────────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────────────── */

function buildSearchRequest(
  state: PlaygroundState,
  vectorValues: number[] | null,
): SearchRequest | null {
  const body: SearchRequest = {
    query: {},
    limit: state.topK,
  }
  if (state.filter.trim()) body.filter = state.filter
  if (state.explainOn) body.explain = true
  if (state.truncateTo) body.truncate_to = state.truncateTo
  if (state.rerankOn && state.rerankField) {
    body.rerank = {
      field: state.rerankField,
      query: state.text,
      top_k: state.topK,
    }
  }

  switch (state.mode) {
    case 'dense':
      if (!vectorValues || !state.vectorField) return null
      body.query.vector = { field: state.vectorField, values: vectorValues }
      return body
    case 'bm25':
      if (!state.bm25Field) return null
      body.query.bm25 = { field: state.bm25Field, query: state.text }
      return body
    case 'sparse':
      // Sparse queries currently require client-side SPLADE
      // encoding; until Phase 2 adds an /embed-sparse companion we
      // skip the request and let the page show an empty-state.
      toast.error(
        'Sparse playground composer needs the upcoming /v1/embed-sparse endpoint. Pick BM25 or Hybrid for now.',
      )
      return null
    case 'hybrid':
      if (!vectorValues || !state.vectorField || !state.bm25Field) return null
      // UI exposes "rrf" | "weighted" — the gateway speaks
      // "rrf" | "weighted_sum". Translate at the boundary.
      body.query.hybrid = {
        dense: { field: state.vectorField, values: vectorValues },
        bm25: { field: state.bm25Field, query: state.text },
        dense_weight:
          state.fusion === 'weighted' ? state.vectorWeight : 0.5,
        fusion: state.fusion === 'weighted' ? 'weighted_sum' : 'rrf',
      }
      return body
  }
}

function highlightSnippet(
  snippet: string,
  query: string,
  mode: QueryMode,
): string {
  const escaped = escapeHtml(snippet)
  if (!query.trim() || mode === 'dense') return escaped
  // For BM25 / sparse / hybrid, lit up the matched tokens.
  const tokens = query
    .toLowerCase()
    .split(/\W+/)
    .filter((t) => t.length >= 2)
  if (tokens.length === 0) return escaped
  const re = new RegExp(`(${tokens.map(escapeRegex).join('|')})`, 'gi')
  return escaped.replace(
    re,
    (m) =>
      `<mark class="bg-[color-mix(in_oklch,var(--color-accent)_30%,transparent)] text-[var(--color-text)] rounded-[2px] px-0.5">${m}</mark>`,
  )
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function copyCurl(state: PlaygroundState, embed: EmbedResponse | null) {
  if (!state.collection) {
    toast.error('Pick a collection first.')
    return
  }
  const body = buildSearchRequest(state, embed?.values ?? null)
  if (!body) {
    toast.error('Run a query first so cURL captures the actual values.')
    return
  }
  const cmd = `curl -sS -X POST 'http://localhost:6340/v1/collections/${state.collection}/search' \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(body, null, 2).replace(/'/g, "'\\''")}'`
  navigator.clipboard.writeText(cmd)
  toast.success('cURL copied')
}

function saveCurrentView(state: PlaygroundState) {
  const name = window.prompt('Name this search')
  if (!name) return
  useSavedObjects.getState().upsert({
    id: '',
    type: 'search',
    name,
    body: state,
  })
  toast.success(`Saved search "${name}"`)
}
