import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight,
  ChevronRight,
  CircleStop,
  FileText,
  FlaskConical,
  GitBranch,
  Plus,
  Play,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import {
  useSavedObjects,
  type SavedObject,
} from '../../stores/savedObjectStore'
import {
  Badge,
  Button,
  Input,
  Modal,
  Select,
  toast,
} from '../../components/primitives'
import { useUrlState } from '../../hooks/useUrlState'
import { fetchCollection, llmComplete } from '../../api/collections'
import type { FieldSchema } from '../../api/collections'
import { cn } from '../../lib/cn'
import { parseEvalSet } from './parseEvalSet'
import { runEval, type RetrievalConfig, type RunnerEvent } from './runner'
import type {
  EvalItem,
  EvalRunBody,
  EvalSetBody,
  PerQueryResult,
} from './types'
import { CompareRuns } from './CompareRuns'
import { RouterExplorer } from './RouterExplorer'
import { Campaigns } from './Campaigns'

interface QualityLabUrlState {
  setId?: string
  runId?: string
  compareId?: string
  /** When true, the active set is shown in the Router-decision view. */
  router?: boolean
  /** When true, the active set is shown in the Campaigns view. */
  campaigns?: boolean
}

export function QualityLab() {
  const [state, setState] = useUrlState<QualityLabUrlState>('q', {})

  const getSaved = useSavedObjects((s) => s.get)
  const objectsMap = useSavedObjects((s) => s.objects)
  const sets = useMemo<SavedObject<EvalSetBody>[]>(
    () =>
      Object.values(objectsMap)
        .filter((o): o is SavedObject<EvalSetBody> => o.type === 'eval')
        .sort(
          (a, b) => (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
        ),
    [objectsMap],
  )
  const runs = useMemo<SavedObject<EvalRunBody>[]>(
    () =>
      Object.values(objectsMap)
        .filter((o): o is SavedObject<EvalRunBody> => o.type === 'eval_run')
        .sort((a, b) => b.createdAt - a.createdAt),
    [objectsMap],
  )

  const [showCreate, setShowCreate] = useState(false)

  const activeSet = state.setId
    ? getSaved<EvalSetBody>(state.setId)
    : undefined
  const activeRun = state.runId
    ? (objectsMap[state.runId] as SavedObject<EvalRunBody> | undefined)
    : undefined
  const compareRun = state.compareId
    ? (objectsMap[state.compareId] as SavedObject<EvalRunBody> | undefined)
    : undefined

  if (activeRun && compareRun) {
    return (
      <CompareRuns
        a={activeRun}
        b={compareRun}
        onClose={() => setState({ ...state, compareId: undefined })}
      />
    )
  }

  if (state.router && activeSet) {
    return (
      <RouterExplorer
        set={activeSet}
        runs={runs.filter((r) => r.body.evalSetId === activeSet.id)}
        onClose={() => setState({ ...state, router: undefined })}
      />
    )
  }

  if (state.campaigns && activeSet) {
    return (
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
        <div className="flex items-center gap-2 border-b border-[var(--color-border)] px-4 py-2">
          <Button
            size="xs"
            variant="ghost"
            onClick={() => setState({ ...state, campaigns: undefined })}
          >
            ← Back
          </Button>
        </div>
        <div className="flex-1 min-h-0">
          <Campaigns
            collection={activeSet.body.collection}
            evalSetIds={sets
              .filter((s) => s.body.collection === activeSet.body.collection)
              .map((s) => s.id)}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--color-bg)] overflow-hidden">
      <SetsRail
        sets={sets}
        activeId={state.setId}
        onSelect={(id) =>
          setState({ ...state, setId: id, runId: undefined, compareId: undefined })
        }
        onCreate={() => setShowCreate(true)}
      />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {activeSet ? (
          <SetPanel
            set={activeSet}
            runs={runs.filter((r) => r.body.evalSetId === activeSet.id)}
            activeRunId={state.runId}
            onRunSelect={(id) => setState({ ...state, runId: id })}
            onCompare={(otherId) =>
              setState({
                ...state,
                runId: state.runId ?? otherId,
                compareId:
                  state.runId && state.runId !== otherId ? otherId : undefined,
              })
            }
            onRouter={() => setState({ ...state, router: true })}
            onCampaigns={() => setState({ ...state, campaigns: true })}
          />
        ) : (
          <EmptyMain onCreate={() => setShowCreate(true)} hasSets={sets.length > 0} />
        )}
      </div>
      <RunDetail run={activeRun} />
      <CreateEvalSet
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={(id) => {
          setState({ setId: id })
          setShowCreate(false)
        }}
      />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Left rail — eval sets list
   ──────────────────────────────────────────────────────────────────── */

function SetsRail({
  sets,
  activeId,
  onSelect,
  onCreate,
}: {
  sets: SavedObject<EvalSetBody>[]
  activeId?: string
  onSelect: (id: string) => void
  onCreate: () => void
}) {
  return (
    <aside className="w-[260px] flex-none border-r border-[var(--color-border)] bg-[var(--color-bg)] flex flex-col min-h-0">
      <header className="flex items-center gap-2 px-3 h-12 border-b border-[var(--color-border)] flex-none">
        <FlaskConical
          size={14}
          strokeWidth={2}
          className="text-[var(--color-accent-9)]"
        />
        <h1 className="text-[13px] font-medium text-[var(--color-text)] flex-1">
          Quality Lab
        </h1>
        <Button
          size="xs"
          variant="primary"
          leading={<Plus size={11} strokeWidth={2.5} />}
          onClick={onCreate}
        >
          New
        </Button>
      </header>
      <div className="px-3 pt-2.5 pb-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
        Eval sets
        <span className="ml-1.5 font-mono text-[var(--color-text-subtle)]">
          {sets.length}
        </span>
      </div>
      <ul className="flex-1 overflow-y-auto">
        {sets.length === 0 && (
          <li className="px-3 py-2 text-[11.5px] text-[var(--color-text-subtle)] italic">
            None yet. Create your first to run an eval.
          </li>
        )}
        {sets.map((s) => (
          <li key={s.id}>
            <button
              onClick={() => onSelect(s.id)}
              className={cn(
                'group w-full flex flex-col gap-0.5 px-3 py-2 text-left border-l-2 border-transparent',
                'hover:bg-[var(--color-bg-hover)] transition-colors',
                activeId === s.id &&
                  'bg-[var(--color-bg-hover)] border-[var(--color-accent)]',
              )}
            >
              <span className="text-[12.5px] text-[var(--color-text)] truncate">
                {s.name}
              </span>
              <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
                {s.body.items.length} q · {s.body.collection}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Center — selected set + items + recent runs + Run button
   ──────────────────────────────────────────────────────────────────── */

function SetPanel({
  set,
  runs,
  activeRunId,
  onRunSelect,
  onCompare,
  onRouter,
  onCampaigns,
}: {
  set: SavedObject<EvalSetBody>
  runs: SavedObject<EvalRunBody>[]
  activeRunId?: string
  onRunSelect: (id: string) => void
  onCompare: (id: string) => void
  onRouter: () => void
  onCampaigns: () => void
}) {
  const { upsert, remove, touch } = useSavedObjects.getState()
  const [schema, setSchema] = useState<FieldSchema[] | undefined>()
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [streamingResults, setStreamingResults] = useState<PerQueryResult[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const [vectorField, setVectorField] = useState('')
  const [topK, setTopK] = useState(10)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    setRunning(false)
    setProgress({ done: 0, total: 0 })
    setStreamingResults([])
    abortRef.current?.abort()
    abortRef.current = null
    setSchema(undefined)
    fetchCollection(set.body.collection)
      .then(({ data }) => {
        setSchema(data.fields)
        const firstVec = data.fields.find((f) => f.type === 'vector')
        setVectorField((prev) => prev || firstVec?.name || '')
      })
      .catch(() => setSchema([]))
  }, [set.id, set.body.collection])

  const vectorFields = useMemo(
    () => (schema ?? []).filter((f) => f.type === 'vector'),
    [schema],
  )

  const start = async () => {
    if (!vectorField) {
      toast.error('Pick a vector field first.')
      return
    }
    const config: RetrievalConfig = {
      collection: set.body.collection,
      vectorField,
      topK,
      mode: 'dense',
    }
    if (filter.trim()) config.filter = filter.trim()
    const vf = vectorFields.find((f) => f.name === vectorField)
    if (vf?.model) config.model = vf.model

    abortRef.current = new AbortController()
    setRunning(true)
    setStreamingResults([])
    setProgress({ done: 0, total: set.body.items.length })
    const startedAt = Date.now()

    const final: { results: PerQueryResult[]; summary?: EvalRunBody['summary'] } = {
      results: [],
    }

    await runEval(
      set.body.items,
      config,
      schema,
      (e: RunnerEvent) => {
        if (e.type === 'progress') setProgress({ done: e.done, total: e.total })
        if (e.type === 'item')
          setStreamingResults((r) => [...r, e.result])
        if (e.type === 'done') {
          final.results = e.perQuery
          final.summary = e.summary
        }
        if (e.type === 'error') toast.error(e.message)
      },
      abortRef.current.signal,
    )
    setRunning(false)
    abortRef.current = null
    touch(set.id)

    if (final.summary && final.results.length > 0) {
      const summary = final.summary
      const run = upsert<EvalRunBody>({
        id: '',
        type: 'eval_run',
        name: `${set.name} · ${new Date(startedAt).toLocaleString([], {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        })}`,
        body: {
          evalSetId: set.id,
          evalSetName: set.name,
          collection: set.body.collection,
          searchConfig: config,
          topK,
          perQuery: final.results,
          summary,
          startedAt,
          finishedAt: Date.now(),
        },
      })
      onRunSelect(run.id)
      toast.success(`Run saved · MRR ${summary.mrr.toFixed(3)}`)
    }
  }

  const abort = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setRunning(false)
  }

  return (
    <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <FlaskConical
          size={13}
          strokeWidth={2}
          className="text-[var(--color-accent-9)]"
        />
        <span className="text-[13.5px] font-medium text-[var(--color-text)] truncate">
          {set.name}
        </span>
        <Badge label={`${set.body.items.length} q`} variant="ghost" />
        <Badge label={set.body.collection} variant="default" />
        <Button
          size="xs"
          variant="ghost"
          leading={<GitBranch size={11} strokeWidth={2.25} />}
          onClick={onRouter}
          className="ml-auto"
        >
          Router decisions
        </Button>
        <Button
          size="xs"
          variant="ghost"
          leading={<Sparkles size={11} strokeWidth={2.25} />}
          onClick={onCampaigns}
        >
          Campaigns
        </Button>
        <Button
          size="xs"
          variant="ghost"
          leading={<Trash2 size={11} strokeWidth={2.5} />}
          onClick={() => {
            if (confirm(`Delete eval set "${set.name}"?`)) {
              remove(set.id)
            }
          }}
          className="text-[var(--color-text-subtle)] hover:text-[var(--color-error)]"
        >
          Delete
        </Button>
      </header>

      {/* Run config bar */}
      <div className="px-4 py-2.5 border-b border-[var(--color-border)] bg-[var(--color-bg)] flex-none flex items-end gap-2">
        <div className="flex-1 grid grid-cols-3 gap-2">
          <Knob label="Vector field">
            <Select
              size="sm"
              value={vectorField}
              onValueChange={setVectorField}
              options={vectorFields.map((f) => ({
                value: f.name,
                label: f.name,
                description: `${f.dimension ?? '?'}d · ${f.metric ?? '?'}`,
              }))}
              placeholder="pick"
            />
          </Knob>
          <Knob label={`Top-K · ${topK}`}>
            <input
              type="range"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              className="w-full accent-[var(--color-accent)]"
            />
          </Knob>
          <Knob label="Filter (DSL)">
            <Input
              mono
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder='field = "value"'
            />
          </Knob>
        </div>
        {running ? (
          <Button
            variant="destructive"
            size="sm"
            leading={<CircleStop size={11} strokeWidth={2.5} />}
            onClick={abort}
          >
            Abort {progress.done}/{progress.total}
          </Button>
        ) : (
          <Button
            variant="primary"
            size="sm"
            leading={<Play size={11} strokeWidth={2.5} />}
            onClick={start}
            disabled={!vectorField}
          >
            Run eval
          </Button>
        )}
      </div>

      {/* Streaming progress + items */}
      <div className="flex-1 min-h-0 grid grid-rows-[auto_1fr] overflow-hidden">
        {running && (
          <div className="px-4 py-2 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
            <div className="flex items-baseline justify-between text-[11px] font-mono tabular-nums">
              <span className="text-[var(--color-text-muted)]">
                <Sparkles
                  size={10}
                  strokeWidth={2.25}
                  className="inline mr-1 animate-pulse text-[var(--color-accent)]"
                />
                Running query {progress.done + 1} of {progress.total}
              </span>
              <span className="text-[var(--color-text-subtle)]">
                {progress.total > 0
                  ? `${Math.round((progress.done / progress.total) * 100)}%`
                  : ''}
              </span>
            </div>
            <div className="mt-1 h-0.5 bg-[var(--color-base-2)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--color-accent)] transition-[width] duration-[var(--duration-base)]"
                style={{
                  width:
                    progress.total > 0
                      ? `${(progress.done / progress.total) * 100}%`
                      : '0%',
                }}
              />
            </div>
          </div>
        )}
        <ItemsTable
          items={set.body.items}
          streamingResults={streamingResults}
        />
      </div>

      {/* Recent runs */}
      <RunsList
        runs={runs}
        activeRunId={activeRunId}
        onSelect={onRunSelect}
        onCompare={onCompare}
      />
    </div>
  )
}

function Knob({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-[10px] text-[var(--color-text-subtle)] uppercase tracking-wider mb-0.5">
        {label}
      </label>
      {children}
    </div>
  )
}

function ItemsTable({
  items,
  streamingResults,
}: {
  items: EvalItem[]
  streamingResults: PerQueryResult[]
}) {
  const lookup = useMemo(() => {
    const m = new Map<string, PerQueryResult>()
    for (const r of streamingResults) m.set(r.query, r)
    return m
  }, [streamingResults])
  return (
    <div className="overflow-auto h-full">
      <table className="w-full text-[12px] tabular-nums">
        <thead className="sticky top-0 bg-[var(--color-bg)] border-b border-[var(--color-border)]">
          <tr className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
            <th className="w-9 px-2 py-1.5 text-left font-medium">#</th>
            <th className="px-3 py-1.5 text-left font-medium">Query</th>
            <th className="w-16 px-2 py-1.5 text-right font-medium">Expected</th>
            <th className="w-16 px-2 py-1.5 text-right font-medium">Hits</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">Recall</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">RR</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">Took</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--color-border)]">
          {items.map((it, i) => {
            const r = lookup.get(it.query)
            return (
              <tr key={i} className="hover:bg-[var(--color-bg-hover)]">
                <td className="px-2 py-1 text-[var(--color-text-subtle)] font-mono text-[10.5px]">
                  {(i + 1).toString().padStart(3, '0')}
                </td>
                <td className="px-3 py-1 text-[var(--color-text)] truncate max-w-[400px]">
                  {it.query}
                </td>
                <td className="px-2 py-1 text-right text-[var(--color-text-muted)] font-mono text-[10.5px]">
                  {it.expected_doc_ids?.length ?? '—'}
                </td>
                <td className="px-2 py-1 text-right font-mono text-[10.5px]">
                  {r ? <HitCell hits={r.hits} expected={r.expected.length} /> : '—'}
                </td>
                <td className="px-2 py-1 text-right font-mono text-[10.5px]">
                  {r ? r.recall.toFixed(3) : '—'}
                </td>
                <td className="px-2 py-1 text-right font-mono text-[10.5px]">
                  {r ? r.rr.toFixed(3) : '—'}
                </td>
                <td className="px-2 py-1 text-right font-mono text-[10.5px] text-[var(--color-text-muted)]">
                  {r ? `${r.took_ms.toFixed(0)}ms` : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function HitCell({ hits, expected }: { hits: number; expected: number }) {
  if (expected === 0)
    return <span className="text-[var(--color-text-subtle)]">—</span>
  return (
    <span
      className={cn(
        hits === expected && 'text-[var(--color-success)]',
        hits === 0 && 'text-[var(--color-error)]',
        hits > 0 && hits < expected && 'text-[var(--color-warning)]',
      )}
    >
      {hits}/{expected}
    </span>
  )
}

function RunsList({
  runs,
  activeRunId,
  onSelect,
  onCompare,
}: {
  runs: SavedObject<EvalRunBody>[]
  activeRunId?: string
  onSelect: (id: string) => void
  onCompare: (id: string) => void
}) {
  if (runs.length === 0) return null
  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-elevated)] flex-none">
      <div className="px-4 pt-2 pb-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
        Runs · {runs.length}
      </div>
      <ul className="max-h-44 overflow-y-auto">
        {runs.map((r) => (
          <li key={r.id}>
            <div
              className={cn(
                'flex items-center gap-3 px-4 h-7 text-[11.5px] font-mono tabular-nums',
                'hover:bg-[var(--color-bg-hover)] cursor-pointer',
                activeRunId === r.id && 'bg-[var(--color-bg-hover)]',
              )}
              onClick={() => onSelect(r.id)}
            >
              <span className="text-[var(--color-text-subtle)] w-32 truncate">
                {new Date(r.body.startedAt).toLocaleString([], {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
              <Metric label="mrr" value={r.body.summary.mrr.toFixed(3)} />
              <Metric
                label="recall"
                value={r.body.summary.recall_mean.toFixed(3)}
              />
              <Metric
                label="hit@1"
                value={(r.body.summary.hit_at_1_pct * 100).toFixed(0) + '%'}
              />
              <Metric
                label="p95"
                value={`${r.body.summary.p95_ms.toFixed(0)}ms`}
              />
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onCompare(r.id)
                }}
                className="ml-auto text-[var(--color-text-subtle)] hover:text-[var(--color-accent)] inline-flex items-center gap-0.5 text-[10.5px]"
                title="A/B compare with the currently-selected run"
              >
                compare
                <ArrowRight size={10} strokeWidth={2.5} />
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1">
      <span className="text-[var(--color-text-subtle)] text-[9.5px] uppercase tracking-wider">
        {label}
      </span>
      <span className="text-[var(--color-text)]">{value}</span>
    </span>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Right — run detail
   ──────────────────────────────────────────────────────────────────── */

function RunDetail({ run }: { run: SavedObject<EvalRunBody> | undefined }) {
  const [explanations, setExplanations] = useState<Map<string, string>>(new Map())
  const [explaining, setExplaining] = useState<Set<string>>(new Set())

  // Reset state when the active run changes — explanations are run-scoped.
  useEffect(() => {
    setExplanations(new Map())
    setExplaining(new Set())
  }, [run?.id])

  if (!run) {
    return (
      <aside className="w-[300px] flex-none border-l border-[var(--color-border)] bg-[var(--color-bg)] flex flex-col">
        <header className="px-3 h-12 border-b border-[var(--color-border)] flex items-center">
          <span className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
            Run details
          </span>
        </header>
        <div className="flex-1 flex items-center justify-center px-4 text-center">
          <p className="text-[11.5px] text-[var(--color-text-subtle)] italic">
            Pick a run from the list below to see per-query metrics.
          </p>
        </div>
      </aside>
    )
  }
  const s = run.body.summary
  return (
    <aside className="w-[300px] flex-none border-l border-[var(--color-border)] bg-[var(--color-bg)] flex flex-col overflow-y-auto">
      <header className="px-3 h-12 border-b border-[var(--color-border)] flex items-center gap-2">
        <FileText
          size={12}
          strokeWidth={2.25}
          className="text-[var(--color-text-subtle)]"
        />
        <span className="text-[12px] font-medium text-[var(--color-text)] truncate">
          {run.name}
        </span>
      </header>
      <section className="px-3 py-3 border-b border-[var(--color-border)] grid grid-cols-2 gap-3 text-[12px]">
        <Stat label="MRR" value={s.mrr.toFixed(3)} highlight />
        <Stat label="Recall@k" value={s.recall_mean.toFixed(3)} />
        <Stat
          label="Precision@k"
          value={s.precision_mean.toFixed(3)}
        />
        <Stat
          label="Hit@1"
          value={`${(s.hit_at_1_pct * 100).toFixed(1)}%`}
        />
        <Stat label="p50" value={`${s.p50_ms.toFixed(0)}ms`} />
        <Stat label="p95" value={`${s.p95_ms.toFixed(0)}ms`} />
        <Stat label="n" value={String(s.n)} />
        <Stat label="failed" value={String(s.failed)} />
      </section>
      <section className="px-3 py-2 border-b border-[var(--color-border)]">
        <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1">
          Per-query
        </div>
        <ul className="space-y-1 max-h-[460px] overflow-y-auto">
          {run.body.perQuery
            .slice()
            .sort((a, b) => a.rr - b.rr)
            .map((r, i) => (
              <PerQueryItem
                key={i}
                r={r}
                explanation={explanations.get(r.query)}
                explaining={explaining.has(r.query)}
                onExplain={async () => {
                  setExplaining((prev) => {
                    const next = new Set(prev)
                    next.add(r.query)
                    return next
                  })
                  const text = await classifyFailure(r)
                  setExplanations((prev) => {
                    const next = new Map(prev)
                    next.set(r.query, text)
                    return next
                  })
                  setExplaining((prev) => {
                    const next = new Set(prev)
                    next.delete(r.query)
                    return next
                  })
                }}
              />
            ))}
        </ul>
      </section>
    </aside>
  )
}

function PerQueryItem({
  r,
  explanation,
  explaining,
  onExplain,
}: {
  r: PerQueryResult
  explanation?: string
  explaining: boolean
  onExplain: () => void
}) {
  const dotColor = r.error
    ? 'bg-[var(--color-error)]'
    : r.rr >= 0.5
      ? 'bg-[var(--color-success)]'
      : r.rr > 0
        ? 'bg-[var(--color-warning)]'
        : 'bg-[var(--color-text-subtle)]'
  // Offer the classifier for anything not at rank 1 (rr < 1) and for
  // errors. Perfect-rank hits don't need an explanation.
  const couldClassify = r.error || r.rr < 1
  return (
    <li className="group flex flex-col gap-0.5 text-[11px] tabular-nums">
      <div className="flex items-center gap-2 font-mono">
        <span className={cn('inline-block w-1.5 h-1.5 rounded-full flex-none', dotColor)} />
        <span className="flex-1 truncate text-[var(--color-text-muted)]">
          {r.query}
        </span>
        <span className="text-[var(--color-text-subtle)] w-10 text-right">
          rr {r.rr.toFixed(2)}
        </span>
        {couldClassify && !explanation && (
          <button
            onClick={onExplain}
            disabled={explaining}
            title="Ask LLM to classify the failure mode"
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-accent)] disabled:text-[var(--color-text-disabled)]"
          >
            {explaining ? '…' : '🪄'}
          </button>
        )}
      </div>
      {explanation && (
        <div className="ml-3.5 text-[10.5px] font-sans text-[var(--color-accent-11)] italic leading-snug">
          {explanation}
        </div>
      )}
    </li>
  )
}

/**
 * Ask the LLM to classify the failure mode for one per-query result.
 * Returns a short label (≤80 chars) — "vocabulary mismatch", "no gold
 * present in corpus", "ambiguous query", "retrieval scored wrong
 * chunk first", etc.
 */
async function classifyFailure(r: PerQueryResult): Promise<string> {
  const expected = r.expected.length > 0 ? r.expected.join(', ') : '(none)'
  const retrieved =
    r.retrieved_ids.length > 0
      ? r.retrieved_ids.slice(0, 5).join(', ')
      : '(none — empty result set)'
  try {
    const { data } = await llmComplete(
      [
        {
          role: 'system',
          content:
            'You diagnose RAG retrieval failures in 6–12 words. ' +
            'Pick the single most likely failure mode (vocabulary ' +
            'mismatch, ambiguous query, missing-from-corpus, wrong ' +
            'first-rank, multi-hop, etc) and state it tersely. No ' +
            'preamble, no period at end.',
        },
        {
          role: 'user',
          content:
            `Query: ${r.query}\n` +
            `Expected doc ids: ${expected}\n` +
            `Top-K retrieved: ${retrieved}\n` +
            `RR=${r.rr.toFixed(3)} hits=${r.hits} took=${r.took_ms.toFixed(0)}ms` +
            (r.error ? `\nError: ${r.error}` : ''),
        },
      ],
      { temperature: 0.2, max_tokens: 48 },
    )
    return data.content.trim().replace(/^["'`]+|["'`.]+$/g, '').slice(0, 120)
  } catch (e) {
    return e instanceof Error ? `(LLM error: ${e.message})` : 'LLM call failed'
  }
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
        {label}
      </div>
      <div
        className={cn(
          'font-mono tabular-nums',
          highlight
            ? 'text-[15px] text-[var(--color-accent-9)] font-semibold'
            : 'text-[13px] text-[var(--color-text)]',
        )}
      >
        {value}
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Create eval-set modal
   ──────────────────────────────────────────────────────────────────── */

function CreateEvalSet({
  open,
  onClose,
  onCreated,
}: {
  open: boolean
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const { collections, fetchCollections } = useCollectionStore()
  const { upsert } = useSavedObjects.getState()
  const [name, setName] = useState('')
  const [collection, setCollection] = useState('')
  const [filename, setFilename] = useState('')
  const [items, setItems] = useState<EvalItem[]>([])
  const [parseError, setParseError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      fetchCollections()
      if (!collection && collections.length > 0) {
        setCollection(collections[0]!.name)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const tryParse = (text: string, fname: string) => {
    setFilename(fname)
    setParseError(null)
    try {
      setItems(parseEvalSet(text, fname))
    } catch (e) {
      setItems([])
      setParseError(e instanceof Error ? e.message : 'Failed to parse')
    }
  }

  const submit = () => {
    if (!name.trim() || !collection || items.length === 0) return
    const created = upsert<EvalSetBody>({
      id: '',
      type: 'eval',
      name: name.trim(),
      body: {
        collection,
        items,
        source: filename ? 'upload' : 'manual',
      },
    })
    setName('')
    setFilename('')
    setItems([])
    onCreated(created.id)
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="New eval set"
      description="Upload a CSV/JSON/JSONL with a `query` column (and optional `expected_doc_ids`). Each row becomes one eval item."
      width={560}
    >
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
              Name
            </label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="rag_eval_2026_05_18"
              mono
            />
          </div>
          <div>
            <label className="block text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
              Target collection
            </label>
            <Select
              value={collection}
              onValueChange={setCollection}
              options={collections.map((c) => ({ value: c.name, label: c.name }))}
              placeholder="pick"
            />
          </div>
        </div>

        <div
          className={cn(
            'rounded-[var(--radius-md)] border border-dashed border-[var(--color-border-strong)]',
            'bg-[var(--color-bg-elevated)] px-3 py-4 text-center cursor-pointer',
            'hover:border-[var(--color-accent-5)]',
          )}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files[0]
            if (!f) return
            f.text().then((t) => tryParse(t, f.name))
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (!f) return
              f.text().then((t) => tryParse(t, f.name))
            }}
          />
          <Upload
            size={16}
            strokeWidth={1.75}
            className="text-[var(--color-text-subtle)] mx-auto mb-1"
          />
          <p className="text-[12px] text-[var(--color-text)]">
            {filename ? `Loaded ${filename}` : 'Drop CSV, JSON, or JSONL — or click to pick'}
          </p>
          <p className="text-[10.5px] text-[var(--color-text-subtle)] mt-0.5">
            Required column: <span className="font-mono">query</span>. Optional:{' '}
            <span className="font-mono">expected_doc_ids</span> (pipe- or
            JSON-list).
          </p>
        </div>

        {parseError && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] font-mono">
            {parseError}
          </div>
        )}

        {items.length > 0 && (
          <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-2">
            <div className="text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1 flex items-center justify-between">
              <span>Preview · {items.length} items</span>
              <span className="text-[var(--color-text-muted)] font-mono normal-case tracking-normal">
                with gold:{' '}
                {items.filter((i) => i.expected_doc_ids?.length).length}
              </span>
            </div>
            <ul className="text-[11px] space-y-0.5 max-h-40 overflow-y-auto">
              {items.slice(0, 8).map((it, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-[var(--color-text-subtle)] font-mono w-5 tabular-nums">
                    {i + 1}
                  </span>
                  <span className="flex-1 truncate text-[var(--color-text)]">
                    {it.query}
                  </span>
                  {it.expected_doc_ids?.length ? (
                    <span className="text-[var(--color-text-subtle)] font-mono">
                      {it.expected_doc_ids.length} gold
                    </span>
                  ) : null}
                </li>
              ))}
              {items.length > 8 && (
                <li className="text-[var(--color-text-subtle)] italic text-[10.5px] pl-7">
                  … +{items.length - 8} more
                </li>
              )}
            </ul>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={submit}
            disabled={!name.trim() || !collection || items.length === 0}
          >
            Create eval set
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function EmptyMain({
  onCreate,
  hasSets,
}: {
  onCreate: () => void
  hasSets: boolean
}) {
  return (
    <div className="relative flex-1 flex items-center justify-center">
      <div className="dot-grid absolute inset-0 opacity-30" aria-hidden />
      <div className="relative text-center max-w-md px-6">
        <FlaskConical
          size={28}
          strokeWidth={1.5}
          className="text-[var(--color-text-subtle)] mx-auto mb-3"
        />
        <h2 className="text-[14px] font-medium text-[var(--color-text)] mb-1">
          {hasSets ? 'Pick an eval set' : 'Quality Lab'}
        </h2>
        <p className="text-[12px] text-[var(--color-text-muted)] mb-4 leading-relaxed">
          Eval sets are the gold queries you run against a retrieval
          configuration. Veda embeds each query server-side, runs the
          search, and computes recall@k, MRR, hit@1 + per-phase latency.
          Pick a set to start, or upload a CSV / JSON / JSONL.
        </p>
        <Button
          variant="primary"
          onClick={onCreate}
          leading={<Plus size={12} strokeWidth={2.5} />}
        >
          {hasSets ? 'Add another set' : 'Create your first eval set'}
        </Button>
        {!hasSets && (
          <p className="mt-2 text-[10.5px] text-[var(--color-text-subtle)] flex items-center gap-1 justify-center">
            <ChevronRight size={10} strokeWidth={2.25} />
            Same sample is fine — start with 10-20 representative queries.
          </p>
        )}
      </div>
    </div>
  )
}
