import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  FlaskConical,
  Play,
  Plus,
  RotateCw,
  Sparkles,
  Trash2,
} from 'lucide-react'
import {
  caseFromConversation,
  createEvalSet,
  deleteEvalSet,
  generateEvalSet,
  getEvalSetSizeRecommendation,
  listEvalSets,
  listRuns,
  pct,
  runEvalSet,
  type EvalRun,
  type EvalSet,
  type EvalSetSizeRecommendation,
} from '../../api/ragEval'
import {
  Badge,
  Button,
  EmptyState,
  Input,
  Modal,
  Skeleton,
  Sparkline,
  StatTile,
  toast,
} from '../../components/primitives'
import { cn } from '../../lib/cn'

/**
 * Eval Harness — the server-backed RAG/agent evaluation surface.
 *
 * Lists eval sets, runs one against its agent, and shows the latest run's
 * aggregate (recall@k / faithfulness / answer-relevancy / pass-rate) plus
 * per-case rows and a pass-rate-over-time trend (drift). Backed by
 * `/v1/eval-sets` (SavedObjects, tenant-scoped). All scoring runs under
 * the caller's identity, like `/converse`.
 *
 * recall@k is a retrieval *proxy* (cited ∩ expected / |expected|) and is
 * labeled as such.
 */
export function EvalHarness() {
  const [sets, setSets] = useState<EvalSet[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<EvalSet | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [running, setRunning] = useState(false)

  const load = useCallback(async (spin = false) => {
    if (spin) setLoading(true)
    setError(null)
    try {
      const { data } = await listEvalSets()
      setSets(data.eval_sets)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load eval sets')
    } finally {
      if (spin) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(true)
  }, [load])

  const loadRuns = useCallback(async (id: string) => {
    setRunsLoading(true)
    try {
      const { data } = await listRuns(id)
      setRuns(data.runs)
    } catch {
      setRuns([])
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (activeId) void loadRuns(activeId)
    else setRuns([])
  }, [activeId, loadRuns])

  const activeSet = useMemo(
    () => sets?.find((s) => s.id === activeId) ?? null,
    [sets, activeId],
  )
  // Runs are returned newest-first; the latest is the head.
  const latestRun = runs[0] ?? null

  const onRun = useCallback(async () => {
    if (!activeId) return
    setRunning(true)
    try {
      await runEvalSet(activeId, {})
      toast.success('Eval run complete')
      await loadRuns(activeId)
    } catch (e) {
      const msg =
        (e as { response?: { status?: number } })?.response?.status === 503
          ? 'No LLM configured — set VEDA_LLM_ENDPOINT to run evals.'
          : e instanceof Error
            ? e.message
            : 'Eval run failed'
      toast.error(msg)
    } finally {
      setRunning(false)
    }
  }, [activeId, loadRuns])

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <FlaskConical size={16} strokeWidth={2} className="text-[var(--color-accent)]" />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
          Eval Harness
        </h1>
        <div className="ml-auto flex items-center gap-2">
          <Button
            size="xs"
            variant="primary"
            leading={<Plus size={11} strokeWidth={2.5} />}
            onClick={() => setCreating(true)}
          >
            New set
          </Button>
          <Button
            variant="secondary"
            size="sm"
            leading={<Sparkles size={14} />}
            onClick={() => setGenerating(true)}
          >
            Generate from documents
          </Button>
          <Button
            size="xs"
            variant="ghost"
            leading={
              <RotateCw
                size={11}
                strokeWidth={2.25}
                className={loading ? 'animate-spin' : ''}
              />
            }
            onClick={() => load(true)}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-4">
        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
            <AlertCircle size={12} strokeWidth={2.25} />
            {error}
          </div>
        )}

        {sets === null ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : sets.length === 0 ? (
          <EmptyState
            icon={<FlaskConical size={20} strokeWidth={2} />}
            title="No eval sets yet"
            description="Create a labeled set of queries, then run an agent over it to score recall@k (proxy), faithfulness, and answer-relevancy."
          />
        ) : (
          <div className="grid grid-cols-[260px_1fr] gap-4">
            {/* Set list */}
            <div className="space-y-1.5">
              {sets.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setActiveId(s.id)}
                  className={cn(
                    'w-full text-left rounded-[var(--radius-sm)] border px-3 py-2.5 transition-colors',
                    s.id === activeId
                      ? 'border-[var(--color-accent)] bg-[var(--color-accent-subtle)]'
                      : 'border-[var(--color-border)] hover:bg-[var(--color-bg-subtle)]',
                  )}
                >
                  <div className="text-[13px] font-medium text-[var(--color-text)] truncate">
                    {s.name}
                  </div>
                  <div className="mt-0.5 text-[11px] text-[var(--color-text-subtle)]">
                    {s.cases.length} case{s.cases.length === 1 ? '' : 's'}
                    {s.agent_id ? ` · ${s.agent_id}` : ' · veda-agent'}
                  </div>
                </button>
              ))}
            </div>

            {/* Active set detail */}
            <div className="min-w-0">
              {!activeSet ? (
                <EmptyState
                  title="Select an eval set"
                  description="Pick a set on the left to view its latest run and per-case scores."
                />
              ) : (
                <ActiveSetPanel
                  set={activeSet}
                  runs={runs}
                  latestRun={latestRun}
                  runsLoading={runsLoading}
                  running={running}
                  onRun={onRun}
                  onDelete={() => setConfirmDelete(activeSet)}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {creating && (
        <CreateSetModal
          onClose={() => setCreating(false)}
          onCreated={async (id) => {
            setCreating(false)
            await load(false)
            setActiveId(id)
          }}
        />
      )}

      {generating && (
        <GenerateSetModal
          onClose={() => setGenerating(false)}
          onCreated={(id) => {
            setActiveId(id)
            setGenerating(false)
            load(true)
          }}
        />
      )}

      {confirmDelete && (
        <Modal open title="Delete eval set" onClose={() => setConfirmDelete(null)}>
          <div className="space-y-4">
            <p className="text-[13px] text-[var(--color-text-muted)]">
              Delete <span className="font-medium">{confirmDelete.name}</span>?
              Its runs are kept but orphaned.
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(null)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="sm"
                leading={<Trash2 size={12} />}
                onClick={async () => {
                  try {
                    await deleteEvalSet(confirmDelete.id)
                    if (activeId === confirmDelete.id) setActiveId(null)
                    setConfirmDelete(null)
                    await load(false)
                  } catch (e) {
                    toast.error(e instanceof Error ? e.message : 'Delete failed')
                  }
                }}
              >
                Delete
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  )
}

function ActiveSetPanel({
  set,
  runs,
  latestRun,
  runsLoading,
  running,
  onRun,
  onDelete,
}: {
  set: EvalSet
  runs: EvalRun[]
  latestRun: EvalRun | null
  runsLoading: boolean
  running: boolean
  onRun: () => void
  onDelete: () => void
}) {
  // Pass-rate over time (oldest → newest) for the drift trend.
  const trend = useMemo(
    () => [...runs].reverse().map((r) => r.aggregate.pass_rate),
    [runs],
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[15px] font-semibold text-[var(--color-text)]">
            {set.name}
          </div>
          <div className="text-[12px] text-[var(--color-text-subtle)]">
            {set.cases.length} case{set.cases.length === 1 ? '' : 's'} ·
            agent {set.agent_id ?? 'veda-agent'}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="xs" leading={<Trash2 size={11} />} onClick={onDelete}>
            Delete
          </Button>
          <Button
            variant="primary"
            size="sm"
            leading={<Play size={12} strokeWidth={2.5} />}
            onClick={onRun}
            disabled={running || set.cases.length === 0}
          >
            {running ? 'Running…' : 'Run eval'}
          </Button>
        </div>
      </div>

      {/* Aggregate KPIs */}
      <div className="grid grid-cols-4 gap-px rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-border)] overflow-hidden">
        <div className="bg-[var(--color-surface)]">
          <StatTile label="Recall@k (proxy)" value={pct(latestRun?.aggregate.mean_recall_at_k)} />
        </div>
        <div className="bg-[var(--color-surface)]">
          <StatTile label="Faithfulness" value={pct(latestRun?.aggregate.mean_faithfulness)} />
        </div>
        <div className="bg-[var(--color-surface)]">
          <StatTile
            label="Answer relevancy"
            value={pct(latestRun?.aggregate.mean_answer_relevancy)}
          />
        </div>
        <div className="bg-[var(--color-surface)]">
          <StatTile
            label="Pass rate"
            value={pct(latestRun?.aggregate.pass_rate)}
            foot={
              trend.length > 1 ? (
                <span className="inline-flex items-center gap-1.5">
                  <Sparkline values={trend} width={64} height={14} />
                  drift over {trend.length} runs
                </span>
              ) : undefined
            }
          />
        </div>
      </div>

      {/* Per-case rows of the latest run */}
      {runsLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : !latestRun ? (
        <EmptyState
          title="No runs yet"
          description="Run this set to score each case and see the aggregate."
        />
      ) : (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[var(--color-text-subtle)] border-b border-[var(--color-border)] bg-[var(--color-bg-subtle)]">
                <th className="text-left font-medium px-3 py-2">Query</th>
                <th className="text-right font-medium px-3 py-2">Recall@k</th>
                <th className="text-right font-medium px-3 py-2">Faithfulness</th>
                <th className="text-right font-medium px-3 py-2">Relevancy</th>
                <th className="text-center font-medium px-3 py-2">Result</th>
              </tr>
            </thead>
            <tbody>
              {latestRun.cases.map((c) => (
                <tr key={c.case_id} className="border-b border-[var(--color-border)] last:border-0">
                  <td className="px-3 py-2 text-[var(--color-text)] max-w-[280px] truncate" title={c.query}>
                    {c.query}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(c.recall_at_k)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(c.faithfulness)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(c.answer_relevancy)}</td>
                  <td className="px-3 py-2 text-center">
                    <Badge
                      variant={c.passed ? 'success' : 'error'}
                      label={c.passed ? 'pass' : 'fail'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function CreateSetModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [name, setName] = useState('')
  const [agentId, setAgentId] = useState('')
  // One-query-per-line bootstrap; expected ids / gold answers come via the
  // flywheel (from-conversation) later.
  const [queries, setQueries] = useState('')
  const [saving, setSaving] = useState(false)

  const submit = async () => {
    const cases = queries
      .split('\n')
      .map((q) => q.trim())
      .filter(Boolean)
      .map((q, i) => ({ id: `c${i + 1}`, query: q, expected_chunk_ids: [] as string[] }))
    setSaving(true)
    try {
      const { data } = await createEvalSet({
        name: name.trim() || 'Untitled set',
        agent_id: agentId.trim() || undefined,
        cases,
      })
      toast.success('Eval set created')
      onCreated(data.id)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Create failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal open title="New eval set" onClose={onClose}>
      <div className="space-y-3">
        <label className="block">
          <span className="text-[12px] text-[var(--color-text-muted)]">Name</span>
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Golden set" />
        </label>
        <label className="block">
          <span className="text-[12px] text-[var(--color-text-muted)]">
            Agent id (optional)
          </span>
          <Input
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="veda-agent"
          />
        </label>
        <label className="block">
          <span className="text-[12px] text-[var(--color-text-muted)]">
            Queries (one per line)
          </span>
          <textarea
            value={queries}
            onChange={(e) => setQueries(e.target.value)}
            rows={5}
            placeholder={'what index does veda use?\nhow does chunking work?'}
            className="mt-1 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-[13px] text-[var(--color-text)]"
          />
        </label>
        <p className="text-[11px] text-[var(--color-text-subtle)]">
          Tip: graduate good production turns into regression cases (with
          known-good retrieval + gold answers) via the conversation flywheel.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={submit} disabled={saving}>
            {saving ? 'Creating…' : 'Create'}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function GenerateSetModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [collection, setCollection] = useState('')
  const [numCases, setNumCases] = useState(20)
  const [recommendation, setRecommendation] = useState<EvalSetSizeRecommendation | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadRecommendation = async (name: string) => {
    if (!name.trim()) {
      setRecommendation(null)
      return
    }
    try {
      const { data } = await getEvalSetSizeRecommendation(name.trim())
      setRecommendation(data)
      setNumCases(data.recommended)
    } catch {
      setRecommendation(null)
    }
  }

  const submit = async () => {
    if (!collection.trim()) {
      setError('Enter a collection name.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const { data } = await generateEvalSet(collection.trim(), { num_cases: numCases })
      onCreated(data.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed. Is ANSWER_GEN_* configured?')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal open onClose={onClose}>
      <div className="p-4">
        <h2 className="mb-3 text-sm font-semibold">Generate an eval set from documents</h2>
        <label className="mb-1 block text-xs font-medium">Collection</label>
        <input
          className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
          value={collection}
          onChange={(e) => {
            setCollection(e.target.value)
            loadRecommendation(e.target.value)
          }}
          placeholder="e.g. rag_test"
        />
        <label className="mb-1 block text-xs font-medium">
          Number of questions
          {recommendation && (
            <span className="ml-1.5 font-normal text-[var(--color-text-muted)]">
              ({recommendation.recommended} recommended for this index's size —{' '}
              {recommendation.document_count} documents, {recommendation.chunk_count} chunks)
            </span>
          )}
        </label>
        <input
          type="number"
          className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
          value={numCases}
          min={1}
          max={recommendation?.max}
          onChange={(e) => {
            const n = Number(e.target.value)
            const capped = recommendation ? Math.min(n, recommendation.max) : n
            setNumCases(Math.max(1, capped))
          }}
        />
        {error && <div className="mb-3 text-xs text-red-500">{error}</div>}
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={submit} disabled={saving}>
            {saving ? 'Generating…' : 'Generate'}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

// Re-exported for potential programmatic flywheel use from a conversation
// view; not yet wired to a button here.
export { caseFromConversation }
