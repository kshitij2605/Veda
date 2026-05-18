import { useCallback, useState } from 'react'
import { Beaker, Plus, Play, Trash2 } from 'lucide-react'
import {
  chunkPreview,
  type ChunkPreviewRun,
  type ChunkingStrategyName,
} from '../api/pipelines'
import {
  Badge,
  Button,
  Input,
  Select,
  toast,
} from '../components/primitives'
import { cn } from '../lib/cn'

const STRATEGY_OPTIONS: {
  value: ChunkingStrategyName
  label: string
  takesSize: boolean
}[] = [
  { value: 'markdown_recursive', label: 'Markdown recursive', takesSize: true },
  { value: 'markdown_h1', label: 'Heading H1', takesSize: true },
  { value: 'markdown_h2', label: 'Heading H2', takesSize: true },
  { value: 'markdown_h3', label: 'Heading H3', takesSize: true },
  { value: 'fixed_size', label: 'Fixed size', takesSize: true },
  { value: 'sentence', label: 'Sentence', takesSize: true },
  { value: 'page', label: 'Page', takesSize: false },
  { value: 'semantic', label: 'Semantic', takesSize: true },
  { value: 'hierarchical', label: 'Hierarchical', takesSize: true },
  { value: 'none', label: 'No chunking', takesSize: false },
]

interface RunSpec {
  id: string
  label: string
  strategy: ChunkingStrategyName
  chunk_size?: number
  chunk_overlap?: number
}

const SAMPLE_TEXT = `# Cloud Security Overview

Cloud security is a shared responsibility between the provider and the customer. Most breaches happen on the customer side — misconfigured buckets, leaked keys, or over-permissive IAM roles.

## Identity and access management

Use short-lived credentials. Rotate keys. Audit who has admin rights and remove access for departed contractors within 24 hours of offboarding.

## Network controls

Default-deny ingress at the edge. Allow-list outbound. Never expose database ports directly to the internet — front everything with a load balancer or service mesh.

# Application Security

Treat every input as hostile. Validate types, lengths, and shapes at the boundary. Output-encode for the destination context.

## Common pitfalls

1. SQL injection via string concatenation.
2. XSS from unescaped HTML.
3. Path traversal in file uploads.
`

function newRun(label: string, strategy: ChunkingStrategyName): RunSpec {
  return {
    id: Math.random().toString(36).slice(2, 8),
    label,
    strategy,
    chunk_size: strategy === 'fixed_size' ? 400 : 2000,
    chunk_overlap: 100,
  }
}

export function ChunkerAB() {
  const [text, setText] = useState(SAMPLE_TEXT)
  const [runs, setRuns] = useState<RunSpec[]>([
    newRun('Default · 2000/100', 'markdown_recursive'),
    newRun('Fixed · 400/100', 'fixed_size'),
  ])
  const [results, setResults] = useState<ChunkPreviewRun[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const updateRun = (id: string, patch: Partial<RunSpec>) =>
    setRuns((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)))

  const addRun = () => {
    if (runs.length >= 6) {
      toast.error('Up to 6 strategies at a time.')
      return
    }
    setRuns((prev) => [...prev, newRun(`Run ${prev.length + 1}`, 'sentence')])
  }

  const removeRun = (id: string) =>
    setRuns((prev) => prev.filter((r) => r.id !== id))

  const run = useCallback(async () => {
    setError(null)
    setLoading(true)
    try {
      const { data } = await chunkPreview(
        text,
        runs.map((r) => ({
          label: r.label,
          strategy_name: r.strategy,
          chunk_size: r.chunk_size,
          chunk_overlap: r.chunk_overlap,
        })),
      )
      setResults(data.runs)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'preview failed')
    } finally {
      setLoading(false)
    }
  }, [text, runs])

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--color-bg)] overflow-hidden">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <Beaker size={14} strokeWidth={2.25} className="text-[var(--color-accent-9)]" />
        <h1 className="text-[14px] font-semibold tracking-tight text-[var(--color-text)] flex-1">
          Chunker A/B
        </h1>
        <Button
          variant="primary"
          size="sm"
          onClick={() => void run()}
          disabled={loading || text.trim().length === 0 || runs.length === 0}
          leading={<Play size={11} strokeWidth={2.25} />}
        >
          {loading ? 'Running…' : 'Run comparison'}
        </Button>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-4 min-h-0">
        <section className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1.5">
              Input markdown ({text.length} chars)
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              spellCheck={false}
              className="w-full h-[260px] resize-y rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2 text-[12px] font-mono leading-snug text-[var(--color-text)] outline-none focus:border-[var(--color-accent-9)]"
              placeholder="Paste markdown to compare chunking strategies…"
            />
          </div>
          <div>
            <div className="flex items-center mb-1.5">
              <span className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] flex-1">
                Strategies ({runs.length} / 6)
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={addRun}
                leading={<Plus size={11} strokeWidth={2.25} />}
                disabled={runs.length >= 6}
              >
                Add
              </Button>
            </div>
            <ul className="space-y-2">
              {runs.map((r) => {
                const opt = STRATEGY_OPTIONS.find((o) => o.value === r.strategy)
                return (
                  <li
                    key={r.id}
                    className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-2.5 py-2"
                  >
                    <div className="flex items-center gap-2">
                      <Input
                        value={r.label}
                        onChange={(e) => updateRun(r.id, { label: e.target.value })}
                        className="flex-1 h-7 text-[12px]"
                        placeholder="Label"
                      />
                      <button
                        onClick={() => removeRun(r.id)}
                        disabled={runs.length <= 1}
                        className="p-1 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-error)] disabled:opacity-30 disabled:cursor-not-allowed"
                        title="Remove run"
                      >
                        <Trash2 size={12} strokeWidth={2.25} />
                      </button>
                    </div>
                    <div className="flex items-center gap-2 mt-1.5">
                      <Select
                        value={r.strategy}
                        onValueChange={(v) =>
                          updateRun(r.id, { strategy: v as ChunkingStrategyName })
                        }
                        options={STRATEGY_OPTIONS.map((o) => ({
                          value: o.value,
                          label: o.label,
                        }))}
                        size="sm"
                      />
                      {opt?.takesSize && (
                        <>
                          <Input
                            type="number"
                            value={r.chunk_size ?? ''}
                            onChange={(e) =>
                              updateRun(r.id, {
                                chunk_size: Number(e.target.value) || undefined,
                              })
                            }
                            placeholder="size"
                            className="w-20 h-7 text-[11.5px] font-mono"
                          />
                          <Input
                            type="number"
                            value={r.chunk_overlap ?? ''}
                            onChange={(e) =>
                              updateRun(r.id, {
                                chunk_overlap: Number(e.target.value) || undefined,
                              })
                            }
                            placeholder="overlap"
                            className="w-20 h-7 text-[11.5px] font-mono"
                          />
                        </>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        </section>

        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
            {error}
          </div>
        )}

        {results.length > 0 && (
          <section>
            <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-2">
              Results · side-by-side
            </div>
            <div
              className="grid gap-3"
              style={{
                gridTemplateColumns: `repeat(${results.length}, minmax(0, 1fr))`,
              }}
            >
              {results.map((r) => (
                <RunColumn key={r.label} run={r} />
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

function RunColumn({ run }: { run: ChunkPreviewRun }) {
  return (
    <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] overflow-hidden flex flex-col min-h-0">
      <header className="px-3 py-2 border-b border-[var(--color-border)] flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-[12.5px] font-semibold text-[var(--color-text)] truncate flex-1">
            {run.label}
          </span>
          <Badge
            variant="default"
            mono
            label={run.strategy_name}
            className="lowercase"
          />
        </div>
        <div className="flex items-center gap-3 text-[10.5px] font-mono tabular-nums text-[var(--color-text-muted)]">
          <span>{run.total_chunks} chunks</span>
          <span>·</span>
          <span>avg {run.avg_chars.toFixed(0)}c</span>
          <span>·</span>
          <span>
            {run.min_chars}-{run.max_chars}c
          </span>
          <span>·</span>
          <span>{run.elapsed_ms.toFixed(1)}ms</span>
        </div>
        {run.truncated && (
          <div className="text-[10.5px] italic text-[var(--color-warning)]">
            Showing first {run.chunks.length} of {run.total_chunks} chunks.
          </div>
        )}
      </header>
      <ul className="flex-1 overflow-auto divide-y divide-[var(--color-border)]">
        {run.chunks.map((c) => (
          <li
            key={c.index}
            className="px-3 py-2 text-[11.5px] leading-snug text-[var(--color-text-muted)]"
          >
            <div className="flex items-center gap-2 mb-1 text-[10px] font-mono tabular-nums text-[var(--color-text-subtle)]">
              <span>#{c.index}</span>
              <span>·</span>
              <span>{c.char_count}c</span>
              <span>·</span>
              <span>
                {c.start_offset}-{c.end_offset}
              </span>
              {c.heading_path.length > 0 && (
                <span
                  className={cn(
                    'truncate text-[var(--color-accent-11)]',
                  )}
                  title={c.heading_path.join(' / ')}
                >
                  {c.heading_path.join(' / ')}
                </span>
              )}
            </div>
            <div className="whitespace-pre-wrap font-mono text-[11px] line-clamp-6">
              {c.text}
            </div>
          </li>
        ))}
        {run.chunks.length === 0 && (
          <li className="px-3 py-4 text-[11.5px] italic text-[var(--color-text-subtle)]">
            No chunks produced.
          </li>
        )}
      </ul>
    </div>
  )
}
