import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  CheckCircle2,
  ChevronRight,
  FileText,
  GitBranch,
  Loader,
  Play,
  Upload,
  X,
} from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import {
  processDocument,
  type ChunkSummary,
  type ChunkingStrategyName,
  type ProcessDocumentResponse,
  type ProcessDocumentSettings,
} from '../api/pipelines'
import {
  Badge,
  Button,
  Input,
  Select,
  toast,
} from '../components/primitives'
import { cn } from '../lib/cn'
import { useUrlState } from '../hooks/useUrlState'

const STRATEGIES: {
  id: ChunkingStrategyName
  label: string
  description: string
}[] = [
  {
    id: 'markdown_recursive',
    label: 'Markdown recursive',
    description:
      'Default. Recurses on heading + paragraph + sentence with size + overlap fallback.',
  },
  {
    id: 'markdown_h1',
    label: 'Heading H1',
    description: 'One chunk per top-level heading. Best for long-form docs.',
  },
  {
    id: 'markdown_h2',
    label: 'Heading H2',
    description: 'One chunk per H2. Tighter than H1.',
  },
  {
    id: 'markdown_h3',
    label: 'Heading H3',
    description: 'One chunk per H3 — tightest heading-based.',
  },
  {
    id: 'sentence',
    label: 'Sentence',
    description: 'Pack sentences into chunks up to size with overlap.',
  },
  {
    id: 'fixed_size',
    label: 'Fixed size',
    description: 'Character window of size + overlap. Predictable.',
  },
  {
    id: 'semantic',
    label: 'Semantic',
    description:
      'Embedding-similarity-based break detection. Latency cost: extra embedding pass.',
  },
  {
    id: 'page',
    label: 'Page',
    description: 'One chunk per OCR page. Good when pages are self-contained.',
  },
  {
    id: 'hierarchical',
    label: 'Hierarchical',
    description: 'Coarse outer chunks + finer inner chunks (parent / child).',
  },
  {
    id: 'late_chunking',
    label: 'Late chunking',
    description:
      'Embed first, chunk after — preserves cross-chunk context. Costlier.',
  },
  {
    id: 'contextual_chunking',
    label: 'Contextual',
    description: 'Prepends contextual headers per chunk before embedding.',
  },
  {
    id: 'none',
    label: 'None',
    description: 'One chunk per file. Use for short docs.',
  },
]

interface PipelineConfig {
  collection: string
  strategy: ChunkingStrategyName
  chunkSize: number
  chunkOverlap: number
  dpi: number
  skipCaptions: boolean
}

const DEFAULT_CFG: PipelineConfig = {
  collection: '',
  strategy: 'markdown_recursive',
  chunkSize: 2000,
  chunkOverlap: 300,
  dpi: 200,
  skipCaptions: false,
}

interface JobState {
  id: string
  file: File
  status: 'queued' | 'running' | 'done' | 'failed'
  progress?: number
  response?: ProcessDocumentResponse
  error?: string
  startedAt?: number
  finishedAt?: number
}

export function Pipelines() {
  const { collections, fetchCollections } = useCollectionStore()
  const [cfg, setCfg] = useUrlState<PipelineConfig>('cfg', DEFAULT_CFG)
  const setC = <K extends keyof PipelineConfig>(k: K, v: PipelineConfig[K]) =>
    setCfg({ ...cfg, [k]: v })

  const [jobs, setJobs] = useState<JobState[]>([])
  const [running, setRunning] = useState(false)

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  useEffect(() => {
    if (!cfg.collection && collections.length > 0) {
      setC('collection', collections[0]!.name)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collections])

  const addFiles = (files: FileList | File[]) => {
    const incoming: JobState[] = Array.from(files).map((f) => ({
      id: `${f.name}_${f.size}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      file: f,
      status: 'queued',
    }))
    setJobs((prev) => [...prev, ...incoming])
  }

  const removeJob = (id: string) =>
    setJobs((prev) => prev.filter((j) => j.id !== id))

  const runAll = async () => {
    if (!cfg.collection) {
      toast.error('Pick a target collection first.')
      return
    }
    if (jobs.filter((j) => j.status === 'queued').length === 0) {
      toast.error('Drop a file first.')
      return
    }
    setRunning(true)
    for (const job of jobs.filter((j) => j.status === 'queued')) {
      setJobs((prev) =>
        prev.map((j) =>
          j.id === job.id
            ? { ...j, status: 'running', startedAt: Date.now() }
            : j,
        ),
      )
      try {
        const settings: ProcessDocumentSettings = {
          chunking_strategy: cfg.strategy,
          chunk_size: cfg.chunkSize,
          chunk_overlap: cfg.chunkOverlap,
          dpi: cfg.dpi,
          skip_captions: cfg.skipCaptions,
        }
        const res = await processDocument(
          cfg.collection,
          job.file,
          settings,
          (loaded, total) => {
            setJobs((prev) =>
              prev.map((j) =>
                j.id === job.id
                  ? { ...j, progress: total > 0 ? loaded / total : 0 }
                  : j,
              ),
            )
          },
        )
        setJobs((prev) =>
          prev.map((j) =>
            j.id === job.id
              ? {
                  ...j,
                  status: 'done',
                  response: res,
                  finishedAt: Date.now(),
                }
              : j,
          ),
        )
      } catch (e) {
        const msg =
          (e as { response?: { data?: { error?: { message?: string } } } })
            ?.response?.data?.error?.message ??
          (e instanceof Error ? e.message : 'Unknown error')
        setJobs((prev) =>
          prev.map((j) =>
            j.id === job.id
              ? { ...j, status: 'failed', error: msg, finishedAt: Date.now() }
              : j,
          ),
        )
      }
    }
    setRunning(false)
    toast.success('Pipeline complete')
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--color-bg)] overflow-hidden">
      <div className="flex-1 flex flex-col overflow-hidden">
        <PageHeader
          collection={cfg.collection}
          onCollection={(c) => setC('collection', c)}
          collections={collections.map((c) => c.name)}
          queued={jobs.filter((j) => j.status === 'queued').length}
          onRun={runAll}
          running={running}
        />
        <div className="flex-1 overflow-auto px-4 py-4 space-y-3">
          <DropZone onFiles={addFiles} />
          {jobs.map((j) => (
            <JobCard
              key={j.id}
              job={j}
              collection={cfg.collection}
              onRemove={() => removeJob(j.id)}
            />
          ))}
          {jobs.length === 0 && <PipelineGuide />}
        </div>
      </div>
      <Sidebar cfg={cfg} setC={setC} />
    </div>
  )
}

function PageHeader({
  collection,
  onCollection,
  collections,
  queued,
  onRun,
  running,
}: {
  collection: string
  onCollection: (c: string) => void
  collections: string[]
  queued: number
  onRun: () => void
  running: boolean
}) {
  return (
    <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none bg-[var(--color-bg)]">
      <GitBranch
        size={16}
        strokeWidth={2}
        className="text-[var(--color-accent)]"
      />
      <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
        Pipelines
      </h1>
      <span className="text-[var(--color-text-subtle)]">·</span>
      <Select
        value={collection}
        onValueChange={onCollection}
        placeholder="Target collection"
        options={collections.map((c) => ({ value: c, label: c }))}
      />
      <span className="ml-auto" />
      <Badge
        label={`${queued} queued`}
        variant={queued > 0 ? 'accent' : 'ghost'}
      />
      <Button
        variant="primary"
        size="sm"
        leading={
          running ? (
            <Loader size={11} strokeWidth={2.25} className="animate-spin" />
          ) : (
            <Play size={11} strokeWidth={2.5} />
          )
        }
        onClick={onRun}
        disabled={running || queued === 0 || !collection}
      >
        {running ? 'Running…' : `Run ${queued}`}
      </Button>
    </header>
  )
}

function DropZone({ onFiles }: { onFiles: (f: FileList | File[]) => void }) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setOver(false)
      if (e.dataTransfer.files.length > 0) onFiles(e.dataTransfer.files)
    },
    [onFiles],
  )

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
      onClick={() => input.current?.click()}
      className={cn(
        'relative cursor-pointer rounded-[var(--radius-md)] border border-dashed',
        'flex flex-col items-center justify-center py-8 px-6 text-center',
        'transition-[border-color,background] duration-[var(--duration-base)]',
        over
          ? 'border-[var(--color-accent)] bg-[var(--color-accent-0)]'
          : 'border-[var(--color-border-strong)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-accent-5)]',
      )}
    >
      <div className="dot-grid absolute inset-0 opacity-20 rounded-[var(--radius-md)]" aria-hidden />
      <input
        ref={input}
        type="file"
        multiple
        accept="application/pdf,image/*,.txt,.md"
        className="hidden"
        onChange={(e) => e.target.files && onFiles(e.target.files)}
      />
      <Upload
        size={20}
        strokeWidth={1.75}
        className="text-[var(--color-text-subtle)] mb-2 relative"
      />
      <p className="relative text-[13px] text-[var(--color-text)] font-medium">
        Drop PDFs, images, or text here
      </p>
      <p className="relative text-[11.5px] text-[var(--color-text-muted)] mt-0.5">
        Veda runs OCR via FlashParse, chunks per the config on the right, and
        embeds with the field's model.
      </p>
    </div>
  )
}

function JobCard({
  job,
  collection,
  onRemove,
}: {
  job: JobState
  collection: string
  onRemove: () => void
}) {
  const took =
    job.finishedAt && job.startedAt
      ? ((job.finishedAt - job.startedAt) / 1000).toFixed(1) + 's'
      : null
  return (
    <div
      className={cn(
        'rounded-[var(--radius-md)] border bg-[var(--color-bg-elevated)] overflow-hidden',
        job.status === 'failed'
          ? 'border-[color-mix(in_oklch,var(--color-error)_30%,transparent)]'
          : 'border-[var(--color-border)]',
      )}
    >
      <div className="flex items-center gap-3 px-3.5 py-2.5">
        <FileText
          size={14}
          strokeWidth={2}
          className="text-[var(--color-text-subtle)] flex-none"
        />
        <div className="flex-1 min-w-0">
          <div className="font-mono text-[12.5px] text-[var(--color-text)] truncate">
            {job.file.name}
          </div>
          <div className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums mt-0.5">
            {(job.file.size / 1024).toFixed(0)} KB
            {took && ` · ${took}`}
          </div>
        </div>
        <JobStatus job={job} />
        {job.status === 'queued' && (
          <button
            onClick={onRemove}
            title="Remove from queue"
            className="p-1 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)]"
          >
            <X size={12} strokeWidth={2.25} />
          </button>
        )}
      </div>

      {job.status === 'running' && job.progress != null && (
        <div className="h-0.5 bg-[var(--color-base-2)]">
          <div
            className="h-full bg-[var(--color-accent)] transition-[width] duration-[var(--duration-base)]"
            style={{ width: `${job.progress * 100}%` }}
          />
        </div>
      )}

      {job.status === 'done' && job.response && (
        <DoneSummary
          response={job.response}
          collection={collection}
        />
      )}
      {job.status === 'failed' && job.error && (
        <div className="px-3.5 py-2.5 border-t border-[var(--color-border)] text-[12px] text-[var(--color-error)] font-mono whitespace-pre-wrap break-words">
          {job.error}
        </div>
      )}
    </div>
  )
}

function JobStatus({ job }: { job: JobState }) {
  if (job.status === 'queued') return <Badge label="queued" variant="ghost" dot />
  if (job.status === 'running')
    return (
      <span className="inline-flex items-center gap-1 text-[11px] text-[var(--color-accent)]">
        <Loader size={11} strokeWidth={2.25} className="animate-spin" />
        running
      </span>
    )
  if (job.status === 'done')
    return (
      <span className="inline-flex items-center gap-1 text-[11px] text-[var(--color-success)]">
        <CheckCircle2 size={11} strokeWidth={2.25} />
        done
      </span>
    )
  return <Badge label="failed" variant="error" dot />
}

function DoneSummary({
  response,
  collection,
}: {
  response: ProcessDocumentResponse
  collection: string
}) {
  // The first record id lets us link into the Records browser with a filter
  // that scopes to this document (provided the gateway tags `parent_doc_id`).
  const sample = response.chunks.slice(0, 5)
  return (
    <div className="border-t border-[var(--color-border)]">
      <div className="px-3.5 py-2.5 flex items-center gap-3 text-[11px] font-mono text-[var(--color-text-muted)] tabular-nums">
        <Stat n={response.pages} label="pages" />
        <Sep />
        <Stat n={response.records_inserted} label="chunks" />
        {response.images_extracted > 0 && (
          <>
            <Sep />
            <Stat n={response.images_extracted} label="images" />
          </>
        )}
        {response.tables_extracted > 0 && (
          <>
            <Sep />
            <Stat n={response.tables_extracted} label="tables" />
          </>
        )}
        <Link
          to={`/collections/${collection}#f=%5B%7B%22field%22%3A%22parent_doc_id%22%2C%22op%22%3A%22%3D%22%2C%22value%22%3A%22${encodeURIComponent(response.source_file)}%22%7D%5D`}
          className="ml-auto inline-flex items-center gap-1 text-[11px] text-[var(--color-accent)] hover:underline"
        >
          Open in Records
          <ChevronRight size={11} strokeWidth={2.25} />
        </Link>
      </div>
      {sample.length > 0 && (
        <ul className="border-t border-[var(--color-border)] divide-y divide-[var(--color-border)] max-h-72 overflow-auto">
          {sample.map((c, i) => (
            <ChunkPreview key={c.id} chunk={c} index={i + 1} />
          ))}
        </ul>
      )}
    </div>
  )
}

function ChunkPreview({ chunk, index }: { chunk: ChunkSummary; index: number }) {
  return (
    <li className="flex gap-3 px-3.5 py-2 hover:bg-[var(--color-bg-hover)] cursor-default">
      <span className="flex-none w-6 text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums pt-0.5">
        {index.toString().padStart(2, '0')}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-[10.5px] text-[var(--color-text-subtle)] truncate">
            {chunk.id}
          </span>
          <Badge label={chunk.type} variant="ghost" mono />
        </div>
        <div className="text-[12px] text-[var(--color-text)] leading-relaxed line-clamp-2 font-sans">
          {chunk.preview || <em className="text-[var(--color-text-subtle)]">(empty preview)</em>}
        </div>
      </div>
    </li>
  )
}

function Stat({ n, label }: { n: number; label: string }) {
  return (
    <span>
      <span className="text-[var(--color-text)] font-medium">
        {n.toLocaleString()}
      </span>{' '}
      <span className="text-[var(--color-text-subtle)]">{label}</span>
    </span>
  )
}
function Sep() {
  return <span className="text-[var(--color-border-strong)]">·</span>
}

function Sidebar({
  cfg,
  setC,
}: {
  cfg: PipelineConfig
  setC: <K extends keyof PipelineConfig>(k: K, v: PipelineConfig[K]) => void
}) {
  const strategy = STRATEGIES.find((s) => s.id === cfg.strategy)
  return (
    <aside className="flex flex-col w-[300px] flex-none border-l border-[var(--color-border)] bg-[var(--color-bg)] overflow-y-auto">
      <header className="px-3 py-2 border-b border-[var(--color-border)]">
        <h3 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Pipeline config
        </h3>
      </header>

      <section className="px-3 py-2.5 border-b border-[var(--color-border)] space-y-2.5">
        <div>
          <label className="block text-[10.5px] text-[var(--color-text-muted)] mb-1">
            Chunking strategy
          </label>
          <Select
            value={cfg.strategy}
            onValueChange={(v) => setC('strategy', v as ChunkingStrategyName)}
            options={STRATEGIES.map((s) => ({
              value: s.id,
              label: s.label,
              description: s.description,
            }))}
          />
          {strategy && (
            <p className="text-[10.5px] text-[var(--color-text-subtle)] mt-1.5 leading-relaxed">
              {strategy.description}
            </p>
          )}
        </div>
      </section>

      <section className="px-3 py-2.5 border-b border-[var(--color-border)] space-y-2.5">
        <div>
          <label className="block text-[10.5px] text-[var(--color-text-muted)] mb-1">
            Chunk size (chars)
          </label>
          <Input
            type="number"
            mono
            value={cfg.chunkSize}
            min={100}
            max={20000}
            onChange={(e) =>
              setC('chunkSize', Math.max(100, parseInt(e.target.value) || 2000))
            }
          />
        </div>
        <div>
          <label className="block text-[10.5px] text-[var(--color-text-muted)] mb-1">
            Chunk overlap (chars)
          </label>
          <Input
            type="number"
            mono
            value={cfg.chunkOverlap}
            min={0}
            max={5000}
            onChange={(e) =>
              setC(
                'chunkOverlap',
                Math.max(0, parseInt(e.target.value) || 0),
              )
            }
          />
        </div>
      </section>

      <section className="px-3 py-2.5 border-b border-[var(--color-border)] space-y-2.5">
        <h4 className="text-[9.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] font-medium">
          OCR (FlashParse)
        </h4>
        <div>
          <label className="block text-[10.5px] text-[var(--color-text-muted)] mb-1">
            DPI
          </label>
          <Input
            type="number"
            mono
            value={cfg.dpi}
            min={72}
            max={600}
            onChange={(e) => setC('dpi', parseInt(e.target.value) || 200)}
          />
        </div>
        <label className="flex items-center gap-2 text-[11.5px] text-[var(--color-text-muted)]">
          <input
            type="checkbox"
            checked={cfg.skipCaptions}
            onChange={(e) => setC('skipCaptions', e.target.checked)}
          />
          Skip VLM captions (faster)
        </label>
      </section>
    </aside>
  )
}

function PipelineGuide() {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-5 text-[12px] text-[var(--color-text-muted)] leading-relaxed">
      <h3 className="text-[13px] font-medium text-[var(--color-text)] mb-2 flex items-center gap-2">
        How processing works
        <Badge label="end-to-end" variant="accent" />
      </h3>
      <ol className="list-decimal list-inside space-y-1.5">
        <li>
          File is sent to <span className="font-mono">FlashParse</span> for OCR
          and structure extraction.
        </li>
        <li>
          Resulting markdown is split per the chunking strategy on the right.
        </li>
        <li>
          Each chunk is embedded with the target collection's vector field
          model (or auto-detected by modality).
        </li>
        <li>
          Records are inserted into the chosen collection. A preview of the
          first 5 chunks appears under each job once it completes.
        </li>
      </ol>
    </div>
  )
}

