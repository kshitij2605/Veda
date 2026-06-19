import { useEffect, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Globe,
  HardDrive,
  Image as ImageIcon,
  Layers,
  Network,
  Pencil,
  Plus,
  Play,
  RefreshCw,
  ScanText,
  Search,
  Settings as SettingsIcon,
  Sparkles,
  Trash2,
  Type,
} from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import { Badge, Button, Skeleton } from '../components/primitives'
import {
  type Connector as DataConnector,
  type ConnectorCreate,
  type RunResponse,
  asWebConfig,
  connectorKindLabel,
  createConnector,
  deleteConnector,
  listConnectors,
  runConnector,
  updateConnector,
} from '../api/connectors'

interface InfraService {
  id: string
  name: string
  type:
    | 'embedding-text'
    | 'embedding-multimodal'
    | 'ocr'
    | 'reranker'
    | 'llm'
    | 'blob-storage'
    | 'messaging'
  endpoint?: string
  status?: 'healthy' | 'degraded' | 'unavailable' | 'unknown'
  /** Free-form rows for the detail panel. */
  details: { label: string; value: string }[]
}

/**
 * Settings → Connectors inventory.
 *
 * Read-only for v1. The CRUD path lands when the gateway connectors
 * module is added (see `niriksha/AUDIT.md`). Until then, this page
 * surfaces what's *actually* wired today by reading /admin/health,
 * cross-referencing the collection store's embedding registry, and
 * decorating each row with status pulled from the subsystem checks.
 */
export function Settings() {
  const { health, fetchHealth } = useCollectionStore()
  const [connectors, setConnectors] = useState<InfraService[]>([])
  const [refreshing, setRefreshing] = useState(false)

  const reload = async () => {
    setRefreshing(true)
    await fetchHealth()
    setRefreshing(false)
  }

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setConnectors(buildConnectors(health))
  }, [health])

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)] overflow-hidden">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <SettingsIcon
          size={14}
          strokeWidth={2}
          className="text-[var(--color-accent-9)]"
        />
        <h1 className="text-[13.5px] font-medium text-[var(--color-text)]">
          Settings
        </h1>
        <span className="text-[var(--color-text-subtle)]">·</span>
        <span className="text-[12px] text-[var(--color-text-muted)]">
          Connectors
        </span>
        <Button
          size="xs"
          variant="ghost"
          leading={
            <RefreshCw
              size={11}
              strokeWidth={2.25}
              className={refreshing ? 'animate-spin' : ''}
            />
          }
          onClick={reload}
          disabled={refreshing}
          className="ml-auto"
        >
          Refresh
        </Button>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-6">
        {/* Data source connectors — managed CRUD framework. */}
        <DataConnectorsSection />

        {/* Infrastructure services — read-only status from /admin/health. */}
        <section>
          <h2 className="text-[12.5px] font-medium text-[var(--color-text)] mb-1">
            Infrastructure services
          </h2>
          <p className="text-[12px] text-[var(--color-text-muted)] max-w-2xl mb-3 leading-relaxed">
            Named credentials and endpoints the Veda daemon is wired against.
            This inventory is read-only — it reflects what <code className="font-mono text-[11.5px] text-[var(--color-text)]">/admin/health</code> reports.
          </p>

          {!health && <Skeleton className="h-32 w-full" />}
          {health && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {connectors.map((c) => (
                <InfraCard key={c.id} c={c} />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function InfraCard({ c }: { c: InfraService }) {
  const Icon = ICONS[c.type] ?? Network
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3.5">
      <div className="flex items-center gap-2 mb-2.5">
        <Icon size={13} strokeWidth={2} className="text-[var(--color-accent-9)]" />
        <span className="text-[12.5px] font-medium text-[var(--color-text)] truncate flex-1">
          {c.name}
        </span>
        <StatusBadge status={c.status} />
      </div>
      <div className="text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
        {labelForType(c.type)}
      </div>
      {c.endpoint && (
        <div className="font-mono text-[11px] text-[var(--color-text-muted)] truncate mb-2">
          {c.endpoint}
        </div>
      )}
      {c.details.length > 0 && (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
          {c.details.map((d) => (
            <div key={d.label}>
              <dt className="text-[10px] text-[var(--color-text-subtle)] uppercase tracking-wider">
                {d.label}
              </dt>
              <dd className="text-[var(--color-text)] font-mono truncate" title={d.value}>
                {d.value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status?: InfraService['status'] }) {
  if (!status || status === 'unknown') {
    return <Badge label="unknown" variant="ghost" dot />
  }
  if (status === 'healthy') {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)] bg-[color-mix(in_oklch,var(--color-success)_12%,transparent)] text-[var(--color-success)] border border-[color-mix(in_oklch,var(--color-success)_28%,transparent)] text-[10.5px] font-mono">
        <CheckCircle2 size={10} strokeWidth={2.25} />
        healthy
      </span>
    )
  }
  if (status === 'degraded') {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)] bg-[color-mix(in_oklch,var(--color-warning)_12%,transparent)] text-[var(--color-warning)] border border-[color-mix(in_oklch,var(--color-warning)_28%,transparent)] text-[10.5px] font-mono">
        <AlertCircle size={10} strokeWidth={2.25} />
        degraded
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)] bg-[color-mix(in_oklch,var(--color-error)_12%,transparent)] text-[var(--color-error)] border border-[color-mix(in_oklch,var(--color-error)_28%,transparent)] text-[10.5px] font-mono">
    <AlertCircle size={10} strokeWidth={2.25} />
      unavailable
    </span>
  )
}

const ICONS: Record<InfraService['type'], React.ComponentType<{ size?: number; strokeWidth?: number; className?: string }>> = {
  'embedding-text': Type,
  'embedding-multimodal': ImageIcon,
  ocr: ScanText,
  reranker: Layers,
  llm: Sparkles,
  'blob-storage': HardDrive,
  messaging: Network,
}

function labelForType(t: InfraService['type']): string {
  switch (t) {
    case 'embedding-text':
      return 'Text embedding'
    case 'embedding-multimodal':
      return 'Multimodal embedding'
    case 'ocr':
      return 'OCR (FlashParse)'
    case 'reranker':
      return 'Reranker'
    case 'llm':
      return 'LLM (chat completion)'
    case 'blob-storage':
      return 'Blob storage (S3-compat)'
    case 'messaging':
      return 'Messaging (NATS)'
  }
}

function buildConnectors(
  health:
    | { status: string; version?: string; checks?: Record<string, string> }
    | null,
): InfraService[] {
  const checks = health?.checks ?? {}
  const toStatus = (k: string): InfraService['status'] => {
    const v = checks[k]
    if (!v) return 'unknown'
    if (v === 'healthy') return 'healthy'
    if (v === 'degraded') return 'degraded'
    return 'unavailable'
  }
  return [
    {
      id: 'embed-text',
      name: 'Text embedder',
      type: 'embedding-text',
      status: toStatus('embedding'),
      endpoint: 'configured via EMBEDDING_ENDPOINT env',
      details: [
        { label: 'Default model', value: 'BAAI/bge-m3' },
        { label: 'Dimension', value: '1024' },
        { label: 'Modality', value: 'text' },
      ],
    },
    {
      id: 'embed-multimodal',
      name: 'Multimodal embedder',
      type: 'embedding-multimodal',
      status: toStatus('multimodal_embedding'),
      endpoint: 'docker compose --profile embedding',
      details: [
        { label: 'Default model', value: 'Qwen3-VL-Embedding-2B' },
        { label: 'Modality', value: 'text · image' },
      ],
    },
    {
      id: 'ocr',
      name: 'FlashParse OCR',
      type: 'ocr',
      status: toStatus('flashparse') ?? 'unknown',
      endpoint: 'VEDA_FLASHPARSE_ENDPOINT',
      details: [
        { label: 'Profile', value: 'ocr' },
        { label: 'DPI default', value: '200' },
      ],
    },
    {
      id: 'reranker',
      name: 'Cross-encoder reranker',
      type: 'reranker',
      status: 'unknown',
      endpoint: 'configured per-call in /search { rerank }',
      details: [
        { label: 'Wire', value: 'POST /v1/.../rerank' },
        { label: 'Auth', value: 'X-API-Key per provider' },
      ],
    },
    {
      id: 'llm-vlm',
      name: 'LLM (chat completions)',
      type: 'llm',
      // /admin/health doesn't probe the LLM yet; surface as "unknown"
      // until a probe lands. The endpoint itself is wired (Phase 5.8).
      status: 'unknown',
      endpoint: 'POST /v1/llm/complete · proxies VEDA_LLM_ENDPOINT',
      details: [
        { label: 'Model', value: 'VEDA_LLM_MODEL env' },
        { label: 'Wire', value: 'OpenAI-compatible chat/completions' },
        { label: 'Use', value: 'cluster labels · failure mode · NL→DSL' },
        { label: 'JSON mode', value: 'supported (json_mode flag)' },
      ],
    },
    {
      id: 'blob',
      name: 'Blob store (MinIO)',
      type: 'blob-storage',
      status: toStatus('minio'),
      endpoint: 'minio:9000 (compose-internal)',
      details: [
        { label: 'Bucket', value: 'veda' },
        { label: 'Tier', value: 'warm + cold' },
      ],
    },
    {
      id: 'nats',
      name: 'NATS messaging',
      type: 'messaging',
      status: toStatus('nats'),
      endpoint: 'nats:4222 (compose-internal)',
      details: [
        { label: 'JetStream', value: 'on' },
        { label: 'Use', value: 'streaming ingestion' },
      ],
    },
  ]
}

// ── Data source connectors (CRUD) ─────────────────────────────────────────

interface DraftConnector {
  id?: string
  kind: 'web' | 'websearch'
  name: string
  urls: string // newline / comma separated in the form
  collection: string
  deep_crawl: boolean
  max_pages: string
  enabled: boolean
}

const EMPTY_DRAFT: DraftConnector = {
  kind: 'web',
  name: '',
  urls: '',
  collection: '',
  deep_crawl: false,
  max_pages: '',
  enabled: true,
}

function DataConnectorsSection() {
  const [connectors, setConnectors] = useState<DataConnector[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState<DraftConnector | null>(null)
  const [runResult, setRunResult] = useState<Record<string, RunResponse | string>>({})
  const [running, setRunning] = useState<string | null>(null)

  const reload = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await listConnectors()
      setConnectors(data.connectors ?? [])
    } catch (e) {
      setError(errMsg(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    reload()
  }, [])

  const onSave = async (draft: DraftConnector) => {
    const urls = draft.urls
      .split(/[\n,]/)
      .map((s) => s.trim())
      .filter(Boolean)
    const body: ConnectorCreate = {
      kind: draft.kind,
      name: draft.name.trim(),
      enabled: draft.enabled,
      config:
        draft.kind === 'web'
          ? {
              urls,
              collection: draft.collection.trim(),
              deep_crawl: draft.deep_crawl,
              ...(draft.max_pages.trim()
                ? { max_pages: Number(draft.max_pages) }
                : {}),
            }
          : {},
    }
    try {
      if (draft.id) await updateConnector(draft.id, body)
      else await createConnector(body)
      setEditing(null)
      await reload()
    } catch (e) {
      setError(errMsg(e))
    }
  }

  const onDelete = async (id: string) => {
    try {
      await deleteConnector(id)
      await reload()
    } catch (e) {
      setError(errMsg(e))
    }
  }

  const onRun = async (id: string) => {
    setRunning(id)
    setRunResult((r) => ({ ...r, [id]: '' }))
    try {
      const { data } = await runConnector(id)
      setRunResult((r) => ({ ...r, [id]: data }))
    } catch (e) {
      setRunResult((r) => ({ ...r, [id]: errMsg(e) }))
    } finally {
      setRunning(null)
    }
  }

  return (
    <section>
      <div className="flex items-center gap-2 mb-1">
        <h2 className="text-[12.5px] font-medium text-[var(--color-text)]">
          Data source connectors
        </h2>
        <Button
          size="xs"
          variant="ghost"
          leading={<Plus size={11} strokeWidth={2.25} />}
          onClick={() => setEditing({ ...EMPTY_DRAFT })}
          className="ml-auto"
        >
          Add web source
        </Button>
      </div>
      <p className="text-[12px] text-[var(--color-text-muted)] max-w-2xl mb-3 leading-relaxed">
        Managed data sources. A <strong>web</strong> connector crawls its
        seed URLs into a target collection (use <em>Crawl now</em> to run it);
        a <strong>web search</strong> connector marks agentic web search as
        enabled for this tenant.
      </p>

      {error && (
        <div className="mb-3 px-3 py-2 rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_28%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_10%,transparent)] text-[11.5px] text-[var(--color-error)]">
          {error}
        </div>
      )}

      {editing && (
        <ConnectorForm
          draft={editing}
          onChange={setEditing}
          onSave={onSave}
          onCancel={() => setEditing(null)}
        />
      )}

      {loading && <Skeleton className="h-20 w-full" />}
      {!loading && connectors.length === 0 && !editing && (
        <div className="text-[12px] text-[var(--color-text-subtle)] px-3 py-4 rounded-[var(--radius-md)] border border-dashed border-[var(--color-border)]">
          No data source connectors yet. Add a web source to start crawling.
        </div>
      )}
      {!loading && connectors.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {connectors.map((c) => (
            <DataConnectorCard
              key={c.id}
              c={c}
              running={running === c.id}
              result={runResult[c.id]}
              onEdit={() => setEditing(toDraft(c))}
              onDelete={() => onDelete(c.id)}
              onRun={() => onRun(c.id)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

function toDraft(c: DataConnector): DraftConnector {
  const web = asWebConfig(c)
  return {
    id: c.id,
    kind: c.kind,
    name: c.name,
    urls: web ? web.urls.join('\n') : '',
    collection: web?.collection ?? '',
    deep_crawl: web?.deep_crawl ?? false,
    max_pages: web?.max_pages != null ? String(web.max_pages) : '',
    enabled: c.enabled,
  }
}

function DataConnectorCard({
  c,
  running,
  result,
  onEdit,
  onDelete,
  onRun,
}: {
  c: DataConnector
  running: boolean
  result?: RunResponse | string
  onEdit: () => void
  onDelete: () => void
  onRun: () => void
}) {
  const web = asWebConfig(c)
  const Icon = c.kind === 'web' ? Globe : Search
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3.5">
      <div className="flex items-center gap-2 mb-2.5">
        <Icon size={13} strokeWidth={2} className="text-[var(--color-accent-9)]" />
        <span className="text-[12.5px] font-medium text-[var(--color-text)] truncate flex-1">
          {c.name || c.id}
        </span>
        {c.enabled ? (
          <Badge label="enabled" variant="ghost" dot />
        ) : (
          <Badge label="disabled" variant="ghost" />
        )}
      </div>
      <div className="text-[10.5px] uppercase tracking-wider text-[var(--color-text-subtle)] mb-1">
        {connectorKindLabel(c.kind)}
      </div>
      {web && (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px] mb-2">
          <div>
            <dt className="text-[10px] text-[var(--color-text-subtle)] uppercase tracking-wider">
              Collection
            </dt>
            <dd className="text-[var(--color-text)] font-mono truncate" title={web.collection}>
              {web.collection || '—'}
            </dd>
          </div>
          <div>
            <dt className="text-[10px] text-[var(--color-text-subtle)] uppercase tracking-wider">
              URLs
            </dt>
            <dd className="text-[var(--color-text)] font-mono">{web.urls.length}</dd>
          </div>
        </dl>
      )}
      <div className="flex items-center gap-1.5">
        {c.kind === 'web' && (
          <Button
            size="xs"
            variant="secondary"
            leading={<Play size={11} strokeWidth={2.25} className={running ? 'animate-pulse' : ''} />}
            onClick={onRun}
            disabled={running}
          >
            {running ? 'Crawling…' : 'Crawl now'}
          </Button>
        )}
        <Button size="xs" variant="ghost" leading={<Pencil size={11} strokeWidth={2.25} />} onClick={onEdit}>
          Edit
        </Button>
        <Button
          size="xs"
          variant="ghost"
          leading={<Trash2 size={11} strokeWidth={2.25} />}
          onClick={onDelete}
          className="text-[var(--color-error)]"
        >
          Delete
        </Button>
      </div>
      {result != null && (
        <div className="mt-2 text-[11px] font-mono">
          {typeof result === 'string' ? (
            result ? (
              <span className="text-[var(--color-error)]">{result}</span>
            ) : null
          ) : (
            <span className="text-[var(--color-success)]">
              Crawled {result.pages} page(s), inserted {result.chunks_inserted} chunk(s).
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function ConnectorForm({
  draft,
  onChange,
  onSave,
  onCancel,
}: {
  draft: DraftConnector
  onChange: (d: DraftConnector) => void
  onSave: (d: DraftConnector) => void
  onCancel: () => void
}) {
  const set = <K extends keyof DraftConnector>(k: K, v: DraftConnector[K]) =>
    onChange({ ...draft, [k]: v })
  const canSave =
    draft.name.trim().length > 0 &&
    (draft.kind === 'websearch' ||
      (draft.collection.trim().length > 0 && draft.urls.trim().length > 0))

  const inputCls =
    'w-full px-2 h-7 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg)] text-[12px] text-[var(--color-text)] outline-none focus:border-[var(--color-accent-8)]'

  return (
    <div className="mb-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3.5 space-y-2.5">
      <div className="grid grid-cols-2 gap-2.5">
        <label className="block">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">Name</span>
          <input
            className={inputCls}
            value={draft.name}
            onChange={(e) => set('name', e.target.value)}
            placeholder="Docs crawler"
          />
        </label>
        <label className="block">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">Kind</span>
          <select
            className={inputCls}
            value={draft.kind}
            onChange={(e) => set('kind', e.target.value as DraftConnector['kind'])}
          >
            <option value="web">Web crawl</option>
            <option value="websearch">Web search</option>
          </select>
        </label>
      </div>

      {draft.kind === 'web' && (
        <>
          <label className="block">
            <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
              URL(s) — one per line
            </span>
            <textarea
              className={`${inputCls} h-16 py-1`}
              value={draft.urls}
              onChange={(e) => set('urls', e.target.value)}
              placeholder="https://example.com/docs"
            />
          </label>
          <div className="grid grid-cols-2 gap-2.5">
            <label className="block">
              <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
                Target collection
              </span>
              <input
                className={inputCls}
                value={draft.collection}
                onChange={(e) => set('collection', e.target.value)}
                placeholder="web_docs"
              />
            </label>
            <label className="block">
              <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-subtle)]">
                Max pages (deep crawl)
              </span>
              <input
                className={inputCls}
                value={draft.max_pages}
                onChange={(e) => set('max_pages', e.target.value)}
                placeholder="10"
                inputMode="numeric"
              />
            </label>
          </div>
          <label className="flex items-center gap-2 text-[11.5px] text-[var(--color-text-muted)]">
            <input
              type="checkbox"
              checked={draft.deep_crawl}
              onChange={(e) => set('deep_crawl', e.target.checked)}
            />
            Follow links (deep crawl)
          </label>
        </>
      )}

      <label className="flex items-center gap-2 text-[11.5px] text-[var(--color-text-muted)]">
        <input
          type="checkbox"
          checked={draft.enabled}
          onChange={(e) => set('enabled', e.target.checked)}
        />
        Enabled
      </label>

      <div className="flex items-center gap-2 pt-1">
        <Button size="xs" variant="primary" onClick={() => onSave(draft)} disabled={!canSave}>
          {draft.id ? 'Save changes' : 'Create connector'}
        </Button>
        <Button size="xs" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

function errMsg(e: unknown): string {
  const anyE = e as { response?: { data?: { error?: string } }; message?: string }
  return anyE?.response?.data?.error ?? anyE?.message ?? 'Request failed'
}

// Exported for the upcoming Cmd-K command "open settings → connector by name".
export { ChevronRight, Search }
