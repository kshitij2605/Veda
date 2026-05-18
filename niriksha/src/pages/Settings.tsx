import { useEffect, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  HardDrive,
  Image as ImageIcon,
  Layers,
  Network,
  RefreshCw,
  ScanText,
  Search,
  Settings as SettingsIcon,
  Sparkles,
  Type,
} from 'lucide-react'
import { useCollectionStore } from '../stores/collectionStore'
import { Badge, Button, Skeleton } from '../components/primitives'

interface Connector {
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
  const [connectors, setConnectors] = useState<Connector[]>([])
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

      <div className="flex-1 overflow-auto px-4 py-4">
        <p className="text-[12px] text-[var(--color-text-muted)] max-w-2xl mb-4 leading-relaxed">
          Named credentials and endpoints the Veda daemon is wired against.
          This view is read-only for now — full CRUD lands when the gateway
          connectors module is added (see <code className="font-mono text-[11.5px] text-[var(--color-text)]">niriksha/AUDIT.md</code>).
        </p>

        {!health && <Skeleton className="h-32 w-full" />}
        {health && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {connectors.map((c) => (
              <ConnectorCard key={c.id} c={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ConnectorCard({ c }: { c: Connector }) {
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

function StatusBadge({ status }: { status?: Connector['status'] }) {
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

const ICONS: Record<Connector['type'], React.ComponentType<{ size?: number; strokeWidth?: number; className?: string }>> = {
  'embedding-text': Type,
  'embedding-multimodal': ImageIcon,
  ocr: ScanText,
  reranker: Layers,
  llm: Sparkles,
  'blob-storage': HardDrive,
  messaging: Network,
}

function labelForType(t: Connector['type']): string {
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
): Connector[] {
  const checks = health?.checks ?? {}
  const toStatus = (k: string): Connector['status'] => {
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
      name: 'LLM / VLM (Qwen3.5-122B-A10B)',
      type: 'llm',
      status: 'unknown',
      endpoint: 'llm-host.tdc:9000/v1/chat/completions',
      details: [
        { label: 'Model', value: 'Qwen/Qwen3.5-122B-A10B' },
        { label: 'Use', value: 'cluster-labels · failure-mode · NL→DSL' },
        { label: 'Connector wire', value: 'planned, Phase 5.8' },
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

// Exported for the upcoming Cmd-K command "open settings → connector by name".
export { ChevronRight, Search }
