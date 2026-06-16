import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Command } from 'cmdk'
import * as Dialog from '@radix-ui/react-dialog'
import {
  Activity,
  Atom,
  BarChart3,
  Beaker,
  Database,
  FileText,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers,
  ScrollText,
  Search,
  Settings,
  ShieldAlert,
  Sparkles,
  TerminalSquare,
  type LucideIcon,
} from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSavedObjects, type SavedObject } from '../../stores/savedObjectStore'
import { KBD, modKey } from '../primitives/KBD'
import { cn } from '../../lib/cn'
import { fetchCollection, llmComplete } from '../../api/collections'

interface CmdItem {
  id: string
  label: string
  hint?: string
  icon: LucideIcon
  group: string
  keywords?: string[]
  onSelect: () => void
}

interface CommandPaletteProps {
  open: boolean
  onOpenChange: (v: boolean) => void
}

/**
 * Cmd-K command palette. Fuzzy search over: navigation, collections,
 * recent saved objects, slash sub-commands.
 */
export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const navigate = useNavigate()
  const { collections, fetchCollections } = useCollectionStore()
  const objectsMap = useSavedObjects((s) => s.objects)
  const touch = useSavedObjects((s) => s.touch)
  const savedObjects = useMemo<SavedObject[]>(
    () =>
      Object.values(objectsMap).sort(
        (a, b) => (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
      ),
    [objectsMap],
  )
  const [query, setQuery] = useState('')
  const [drafting, setDrafting] = useState<string | null>(null)
  const [draftError, setDraftError] = useState<string | null>(null)

  useEffect(() => {
    if (open) fetchCollections()
  }, [open, fetchCollections])

  useEffect(() => {
    if (!open) {
      setQuery('')
      setDrafting(null)
      setDraftError(null)
    }
  }, [open])

  const close = () => onOpenChange(false)

  // When query starts with "?", treat it as NL → search-query draft.
  const nlText = query.startsWith('?') ? query.slice(1).trim() : ''

  const draftQuery = async (collectionName: string) => {
    setDraftError(null)
    setDrafting(collectionName)
    try {
      const { data: schema } = await fetchCollection(collectionName)
      const fieldDescs = schema.fields
        .map((f) => {
          const tags: string[] = [f.type]
          if (f.filterable) tags.push('filterable')
          if (f.type === 'vector' && f.dimension) tags.push(`${f.dimension}d`)
          return `- ${f.name} (${tags.join(', ')})`
        })
        .join('\n')
      const draft = await callDraftLlm(collectionName, fieldDescs, nlText)
      const url = buildSearchUrl(collectionName, draft, schema.fields)
      navigate(url)
      close()
    } catch (e) {
      setDraftError(e instanceof Error ? e.message : 'draft failed')
    } finally {
      setDrafting(null)
    }
  }

  const items = useMemo<CmdItem[]>(() => {
    const navItems: CmdItem[] = [
      {
        id: 'nav-collections',
        label: 'Collections',
        hint: 'G C',
        icon: Database,
        group: 'Navigate',
        onSelect: () => navigate('/collections'),
      },
      {
        id: 'nav-search',
        label: 'Search Playground',
        hint: 'G S',
        icon: Search,
        group: 'Navigate',
        onSelect: () => navigate('/search'),
      },
      {
        id: 'nav-atlas',
        label: 'Atlas',
        hint: 'G A',
        icon: Atom,
        group: 'Navigate',
        onSelect: () => navigate('/atlas'),
      },
      {
        id: 'nav-quality',
        label: 'Quality Lab',
        icon: FlaskConical,
        group: 'Navigate',
        onSelect: () => navigate('/quality'),
      },
      {
        id: 'nav-evaluation',
        label: 'Evaluation',
        icon: Beaker,
        group: 'Navigate',
        onSelect: () => navigate('/evaluation'),
      },
      {
        id: 'nav-pipelines',
        label: 'Pipelines',
        icon: GitBranch,
        group: 'Navigate',
        onSelect: () => navigate('/pipelines'),
      },
      {
        id: 'nav-health',
        label: 'Health',
        icon: Activity,
        group: 'Navigate',
        onSelect: () => navigate('/health'),
      },
      {
        id: 'nav-indexes',
        label: 'Indexes',
        icon: Layers,
        group: 'Navigate',
        onSelect: () => navigate('/indexes'),
      },
      {
        id: 'nav-observability',
        label: 'Observability',
        icon: BarChart3,
        group: 'Navigate',
        onSelect: () => navigate('/observability'),
      },
      {
        id: 'nav-insights',
        label: 'Insights',
        icon: ShieldAlert,
        group: 'Navigate',
        keywords: ['advisor', 'ops', 'events', 'alerts', 'remediation'],
        onSelect: () => navigate('/insights'),
      },
      {
        id: 'nav-slos',
        label: 'SLOs',
        icon: Gauge,
        group: 'Navigate',
        keywords: ['slo', 'objective', 'sli', 'error budget', 'burn rate', 'reliability'],
        onSelect: () => navigate('/slos'),
      },
      {
        id: 'nav-logs',
        label: 'Logs',
        icon: ScrollText,
        group: 'Navigate',
        keywords: ['logs', 'log', 'tail', 'trace', 'slow queries', 'errors', 'warnings'],
        onSelect: () => navigate('/logs'),
      },
      {
        id: 'nav-console',
        label: 'Dev Console',
        icon: TerminalSquare,
        group: 'Navigate',
        onSelect: () => navigate('/console'),
      },
      {
        id: 'nav-settings',
        label: 'Settings',
        icon: Settings,
        group: 'Navigate',
        onSelect: () => navigate('/settings'),
      },
    ]

    const colItems: CmdItem[] = collections.map((c) => ({
      id: `col-${c.name}`,
      label: c.name,
      hint: `${c.field_count}f · ${c.shard_count}s`,
      icon: Database,
      group: 'Collections',
      keywords: ['collection', c.name],
      onSelect: () => navigate(`/collections/${c.name}`),
    }))

    const slashItems: CmdItem[] = [
      {
        id: 'slash-new',
        label: '/new — Create a collection',
        icon: Database,
        group: 'Actions',
        keywords: ['create', 'new', 'collection'],
        onSelect: () => navigate('/collections?new=1'),
      },
      {
        id: 'slash-process',
        label: '/process — Upload & process a document',
        icon: FileText,
        group: 'Actions',
        keywords: ['upload', 'pdf', 'ocr', 'process'],
        onSelect: () => navigate('/pipelines'),
      },
    ]

    const recentItems: CmdItem[] = savedObjects.slice(0, 6).map((obj) => ({
      id: `saved-${obj.id}`,
      label: obj.name,
      hint: obj.type,
      icon: iconForSaved(obj),
      group: 'Recent',
      keywords: [obj.type, obj.name],
      onSelect: () => {
        touch(obj.id)
        navigateToSaved(obj, navigate)
      },
    }))

    // NL-draft mode — when the query begins with "?", offer one
    // "Ask LLM" item per (top-N) collection. Selection runs the
    // LLM call against that collection's schema.
    const nlItems: CmdItem[] =
      nlText.length > 0
        ? collections.slice(0, 5).map((c) => ({
            id: `nl-${c.name}`,
            label: `Ask ${c.name}: ${nlText}`,
            hint: drafting === c.name ? 'drafting…' : '↵ draft',
            icon: Sparkles,
            group: 'Ask Veda',
            keywords: ['ask', 'nl', 'natural', c.name],
            onSelect: () => {
              if (drafting) return
              void draftQuery(c.name)
            },
          }))
        : []

    if (nlItems.length > 0) {
      return nlItems
    }
    return [...recentItems, ...navItems, ...colItems, ...slashItems]
  }, [collections, savedObjects, navigate, touch, nlText, drafting])

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/45 backdrop-blur-[2px] data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=closed]:animate-out data-[state=closed]:fade-out-0" style={{ animationDuration: '120ms' }} />
        <Dialog.Content
          className={cn(
            'fixed left-1/2 top-[22%] z-50 -translate-x-1/2 w-[min(640px,calc(100vw-32px))]',
            'rounded-[var(--radius-lg)] border border-[var(--color-border-strong)]',
            'bg-[var(--color-bg-elevated)] shadow-[var(--shadow-lg)] overflow-hidden',
            'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
            'data-[state=closed]:animate-out data-[state=closed]:fade-out-0',
          )}
          style={{ animationDuration: '120ms' }}
          aria-label="Command palette"
        >
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Command label="Command Palette" loop shouldFilter={nlText.length === 0}>
            <div className="flex items-center gap-2 px-3 h-11 border-b border-[var(--color-border)]">
              <Search
                size={14}
                strokeWidth={2.25}
                className="text-[var(--color-text-subtle)]"
              />
              <Command.Input
                autoFocus
                value={query}
                onValueChange={setQuery}
                placeholder="Jump to a collection, action, or saved view…"
                className="flex-1 bg-transparent text-[14px] text-[var(--color-text)] placeholder:text-[var(--color-text-subtle)] outline-none"
              />
              <KBD>esc</KBD>
            </div>
            <Command.List className="max-h-[60vh] overflow-y-auto py-1.5">
              <Command.Empty className="text-[12px] text-[var(--color-text-subtle)] px-3 py-6 text-center">
                No matches.
              </Command.Empty>
              {groupBy(items, 'group').map(([group, list]) => (
                <Command.Group key={group} heading={group} className="px-1">
                  <div className="px-2 pt-2 pb-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
                    {group}
                  </div>
                  {list.map((item) => (
                    <Command.Item
                      key={item.id}
                      value={[item.label, ...(item.keywords ?? [])].join(' ')}
                      onSelect={() => {
                        item.onSelect()
                        close()
                      }}
                      className={cn(
                        'flex items-center gap-2 px-2 py-1.5 mx-0.5 rounded-[var(--radius-sm)] cursor-pointer',
                        'text-[12.5px] text-[var(--color-text)]',
                        'data-[selected=true]:bg-[var(--color-bg-hover)]',
                        'data-[selected=true]:text-[var(--color-text)]',
                      )}
                    >
                      <item.icon
                        size={13}
                        strokeWidth={2.25}
                        className="text-[var(--color-text-subtle)] flex-none"
                      />
                      <span className="flex-1 truncate">{item.label}</span>
                      {item.hint && (
                        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)]">
                          {item.hint}
                        </span>
                      )}
                    </Command.Item>
                  ))}
                </Command.Group>
              ))}
            </Command.List>
            <div className="flex items-center gap-3 px-3 h-7 border-t border-[var(--color-border)] text-[10.5px] text-[var(--color-text-subtle)]">
              {draftError ? (
                <span className="text-[var(--color-error)]">
                  ✗ {draftError}
                </span>
              ) : drafting ? (
                <span className="text-[var(--color-accent-11)]">
                  ✨ Drafting query for {drafting}…
                </span>
              ) : nlText.length > 0 ? (
                <span>Tip: ↵ to ask the LLM</span>
              ) : (
                <>
                  <span className="flex items-center gap-1">
                    <KBD>↑</KBD>
                    <KBD>↓</KBD>
                    navigate
                  </span>
                  <span className="flex items-center gap-1">
                    <KBD>↵</KBD>
                    select
                  </span>
                </>
              )}
              <span className="flex items-center gap-1 ml-auto">
                <KBD>{modKey()}</KBD>
                <KBD>K</KBD>
                toggle
              </span>
            </div>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function iconForSaved(obj: SavedObject): LucideIcon {
  switch (obj.type) {
    case 'search':
      return Search
    case 'atlas':
      return Atom
    case 'eval':
      return FlaskConical
    case 'console_query':
      return TerminalSquare
    case 'pipeline':
      return GitBranch
    default:
      return FileText
  }
}

function navigateToSaved(obj: SavedObject, nav: (to: string) => void) {
  // For now, route to the canonical surface for each type. The actual
  // restore-state-from-body happens in each page once Phase 1+ rewires them.
  switch (obj.type) {
    case 'search':
      nav(`/search?saved=${obj.id}`)
      break
    case 'atlas':
      nav(`/atlas?saved=${obj.id}`)
      break
    case 'eval':
      nav(`/quality?saved=${obj.id}`)
      break
    case 'console_query':
      nav(`/console?saved=${obj.id}`)
      break
    case 'pipeline':
      nav(`/pipelines?saved=${obj.id}`)
      break
  }
}

interface DraftResult {
  text: string
  filter?: string
  mode: 'dense' | 'hybrid' | 'bm25'
}

/**
 * Ask the LLM to convert a natural-language question into Veda search
 * parameters. Returns { text, filter?, mode }. The system prompt
 * pins the JSON shape; the user prompt ships the collection schema +
 * the question. Filter uses the SQL-like DSL the gateway already
 * parses (e.g. `category = 'docs' AND year >= 2024`).
 */
async function callDraftLlm(
  collectionName: string,
  fieldDescs: string,
  question: string,
): Promise<DraftResult> {
  const { data } = await llmComplete(
    [
      {
        role: 'system',
        content:
          'You convert natural-language questions into Veda search ' +
          'parameters. Reply with a single JSON object — no prose, no ' +
          'markdown fence — with exactly these keys:\n' +
          '  "text": string  // the semantic search text (required)\n' +
          '  "filter": string or null  // optional SQL-like Veda filter, e.g. "category = \'pdf\' AND year >= 2024". Only reference filterable fields. Use null when no obvious filter is implied.\n' +
          '  "mode": "dense"|"hybrid"|"bm25"  // pick "hybrid" by default; "bm25" only when the user explicitly asks for keyword/exact matching\n' +
          'Filter operators supported: =, !=, >, >=, <, <=, AND, OR, ' +
          'NOT, IN (...), BETWEEN, IS NULL, IS NOT NULL. Strings ' +
          'use single quotes.',
      },
      {
        role: 'user',
        content:
          `Collection: ${collectionName}\nFields:\n${fieldDescs}\n\n` +
          `Question: ${question}`,
      },
    ],
    { temperature: 0.1, max_tokens: 200, json_mode: true },
  )
  let parsed: unknown
  try {
    parsed = JSON.parse(data.content)
  } catch {
    // Fallback — strip ```json fences if upstream ignored response_format.
    const stripped = data.content
      .trim()
      .replace(/^```(?:json)?/i, '')
      .replace(/```$/, '')
      .trim()
    parsed = JSON.parse(stripped)
  }
  const obj = parsed as Record<string, unknown>
  const text = typeof obj.text === 'string' ? obj.text : question
  const filter =
    typeof obj.filter === 'string' && obj.filter.trim().length > 0
      ? obj.filter
      : undefined
  const modeRaw = typeof obj.mode === 'string' ? obj.mode.toLowerCase() : ''
  const mode: DraftResult['mode'] =
    modeRaw === 'dense' || modeRaw === 'bm25' ? modeRaw : 'hybrid'
  return { text, filter, mode }
}

interface FieldShape {
  name: string
  type: string
}

/**
 * Build a /search/<col> URL with a hash that pre-fills the
 * SearchPlayground useUrlState `s` key. The shape must match
 * PlaygroundState in SearchPlayground.tsx.
 */
function buildSearchUrl(
  collectionName: string,
  draft: DraftResult,
  fields: FieldShape[],
): string {
  const firstVec = fields.find((f) => f.type === 'vector')?.name ?? ''
  const firstStr = fields.find((f) => f.type === 'string')?.name ?? ''
  const state = {
    collection: collectionName,
    mode: draft.mode,
    text: draft.text,
    vectorField: firstVec,
    bm25Field: firstStr,
    filter: draft.filter ?? '',
    topK: 10,
    truncateTo: null,
    fusion: 'rrf',
    rrfK: 60,
    vectorWeight: 0.5,
    bm25Weight: 0.5,
    rerankOn: false,
    rerankField: firstStr,
    explainOn: true,
  }
  const hash = new URLSearchParams()
  hash.set('s', JSON.stringify(state))
  return `/search/${encodeURIComponent(collectionName)}#${hash.toString()}`
}

function groupBy<T>(items: T[], key: keyof T): [string, T[]][] {
  const map = new Map<string, T[]>()
  for (const item of items) {
    const k = String(item[key])
    if (!map.has(k)) map.set(k, [])
    map.get(k)!.push(item)
  }
  return Array.from(map.entries())
}
