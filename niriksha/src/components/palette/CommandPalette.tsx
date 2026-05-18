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
  GitBranch,
  Layers,
  Search,
  Settings,
  TerminalSquare,
  type LucideIcon,
} from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSavedObjects, type SavedObject } from '../../stores/savedObjectStore'
import { KBD, modKey } from '../primitives/KBD'
import { cn } from '../../lib/cn'

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
  const savedObjects = useSavedObjects((s) => s.list())
  const touch = useSavedObjects((s) => s.touch)
  const [query, setQuery] = useState('')

  useEffect(() => {
    if (open) fetchCollections()
  }, [open, fetchCollections])

  useEffect(() => {
    if (!open) setQuery('')
  }, [open])

  const close = () => onOpenChange(false)

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

    return [...recentItems, ...navItems, ...colItems, ...slashItems]
  }, [collections, savedObjects, navigate, touch])

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
          <Command label="Command Palette" loop>
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
              <span className="flex items-center gap-1">
                <KBD>↑</KBD>
                <KBD>↓</KBD>
                navigate
              </span>
              <span className="flex items-center gap-1">
                <KBD>↵</KBD>
                select
              </span>
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

function groupBy<T>(items: T[], key: keyof T): [string, T[]][] {
  const map = new Map<string, T[]>()
  for (const item of items) {
    const k = String(item[key])
    if (!map.has(k)) map.set(k, [])
    map.get(k)!.push(item)
  }
  return Array.from(map.entries())
}
