import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Atom,
  ChevronLeft,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  Plus,
  Search,
  TerminalSquare,
  X,
} from 'lucide-react'
import {
  useSavedObjects,
  type DashboardBody,
  type SavedObject,
  type SavedObjectType,
} from '../../stores/savedObjectStore'
import { Button, Input, Modal, toast } from '../../components/primitives'
import { cn } from '../../lib/cn'

const TYPE_LABELS: Record<SavedObjectType, string> = {
  search: 'Search',
  atlas: 'Atlas view',
  eval: 'Eval set',
  eval_run: 'Eval run',
  console_query: 'Console query',
  pipeline: 'Pipeline',
  dashboard: 'Dashboard',
}

const TYPE_ICONS: Record<SavedObjectType, typeof Search> = {
  search: Search,
  atlas: Atom,
  eval: FlaskConical,
  eval_run: FlaskConical,
  console_query: TerminalSquare,
  pipeline: GitBranch,
  dashboard: LayoutDashboard,
}

const TILE_LINK: Record<SavedObjectType, (id: string) => string> = {
  search: (id) => `/search?saved=${id}`,
  atlas: (id) => `/atlas?saved=${id}`,
  eval: (id) => `/quality?saved=${id}`,
  eval_run: (id) => `/quality?saved=${id}`,
  console_query: (id) => `/console?saved=${id}`,
  pipeline: (id) => `/pipelines?saved=${id}`,
  dashboard: (id) => `/dashboards/${id}`,
}

export function DashboardPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const get = useSavedObjects((s) => s.get)
  const upsert = useSavedObjects((s) => s.upsert)
  const objects = useSavedObjects((s) => s.objects)

  const dashboard = id
    ? (get<DashboardBody>(id) as SavedObject<DashboardBody> | undefined)
    : undefined

  const [picking, setPicking] = useState(false)
  const [editingName, setEditingName] = useState(false)
  const [draftName, setDraftName] = useState(dashboard?.name ?? '')

  if (!dashboard) {
    return (
      <div className="px-6 py-10 text-center">
        <p className="text-[12.5px] text-[var(--color-text-muted)] mb-3">
          Dashboard not found.
        </p>
        <Button
          variant="secondary"
          size="sm"
          leading={<ChevronLeft size={11} strokeWidth={2.25} />}
          onClick={() => navigate('/dashboards')}
        >
          Back to dashboards
        </Button>
      </div>
    )
  }

  const tiles = dashboard.body.items
    .map((it) => objects[it.savedObjectId])
    .filter((o): o is SavedObject => Boolean(o))

  const update = (body: DashboardBody, name?: string) =>
    upsert<DashboardBody>({
      id: dashboard.id,
      type: 'dashboard',
      name: name ?? dashboard.name,
      body,
    })

  const addTile = (objId: string) => {
    if (dashboard.body.items.some((i) => i.savedObjectId === objId)) {
      toast.error('Already on this dashboard.')
      return
    }
    update({
      items: [...dashboard.body.items, { savedObjectId: objId }],
    })
    setPicking(false)
  }

  const removeTile = (objId: string) =>
    update({
      items: dashboard.body.items.filter((i) => i.savedObjectId !== objId),
    })

  const moveTile = (objId: string, dir: -1 | 1) => {
    const items = [...dashboard.body.items]
    const i = items.findIndex((t) => t.savedObjectId === objId)
    if (i < 0) return
    const j = i + dir
    if (j < 0 || j >= items.length) return
    ;[items[i], items[j]] = [items[j]!, items[i]!]
    update({ items })
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--color-bg)] overflow-hidden">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <button
          type="button"
          onClick={() => navigate('/dashboards')}
          className="p-1 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-text)]"
          title="Back"
        >
          <ChevronLeft size={14} strokeWidth={2.25} />
        </button>
        <LayoutDashboard
          size={14}
          strokeWidth={2.25}
          className="text-[var(--color-accent-9)]"
        />
        {editingName ? (
          <Input
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                update(dashboard.body, draftName.trim() || dashboard.name)
                setEditingName(false)
              }
              if (e.key === 'Escape') setEditingName(false)
            }}
            onBlur={() => {
              update(dashboard.body, draftName.trim() || dashboard.name)
              setEditingName(false)
            }}
            autoFocus
            className="flex-1 max-w-[420px]"
          />
        ) : (
          <button
            type="button"
            onClick={() => {
              setDraftName(dashboard.name)
              setEditingName(true)
            }}
            className="flex-1 text-left text-[14px] font-semibold text-[var(--color-text)] truncate hover:text-[var(--color-accent-11)]"
            title="Rename"
          >
            {dashboard.name}
          </button>
        )}
        <Button
          variant="primary"
          size="sm"
          leading={<Plus size={11} strokeWidth={2.25} />}
          onClick={() => setPicking(true)}
        >
          Add tile
        </Button>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4">
        {tiles.length === 0 ? (
          <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-6 py-10 text-center">
            <p className="text-[13px] text-[var(--color-text-muted)] mb-1">
              Empty dashboard.
            </p>
            <p className="text-[11.5px] text-[var(--color-text-subtle)] mb-4">
              Add a saved search, an atlas view, or an eval run to
              start composing.
            </p>
            <Button
              variant="primary"
              size="sm"
              leading={<Plus size={11} strokeWidth={2.25} />}
              onClick={() => setPicking(true)}
            >
              Add tile
            </Button>
          </div>
        ) : (
          <ul className="grid grid-cols-2 gap-3">
            {tiles.map((obj, i) => (
              <li key={obj.id}>
                <Tile
                  obj={obj}
                  onOpen={() => navigate(TILE_LINK[obj.type](obj.id))}
                  onRemove={() => removeTile(obj.id)}
                  onUp={i > 0 ? () => moveTile(obj.id, -1) : undefined}
                  onDown={
                    i < tiles.length - 1
                      ? () => moveTile(obj.id, 1)
                      : undefined
                  }
                />
              </li>
            ))}
          </ul>
        )}
      </div>

      <Modal
        open={picking}
        onClose={() => setPicking(false)}
        title="Add a tile"
        description="Pick a saved object to drop on this dashboard."
      >
        <TilePicker
          existing={new Set(dashboard.body.items.map((i) => i.savedObjectId))}
          onPick={addTile}
        />
      </Modal>
    </div>
  )
}

function Tile({
  obj,
  onOpen,
  onRemove,
  onUp,
  onDown,
}: {
  obj: SavedObject
  onOpen: () => void
  onRemove: () => void
  onUp?: () => void
  onDown?: () => void
}) {
  const Icon = TYPE_ICONS[obj.type] ?? Database
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onOpen()
      }}
      className="group rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-strong)] px-3 py-3 cursor-pointer flex flex-col gap-1 min-h-[100px]"
    >
      <div className="flex items-start gap-2">
        <Icon
          size={13}
          strokeWidth={2}
          className="text-[var(--color-accent-9)] mt-0.5 flex-none"
        />
        <span className="flex-1 text-[12.5px] font-medium text-[var(--color-text)] truncate">
          {obj.name}
        </span>
        <span
          onClick={(e) => e.stopPropagation()}
          className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5"
        >
          {onUp && (
            <TileBtn onClick={onUp} title="Move up">
              ↑
            </TileBtn>
          )}
          {onDown && (
            <TileBtn onClick={onDown} title="Move down">
              ↓
            </TileBtn>
          )}
          <TileBtn onClick={onRemove} title="Remove" danger>
            <X size={11} strokeWidth={2.25} />
          </TileBtn>
        </span>
      </div>
      <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
        {TYPE_LABELS[obj.type]}
      </div>
      <div className="mt-auto text-[10.5px] font-mono tabular-nums text-[var(--color-text-subtle)]">
        {obj.id.slice(0, 28)}
        {obj.id.length > 28 ? '…' : ''}
      </div>
    </div>
  )
}

function TileBtn({
  onClick,
  title,
  danger,
  children,
}: {
  onClick: () => void
  title: string
  danger?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={cn(
        'p-1 rounded text-[10px] text-[var(--color-text-subtle)]',
        danger
          ? 'hover:text-[var(--color-error)]'
          : 'hover:text-[var(--color-text)]',
      )}
    >
      {children}
    </button>
  )
}

function TilePicker({
  existing,
  onPick,
}: {
  existing: Set<string>
  onPick: (id: string) => void
}) {
  const objects = useSavedObjects((s) => s.objects)
  const [filter, setFilter] = useState('')
  const candidates = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return Object.values(objects)
      .filter((o) => o.type !== 'dashboard')
      .filter((o) => !f || o.name.toLowerCase().includes(f))
      .sort(
        (a, b) =>
          (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
      )
  }, [objects, filter])

  return (
    <div className="space-y-2">
      <Input
        placeholder="Filter saved objects…"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        leading={<Search size={11} strokeWidth={2.25} />}
        autoFocus
      />
      <ul className="max-h-[320px] overflow-y-auto divide-y divide-[var(--color-border)] rounded-[var(--radius-sm)] border border-[var(--color-border)]">
        {candidates.length === 0 && (
          <li className="px-3 py-6 text-center text-[12px] text-[var(--color-text-subtle)] italic">
            No saved objects match.
          </li>
        )}
        {candidates.map((o) => {
          const Icon = TYPE_ICONS[o.type] ?? FileText
          const taken = existing.has(o.id)
          return (
            <li
              key={o.id}
              className={cn(
                'px-3 py-2 flex items-center gap-2 text-[12px]',
                taken
                  ? 'opacity-50'
                  : 'hover:bg-[var(--color-bg-hover)] cursor-pointer',
              )}
              onClick={() => !taken && onPick(o.id)}
            >
              <Icon
                size={12}
                strokeWidth={2.25}
                className="text-[var(--color-text-subtle)] flex-none"
              />
              <span className="flex-1 truncate text-[var(--color-text)]">
                {o.name}
              </span>
              <span className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
                {TYPE_LABELS[o.type]}
              </span>
              {taken && (
                <span className="text-[10px] text-[var(--color-text-subtle)] italic">
                  added
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
