import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { LayoutDashboard, Plus, Trash2 } from 'lucide-react'
import {
  useSavedObjects,
  type DashboardBody,
  type SavedObject,
} from '../../stores/savedObjectStore'
import { Button, Input, toast } from '../../components/primitives'

/**
 * Phase 6.5 — list of saved dashboards. Each dashboard is a
 * SavedObject<DashboardBody> backed by the existing zustand+persist
 * store, so creation / rename / delete piggyback on the same code
 * path used by saved searches and eval runs.
 */
export function DashboardsList() {
  const navigate = useNavigate()
  const list = useSavedObjects((s) => s.list)
  const upsert = useSavedObjects((s) => s.upsert)
  const remove = useSavedObjects((s) => s.remove)
  const dashboards = list('dashboard') as SavedObject<DashboardBody>[]

  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')

  const create = () => {
    const trimmed = name.trim() || 'Untitled dashboard'
    const obj = upsert<DashboardBody>({
      id: '',
      type: 'dashboard',
      name: trimmed,
      body: { items: [] },
    })
    setName('')
    setCreating(false)
    navigate(`/dashboards/${obj.id}`)
  }

  return (
    <div className="px-6 py-5">
      <header className="flex items-center gap-3 mb-5">
        <div className="flex-1">
          <h1 className="text-[20px] font-semibold tracking-tight text-[var(--color-text)] mb-0.5">
            Dashboards
          </h1>
          <p className="text-[12.5px] text-[var(--color-text-muted)]">
            {dashboards.length} saved dashboard
            {dashboards.length !== 1 ? 's' : ''} · drag your saved
            searches, atlas views, eval runs, and coverage cards onto
            a single shareable URL.
          </p>
        </div>
        <Button
          variant="primary"
          leading={<Plus size={12} strokeWidth={2.25} />}
          onClick={() => setCreating((v) => !v)}
        >
          New dashboard
        </Button>
      </header>

      {creating && (
        <div className="mb-4 flex items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2">
          <Input
            placeholder="Name your dashboard…"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') create()
              if (e.key === 'Escape') setCreating(false)
            }}
            autoFocus
            className="flex-1"
          />
          <Button variant="primary" size="sm" onClick={create}>
            Create
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCreating(false)}
          >
            Cancel
          </Button>
        </div>
      )}

      {dashboards.length === 0 && !creating && (
        <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-6 py-10 text-center">
          <LayoutDashboard
            size={32}
            strokeWidth={1.5}
            className="text-[var(--color-text-subtle)] mx-auto mb-3"
          />
          <p className="text-[13px] text-[var(--color-text-muted)] mb-1">
            No dashboards yet.
          </p>
          <p className="text-[11.5px] text-[var(--color-text-subtle)] mb-4">
            A dashboard is a grid of saved searches, atlas views,
            and eval runs you want to look at together.
          </p>
          <Button
            variant="primary"
            size="sm"
            leading={<Plus size={11} strokeWidth={2.25} />}
            onClick={() => setCreating(true)}
          >
            Create one
          </Button>
        </div>
      )}

      <ul className="grid grid-cols-3 gap-3">
        {dashboards.map((d) => (
          <li key={d.id}>
            <div
              role="button"
              tabIndex={0}
              onClick={() => navigate(`/dashboards/${d.id}`)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') navigate(`/dashboards/${d.id}`)
              }}
              className="group rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-strong)] px-3 py-3 cursor-pointer"
            >
              <div className="flex items-start gap-2 mb-1.5">
                <LayoutDashboard
                  size={13}
                  strokeWidth={2}
                  className="text-[var(--color-accent-9)] mt-0.5"
                />
                <span className="flex-1 text-[13px] font-medium text-[var(--color-text)] truncate">
                  {d.name}
                </span>
                <button
                  type="button"
                  className="opacity-0 group-hover:opacity-100 p-1 rounded text-[var(--color-text-subtle)] hover:text-[var(--color-error)]"
                  title="Delete dashboard"
                  onClick={(e) => {
                    e.stopPropagation()
                    if (!confirm(`Delete "${d.name}"?`)) return
                    remove(d.id)
                    toast.success(`Deleted ${d.name}`)
                  }}
                >
                  <Trash2 size={11} strokeWidth={2.25} />
                </button>
              </div>
              <div className="text-[10.5px] font-mono tabular-nums text-[var(--color-text-subtle)]">
                {d.body.items.length} tile
                {d.body.items.length !== 1 ? 's' : ''} · updated{' '}
                {timeAgo(d.updatedAt)}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function timeAgo(ts: number): string {
  const dt = Date.now() - ts
  const m = Math.floor(dt / 60_000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}
