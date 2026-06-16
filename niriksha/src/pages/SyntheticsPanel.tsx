import { useCallback, useEffect, useMemo, useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import {
  createSynthetic,
  deleteSynthetic,
  getSyntheticResults,
  listSynthetics,
  probeKindLabel,
  type ProbeKind,
  type ProbeKindTag,
  type SyntheticCreate,
  type SyntheticProbe,
  type SyntheticSummary,
} from '../api/synthetics'
import {
  Badge,
  Button,
  Input,
  Modal,
  Select,
  toast,
} from '../components/primitives'

/**
 * Synthetics — per-probe uptime, last result (ok/fail + latency),
 * consecutive failures (warning-colored), and last-checked time. A create
 * form (probe kind + interval + timeout) and delete. Null-safe ("—") like
 * the Resources panel. Designed to slot into the System Health page.
 */
export function SyntheticsPanel() {
  const [probes, setProbes] = useState<SyntheticProbe[] | null>(null)
  const [summaries, setSummaries] = useState<Record<string, SyntheticSummary>>({})
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<SyntheticProbe | null>(null)

  const load = useCallback(async () => {
    setError(null)
    try {
      const { data } = await listSynthetics()
      setProbes(data.synthetics)
      const entries = await Promise.all(
        data.synthetics.map(async (p) => {
          try {
            const { data: r } = await getSyntheticResults(p.id)
            return [p.id, r.summary] as const
          } catch {
            return null
          }
        }),
      )
      const next: Record<string, SyntheticSummary> = {}
      for (const e of entries) if (e) next[e[0]] = e[1]
      setSummaries(next)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load synthetics')
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const onDelete = async (p: SyntheticProbe) => {
    try {
      await deleteSynthetic(p.id)
      toast.success(`Deleted “${p.name}”`)
      setConfirmDelete(null)
      await load()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
          Synthetics
        </h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setCreating(true)}
            className="inline-flex items-center gap-1 text-xs text-accent hover:text-accent-hover transition-colors"
          >
            <Plus size={12} strokeWidth={2.5} />
            New probe
          </button>
          <button
            onClick={load}
            className="text-xs text-accent hover:text-accent-hover transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg bg-error/10 border border-error/20 px-4 py-3 text-sm text-error">
          Synthetics lookup failed: {error}
        </div>
      ) : probes === null ? (
        <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
          Loading synthetics...
        </div>
      ) : probes.length === 0 ? (
        <div className="rounded-lg border border-border bg-bg-secondary p-5 text-sm text-text-muted">
          No synthetic probes configured.
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-bg-secondary text-text-muted text-left">
                <th className="px-4 py-3 font-medium">Probe</th>
                <th className="px-4 py-3 font-medium">Kind</th>
                <th className="px-4 py-3 font-medium text-right">Uptime</th>
                <th className="px-4 py-3 font-medium">Last result</th>
                <th className="px-4 py-3 font-medium text-right">Failures</th>
                <th className="px-4 py-3 font-medium">Last checked</th>
                <th className="px-4 py-3 font-medium"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {probes.map((p) => {
                const s = summaries[p.id]
                return (
                  <tr key={p.id} className="hover:bg-bg-tertiary transition-colors">
                    <td className="px-4 py-3 font-medium text-text-primary">
                      {p.name}
                    </td>
                    <td className="px-4 py-3 text-text-secondary">
                      {probeKindLabel(p.kind)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-text-secondary">
                      {fmtPct(s?.uptime_pct ?? null)}
                    </td>
                    <td className="px-4 py-3">
                      {s?.last_ok == null ? (
                        <span className="text-text-muted">—</span>
                      ) : (
                        <span className="inline-flex items-center gap-2">
                          <Badge
                            label={s.last_ok ? 'ok' : 'fail'}
                            variant={s.last_ok ? 'success' : 'error'}
                          />
                          <span className="text-xs text-text-muted tabular-nums">
                            {fmtMs(s.last_latency_ms)}
                          </span>
                        </span>
                      )}
                    </td>
                    <td
                      className={`px-4 py-3 text-right tabular-nums ${
                        (s?.consecutive_failures ?? 0) > 0
                          ? 'text-warning font-medium'
                          : 'text-text-secondary'
                      }`}
                    >
                      {s?.consecutive_failures ?? 0}
                    </td>
                    <td className="px-4 py-3 text-xs text-text-muted tabular-nums">
                      {s?.last_ts_ms ? fmtAgo(s.last_ts_ms) : '—'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => setConfirmDelete(p)}
                        className="text-text-muted hover:text-error transition-colors"
                        aria-label={`Delete ${p.name}`}
                      >
                        <Trash2 size={14} strokeWidth={2} />
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {creating && (
        <SyntheticFormModal
          onClose={() => setCreating(false)}
          onSaved={async () => {
            setCreating(false)
            await load()
          }}
        />
      )}

      <Modal
        open={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        title="Delete probe?"
      >
        <p className="text-[13px] text-[var(--color-text-muted)] leading-relaxed">
          This removes the synthetic probe{' '}
          <span className="font-medium text-[var(--color-text)]">
            {confirmDelete?.name}
          </span>
          .
        </p>
        <div className="flex justify-end gap-2 mt-5">
          <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(null)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="sm"
            leading={<Trash2 size={13} strokeWidth={2.25} />}
            onClick={() => confirmDelete && onDelete(confirmDelete)}
          >
            Delete
          </Button>
        </div>
      </Modal>
    </div>
  )
}

const KIND_OPTIONS: { value: ProbeKindTag; label: string }[] = [
  { value: 'health', label: 'Health' },
  { value: 'search', label: 'Search' },
  { value: 'embed', label: 'Embed' },
]

function SyntheticFormModal({
  onClose,
  onSaved,
}: {
  onClose: () => void
  onSaved: () => void
}) {
  const [name, setName] = useState('')
  const [kindTag, setKindTag] = useState<ProbeKindTag>('health')
  const [collection, setCollection] = useState('')
  const [text, setText] = useState('')
  const [intervalSecs, setIntervalSecs] = useState('60')
  const [timeoutMs, setTimeoutMs] = useState('2000')
  const [saving, setSaving] = useState(false)

  const kind = useMemo<ProbeKind>(() => {
    switch (kindTag) {
      case 'search':
        return { kind: 'search', collection: collection.trim() }
      case 'embed':
        return { kind: 'embed', text: text.trim() }
      default:
        return { kind: 'health' }
    }
  }, [kindTag, collection, text])

  const valid =
    name.trim().length > 0 &&
    Number(intervalSecs) > 0 &&
    Number(timeoutMs) > 0 &&
    (kindTag !== 'search' || collection.trim().length > 0) &&
    (kindTag !== 'embed' || text.trim().length > 0)

  const submit = async () => {
    if (!valid) return
    setSaving(true)
    try {
      const body: SyntheticCreate = {
        name: name.trim(),
        kind,
        interval_secs: Math.round(Number(intervalSecs)),
        timeout_ms: Math.round(Number(timeoutMs)),
      }
      await createSynthetic(body)
      toast.success('Probe created')
      onSaved()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal open onClose={onClose} title="New synthetic probe">
      <div className="space-y-4">
        <FormRow label="Name">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="readiness check"
          />
        </FormRow>
        <FormRow label="Kind">
          <Select
            value={kindTag}
            onValueChange={(v) => setKindTag(v as ProbeKindTag)}
            options={KIND_OPTIONS}
          />
        </FormRow>
        {kindTag === 'search' && (
          <FormRow label="Collection">
            <Input
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              placeholder="docs"
              mono
            />
          </FormRow>
        )}
        {kindTag === 'embed' && (
          <FormRow label="Text to embed">
            <Input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="health check probe"
            />
          </FormRow>
        )}
        <div className="grid grid-cols-2 gap-3">
          <FormRow label="Interval (s)">
            <Input
              type="number"
              value={intervalSecs}
              onChange={(e) => setIntervalSecs(e.target.value)}
              mono
            />
          </FormRow>
          <FormRow label="Timeout (ms)">
            <Input
              type="number"
              value={timeoutMs}
              onChange={(e) => setTimeoutMs(e.target.value)}
              mono
            />
          </FormRow>
        </div>
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={!valid || saving}
            onClick={submit}
          >
            {saving ? 'Saving…' : 'Create'}
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function FormRow({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1.5">
        {label}
      </div>
      {children}
    </div>
  )
}

/** Percentage → "X.X%"; null → "—". */
function fmtPct(pct: number | null): string {
  if (pct == null) return '—'
  return `${pct.toFixed(1)}%`
}

/** Milliseconds → "X ms"; null → "—". */
function fmtMs(ms: number | null): string {
  if (ms == null) return '—'
  return `${ms.toFixed(0)} ms`
}

/** Epoch ms → "Xs/m/h ago". */
function fmtAgo(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}
