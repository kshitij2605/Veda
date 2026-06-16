import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertCircle, Flame, Gauge, Plus, RotateCw, Trash2 } from 'lucide-react'
import {
  createSlo,
  deleteSlo,
  DEFAULT_FAST_BURN,
  DEFAULT_SLOW_BURN,
  getSloStatus,
  listSlos,
  sloKindLabel,
  updateSlo,
  type SloCreate,
  type SloDef,
  type SloKind,
  type SloKindTag,
  type SloStatus,
} from '../api/slos'
import {
  Badge,
  Button,
  EmptyState,
  Input,
  Modal,
  Select,
  Skeleton,
  toast,
} from '../components/primitives'

/**
 * SLOs — the service-level-objective surface. Each SLO shows its kind,
 * target, live SLI %, an error-budget gauge (green→amber→red), the
 * fast/slow burn rates (highlighted when breaching), and a compliant
 * badge. A create/edit form (kind selector + kind-specific field, target
 * %, window, burn configs) and delete-with-confirm. Mirrors the Insights
 * page IA (custom header + the Resources panel's null-safe rendering).
 */
export function Slos() {
  const [slos, setSlos] = useState<SloDef[] | null>(null)
  const [statuses, setStatuses] = useState<Record<string, SloStatus>>({})
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [editing, setEditing] = useState<SloDef | null>(null)
  const [creating, setCreating] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<SloDef | null>(null)

  const load = useCallback(async (spin = false) => {
    if (spin) setLoading(true)
    setError(null)
    try {
      const { data } = await listSlos()
      setSlos(data.slos)
      // Fetch live status per SLO; a per-SLO failure is non-fatal.
      const entries = await Promise.all(
        data.slos.map(async (s) => {
          try {
            const { data: st } = await getSloStatus(s.id)
            return [s.id, st] as const
          } catch {
            return null
          }
        }),
      )
      const next: Record<string, SloStatus> = {}
      for (const e of entries) if (e) next[e[0]] = e[1]
      setStatuses(next)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load SLOs')
    } finally {
      if (spin) setLoading(false)
    }
  }, [])

  useEffect(() => {
    load(true)
  }, [load])

  // Poll live status every 10 s.
  useEffect(() => {
    const id = setInterval(() => load(false), 10_000)
    return () => clearInterval(id)
  }, [load])

  const onDelete = async (s: SloDef) => {
    try {
      await deleteSlo(s.id)
      toast.success(`Deleted “${s.name}”`)
      setConfirmDelete(null)
      await load(false)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <Gauge size={16} strokeWidth={2} className="text-[var(--color-accent)]" />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
          SLOs
        </h1>
        <div className="ml-auto flex items-center gap-2">
          <Button
            size="xs"
            variant="primary"
            leading={<Plus size={11} strokeWidth={2.5} />}
            onClick={() => setCreating(true)}
          >
            New SLO
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

        {slos === null ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        ) : slos.length === 0 ? (
          <EmptyState
            icon={<Gauge size={20} strokeWidth={2} />}
            title="No SLOs defined"
            description="Create an objective to track SLI, error budget, and burn rate."
          />
        ) : (
          <div className="space-y-3">
            {slos.map((s) => (
              <SloCard
                key={s.id}
                slo={s}
                status={statuses[s.id]}
                onEdit={() => setEditing(s)}
                onDelete={() => setConfirmDelete(s)}
              />
            ))}
          </div>
        )}
      </div>

      {(creating || editing) && (
        <SloFormModal
          slo={editing}
          onClose={() => {
            setCreating(false)
            setEditing(null)
          }}
          onSaved={async () => {
            setCreating(false)
            setEditing(null)
            await load(false)
          }}
        />
      )}

      <Modal
        open={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        title="Delete SLO?"
      >
        <p className="text-[13px] text-[var(--color-text-muted)] leading-relaxed">
          This permanently removes the objective{' '}
          <span className="font-medium text-[var(--color-text)]">
            {confirmDelete?.name}
          </span>
          . Its rolling counters are retained but no longer evaluated.
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

/* ────────────────────────────────────────────────────────────────────────
   SLO card — kind / target / live SLI / error-budget gauge / burn rates
   ──────────────────────────────────────────────────────────────────── */

function SloCard({
  slo,
  status,
  onEdit,
  onDelete,
}: {
  slo: SloDef
  status: SloStatus | undefined
  onEdit: () => void
  onDelete: () => void
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-base-1)] p-4">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[14px] font-medium text-[var(--color-text)] truncate">
              {slo.name}
            </span>
            <Badge label={sloKindLabel(slo.kind)} variant="default" mono />
            {status ? (
              <Badge
                label={status.compliant ? 'compliant' : 'breaching'}
                variant={status.compliant ? 'success' : 'error'}
              />
            ) : (
              <Badge label="no data" variant="ghost" />
            )}
            {status?.budget_exhausted && (
              <Badge label="budget exhausted" variant="error" />
            )}
          </div>
          <div className="mt-1 text-[11.5px] text-[var(--color-text-subtle)] font-mono">
            target {slo.target_pct}% · window {fmtWindow(slo.window_secs)}
          </div>
        </div>
        <Button size="xs" variant="ghost" onClick={onEdit}>
          Edit
        </Button>
        <Button
          size="xs"
          variant="ghost"
          leading={<Trash2 size={12} strokeWidth={2.25} />}
          onClick={onDelete}
        >
          Delete
        </Button>
      </div>

      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1">
            SLI
          </div>
          <div className="text-[15px] font-medium tabular-nums text-[var(--color-text)]">
            {fmtPct(status?.sli_pct ?? null)}
          </div>
          {status && (
            <div className="text-[11px] text-[var(--color-text-subtle)] tabular-nums mt-0.5">
              {status.good}/{status.total} good
            </div>
          )}
        </div>

        <div className="md:col-span-1 col-span-1">
          <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1">
            Error budget
          </div>
          <BudgetGauge
            remaining={status?.error_budget_remaining_pct ?? null}
            exhausted={status?.budget_exhausted ?? false}
          />
        </div>

        <BurnCell
          label="Fast burn"
          rate={status?.burn_rate_fast ?? null}
          threshold={status?.fast_burn_threshold ?? slo.fast_burn.threshold}
          breaching={status?.breaching_fast ?? false}
          window={status ? slo.fast_burn.window_secs : slo.fast_burn.window_secs}
        />
        <BurnCell
          label="Slow burn"
          rate={status?.burn_rate_slow ?? null}
          threshold={status?.slow_burn_threshold ?? slo.slow_burn.threshold}
          breaching={status?.breaching_slow ?? false}
          window={slo.slow_burn.window_secs}
        />
      </div>
    </div>
  )
}

/** Error-budget gauge: a bar that fills with remaining budget, colored
 * green → amber → red as it drops; red when exhausted. Null → "—". */
function BudgetGauge({
  remaining,
  exhausted,
}: {
  remaining: number | null
  exhausted: boolean
}) {
  if (remaining == null) {
    return (
      <div className="text-[15px] font-medium tabular-nums text-[var(--color-text-subtle)]">
        —
      </div>
    )
  }
  const pct = Math.max(0, Math.min(100, remaining))
  const color = exhausted || remaining <= 0
    ? 'var(--color-error)'
    : remaining < 25
      ? 'var(--color-error)'
      : remaining < 50
        ? 'var(--color-warning)'
        : 'var(--color-success)'
  const valueColor = exhausted || remaining <= 0
    ? 'text-[var(--color-error)]'
    : remaining < 25
      ? 'text-[var(--color-error)]'
      : remaining < 50
        ? 'text-[var(--color-warning)]'
        : 'text-[var(--color-text)]'
  return (
    <div>
      <div className={`text-[15px] font-medium tabular-nums ${valueColor}`}>
        {remaining.toFixed(1)}%
      </div>
      <div
        className="mt-1.5 h-1.5 w-full rounded-full bg-[var(--color-base-2)] overflow-hidden"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full transition-[width] duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function BurnCell({
  label,
  rate,
  threshold,
  breaching,
  window,
}: {
  label: string
  rate: number | null
  threshold: number
  breaching: boolean
  window: number
}) {
  const color = breaching ? 'text-[var(--color-error)]' : 'text-[var(--color-text)]'
  return (
    <div>
      <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1 inline-flex items-center gap-1">
        {breaching && (
          <Flame size={11} strokeWidth={2.5} className="text-[var(--color-error)]" />
        )}
        {label}
      </div>
      <div className={`text-[15px] font-medium tabular-nums ${color}`}>
        {rate == null ? '—' : `${rate.toFixed(2)}×`}
      </div>
      <div className="text-[11px] text-[var(--color-text-subtle)] tabular-nums mt-0.5">
        {fmtWindow(window)} · ≥{threshold}×
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Create / edit form
   ──────────────────────────────────────────────────────────────────── */

const KIND_OPTIONS: { value: SloKindTag; label: string }[] = [
  { value: 'availability', label: 'Availability' },
  { value: 'latency', label: 'Latency' },
  { value: 'error_rate', label: 'Error rate' },
  { value: 'synthetic', label: 'Synthetic probe' },
]

function SloFormModal({
  slo,
  onClose,
  onSaved,
}: {
  slo: SloDef | null
  onClose: () => void
  onSaved: () => void
}) {
  const [name, setName] = useState(slo?.name ?? '')
  const [kindTag, setKindTag] = useState<SloKindTag>(slo?.kind.kind ?? 'availability')
  const [thresholdMs, setThresholdMs] = useState(
    slo?.kind.kind === 'latency' ? String(slo.kind.threshold_ms) : '100',
  )
  const [probeId, setProbeId] = useState(
    slo?.kind.kind === 'synthetic' ? slo.kind.probe_id : '',
  )
  const [targetPct, setTargetPct] = useState(String(slo?.target_pct ?? 99))
  // Window expressed in minutes for friendlier entry.
  const [windowMin, setWindowMin] = useState(
    String((slo?.window_secs ?? 2_592_000) / 60),
  )
  const [fastWindow, setFastWindow] = useState(
    String(slo?.fast_burn.window_secs ?? DEFAULT_FAST_BURN.window_secs),
  )
  const [fastThreshold, setFastThreshold] = useState(
    String(slo?.fast_burn.threshold ?? DEFAULT_FAST_BURN.threshold),
  )
  const [slowWindow, setSlowWindow] = useState(
    String(slo?.slow_burn.window_secs ?? DEFAULT_SLOW_BURN.window_secs),
  )
  const [slowThreshold, setSlowThreshold] = useState(
    String(slo?.slow_burn.threshold ?? DEFAULT_SLOW_BURN.threshold),
  )
  const [saving, setSaving] = useState(false)

  const kind = useMemo<SloKind>(() => {
    switch (kindTag) {
      case 'latency':
        return { kind: 'latency', threshold_ms: Number(thresholdMs) || 0 }
      case 'synthetic':
        return { kind: 'synthetic', probe_id: probeId.trim() }
      case 'error_rate':
        return { kind: 'error_rate' }
      default:
        return { kind: 'availability' }
    }
  }, [kindTag, thresholdMs, probeId])

  const valid =
    name.trim().length > 0 &&
    Number(targetPct) > 0 &&
    Number(windowMin) > 0 &&
    (kindTag !== 'synthetic' || probeId.trim().length > 0)

  const submit = async () => {
    if (!valid) return
    setSaving(true)
    try {
      const body: SloCreate = {
        name: name.trim(),
        kind,
        target_pct: Number(targetPct),
        window_secs: Math.round(Number(windowMin) * 60),
        fast_burn: {
          window_secs: Math.round(Number(fastWindow)) || DEFAULT_FAST_BURN.window_secs,
          threshold: Number(fastThreshold) || DEFAULT_FAST_BURN.threshold,
        },
        slow_burn: {
          window_secs: Math.round(Number(slowWindow)) || DEFAULT_SLOW_BURN.window_secs,
          threshold: Number(slowThreshold) || DEFAULT_SLOW_BURN.threshold,
        },
      }
      if (slo) {
        await updateSlo(slo.id, body)
        toast.success('SLO updated')
      } else {
        await createSlo(body)
        toast.success('SLO created')
      }
      onSaved()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal open onClose={onClose} title={slo ? 'Edit SLO' : 'New SLO'}>
      <div className="space-y-4">
        <FormRow label="Name">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="API availability"
          />
        </FormRow>

        <FormRow label="Kind">
          <Select
            value={kindTag}
            onValueChange={(v) => setKindTag(v as SloKindTag)}
            options={KIND_OPTIONS}
          />
        </FormRow>

        {kindTag === 'latency' && (
          <FormRow label="Latency threshold (ms)">
            <Input
              type="number"
              value={thresholdMs}
              onChange={(e) => setThresholdMs(e.target.value)}
              mono
            />
          </FormRow>
        )}
        {kindTag === 'synthetic' && (
          <FormRow label="Probe id">
            <Input
              value={probeId}
              onChange={(e) => setProbeId(e.target.value)}
              placeholder="synthetic-…"
              mono
            />
          </FormRow>
        )}

        <div className="grid grid-cols-2 gap-3">
          <FormRow label="Target %">
            <Input
              type="number"
              value={targetPct}
              onChange={(e) => setTargetPct(e.target.value)}
              mono
            />
          </FormRow>
          <FormRow label="Window (minutes)">
            <Input
              type="number"
              value={windowMin}
              onChange={(e) => setWindowMin(e.target.value)}
              mono
            />
          </FormRow>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <FormRow label="Fast burn window (s)">
            <Input
              type="number"
              value={fastWindow}
              onChange={(e) => setFastWindow(e.target.value)}
              mono
            />
          </FormRow>
          <FormRow label="Fast burn ≥">
            <Input
              type="number"
              value={fastThreshold}
              onChange={(e) => setFastThreshold(e.target.value)}
              mono
            />
          </FormRow>
          <FormRow label="Slow burn window (s)">
            <Input
              type="number"
              value={slowWindow}
              onChange={(e) => setSlowWindow(e.target.value)}
              mono
            />
          </FormRow>
          <FormRow label="Slow burn ≥">
            <Input
              type="number"
              value={slowThreshold}
              onChange={(e) => setSlowThreshold(e.target.value)}
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
            {saving ? 'Saving…' : slo ? 'Save' : 'Create'}
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

/** Seconds → compact human window (e.g. "30d", "5m", "1h"). */
function fmtWindow(secs: number): string {
  if (secs % 86400 === 0) return `${secs / 86400}d`
  if (secs % 3600 === 0) return `${secs / 3600}h`
  if (secs % 60 === 0) return `${secs / 60}m`
  return `${secs}s`
}
