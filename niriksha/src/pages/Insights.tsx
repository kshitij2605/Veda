import { useCallback, useEffect, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  Bell,
  Check,
  ChevronRight,
  GitMerge,
  Layers,
  Play,
  RotateCw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import {
  ackInsight,
  fetchChannels,
  fetchEventRca,
  fetchIncidents,
  fetchInsights,
  fetchInsightsHistory,
  fetchRules,
  putChannels,
  putRule,
  runInsights,
  scopeLabel,
  triageEvent,
  type AgentStep,
  type ChannelsConfig,
  type ChannelsView,
  type Incident,
  type NoiseReduction,
  type OpsEvent,
  type RcaBundle,
  type RuleInfo,
  type Severity,
} from '../api/insights'
import {
  Badge,
  Button,
  Drawer,
  EmptyState,
  Input,
  Select,
  Skeleton,
  Switch,
  Textarea,
  toast,
} from '../components/primitives'
import { useInsightsStore } from '../stores/insightsStore'
import { cn } from '../lib/cn'

type View = 'open' | 'incidents' | 'history' | 'rules' | 'channels'

/**
 * Insights — the ops-advisor surface. Open events (severity-sorted) with
 * an ack action + detail drawer, closed-event history, rule settings, and
 * (redacted) alert-channel config. Polls the open-events endpoint every
 * 5 s while the "open" view is active, matching Observability's cadence.
 */
export function Insights() {
  const [view, setView] = useState<View>('open')
  const [events, setEvents] = useState<OpsEvent[] | null>(null)
  const [history, setHistory] = useState<OpsEvent[] | null>(null)
  const [incidents, setIncidents] = useState<Incident[] | null>(null)
  const [noiseReduction, setNoiseReduction] = useState<NoiseReduction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [lastFetched, setLastFetched] = useState<number | null>(null)
  const [selected, setSelected] = useState<OpsEvent | null>(null)

  const setBadge = useInsightsStore((s) => s.setFromEvents)

  const loadOpen = useCallback(
    async (spin = false) => {
      if (spin) setLoading(true)
      setError(null)
      try {
        const { data } = await fetchInsights()
        setEvents(data.events)
        setBadge(data.events)
        setLastFetched(Date.now())
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load insights')
      } finally {
        if (spin) setLoading(false)
      }
    },
    [setBadge],
  )

  const loadHistory = useCallback(async () => {
    try {
      const { data } = await fetchInsightsHistory()
      setHistory(data.events)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load history')
    }
  }, [])

  const loadIncidents = useCallback(async () => {
    try {
      const { data } = await fetchIncidents()
      setIncidents(data.incidents)
      setNoiseReduction(data.noise_reduction ?? null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load incidents')
    }
  }, [])

  // Initial load + 5 s poll of open events while the open view is up.
  useEffect(() => {
    loadOpen(true)
  }, [loadOpen])

  useEffect(() => {
    if (view !== 'open') return
    const id = setInterval(() => loadOpen(false), 5_000)
    return () => clearInterval(id)
  }, [view, loadOpen])

  useEffect(() => {
    if (view === 'history' && history === null) loadHistory()
  }, [view, history, loadHistory])

  useEffect(() => {
    if (view === 'incidents' && incidents === null) loadIncidents()
  }, [view, incidents, loadIncidents])

  const runCheck = async () => {
    setRunning(true)
    try {
      const { data } = await runInsights()
      toast.success(`Advisor tick: ${data.opened} opened, ${data.closed} closed`)
      await loadOpen(false)
      if (view === 'history') await loadHistory()
      if (view === 'incidents') await loadIncidents()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Run failed')
    } finally {
      setRunning(false)
    }
  }

  const ack = async (ev: OpsEvent) => {
    try {
      await ackInsight(ev.id)
      toast.success('Acknowledged')
      // Optimistic local flip; the next poll re-syncs.
      setEvents((prev) =>
        prev
          ? prev.map((e) =>
              e.id === ev.id ? { ...e, acknowledged: true } : e,
            )
          : prev,
      )
      setSelected((s) => (s && s.id === ev.id ? { ...s, acknowledged: true } : s))
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Ack failed')
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <ShieldAlert
          size={16}
          strokeWidth={2}
          className="text-[var(--color-accent)]"
        />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
          Insights
        </h1>
        {lastFetched && view === 'open' && (
          <span className="text-[11px] font-mono text-[var(--color-text-subtle)]">
            polled {formatRelative(lastFetched)} · live
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button
            size="xs"
            variant="secondary"
            leading={<Play size={11} strokeWidth={2.25} />}
            onClick={runCheck}
            disabled={running}
          >
            {running ? 'Running…' : 'Run check now'}
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
            onClick={() => loadOpen(true)}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </header>

      <nav className="flex items-center gap-1 px-4 h-10 border-b border-[var(--color-border)] flex-none">
        <ViewTab id="open" label="Open" view={view} onSelect={setView} />
        <ViewTab id="incidents" label="Incidents" view={view} onSelect={setView} />
        <ViewTab id="history" label="History" view={view} onSelect={setView} />
        <ViewTab id="rules" label="Rules" view={view} onSelect={setView} />
        <ViewTab id="channels" label="Channels" view={view} onSelect={setView} />
      </nav>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-4">
        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
            <AlertCircle size={12} strokeWidth={2.25} />
            {error}
          </div>
        )}

        {view === 'open' &&
          (events === null ? (
            <ListSkeleton />
          ) : events.length === 0 ? (
            <EmptyState
              icon={<Check size={20} strokeWidth={2} />}
              title="All clear"
              description="No open advisor events. Run a check to evaluate now."
            />
          ) : (
            <EventList events={events} onSelect={setSelected} onAck={ack} />
          ))}

        {view === 'incidents' &&
          (incidents === null ? (
            <ListSkeleton />
          ) : incidents.length === 0 ? (
            <EmptyState
              icon={<Layers size={20} strokeWidth={2} />}
              title="No incidents yet"
              description="Recurring events are grouped into incidents by the correlator (runs every few minutes). Once it has run, clusters appear here."
            />
          ) : (
            <IncidentList incidents={incidents} noiseReduction={noiseReduction} />
          ))}

        {view === 'history' &&
          (history === null ? (
            <ListSkeleton />
          ) : history.length === 0 ? (
            <EmptyState
              icon={<Bell size={20} strokeWidth={2} />}
              title="No history yet"
              description="Closed events will appear here as conditions resolve."
            />
          ) : (
            <EventList events={history} onSelect={setSelected} closed />
          ))}

        {view === 'rules' && <RulesPanel onError={setError} />}
        {view === 'channels' && <ChannelsPanel onError={setError} />}
      </div>

      <EventDrawer
        event={selected}
        onClose={() => setSelected(null)}
        onAck={ack}
        onGoToRule={(ruleId) => {
          setSelected(null)
          setView('rules')
          // Rules panel scrolls itself; surface a hint.
          toast.success(`Configure auto-remediation for "${ruleId}" below`)
        }}
      />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Tabs (simple in-page view switch — Tabs primitive remounts content,
   which would reset polling, so we use a controlled segmented nav)
   ──────────────────────────────────────────────────────────────────── */

function ViewTab({
  id,
  label,
  view,
  onSelect,
}: {
  id: View
  label: string
  view: View
  onSelect: (v: View) => void
}) {
  const active = view === id
  return (
    <button
      type="button"
      onClick={() => onSelect(id)}
      className={cn(
        'relative px-3 h-8 text-[12px] rounded-[var(--radius-sm)] transition-colors',
        active
          ? 'text-[var(--color-text)] font-medium'
          : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
      )}
    >
      {label}
      {active && (
        <span className="absolute left-2 right-2 -bottom-[5px] h-[2px] rounded bg-[var(--color-accent)]" />
      )}
    </button>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Event list
   ──────────────────────────────────────────────────────────────────── */

function EventList({
  events,
  onSelect,
  onAck,
  closed,
}: {
  events: OpsEvent[]
  onSelect: (e: OpsEvent) => void
  onAck?: (e: OpsEvent) => void
  closed?: boolean
}) {
  return (
    <ul className="flex flex-col gap-2">
      {events.map((ev) => (
        <li key={`${ev.id}:${ev.opened_at_ms}`}>
          <button
            type="button"
            onClick={() => onSelect(ev)}
            className="w-full text-left rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2.5 hover:border-[var(--color-border-strong)] hover:bg-[var(--color-bg-hover)] transition-colors"
          >
            <div className="flex items-center gap-2">
              <SeverityBadge severity={ev.severity} />
              <span className="text-[13px] font-medium text-[var(--color-text)] truncate">
                {ev.title}
              </span>
              {ev.acknowledged && !closed && (
                <Badge label="ack" variant="success" />
              )}
              {ev.remediation.action !== 'none' && (
                <Badge label={ev.remediation.action} variant="accent" mono />
              )}
              <span className="ml-auto flex items-center gap-2 flex-none">
                {!closed && onAck && !ev.acknowledged && (
                  <Button
                    size="xs"
                    variant="ghost"
                    leading={<Check size={11} strokeWidth={2.25} />}
                    onClick={(e) => {
                      e.stopPropagation()
                      onAck(ev)
                    }}
                  >
                    Ack
                  </Button>
                )}
                <ChevronRight
                  size={14}
                  strokeWidth={2}
                  className="text-[var(--color-text-subtle)]"
                />
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1 text-[11px] text-[var(--color-text-subtle)]">
              <span className="font-mono">{ev.rule_id}</span>
              <span aria-hidden>·</span>
              <span>{scopeLabel(ev.scope)}</span>
              <span aria-hidden>·</span>
              <span>
                {closed && ev.closed_at_ms
                  ? `closed ${formatRelative(ev.closed_at_ms)}`
                  : `opened ${formatRelative(ev.opened_at_ms)}`}
              </span>
            </div>
          </button>
        </li>
      ))}
    </ul>
  )
}

function SeverityBadge({ severity }: { severity: Severity }) {
  const map: Record<Severity, { label: string; variant: 'error' | 'warning' | 'default' }> = {
    high: { label: 'High', variant: 'error' },
    medium: { label: 'Medium', variant: 'warning' },
    low: { label: 'Low', variant: 'default' },
  }
  const { label, variant } = map[severity]
  return <Badge label={label} variant={variant} dot className="flex-none" />
}

/* ────────────────────────────────────────────────────────────────────────
   Incident list — recurring/duplicate events collapsed into one row each
   ──────────────────────────────────────────────────────────────────── */

function IncidentList({
  incidents,
  noiseReduction,
}: {
  incidents: Incident[]
  noiseReduction: NoiseReduction | null
}) {
  return (
    <div className="flex flex-col gap-2">
      {noiseReduction && noiseReduction.suppressed > 0 && (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-3 py-2 text-[12px] text-[var(--color-text-subtle)]">
          Alert-storm noise reduction:{' '}
          <span className="font-medium text-[var(--color-text)] tabular-nums">
            {noiseReduction.event_count}
          </span>{' '}
          alerts collapsed into{' '}
          <span className="font-medium text-[var(--color-text)] tabular-nums">
            {noiseReduction.incident_count}
          </span>{' '}
          incident{noiseReduction.incident_count === 1 ? '' : 's'} —{' '}
          <span className="font-medium text-[var(--color-text)] tabular-nums">
            {noiseReduction.suppressed}
          </span>{' '}
          suppressed.
        </div>
      )}
      <ul className="flex flex-col gap-2">
        {incidents.map((inc) => {
          const ruleIds = inc.rule_ids ?? []
          const count = inc.count ?? inc.member_event_ids?.length ?? 0
          const draft = inc.draft?.trim()
          return (
            <li key={inc.id}>
              <div className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2.5">
                <div className="flex items-center gap-2">
                  <SeverityBadge severity={inc.max_severity} />
                  <span className="text-[13px] font-medium text-[var(--color-text)] truncate">
                    {inc.representative}
                  </span>
                  <Badge
                    label={`${count} alerts → 1 incident`}
                    variant="accent"
                    mono
                    className="flex-none"
                  />
                </div>
                <div className="flex items-center gap-2 mt-1 text-[11px] text-[var(--color-text-subtle)] flex-wrap">
                  <span className="inline-flex items-center gap-1">
                    <Layers size={11} strokeWidth={2} />
                    {ruleIds.length > 0 ? ruleIds.join(', ') : 'correlated'}
                  </span>
                  <span aria-hidden>·</span>
                  <span>
                    seen {count}× since {formatRelative(inc.first_seen_ms)}
                  </span>
                  {inc.last_seen_ms > inc.first_seen_ms && (
                    <>
                      <span aria-hidden>·</span>
                      <span>last {formatRelative(inc.last_seen_ms)}</span>
                    </>
                  )}
                </div>
                {draft && (
                  <div className="mt-2 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-2.5 py-2">
                    <div className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-text-subtle)]">
                      AI draft root-cause
                    </div>
                    <p className="mt-1 text-[12px] leading-relaxed text-[var(--color-text)] whitespace-pre-wrap">
                      {draft}
                    </p>
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Event detail drawer
   ──────────────────────────────────────────────────────────────────── */

function EventDrawer({
  event,
  onClose,
  onAck,
  onGoToRule,
}: {
  event: OpsEvent | null
  onClose: () => void
  onAck: (e: OpsEvent) => void
  onGoToRule: (ruleId: string) => void
}) {
  const ev = event
  return (
    <Drawer
      open={!!ev}
      onClose={onClose}
      title={ev?.title}
      subtitle={ev ? `${ev.rule_id} · ${scopeLabel(ev.scope)}` : undefined}
      width={520}
    >
      {ev && (
        <div className="space-y-5">
          <div className="flex items-center gap-2">
            <SeverityBadge severity={ev.severity} />
            <Badge
              label={ev.status}
              variant={ev.status === 'open' ? 'outline' : 'ghost'}
              mono
            />
            {ev.acknowledged && <Badge label="acknowledged" variant="success" />}
          </div>

          <Field label="Scope" value={scopeLabel(ev.scope)} />
          <Section label="Root cause" body={ev.detail} />
          <Section label="Recommendation" body={ev.recommendation} icon={<Sparkles size={12} strokeWidth={2.25} />} />

          {ev.runbook && (
            <div>
              <SectionLabel>Runbook</SectionLabel>
              <code className="block rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-2 text-[12px] font-mono text-[var(--color-text)] whitespace-pre-wrap break-words">
                {ev.runbook}
              </code>
            </div>
          )}

          {/* Remediation affordance. There is NO per-event manual-trigger
              endpoint — auto-remediation is a backend opt-in per rule.
              So we surface the recommended action + a route to the rule's
              auto_remediate toggle, rather than a destructive button. */}
          {ev.remediation.action !== 'none' && (
            <div className="rounded-[var(--radius-md)] border border-[color-mix(in_oklch,var(--color-accent)_30%,transparent)] bg-[var(--color-accent-subtle)] px-3 py-3">
              <div className="flex items-center gap-2 mb-1.5">
                <AlertTriangle
                  size={13}
                  strokeWidth={2.25}
                  className="text-[var(--color-accent)]"
                />
                <span className="text-[12px] font-medium text-[var(--color-text)]">
                  Suggested remediation
                </span>
                <Badge
                  label={remediationLabel(ev)}
                  variant="accent"
                  mono
                  className="ml-auto"
                />
              </div>
              <p className="text-[11.5px] text-[var(--color-text-muted)] leading-relaxed">
                The advisor can apply this automatically when{' '}
                <span className="font-mono text-[var(--color-text)]">auto_remediate</span>{' '}
                is enabled for rule{' '}
                <span className="font-mono text-[var(--color-text)]">{ev.rule_id}</span>.
                There is no manual one-shot trigger.
              </p>
              <Button
                size="xs"
                variant="secondary"
                className="mt-2"
                onClick={() => onGoToRule(ev.rule_id)}
              >
                Configure rule →
              </Button>
            </div>
          )}

          {ev.remediation_note && (
            <Section label="Last remediation outcome" body={ev.remediation_note} mono />
          )}

          <div className="grid grid-cols-2 gap-3 pt-1">
            <Field label="Opened" value={formatAbsolute(ev.opened_at_ms)} mono />
            <Field
              label="Closed"
              value={ev.closed_at_ms ? formatAbsolute(ev.closed_at_ms) : '—'}
              mono
            />
          </div>

          {ev.status === 'open' && !ev.acknowledged && (
            <Button
              variant="primary"
              size="sm"
              block
              leading={<Check size={13} strokeWidth={2.25} />}
              onClick={() => onAck(ev)}
            >
              Acknowledge
            </Button>
          )}

          {/* Cross-signal root-cause bundle — auto-joined logs + slow/errored
              queries + resource state + related incident. `key` resets the
              section's local state per event. */}
          <RcaSection key={`rca-${ev.id}`} eventId={ev.id} />

          {/* AI triage — runs the built-in veda-ops agent to explain the
              alert. `key` resets the section's local state per event. */}
          <TriageSection key={ev.id} eventId={ev.id} />
        </div>
      )}
    </Drawer>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Cross-signal root-cause bundle — correlated logs + slow/errored queries
   + nearest resource snapshot + related incident
   ──────────────────────────────────────────────────────────────────── */

function RcaSection({ eventId }: { eventId: string }) {
  const [loading, setLoading] = useState(true)
  const [bundle, setBundle] = useState<RcaBundle | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchEventRca(eventId)
      .then(({ data }) => {
        if (!cancelled) setBundle(data)
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : 'Failed to load root-cause bundle')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [eventId])

  const logs = bundle?.logs ?? []
  const queries = bundle?.slow_queries ?? []
  const traceIds = bundle?.trace_ids ?? []
  const snap = bundle?.resource_snapshot ?? null
  const incident = bundle?.related_incident ?? null
  const empty =
    !loading &&
    !error &&
    logs.length === 0 &&
    queries.length === 0 &&
    traceIds.length === 0 &&
    !snap &&
    !incident

  return (
    <div className="pt-1 border-t border-[var(--color-border)]">
      <SectionLabel>
        <span className="inline-flex items-center gap-1">
          <GitMerge size={12} strokeWidth={2.25} />
          Root-cause bundle
        </span>
      </SectionLabel>

      {loading && <Skeleton className="h-16 w-full rounded-[var(--radius-sm)]" />}

      {error && (
        <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
          <AlertCircle size={12} strokeWidth={2.25} />
          {error}
        </div>
      )}

      {empty && (
        <p className="text-[12px] text-[var(--color-text-muted)]">
          No correlated signals found in the incident window.
        </p>
      )}

      {!loading && !error && !empty && (
        <div className="space-y-3">
          {incident && (
            <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-base-1)] px-2.5 py-2">
              <div className="flex items-center gap-1.5 mb-0.5">
                <Badge label="incident" variant="accent" mono />
                <span className="text-[11px] font-mono text-[var(--color-text-subtle)]">
                  {incident.id} · {incident.count} events
                </span>
              </div>
              <p className="text-[12px] text-[var(--color-text)] break-words">
                {incident.representative}
              </p>
            </div>
          )}

          {snap && (
            <div className="grid grid-cols-2 gap-2">
              <Field
                label="Disk used"
                value={snap.disk_used_pct != null ? `${snap.disk_used_pct.toFixed(1)}%` : '—'}
                mono
              />
              <Field label="Query p95" value={`${snap.query_p95_ms.toFixed(1)} ms`} mono />
              <Field
                label="RSS"
                value={snap.rss_bytes != null ? formatBytes(snap.rss_bytes) : '—'}
                mono
              />
              <Field label="Errors (log)" value={String(snap.error_log_count)} mono />
            </div>
          )}

          {traceIds.length > 0 && (
            <div>
              <SectionLabel>Correlated trace ids</SectionLabel>
              <div className="flex flex-wrap gap-1">
                {traceIds.slice(0, 8).map((t) => (
                  <Badge key={t} label={t} variant="ghost" mono />
                ))}
              </div>
            </div>
          )}

          {logs.length > 0 && (
            <div>
              <SectionLabel>Correlated logs ({logs.length})</SectionLabel>
              <div className="space-y-1.5">
                {logs.slice(0, 12).map((l, i) => (
                  <div
                    key={i}
                    className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-1.5"
                  >
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <Badge
                        label={l.level}
                        variant={l.level.toUpperCase() === 'ERROR' ? 'error' : 'default'}
                        mono
                      />
                      <span className="text-[11px] font-mono text-[var(--color-text-subtle)] truncate">
                        {l.target}
                      </span>
                    </div>
                    <p className="text-[12px] text-[var(--color-text)] leading-relaxed whitespace-pre-wrap break-words">
                      {l.message}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {queries.length > 0 && (
            <div>
              <SectionLabel>Slow / errored queries ({queries.length})</SectionLabel>
              <div className="space-y-1.5">
                {queries.slice(0, 12).map((q, i) => (
                  <div
                    key={i}
                    className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-1.5 flex items-center gap-2"
                  >
                    <Badge
                      label={q.status}
                      variant={q.status === 'error' ? 'error' : 'default'}
                      mono
                    />
                    <span className="text-[11.5px] font-mono text-[var(--color-text)]">
                      {q.collection} · {q.query_type}
                    </span>
                    <span className="ml-auto text-[11.5px] font-mono tabular-nums text-[var(--color-text-muted)]">
                      {q.took_ms} ms
                      {q.error_kind ? ` · ${q.error_kind}` : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   AI triage — explanation + reasoning/acting step trace
   ──────────────────────────────────────────────────────────────────── */

function TriageSection({ eventId }: { eventId: string }) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ explanation: string; steps: AgentStep[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [noLlm, setNoLlm] = useState(false)

  const run = async () => {
    setLoading(true)
    setError(null)
    setNoLlm(false)
    try {
      const { data } = await triageEvent(eventId)
      setResult({ explanation: data.explanation, steps: data.steps })
    } catch (e) {
      // 503 → LLM not configured; surface that gracefully (not a hard error).
      const status =
        typeof e === 'object' && e && 'response' in e
          ? (e as { response?: { status?: number; data?: { error?: string } } }).response?.status
          : undefined
      if (status === 503) {
        setNoLlm(true)
      } else {
        setError(e instanceof Error ? e.message : 'Triage failed')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="pt-1 border-t border-[var(--color-border)]">
      <Button
        variant="secondary"
        size="sm"
        block
        className="mt-3"
        disabled={loading}
        leading={
          <Sparkles
            size={13}
            strokeWidth={2.25}
            className={loading ? 'animate-pulse' : ''}
          />
        }
        onClick={run}
      >
        {loading ? 'Triaging…' : result ? 'Re-run triage' : 'Triage with AI'}
      </Button>

      {noLlm && (
        <div className="mt-3 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-base-1)] px-3 py-2 text-[12px] text-[var(--color-text-muted)]">
          LLM not configured. Set{' '}
          <span className="font-mono text-[var(--color-text)]">VEDA_LLM_ENDPOINT</span>{' '}
          to enable AI triage.
        </div>
      )}

      {error && (
        <div className="mt-3 rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
          <AlertCircle size={12} strokeWidth={2.25} />
          {error}
        </div>
      )}

      {result && (
        <div className="mt-3 space-y-3">
          <div>
            <SectionLabel>
              <span className="inline-flex items-center gap-1">
                <Sparkles size={12} strokeWidth={2.25} />
                Explanation
              </span>
            </SectionLabel>
            <p className="text-[12.5px] text-[var(--color-text)] leading-relaxed whitespace-pre-wrap break-words">
              {result.explanation}
            </p>
          </div>

          {result.steps.length > 0 && (
            <div>
              <SectionLabel>Step trace</SectionLabel>
              <div className="space-y-1.5">
                {result.steps.map((s, i) => (
                  <TriageStep key={i} step={s} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function TriageStep({ step }: { step: AgentStep }) {
  if (step.kind === 'reasoning') {
    return (
      <div className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-1.5">
        <div className="flex items-center gap-1.5 mb-0.5">
          <Badge label="reasoning" variant="ghost" mono />
        </div>
        <p className="text-[12px] text-[var(--color-text-muted)] leading-relaxed whitespace-pre-wrap break-words">
          {step.text}
        </p>
      </div>
    )
  }
  if (step.kind === 'tool_call') {
    return (
      <div className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-1.5">
        <div className="flex items-center gap-1.5 mb-0.5">
          <Badge label="tool call" variant="accent" mono />
          <span className="text-[11.5px] font-mono text-[var(--color-text)]">
            {step.tool_id}
          </span>
        </div>
        <code className="block text-[11px] font-mono text-[var(--color-text-subtle)] whitespace-pre-wrap break-words">
          {fmtJson(step.args)}
        </code>
      </div>
    )
  }
  // tool_result
  return (
    <div className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-1.5">
      <div className="flex items-center gap-1.5 mb-0.5">
        <Badge
          label={step.is_error ? 'tool error' : 'tool result'}
          variant={step.is_error ? 'error' : 'default'}
          mono
        />
      </div>
      <code className="block text-[11px] font-mono text-[var(--color-text-subtle)] whitespace-pre-wrap break-words">
        {fmtJson(step.result)}
      </code>
    </div>
  )
}

/** Compact JSON for a trace value; strings pass through verbatim. */
function fmtJson(v: unknown): string {
  if (v == null) return ''
  if (typeof v === 'string') return v
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}

function remediationLabel(ev: OpsEvent): string {
  const r = ev.remediation
  if (r.action === 'none') return 'none'
  return `${r.action} ${r.params.collection}`
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1">
      {children}
    </div>
  )
}

function Section({
  label,
  body,
  icon,
  mono,
}: {
  label: string
  body: string
  icon?: React.ReactNode
  mono?: boolean
}) {
  return (
    <div>
      <SectionLabel>
        <span className="inline-flex items-center gap-1">
          {icon}
          {label}
        </span>
      </SectionLabel>
      <p
        className={cn(
          'text-[12.5px] text-[var(--color-text)] leading-relaxed whitespace-pre-wrap break-words',
          mono && 'font-mono text-[12px]',
        )}
      >
        {body}
      </p>
    </div>
  )
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <div
        className={cn(
          'text-[12.5px] text-[var(--color-text)]',
          mono && 'font-mono text-[12px] tabular-nums',
        )}
      >
        {value}
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Rules panel
   ──────────────────────────────────────────────────────────────────── */

function RulesPanel({ onError }: { onError: (e: string | null) => void }) {
  const [rules, setRules] = useState<RuleInfo[] | null>(null)

  const load = useCallback(async () => {
    try {
      const { data } = await fetchRules()
      setRules(data.rules)
    } catch (e) {
      onError(e instanceof Error ? e.message : 'Failed to load rules')
    }
  }, [onError])

  useEffect(() => {
    load()
  }, [load])

  if (rules === null) return <ListSkeleton />
  if (rules.length === 0)
    return <EmptyState title="No rules" description="The advisor catalog is empty." />

  return (
    <div className="space-y-2">
      {rules.map((r) => (
        <RuleRow key={r.id} rule={r} onError={onError} />
      ))}
    </div>
  )
}

const SEVERITY_OPTIONS = [
  { value: 'inherit', label: 'Inherit default' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
] as const

function RuleRow({ rule, onError }: { rule: RuleInfo; onError: (e: string | null) => void }) {
  const [cfg, setCfg] = useState(rule.effective_config)
  const [thresholdText, setThresholdText] = useState(() =>
    cfg.thresholds == null ? '' : JSON.stringify(cfg.thresholds, null, 2),
  )
  const [thresholdErr, setThresholdErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const patch = async (p: Parameters<typeof putRule>[1]) => {
    setSaving(true)
    try {
      const { data } = await putRule(rule.id, p)
      setCfg(data)
      onError(null)
    } catch (e) {
      onError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  const saveThresholds = async () => {
    let parsed: unknown
    if (thresholdText.trim() === '') {
      parsed = null
    } else {
      try {
        parsed = JSON.parse(thresholdText)
      } catch {
        setThresholdErr('Invalid JSON')
        return
      }
    }
    setThresholdErr(null)
    await patch({ thresholds: parsed })
    toast.success('Thresholds saved')
  }

  const sevValue: string = cfg.severity_override ?? 'inherit'

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[13px] font-mono text-[var(--color-text)]">{rule.id}</span>
        {rule.anomaly ? (
          <Badge label="anomaly" variant="accent" mono />
        ) : (
          rule.default_severity && (
            <Badge label={`default ${rule.default_severity}`} variant="ghost" mono />
          )
        )}
        {saving && (
          <span className="text-[10.5px] text-[var(--color-text-subtle)] ml-auto">
            saving…
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2.5 mt-3">
        <ToggleRow
          label="Enabled"
          hint="Whether the rule runs at all"
          checked={cfg.enabled}
          onChange={(v) => patch({ enabled: v })}
        />
        <ToggleRow
          label="Muted"
          hint="Runs but suppresses events/notifications"
          checked={cfg.muted}
          onChange={(v) => patch({ muted: v })}
        />
        <ToggleRow
          label="Auto-remediate"
          hint="Let the advisor apply the remediation automatically"
          checked={cfg.auto_remediate}
          onChange={(v) => patch({ auto_remediate: v })}
        />
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-[12.5px] text-[var(--color-text)]">Severity override</div>
            <div className="text-[10.5px] text-[var(--color-text-subtle)]">
              Force a severity for every hit
            </div>
          </div>
          <Select
            size="sm"
            value={sevValue}
            options={SEVERITY_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
            onValueChange={(v) =>
              patch({
                severity_override: v === 'inherit' ? null : (v as Severity),
              })
            }
          />
        </div>
      </div>

      <div className="mt-3">
        <SectionLabel>Thresholds (JSON)</SectionLabel>
        <Textarea
          mono
          value={thresholdText}
          onChange={(e) => setThresholdText(e.target.value)}
          invalid={!!thresholdErr}
          placeholder='e.g. { "sealed": 50 }'
          className="min-h-[64px] text-[12px]"
        />
        <div className="flex items-center gap-2 mt-1.5">
          <Button size="xs" variant="secondary" onClick={saveThresholds} disabled={saving}>
            Save thresholds
          </Button>
          {thresholdErr && (
            <span className="text-[11px] text-[var(--color-error)]">{thresholdErr}</span>
          )}
        </div>
      </div>
    </div>
  )
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div>
        <div className="text-[12.5px] text-[var(--color-text)]">{label}</div>
        <div className="text-[10.5px] text-[var(--color-text-subtle)]">{hint}</div>
      </div>
      <Switch checked={checked} onChange={onChange} label={label} />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Channels panel
   ──────────────────────────────────────────────────────────────────── */

const MIN_SEV_OPTIONS = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

function ChannelsPanel({ onError }: { onError: (e: string | null) => void }) {
  const [view, setView] = useState<ChannelsView | null>(null)

  // Write-side form state (secrets are never returned, so they start blank).
  const [webhookUrl, setWebhookUrl] = useState('')
  const [webhookEnabled, setWebhookEnabled] = useState(false)
  const [webhookSev, setWebhookSev] = useState<Severity>('low')

  const [slackUrl, setSlackUrl] = useState('')
  const [slackEnabled, setSlackEnabled] = useState(false)
  const [slackSev, setSlackSev] = useState<Severity>('low')

  const [pdKey, setPdKey] = useState('')
  const [pdEnabled, setPdEnabled] = useState(false)
  const [pdSev, setPdSev] = useState<Severity>('high')

  const [emailEnabled, setEmailEnabled] = useState(false)
  const [smtpHost, setSmtpHost] = useState('')
  const [smtpPort, setSmtpPort] = useState('587')
  const [emailUser, setEmailUser] = useState('')
  const [emailPass, setEmailPass] = useState('')
  const [emailFrom, setEmailFrom] = useState('')
  const [emailTo, setEmailTo] = useState('')
  const [emailSev, setEmailSev] = useState<Severity>('medium')

  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    try {
      const { data } = await fetchChannels()
      setView(data)
      // Seed the non-secret bits we DO know.
      setWebhookEnabled(data.webhook.enabled)
      if (data.webhook.min_severity) setWebhookSev(data.webhook.min_severity)
      setSlackEnabled(data.slack.enabled)
      if (data.slack.min_severity) setSlackSev(data.slack.min_severity)
      setPdEnabled(data.pagerduty.enabled)
      if (data.pagerduty.min_severity) setPdSev(data.pagerduty.min_severity)
      setEmailEnabled(data.email.enabled)
      if (data.email.min_severity) setEmailSev(data.email.min_severity)
      if (data.email.email) {
        setSmtpHost(data.email.email.smtp_host)
        setSmtpPort(String(data.email.email.smtp_port))
        setEmailFrom(data.email.email.from)
      }
    } catch (e) {
      onError(e instanceof Error ? e.message : 'Failed to load channels')
    }
  }, [onError])

  useEffect(() => {
    load()
  }, [load])

  const save = async () => {
    setSaving(true)
    const body: ChannelsConfig = {}
    if (webhookUrl.trim())
      body.webhook = {
        enabled: webhookEnabled,
        url: webhookUrl.trim(),
        min_severity: webhookSev,
      }
    if (slackUrl.trim())
      body.slack = {
        enabled: slackEnabled,
        webhook_url: slackUrl.trim(),
        min_severity: slackSev,
      }
    if (pdKey.trim())
      body.pagerduty = {
        enabled: pdEnabled,
        routing_key: pdKey.trim(),
        min_severity: pdSev,
      }
    if (smtpHost.trim() && emailFrom.trim())
      body.email = {
        enabled: emailEnabled,
        smtp_host: smtpHost.trim(),
        smtp_port: Number(smtpPort) || 587,
        username: emailUser,
        password: emailPass,
        from: emailFrom.trim(),
        to: emailTo
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        min_severity: emailSev,
      }
    try {
      const { data } = await putChannels(body)
      setView(data)
      onError(null)
      toast.success('Channels saved')
      // Secrets stay write-only — clear them from the form after save.
      setWebhookUrl('')
      setSlackUrl('')
      setPdKey('')
      setEmailPass('')
    } catch (e) {
      onError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  if (view === null) return <ListSkeleton />

  return (
    <div className="space-y-3">
      <p className="text-[11.5px] text-[var(--color-text-subtle)]">
        Secrets (URLs, routing key, SMTP password) are write-only — never
        returned. Leave a secret blank to keep the stored value untouched only
        if that channel is left unset; submitting a channel replaces its config.
      </p>

      <ChannelCard
        title="Webhook"
        state={view.webhook}
        enabled={webhookEnabled}
        onEnabled={setWebhookEnabled}
        sev={webhookSev}
        onSev={setWebhookSev}
      >
        <LabeledInput
          label="URL"
          placeholder={view.webhook.configured ? '•••••• (set — re-enter to change)' : 'https://…'}
          value={webhookUrl}
          onChange={setWebhookUrl}
        />
      </ChannelCard>

      <ChannelCard
        title="Slack"
        state={view.slack}
        enabled={slackEnabled}
        onEnabled={setSlackEnabled}
        sev={slackSev}
        onSev={setSlackSev}
      >
        <LabeledInput
          label="Incoming webhook URL"
          placeholder={view.slack.configured ? '•••••• (set)' : 'https://hooks.slack.com/…'}
          value={slackUrl}
          onChange={setSlackUrl}
        />
      </ChannelCard>

      <ChannelCard
        title="PagerDuty"
        state={view.pagerduty}
        enabled={pdEnabled}
        onEnabled={setPdEnabled}
        sev={pdSev}
        onSev={setPdSev}
      >
        <LabeledInput
          label="Routing key"
          placeholder={view.pagerduty.configured ? '•••••• (set)' : 'Events API v2 routing key'}
          value={pdKey}
          onChange={setPdKey}
        />
      </ChannelCard>

      <ChannelCard
        title="Email (SMTP)"
        state={view.email}
        enabled={emailEnabled}
        onEnabled={setEmailEnabled}
        sev={emailSev}
        onSev={setEmailSev}
      >
        <div className="grid grid-cols-2 gap-2.5">
          <LabeledInput label="SMTP host" value={smtpHost} onChange={setSmtpHost} />
          <LabeledInput label="SMTP port" value={smtpPort} onChange={setSmtpPort} />
          <LabeledInput label="Username" value={emailUser} onChange={setEmailUser} />
          <LabeledInput
            label="Password"
            type="password"
            placeholder={view.email.configured ? '•••••• (set)' : ''}
            value={emailPass}
            onChange={setEmailPass}
          />
          <LabeledInput label="From" value={emailFrom} onChange={setEmailFrom} />
          <LabeledInput
            label="Recipients (comma-sep)"
            value={emailTo}
            onChange={setEmailTo}
          />
        </div>
        {view.email.email && (
          <p className="text-[10.5px] text-[var(--color-text-subtle)] mt-1.5">
            Currently: {view.email.email.recipients} recipient(s) ·{' '}
            {view.email.email.from}
          </p>
        )}
      </ChannelCard>

      <Button variant="primary" size="sm" onClick={save} disabled={saving}>
        {saving ? 'Saving…' : 'Save channels'}
      </Button>
    </div>
  )
}

function ChannelCard({
  title,
  state,
  enabled,
  onEnabled,
  sev,
  onSev,
  children,
}: {
  title: string
  state: ChannelsView[keyof ChannelsView]
  enabled: boolean
  onEnabled: (v: boolean) => void
  sev: Severity
  onSev: (v: Severity) => void
  children: React.ReactNode
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-3">
      <div className="flex items-center gap-2">
        <Bell size={13} strokeWidth={2.25} className="text-[var(--color-text-subtle)]" />
        <span className="text-[13px] font-medium text-[var(--color-text)]">{title}</span>
        <Badge
          label={state.configured ? 'configured' : 'not set'}
          variant={state.configured ? 'success' : 'ghost'}
        />
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[11px] text-[var(--color-text-subtle)]">enabled</span>
          <Switch checked={enabled} onChange={onEnabled} label={`${title} enabled`} />
        </div>
      </div>
      <div className="mt-3 space-y-2.5">{children}</div>
      <div className="flex items-center justify-between gap-3 mt-3">
        <span className="text-[11.5px] text-[var(--color-text-muted)]">Min severity</span>
        <Select
          size="sm"
          value={sev}
          options={MIN_SEV_OPTIONS}
          onValueChange={(v) => onSev(v as Severity)}
        />
      </div>
    </div>
  )
}

function LabeledInput({
  label,
  value,
  onChange,
  placeholder,
  type,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  type?: string
}) {
  return (
    <label className="block">
      <span className="block text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)] mb-1">
        {label}
      </span>
      <Input
        value={value}
        type={type}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Shared bits
   ──────────────────────────────────────────────────────────────────── */

function ListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  )
}

function formatRelative(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

function formatAbsolute(ts: number): string {
  return new Date(ts).toLocaleString()
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(1)} ${units[i]}`
}
