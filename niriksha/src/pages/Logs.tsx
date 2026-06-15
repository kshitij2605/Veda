import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  Clock,
  Layers,
  Pause,
  Play,
  RotateCw,
  ScrollText,
  Search as SearchIcon,
} from 'lucide-react'
import {
  fetchLogCategories,
  fetchLogs,
  fetchSlowQueries,
  type LogCategory,
  type LogRecord,
  type SlowQuery,
} from '../api/logs'
import {
  Badge,
  Button,
  CopyButton,
  Drawer,
  EmptyState,
  Input,
  Select,
  Skeleton,
} from '../components/primitives'
import { cn } from '../lib/cn'

/**
 * Logs — server-backed Log Insights surface. Live-tails the in-process
 * log capture buffer (`GET /v1/admin/logs`, polled every 5 s) with a
 * level filter + text search, surfaces the semantic categories of recent
 * WARN/ERROR messages (`GET /v1/admin/logs/categories`), and lists the
 * slowest searches across collections (`GET /v1/admin/slow-queries`).
 * Admin-only — the bearer token is attached by the shared axios client.
 */

const LEVEL_OPTIONS = [
  { value: 'ALL', label: 'All levels' },
  { value: 'INFO', label: 'INFO+' },
  { value: 'WARN', label: 'WARN+' },
  { value: 'ERROR', label: 'ERROR only' },
] as const

const POLL_MS = 5_000

export function Logs() {
  const [logs, setLogs] = useState<LogRecord[] | null>(null)
  const [level, setLevel] = useState<string>('ALL')
  const [search, setSearch] = useState('')
  const [paused, setPaused] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [lastFetched, setLastFetched] = useState<number | null>(null)
  const [selected, setSelected] = useState<LogRecord | null>(null)

  // Debounce the text search so each keystroke doesn't fire a request.
  const [debouncedSearch, setDebouncedSearch] = useState('')
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search.trim()), 300)
    return () => clearTimeout(id)
  }, [search])

  const loadLogs = useCallback(
    async (spin = false) => {
      if (spin) setLoading(true)
      try {
        const { data } = await fetchLogs({
          level: level === 'ALL' ? undefined : level,
          q: debouncedSearch || undefined,
          limit: 500,
        })
        setLogs(data.logs)
        setError(null)
        setLastFetched(Date.now())
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load logs')
      } finally {
        if (spin) setLoading(false)
      }
    },
    [level, debouncedSearch],
  )

  // Reload immediately when filters change.
  useEffect(() => {
    loadLogs(true)
  }, [loadLogs])

  // Live tail — poll every 5 s unless paused.
  useEffect(() => {
    if (paused) return
    const id = setInterval(() => loadLogs(false), POLL_MS)
    return () => clearInterval(id)
  }, [paused, loadLogs])

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <ScrollText
          size={16}
          strokeWidth={2}
          className="text-[var(--color-accent)]"
        />
        <h1 className="text-[16px] font-semibold tracking-tight text-[var(--color-text)]">
          Logs
        </h1>
        {lastFetched && (
          <span className="text-[11px] font-mono text-[var(--color-text-subtle)]">
            polled {formatRelative(lastFetched)} · {paused ? 'paused' : 'live'}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button
            size="xs"
            variant="secondary"
            leading={
              paused ? (
                <Play size={11} strokeWidth={2.25} />
              ) : (
                <Pause size={11} strokeWidth={2.25} />
              )
            }
            onClick={() => setPaused((p) => !p)}
          >
            {paused ? 'Resume' : 'Pause'}
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
            onClick={() => loadLogs(true)}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-auto px-4 py-4 space-y-5">
        {error && (
          <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)] flex items-center gap-2">
            <AlertCircle size={12} strokeWidth={2.25} />
            {error}
          </div>
        )}

        {/* ── Live tail ─────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <SectionHeading icon={<ScrollText size={12} strokeWidth={2.25} />}>
              Live tail
            </SectionHeading>
            <div className="ml-auto flex items-center gap-2">
              <Select
                size="sm"
                value={level}
                options={LEVEL_OPTIONS.map((o) => ({
                  value: o.value,
                  label: o.label,
                }))}
                onValueChange={setLevel}
              />
              <div className="relative">
                <SearchIcon
                  size={12}
                  strokeWidth={2.25}
                  className="absolute left-2 top-1/2 -translate-y-1/2 text-[var(--color-text-subtle)] pointer-events-none"
                />
                <Input
                  value={search}
                  placeholder="Filter message / target…"
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-7 w-[220px]"
                />
              </div>
            </div>
          </div>
          {logs === null ? (
            <TableSkeleton />
          ) : logs.length === 0 ? (
            <EmptyState
              icon={<ScrollText size={20} strokeWidth={2} />}
              title="No matching logs"
              description="Nothing in the capture buffer matches the current filters."
            />
          ) : (
            <LogTable logs={logs} onSelect={setSelected} />
          )}
        </section>

        {/* ── Categories ────────────────────────────────────────────── */}
        <CategoriesPanel />

        {/* ── Slow queries ──────────────────────────────────────────── */}
        <SlowQueriesPanel />
      </div>

      <LogDrawer record={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Live-tail table
   ──────────────────────────────────────────────────────────────────── */

function LogTable({
  logs,
  onSelect,
}: {
  logs: LogRecord[]
  onSelect: (r: LogRecord) => void
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
      <table className="w-full text-[12px] tabular-nums">
        <thead className="bg-[var(--color-bg-elevated)] text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
          <tr>
            <th className="px-3 py-2 text-left font-medium w-[88px]">Time</th>
            <th className="px-3 py-2 text-left font-medium w-[72px]">Level</th>
            <th className="px-3 py-2 text-left font-medium w-[180px]">Target</th>
            <th className="px-3 py-2 text-left font-medium">Message</th>
            <th className="px-3 py-2 text-left font-medium w-[140px]">Trace</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--color-border)]">
          {logs.map((r, i) => (
            <tr
              key={`${r.ts_ms}:${i}`}
              onClick={() => onSelect(r)}
              className="hover:bg-[var(--color-bg-hover)] cursor-pointer"
            >
              <td className="px-3 py-1.5 text-[var(--color-text-subtle)] font-mono text-[11px]">
                {formatRelative(r.ts_ms)}
              </td>
              <td className="px-3 py-1.5">
                <LevelBadge level={r.level} />
              </td>
              <td className="px-3 py-1.5 font-mono text-[11px] text-[var(--color-text-muted)] truncate max-w-[180px]">
                {r.target}
              </td>
              <td className="px-3 py-1.5 text-[var(--color-text)] truncate max-w-[480px]">
                {r.message}
              </td>
              <td className="px-3 py-1.5">
                {r.trace_id ? (
                  <span
                    className="inline-flex items-center gap-1"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <code className="font-mono text-[10.5px] text-[var(--color-text-subtle)]">
                      {r.trace_id.slice(0, 12)}…
                    </code>
                    <CopyButton value={r.trace_id} />
                  </span>
                ) : (
                  <span className="text-[var(--color-text-disabled)]">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LevelBadge({ level }: { level: string }) {
  const up = level.toUpperCase()
  const variant: 'error' | 'warning' | 'accent' | 'ghost' =
    up === 'ERROR'
      ? 'error'
      : up === 'WARN'
        ? 'warning'
        : up === 'INFO'
          ? 'accent'
          : 'ghost'
  return <Badge label={up} variant={variant} mono className="flex-none" />
}

/* ────────────────────────────────────────────────────────────────────────
   Categories — semantic grouping of recent WARN/ERROR logs
   ──────────────────────────────────────────────────────────────────── */

function CategoriesPanel() {
  const [categories, setCategories] = useState<LogCategory[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await fetchLogCategories()
      setCategories(data.categories)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load categories')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  return (
    <section>
      <div className="flex items-center gap-2 mb-2">
        <SectionHeading icon={<Layers size={12} strokeWidth={2.25} />}>
          Problem categories
        </SectionHeading>
        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)]">
          semantic grouping of recent WARN/ERROR
        </span>
        <Button
          size="xs"
          variant="ghost"
          className="ml-auto"
          leading={
            <RotateCw
              size={11}
              strokeWidth={2.25}
              className={loading ? 'animate-spin' : ''}
            />
          }
          onClick={load}
          disabled={loading}
        >
          Recompute
        </Button>
      </div>
      {error && (
        <div className="text-[11.5px] text-[var(--color-error)] mb-2">{error}</div>
      )}
      {categories === null ? (
        <TableSkeleton rows={2} />
      ) : categories.length === 0 ? (
        <EmptyState
          icon={<Layers size={20} strokeWidth={2} />}
          title="No warnings or errors"
          description="The capture buffer holds no recent WARN/ERROR messages to group."
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {categories.map((c) => (
            <li
              key={c.id}
              className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2.5"
            >
              <div className="flex items-center gap-2">
                <LevelBadge level={c.max_severity} />
                <span className="text-[12.5px] text-[var(--color-text)] truncate">
                  {c.representative}
                </span>
                <span className="ml-auto flex-none text-[11px] font-mono text-[var(--color-text-subtle)] tabular-nums">
                  ×{c.count}
                </span>
              </div>
              {c.examples.length > 0 && (
                <ul className="mt-1.5 space-y-0.5">
                  {c.examples.slice(0, 3).map((ex, i) => (
                    <li
                      key={i}
                      className="text-[11px] font-mono text-[var(--color-text-subtle)] truncate"
                    >
                      {ex}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Slow queries — server-backed from per-collection query logs
   ──────────────────────────────────────────────────────────────────── */

function SlowQueriesPanel() {
  const [items, setItems] = useState<SlowQuery[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await fetchSlowQueries({ limit: 20 })
      setItems(data.slow_queries)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load slow queries')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  return (
    <section>
      <div className="flex items-center gap-2 mb-2">
        <SectionHeading icon={<Clock size={12} strokeWidth={2.25} />}>
          Slowest queries
        </SectionHeading>
        <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)]">
          last 24h · server-backed query log
        </span>
        <Button
          size="xs"
          variant="ghost"
          className="ml-auto"
          leading={
            <RotateCw
              size={11}
              strokeWidth={2.25}
              className={loading ? 'animate-spin' : ''}
            />
          }
          onClick={load}
          disabled={loading}
        >
          Refresh
        </Button>
      </div>
      {error && (
        <div className="text-[11.5px] text-[var(--color-error)] mb-2">{error}</div>
      )}
      {items === null ? (
        <TableSkeleton rows={3} />
      ) : items.length === 0 ? (
        <EmptyState
          icon={<Clock size={20} strokeWidth={2} />}
          title="No queries recorded"
          description="Slow searches will appear here as the query log fills."
        />
      ) : (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
          <table className="w-full text-[11.5px] tabular-nums">
            <thead className="bg-[var(--color-bg-elevated)] text-[10px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Collection</th>
                <th className="px-3 py-1.5 text-right font-medium">Duration</th>
                <th className="px-3 py-1.5 text-right font-medium">top_k</th>
                <th className="px-3 py-1.5 text-left font-medium w-[150px]">Trace</th>
                <th className="px-3 py-1.5 text-right font-medium">When</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)]">
              {items.map((q, i) => (
                <tr key={i} className="hover:bg-[var(--color-bg-hover)]">
                  <td className="px-3 py-1 font-mono text-[var(--color-text)] truncate max-w-[280px]">
                    {q.collection}
                  </td>
                  <td className="px-3 py-1 text-right font-mono text-[var(--color-text)]">
                    {q.took_ms}ms
                  </td>
                  <td className="px-3 py-1 text-right font-mono text-[var(--color-text-muted)]">
                    {q.top_k}
                  </td>
                  <td className="px-3 py-1">
                    {q.trace_id ? (
                      <span className="inline-flex items-center gap-1">
                        <code className="font-mono text-[10.5px] text-[var(--color-text-subtle)]">
                          {q.trace_id.slice(0, 12)}…
                        </code>
                        <CopyButton value={q.trace_id} />
                      </span>
                    ) : (
                      <span className="text-[var(--color-text-disabled)]">—</span>
                    )}
                  </td>
                  <td className="px-3 py-1 text-right text-[var(--color-text-subtle)]">
                    {formatRelative(q.ts_ms)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Detail drawer
   ──────────────────────────────────────────────────────────────────── */

function LogDrawer({
  record,
  onClose,
}: {
  record: LogRecord | null
  onClose: () => void
}) {
  const r = record
  const fieldsJson = useMemo(
    () => (r ? JSON.stringify(r.fields ?? {}, null, 2) : ''),
    [r],
  )
  return (
    <Drawer
      open={!!r}
      onClose={onClose}
      title={r ? r.message || '(no message)' : undefined}
      subtitle={r ? `${r.level} · ${r.target}` : undefined}
      width={520}
    >
      {r && (
        <div className="space-y-5">
          <div className="flex items-center gap-2">
            <LevelBadge level={r.level} />
            <span className="text-[11px] font-mono text-[var(--color-text-subtle)]">
              {formatAbsolute(r.ts_ms)}
            </span>
          </div>

          <Detail label="Target" value={r.target} mono />

          <div>
            <DetailLabel>Trace id</DetailLabel>
            {r.trace_id ? (
              <div className="flex items-center gap-1.5">
                <code className="font-mono text-[12px] text-[var(--color-text)] break-all">
                  {r.trace_id}
                </code>
                <CopyButton value={r.trace_id} />
              </div>
            ) : (
              <div className="text-[12px] text-[var(--color-text-subtle)]">—</div>
            )}
          </div>

          <div>
            <DetailLabel>Span id</DetailLabel>
            {r.span_id ? (
              <div className="flex items-center gap-1.5">
                <code className="font-mono text-[12px] text-[var(--color-text)] break-all">
                  {r.span_id}
                </code>
                <CopyButton value={r.span_id} />
              </div>
            ) : (
              <div className="text-[12px] text-[var(--color-text-subtle)]">—</div>
            )}
          </div>

          <div>
            <DetailLabel>Fields</DetailLabel>
            <pre className="rounded-[var(--radius-sm)] bg-[var(--color-base-1)] border border-[var(--color-border)] px-2.5 py-2 text-[11.5px] font-mono text-[var(--color-text)] whitespace-pre-wrap break-words max-h-72 overflow-auto">
              {fieldsJson === '{}' ? '(no structured fields)' : fieldsJson}
            </pre>
          </div>
        </div>
      )}
    </Drawer>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Shared bits
   ──────────────────────────────────────────────────────────────────── */

function SectionHeading({
  icon,
  children,
}: {
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <h2 className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] inline-flex items-center gap-1.5">
      {icon}
      {children}
    </h2>
  )
}

function DetailLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-1">
      {children}
    </div>
  )
}

function Detail({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <DetailLabel>{label}</DetailLabel>
      <div
        className={cn(
          'text-[12.5px] text-[var(--color-text)] break-words',
          mono && 'font-mono text-[12px]',
        )}
      >
        {value}
      </div>
    </div>
  )
}

function TableSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
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
