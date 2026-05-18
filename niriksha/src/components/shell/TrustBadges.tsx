import { useEffect, useState } from 'react'
import { Activity, Clock, Server } from 'lucide-react'
import { useCollectionStore } from '../../stores/collectionStore'
import { Tooltip } from '../primitives/Tooltip'
import { cn } from '../../lib/cn'

/**
 * TrustBadges — sticky in the header. Engineers distrust tools that hide
 * their work; surface freshness + health + version unprompted.
 */
export function TrustBadges() {
  const { health, fetchHealth } = useCollectionStore()

  useEffect(() => {
    fetchHealth()
    const t = setInterval(fetchHealth, 15_000)
    return () => clearInterval(t)
  }, [fetchHealth])

  const healthy = health?.status === 'healthy'
  const degraded = health?.status === 'degraded'

  return (
    <div className="flex items-center gap-2 text-[11px] text-[var(--color-text-muted)]">
      <Tooltip
        label={
          healthy
            ? 'All subsystem checks green'
            : degraded
            ? 'Subsystem(s) reporting degraded'
            : 'Unable to reach Veda /admin/health'
        }
      >
        <span
          className={cn(
            'inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)]',
            'border tabular-nums',
            healthy &&
              'bg-[color-mix(in_oklch,var(--color-success)_10%,transparent)] text-[var(--color-success)] border-[color-mix(in_oklch,var(--color-success)_25%,transparent)]',
            degraded &&
              'bg-[color-mix(in_oklch,var(--color-warning)_10%,transparent)] text-[var(--color-warning)] border-[color-mix(in_oklch,var(--color-warning)_25%,transparent)]',
            !healthy && !degraded &&
              'bg-[color-mix(in_oklch,var(--color-error)_10%,transparent)] text-[var(--color-error)] border-[color-mix(in_oklch,var(--color-error)_25%,transparent)]',
          )}
        >
          <Activity size={10} strokeWidth={2.5} />
          {healthy ? 'healthy' : degraded ? 'degraded' : 'unreachable'}
        </span>
      </Tooltip>

      {health?.version && (
        <Tooltip label="Server version">
          <span className="inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)] bg-[var(--color-base-2)] border border-[var(--color-border)] font-mono tabular-nums">
            <Server size={10} strokeWidth={2.5} />v{health.version}
          </span>
        </Tooltip>
      )}

      <Tooltip label="Local clock — for cross-referencing logs">
        <span className="inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)] bg-[var(--color-base-2)] border border-[var(--color-border)] font-mono tabular-nums text-[var(--color-text-subtle)]">
          <Clock size={10} strokeWidth={2.5} />
          <LocalClock />
        </span>
      </Tooltip>
    </div>
  )
}

function LocalClock() {
  const [time, setTime] = useState(formatTime(new Date()))
  useEffect(() => {
    const id = setInterval(() => setTime(formatTime(new Date())), 60_000)
    return () => clearInterval(id)
  }, [])
  return <>{time}</>
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
