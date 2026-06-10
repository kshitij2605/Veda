import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

/**
 * Surface panel. `glow` adds a gated radial accent glow — reserve it for
 * showcase bands (hero / overview), never behind dense data.
 */
export function Card({
  children,
  className,
  glow,
}: {
  children: ReactNode
  className?: string
  glow?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]',
        glow && 'relative overflow-hidden',
        className,
      )}
    >
      {glow && (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{ background: 'var(--glow-accent)' }}
        />
      )}
      <div className="relative">{children}</div>
    </div>
  )
}
