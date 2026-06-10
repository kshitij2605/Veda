import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

/**
 * Surface panel. `glow` adds a gated radial accent glow — reserve it for
 * showcase bands (hero / overview), never behind dense data. `interactive`
 * lifts the card on hover (use for clickable cards).
 */
export function Card({
  children,
  className,
  glow,
  interactive,
}: {
  children: ReactNode
  className?: string
  glow?: boolean
  interactive?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] shadow-[var(--shadow-sm)]',
        interactive &&
          'transition-[box-shadow,transform] duration-[var(--duration-base)] ease-[var(--ease-out)] hover:shadow-[var(--shadow-lg)] hover:-translate-y-0.5',
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
