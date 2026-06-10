import { type ReactNode } from 'react'
import { cn } from '../lib/cn'

/** Centered section wrapper with an optional overline + display heading + lede. */
export function Section({
  id,
  overline,
  title,
  lede,
  children,
  className,
}: {
  id?: string
  overline?: string
  title?: ReactNode
  lede?: ReactNode
  children?: ReactNode
  className?: string
}) {
  return (
    <section id={id} className={cn('mx-auto max-w-[1200px] px-6 md:px-8 py-20 md:py-28', className)}>
      {(overline || title || lede) && (
        <div className="max-w-2xl">
          {overline && (
            <p className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
              {overline}
            </p>
          )}
          {title && (
            <h2 className="mt-3 text-[clamp(28px,4vw,40px)] [font-family:var(--font-display)] font-semibold leading-[1.1] tracking-[-0.02em] text-[var(--color-text)]">
              {title}
            </h2>
          )}
          {lede && (
            <p className="mt-4 text-[17px] leading-relaxed text-[var(--color-text-muted)]">{lede}</p>
          )}
        </div>
      )}
      {children}
    </section>
  )
}
