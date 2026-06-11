import { type ReactNode } from 'react'
import { cn } from '../lib/cn'

/** Full-bleed section band with a centered container, mono-numbered overline, and display heading. */
export function Section({
  id,
  number,
  overline,
  title,
  lede,
  children,
  className,
  bleed,
}: {
  id?: string
  /** Section index, rendered as `02 /` in saffron mono. */
  number?: string
  overline?: string
  title?: ReactNode
  lede?: ReactNode
  children?: ReactNode
  className?: string
  /** Extra full-bleed layers (textures, rules) rendered behind the container. */
  bleed?: ReactNode
}) {
  return (
    <section id={id} className={cn('relative', className)}>
      {bleed}
      <div className="relative mx-auto max-w-[1320px] px-6 md:px-10 py-20 md:py-28">
        {(overline || title || lede) && (
          <div className="max-w-2xl">
            {overline && (
              <p className="mono-label text-[var(--color-accent)]">
                {number ? `${number} / ` : ''}
                {overline}
              </p>
            )}
            {title && (
              <h2 className="mt-4 text-[clamp(30px,4vw,48px)] [font-family:var(--font-display)] font-bold leading-[1.06] tracking-[-0.02em] text-[var(--color-text)]">
                {title}
              </h2>
            )}
            {lede && (
              <p className="mt-5 text-[17px] leading-relaxed text-[var(--color-text-muted)]">{lede}</p>
            )}
          </div>
        )}
        {children}
      </div>
    </section>
  )
}
