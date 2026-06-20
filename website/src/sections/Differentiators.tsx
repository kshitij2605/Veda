import { Section } from '../components/Section'
import { useReveal } from '../lib/useReveal'
import { FEATURES } from '../content/features'

export function Differentiators() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <Section
      id="features"
      number="02"
      overline="Why Veda"
      title="Built multimodal-first, end to end"
      lede="Most vector databases just store vectors. Veda owns the whole path — from raw files to fused, cross-modal search."
    >
      <div
        ref={ref}
        className="mt-14 grid gap-px overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-border)] sm:grid-cols-2 lg:grid-cols-3"
      >
        {FEATURES.map((f, i) => {
          const Icon = f.icon
          // Keep the bordered grid clean at any card count: have the last card
          // span whatever fills the final row on each breakpoint (sm = 2-col,
          // lg = 3-col) so it never leaves a lonely empty cell.
          const isLast = i === FEATURES.length - 1
          const n = FEATURES.length
          const smSpan = n % 2 === 1 ? 'sm:col-span-2' : ''
          // lg fill: span the gap on the final 3-col row. When no lg span is
          // needed but smSpan is set, reset to 1 col so sm:col-span-2 doesn't
          // cascade up and wrongly widen the card on lg.
          const lgSpan =
            n % 3 === 1 ? 'lg:col-span-3' : n % 3 === 2 ? 'lg:col-span-2' : smSpan ? 'lg:col-span-1' : ''
          const span = isLast ? [smSpan, lgSpan].filter(Boolean).join(' ') : ''
          const wide = n % 3 !== 0 && span !== ''
          return (
            <div
              key={f.title}
              className={`group bg-[var(--color-bg)] p-7 transition-colors duration-[var(--duration-base)] hover:bg-[var(--color-surface)] ${span}`}
            >
              <div className="flex items-center justify-between">
                <span style={{ color: f.accent }}>
                  <Icon size={20} strokeWidth={2} />
                </span>
                <span className="[font-family:var(--font-mono)] text-[12px] text-[var(--color-text-subtle)]">
                  {String(i + 1).padStart(2, '0')}
                </span>
              </div>
              <h3 className="mt-5 text-[18px] [font-family:var(--font-display)] font-bold tracking-tight text-[var(--color-text)] transition-colors group-hover:text-[var(--color-accent)]">
                {f.title}
              </h3>
              <p
                className={`mt-2 text-[14px] leading-relaxed text-[var(--color-text-muted)] ${
                  wide ? 'max-w-3xl' : ''
                }`}
              >
                {f.blurb}
              </p>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
