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
          const last = i === FEATURES.length - 1
          return (
            <div
              key={f.title}
              className={`group bg-[var(--color-bg)] p-7 transition-colors duration-[var(--duration-base)] hover:bg-[var(--color-surface)] ${
                last ? 'sm:col-span-2 lg:col-span-3' : ''
              }`}
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
              <p className={`mt-2 text-[14px] leading-relaxed text-[var(--color-text-muted)] ${last ? 'max-w-3xl' : ''}`}>
                {f.blurb}
              </p>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
