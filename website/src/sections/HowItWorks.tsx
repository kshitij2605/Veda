import { Section } from '../components/Section'
import { useReveal } from '../lib/useReveal'
import { PIPELINE } from '../content/pipeline'

export function HowItWorks() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <Section
      number="04"
      overline="How it works"
      title="From raw files to search, in one pipeline"
      lede="Veda's document endpoint runs the entire ingestion path natively — no Unstructured.io, no LangChain loaders, no glue code."
    >
      <div
        ref={ref}
        className="mt-14 grid gap-px overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-border)] sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6"
      >
        {PIPELINE.map((s, i) => {
          const Icon = s.icon
          return (
            <div
              key={s.title}
              className="group bg-[var(--color-bg)] p-6 transition-colors duration-[var(--duration-base)] hover:bg-[var(--color-surface)]"
            >
              <div className="flex items-center justify-between">
                <span className="[font-family:var(--font-mono)] text-[12px] text-[var(--color-accent)]">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span className="text-[var(--color-text-subtle)] transition-colors group-hover:text-[var(--color-accent)]">
                  <Icon size={18} strokeWidth={2} />
                </span>
              </div>
              <h3 className="mt-5 text-[16px] [font-family:var(--font-display)] font-bold text-[var(--color-text)]">
                {s.title}
              </h3>
              <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--color-text-muted)]">{s.caption}</p>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
