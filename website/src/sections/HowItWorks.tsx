import { Section } from '../components/Section'
import { useReveal } from '../lib/useReveal'
import { PIPELINE } from '../content/pipeline'

export function HowItWorks() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <Section
      overline="How it works"
      title="From raw files to search, in one pipeline"
      lede="Veda's document endpoint runs the entire ingestion path natively — no Unstructured.io, no LangChain loaders, no glue code."
    >
      <div ref={ref} className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {PIPELINE.map((s, i) => {
          const Icon = s.icon
          return (
            <div
              key={s.title}
              className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-5 shadow-[var(--shadow-sm)]"
            >
              <div className="flex items-center gap-2 text-[var(--color-accent)]">
                <span className="text-[12px] [font-family:var(--font-mono)] text-[var(--color-text-subtle)]">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <Icon size={18} strokeWidth={2} />
              </div>
              <h3 className="mt-3 text-[15px] font-semibold text-[var(--color-text)]">{s.title}</h3>
              <p className="mt-1 text-[13px] leading-relaxed text-[var(--color-text-muted)]">{s.caption}</p>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
