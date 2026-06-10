import { Section } from '../components/Section'
import { SpotlightCard } from '../components/reactbits/SpotlightCard'
import { useReveal } from '../lib/useReveal'
import { FEATURES } from '../content/features'

export function Differentiators() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <Section
      id="features"
      overline="Why Veda"
      title="Built multimodal-first, end to end"
      lede="Most vector databases just store vectors. Veda owns the whole path — from raw files to fused, cross-modal search."
    >
      <div ref={ref} className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((f) => {
          const Icon = f.icon
          const color =
            f.modality === 'accent' ? 'var(--color-accent)' : `var(--color-modality-${f.modality})`
          return (
            <SpotlightCard key={f.title} className="p-6">
              <div
                className="flex h-11 w-11 items-center justify-center rounded-[var(--radius-md)]"
                style={{ background: `color-mix(in oklch, ${color} 14%, transparent)`, color }}
              >
                <Icon size={20} strokeWidth={2} />
              </div>
              <h3 className="mt-4 text-[18px] font-semibold tracking-tight text-[var(--color-text)]">
                {f.title}
              </h3>
              <p className="mt-2 text-[14px] leading-relaxed text-[var(--color-text-muted)]">
                {f.blurb}
              </p>
            </SpotlightCard>
          )
        })}
      </div>
    </Section>
  )
}
