import { Section } from '../components/Section'
import { useReveal } from '../lib/useReveal'

/** Short demos recorded against a running Veda — no mockups. */
const DEMOS = [
  {
    href: '/docs/quickstart',
    label: 'Quick start',
    blurb: 'Two calls to a collection you can search.',
  },
  {
    href: '/docs/search',
    label: 'Search & fusion',
    blurb: 'Meaning, keywords, and both blended together.',
  },
  {
    href: '/docs/document-pipeline',
    label: 'Document pipeline',
    blurb: 'A PDF goes in, searchable text comes out.',
  },
  {
    href: '/docs/agents-mcp',
    label: 'Agents & MCP',
    blurb: 'Answers that carry the passages behind them.',
  },
  {
    href: '/docs/auto-tuning',
    label: 'RAG auto-tuning',
    blurb: 'Veda writes the test set and tunes itself against it.',
  },
  {
    href: '/docs/multimodal',
    label: 'Multimodal search',
    blurb: 'Type “a dog barking” and get the recording back.',
  },
]

export function SeeItInAction() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <Section
      id="demos"
      number="05"
      overline="See it in action"
      title="Watch it actually run"
      lede="These are screen recordings of a live Veda instance — real collections, real queries, real timings. Nothing is mocked up."
    >
      <div ref={ref} className="mt-14">
        <figure className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]">
          <video
            src="/videos/multimodal.mp4"
            poster="/videos/multimodal.jpg"
            controls
            muted
            playsInline
            preload="none"
            className="w-full"
          />
        </figure>
        <p className="mt-4 text-[14px] leading-[1.7] text-[var(--color-text-subtle)]">
          One search box, one collection of sound clips and written notes. The query is plain
          English; the top hit is an audio file.
        </p>

        <div className="mt-10 grid gap-px overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-border)] sm:grid-cols-2 lg:grid-cols-3">
          {DEMOS.map((d) => (
            <a
              key={d.href}
              href={d.href}
              className="group bg-[var(--color-bg)] p-6 transition-colors duration-[var(--duration-base)] hover:bg-[var(--color-surface)]"
            >
              <p className="text-[15px] font-semibold text-[var(--color-text)] transition-colors group-hover:text-[var(--color-accent)]">
                {d.label}
              </p>
              <p className="mt-2 text-[14px] leading-[1.65] text-[var(--color-text-muted)]">
                {d.blurb}
              </p>
            </a>
          ))}
        </div>
      </div>
    </Section>
  )
}
