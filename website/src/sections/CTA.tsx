import { Button } from '../components/primitives/Button'
import { GitHubIcon } from '../components/GitHubIcon'
import { DotGrid } from '../components/reactbits/DotGrid'

export function CTA() {
  return (
    <section id="get-started" className="relative overflow-hidden border-t border-[var(--color-border)]">
      <DotGrid className="opacity-60" />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background: 'radial-gradient(60% 70% at 50% 50%, rgba(255,178,36,0.06), transparent 75%)',
        }}
      />

      <div className="relative mx-auto max-w-[860px] px-6 py-28 text-center md:px-10 md:py-36">
        <p className="mono-label text-[var(--color-accent)]">06 / Get started</p>
        <h2 className="mt-5 text-[clamp(34px,5.5vw,64px)] [font-family:var(--font-display)] font-bold leading-[1.04] tracking-[-0.025em] text-[var(--color-text)]">
          Run it on your own machine
        </h2>
        <p className="mt-5 text-[17px] leading-relaxed text-[var(--color-text-muted)]">
          Clone the repo, bring up three services, and ingest your first multimodal document. The
          quick start walks through it end to end.
        </p>
        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <Button variant="primary" size="lg" onClick={() => { window.location.href = '/docs' }}>
            Read the docs
          </Button>
          <Button
            variant="secondary"
            size="lg"
            leading={<GitHubIcon size={16} />}
            onClick={() => window.open('https://github.com/kshitij2605/Veda', '_blank')}
          >
            Star on GitHub
          </Button>
        </div>
      </div>
    </section>
  )
}
