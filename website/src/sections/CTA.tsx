import { Suspense, lazy } from 'react'
import { Button } from '../components/primitives/Button'
import { GitHubIcon } from '../components/GitHubIcon'

const Aurora = lazy(() => import('../components/reactbits/Aurora'))

export function CTA() {
  return (
    <section id="get-started" className="relative overflow-hidden border-t border-[var(--color-border)]">
      <div aria-hidden className="pointer-events-none absolute inset-0 opacity-40">
        <Suspense fallback={null}>
          <Aurora colorStops={['#7c6af0', '#b06bff', '#38bdf8']} amplitude={0.8} blend={0.5} />
        </Suspense>
      </div>
      <div
        aria-hidden
        className="absolute inset-0"
        style={{ background: 'radial-gradient(80% 80% at 50% 45%, transparent, var(--color-bg) 82%)' }}
      />

      <div className="relative mx-auto max-w-[820px] px-6 md:px-8 py-24 md:py-32 text-center">
        <h2 className="text-[clamp(30px,5vw,52px)] [font-family:var(--font-display)] font-semibold leading-[1.05] tracking-[-0.02em] text-[var(--color-text)]">
          Up and running in 30 seconds
        </h2>
        <p className="mt-4 text-[18px] leading-relaxed text-[var(--color-text-muted)]">
          Clone the repo, start three services, and ingest your first multimodal document.
        </p>
        <div className="mt-7 inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-5 py-3 [font-family:var(--font-mono)] text-[14px] text-[var(--color-text)] shadow-[var(--shadow-sm)]">
          <span className="text-[var(--color-text-subtle)]">$</span> docker compose up
        </div>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
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
          <Button variant="ghost" size="lg" onClick={() => { window.location.href = '/niriksha' }}>
            Open the dashboard
          </Button>
        </div>
      </div>
    </section>
  )
}
