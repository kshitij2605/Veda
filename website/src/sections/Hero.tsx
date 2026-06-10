import { Suspense, lazy } from 'react'
import { Button } from '../components/primitives/Button'
import { GitHubIcon } from '../components/GitHubIcon'
import { HeroScatter } from '../components/HeroScatter'

const Aurora = lazy(() => import('../components/reactbits/Aurora'))

export function Hero() {
  return (
    <section id="hero" className="relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0 opacity-45">
        <Suspense fallback={null}>
          <Aurora colorStops={['#7c6af0', '#b06bff', '#38bdf8']} amplitude={0.85} blend={0.6} />
        </Suspense>
      </div>
      <div
        aria-hidden
        className="absolute inset-0"
        style={{ background: 'radial-gradient(75% 55% at 50% 30%, transparent, var(--color-bg) 88%)' }}
      />

      <div className="relative mx-auto grid max-w-[1200px] items-center gap-10 px-6 md:px-8 pt-32 pb-20 md:pt-40 md:pb-28 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <p className="text-[12px] font-semibold uppercase tracking-[0.16em] text-[var(--color-accent)]">
            Multimodal vector database
          </p>
          <h1 className="mt-4 text-[clamp(40px,7vw,76px)] [font-family:var(--font-display)] font-semibold leading-[1.02] tracking-[-0.03em] text-[var(--color-text)]">
            The{' '}
            <span
              className="bg-clip-text text-transparent"
              style={{ backgroundImage: 'linear-gradient(110deg, #7c6af0, #c66be6, #38bdf8)' }}
            >
              multimodal
            </span>{' '}
            vector database for the era of RAG
          </h1>
          <p className="mt-6 max-w-xl text-[18px] leading-relaxed text-[var(--color-text-muted)]">
            Upload a PDF, image, or video and Veda handles the whole pipeline — OCR, chunking,
            modality-routed embeddings, indexing, and fused search — in a single pure-Rust engine.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              size="lg"
              onClick={() =>
                document.getElementById('get-started')?.scrollIntoView({ behavior: 'smooth' })
              }
            >
              Get started
            </Button>
            <Button
              variant="secondary"
              size="lg"
              leading={<GitHubIcon size={16} />}
              onClick={() => window.open('https://github.com/kshitij2605/Veda', '_blank')}
            >
              View on GitHub
            </Button>
          </div>
        </div>

        <div className="relative h-[300px] md:h-[420px]">
          <HeroScatter className="absolute inset-0 h-full w-full" />
        </div>
      </div>
    </section>
  )
}
