import { Button } from '../components/primitives/Button'
import { GitHubIcon } from '../components/GitHubIcon'
import { HeroScatter } from '../components/HeroScatter'
import { DotGrid } from '../components/reactbits/DotGrid'
import { DecryptedText } from '../components/reactbits/DecryptedText'

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-[var(--color-border)]">
      {/* full-bleed texture layer */}
      <DotGrid className="opacity-50" />

      {/* the graph runs to the VIEWPORT's right edge (full-bleed fix) */}
      <div className="absolute inset-y-0 left-0 right-0 opacity-40 lg:left-[42%] lg:opacity-100">
        <HeroScatter className="absolute inset-0 h-full w-full" />
      </div>

      {/* readability fade under the headline */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'linear-gradient(90deg, var(--color-bg) 0%, rgba(10,12,16,0.85) 34%, rgba(10,12,16,0.25) 52%, transparent 66%)',
        }}
      />

      {/* content */}
      <div className="relative mx-auto flex min-h-[92vh] max-w-[1320px] flex-col justify-center px-6 py-32 md:px-10 md:py-40">
        <p className="mono-label text-[var(--color-accent)]">01 / Multimodal vector database</p>
        <h1 className="mt-6 max-w-[13ch] text-[clamp(44px,7.5vw,92px)] [font-family:var(--font-display)] font-bold leading-[1.02] tracking-[-0.03em] text-[var(--color-text)]">
          Search everything you can{' '}
          <DecryptedText
            text="embed."
            className="text-[var(--color-accent)]"
            encryptedClassName="text-[var(--color-accent)] opacity-60"
          />
        </h1>
        <p className="mt-7 max-w-xl text-[18px] leading-relaxed text-[var(--color-text-muted)]">
          Drop in any document, image, audio, or video — Veda runs the whole pipeline (OCR,
          chunking, modality-routed embeddings, indexing, fused search) and makes every modality
          searchable in one cross-modal space, on a single pure-Rust engine.
        </p>
        <div className="mt-9 flex flex-wrap items-center gap-3">
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
        {/* The only install snippet on the page. It must be runnable as shown:
            compose builds Veda from source, so the clone is not optional. */}
        <div className="mt-8 w-fit rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[rgba(13,16,21,0.7)] px-4 py-3 [font-family:var(--font-mono)] text-[13px] text-[var(--color-text-muted)]">
          <div className="flex items-center gap-2">
            <span className="text-[var(--color-accent)]">$</span>
            <span>git clone https://github.com/kshitij2605/Veda.git &amp;&amp; cd Veda</span>
          </div>
          <div className="mt-1.5 flex items-center gap-2">
            <span className="text-[var(--color-accent)]">$</span>
            <span>docker compose up</span>
          </div>
          <p className="mt-2.5 text-[12px] leading-relaxed text-[var(--color-text-subtle)] [font-family:var(--font-sans)]">
            The first run builds the image from source, so allow a few minutes.
          </p>
        </div>
      </div>
    </section>
  )
}
