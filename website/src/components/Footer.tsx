import { withBase } from '../lib/withBase'
import { BrandMark } from './BrandMark'
import { GitHubIcon } from './GitHubIcon'

const COLS: { title: string; links: [string, string][] }[] = [
  {
    title: 'Product',
    links: [
      ['Features', '#features'],
      ['Architecture', '#architecture'],
      ['Dashboard', '/niriksha'],
    ],
  },
  {
    title: 'Developers',
    links: [
      ['Docs', '/docs'],
      ['Python SDK', 'https://github.com/kshitij2605/Veda'],
      ['GitHub', 'https://github.com/kshitij2605/Veda'],
      ['Contributing', 'https://github.com/kshitij2605/Veda/blob/main/CONTRIBUTING.md'],
    ],
  },
]

export function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] bg-[var(--color-bg-raised)]">
      <div className="mx-auto grid max-w-[1320px] gap-10 px-6 md:px-10 py-16 md:grid-cols-[1.6fr_1fr_1fr]">
        <div>
          <div className="flex items-center gap-2.5">
            <BrandMark size={20} />
            <span className="text-[18px] font-bold [font-family:var(--font-display)] text-[var(--color-text)]">
              Veda
            </span>
          </div>
          <p className="mt-3 max-w-xs text-[14px] leading-relaxed text-[var(--color-text-muted)]">
            A next-generation multimodal vector database for the era of RAG.
          </p>
        </div>
        {COLS.map((c) => (
          <div key={c.title}>
            <h3 className="mono-label text-[var(--color-text-subtle)]">{c.title}</h3>
            <ul className="mt-4 space-y-2.5">
              {c.links.map(([label, href]) => (
                <li key={label}>
                  <a
                    href={withBase(href)}
                    className="text-[14px] text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-accent)]"
                  >
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="border-t border-[var(--color-border)]">
        <div className="mx-auto flex max-w-[1320px] items-center justify-between px-6 md:px-10 py-5 text-[12.5px] [font-family:var(--font-mono)] text-[var(--color-text-subtle)]">
          <span>© 2026 Veda · Apache-2.0</span>
          <a
            href="https://github.com/kshitij2605/Veda"
            className="inline-flex items-center gap-1.5 transition-colors hover:text-[var(--color-text)]"
          >
            <GitHubIcon size={14} /> GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
