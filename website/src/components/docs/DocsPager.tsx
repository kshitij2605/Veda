import { Link } from 'react-router-dom'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { FLAT } from '../../docs/registry'

export function DocsPager({ slug }: { slug: string }) {
  const i = FLAT.findIndex((p) => p.slug === slug)
  const prev = i > 0 ? FLAT[i - 1] : undefined
  const next = i >= 0 && i < FLAT.length - 1 ? FLAT[i + 1] : undefined
  if (!prev && !next) return null
  return (
    <div className="mt-14 grid gap-4 border-t border-[var(--color-border)] pt-8 sm:grid-cols-2">
      {prev ? (
        <Link
          to={`/docs/${prev.slug}`}
          className="group rounded-[var(--radius-md)] border border-[var(--color-border)] p-4 transition-colors hover:border-[var(--color-border-strong)]"
        >
          <span className="mono-label flex items-center gap-1.5 text-[var(--color-text-subtle)]">
            <ArrowLeft size={12} /> Previous
          </span>
          <span className="mt-1.5 block text-[15px] font-medium text-[var(--color-text)] transition-colors group-hover:text-[var(--color-accent)]">
            {prev.title}
          </span>
        </Link>
      ) : (
        <span />
      )}
      {next && (
        <Link
          to={`/docs/${next.slug}`}
          className="group rounded-[var(--radius-md)] border border-[var(--color-border)] p-4 text-right transition-colors hover:border-[var(--color-border-strong)]"
        >
          <span className="mono-label flex items-center justify-end gap-1.5 text-[var(--color-text-subtle)]">
            Next <ArrowRight size={12} />
          </span>
          <span className="mt-1.5 block text-[15px] font-medium text-[var(--color-text)] transition-colors group-hover:text-[var(--color-accent)]">
            {next.title}
          </span>
        </Link>
      )}
    </div>
  )
}
