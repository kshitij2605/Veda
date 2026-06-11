import { useEffect, useMemo, useState } from 'react'
import { cn } from '../../lib/cn'
import { parseHeadings } from '../../docs/headings'

/** Right-rail "On this page" with IntersectionObserver scroll-spy. */
export function DocsToc({ md }: { md: string }) {
  const headings = useMemo(() => parseHeadings(md), [md])
  const [active, setActive] = useState<string | null>(null)

  useEffect(() => {
    setActive(headings[0]?.id ?? null)
    const els = headings
      .map((h) => document.getElementById(h.id))
      .filter((el): el is HTMLElement => !!el)
    if (els.length === 0) return
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting)
        if (visible.length > 0) setActive(visible[0]!.target.id)
      },
      { rootMargin: '-80px 0px -70% 0px' },
    )
    els.forEach((el) => obs.observe(el))
    return () => obs.disconnect()
  }, [headings])

  if (headings.length === 0) return null

  return (
    <nav>
      <p className="mono-label text-[var(--color-text-subtle)]">On this page</p>
      <ul className="mt-3 border-l border-[var(--color-border)]">
        {headings.map((h) => (
          <li key={h.id}>
            <a
              href={`#${h.id}`}
              onClick={(e) => {
                e.preventDefault()
                document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                history.replaceState(null, '', `#${h.id}`)
              }}
              className={cn(
                '-ml-px block border-l py-1 pl-3.5 text-[13px] leading-snug transition-colors',
                h.depth === 3 && 'pl-7',
                active === h.id
                  ? 'border-[var(--color-accent)] text-[var(--color-accent)]'
                  : 'border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
              )}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
