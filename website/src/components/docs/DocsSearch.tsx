import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, Hash, Search } from 'lucide-react'
import { cn } from '../../lib/cn'
import { FLAT } from '../../docs/registry'
import { parseHeadings, type DocHeading } from '../../docs/headings'

interface Item {
  slug: string
  pageTitle: string
  heading?: DocHeading
}

const INDEX: Item[] = FLAT.flatMap((p) => [
  { slug: p.slug, pageTitle: p.title },
  ...parseHeadings(p.md)
    .filter((h) => h.depth === 2)
    .map((h) => ({ slug: p.slug, pageTitle: p.title, heading: h })),
])

/** Ctrl/Cmd-K palette over page titles + h2 headings. Dependency-free. */
export function DocsSearch({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (open) {
      setQ('')
      setSel(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const results = useMemo(() => {
    const t = q.trim().toLowerCase()
    if (!t) return INDEX.filter((i) => !i.heading)
    return INDEX.filter(
      (i) =>
        (i.heading?.text ?? '').toLowerCase().includes(t) ||
        i.pageTitle.toLowerCase().includes(t),
    ).slice(0, 12)
  }, [q])

  const go = (item: Item) => {
    navigate(`/docs/${item.slug}${item.heading ? `#${item.heading.id}` : ''}`)
    onClose()
  }

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[100] bg-[rgba(5,6,8,0.72)] backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Search documentation"
    >
      <div
        className="mx-auto mt-[12vh] w-[min(620px,calc(100vw-32px))] overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface)] shadow-[var(--shadow-lg)]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-4">
          <Search size={16} className="text-[var(--color-text-subtle)]" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => {
              setQ(e.target.value)
              setSel(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSel((s) => Math.min(s + 1, results.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSel((s) => Math.max(s - 1, 0))
              } else if (e.key === 'Enter' && results[sel]) {
                go(results[sel]!)
              } else if (e.key === 'Escape') {
                onClose()
              }
            }}
            placeholder="Search the docs…"
            className="h-12 flex-1 bg-transparent text-[15px] text-[var(--color-text)] outline-none placeholder:text-[var(--color-text-subtle)]"
          />
          <kbd className="mono-label rounded-[var(--radius-xs)] border border-[var(--color-border)] px-1.5 py-0.5 text-[var(--color-text-subtle)]">
            esc
          </kbd>
        </div>
        <ul className="max-h-[50vh] overflow-y-auto py-2">
          {results.length === 0 && (
            <li className="px-4 py-6 text-center text-[14px] text-[var(--color-text-subtle)]">
              No matches for “{q}”
            </li>
          )}
          {results.map((r, i) => (
            <li key={`${r.slug}-${r.heading?.id ?? 'page'}`}>
              <button
                type="button"
                onMouseEnter={() => setSel(i)}
                onClick={() => go(r)}
                className={cn(
                  'flex w-full cursor-pointer items-center gap-3 px-4 py-2.5 text-left',
                  i === sel ? 'bg-[var(--color-surface-2)]' : '',
                )}
              >
                {r.heading ? (
                  <Hash size={14} className="flex-none text-[var(--color-accent)]" />
                ) : (
                  <FileText size={14} className="flex-none text-[var(--color-text-subtle)]" />
                )}
                <span className="min-w-0">
                  <span className="block truncate text-[14px] text-[var(--color-text)]">
                    {r.heading?.text ?? r.pageTitle}
                  </span>
                  {r.heading && (
                    <span className="mono-label text-[var(--color-text-subtle)]">{r.pageTitle}</span>
                  )}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
