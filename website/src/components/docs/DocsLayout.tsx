import { useEffect, useState } from 'react'
import { Menu, Search, X } from 'lucide-react'
import { Nav } from '../Nav'
import { Footer } from '../Footer'
import { DocsSidebar } from './DocsSidebar'
import { DocsToc } from './DocsToc'
import { DocsPager } from './DocsPager'
import { DocsSearch } from './DocsSearch'
import { DocMarkdown } from './DocMarkdown'
import type { DocPageEntry } from '../../docs/registry'

export function DocsLayout({ page }: { page: DocPageEntry }) {
  const [searchOpen, setSearchOpen] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setSearchOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => setMenuOpen(false), [page.slug])

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Nav solid />

      {/* mobile docs bar */}
      <div className="glass sticky top-16 z-40 border-b border-[var(--color-border)] lg:hidden">
        <div className="mx-auto flex h-12 max-w-[1320px] items-center gap-3 px-6">
          <button
            type="button"
            onClick={() => setMenuOpen((v) => !v)}
            className="mono-label inline-flex cursor-pointer items-center gap-1.5 text-[var(--color-text-muted)]"
          >
            {menuOpen ? <X size={14} /> : <Menu size={14} />} Menu
          </button>
          <span className="truncate text-[13px] text-[var(--color-text-subtle)]">{page.title}</span>
          <button
            type="button"
            aria-label="Search docs"
            onClick={() => setSearchOpen(true)}
            className="ml-auto inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
          >
            <Search size={15} />
          </button>
        </div>
        {menuOpen && (
          <div className="max-h-[60vh] overflow-y-auto border-t border-[var(--color-border)] bg-[var(--color-bg)] px-6 py-5">
            <DocsSidebar onNavigate={() => setMenuOpen(false)} />
          </div>
        )}
      </div>

      <div className="mx-auto max-w-[1320px] px-6 pt-16 md:px-10 lg:grid lg:grid-cols-[230px_minmax(0,1fr)] lg:gap-10 xl:grid-cols-[230px_minmax(0,1fr)_190px]">
        {/* sidebar */}
        <aside className="sticky top-16 hidden h-[calc(100vh-4rem)] self-start overflow-y-auto py-10 pr-2 lg:block">
          <button
            type="button"
            onClick={() => setSearchOpen(true)}
            className="mb-7 flex w-full cursor-pointer items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-[13px] text-[var(--color-text-subtle)] transition-colors hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-muted)]"
          >
            <Search size={14} />
            <span className="flex-1 text-left">Search docs</span>
            <kbd className="mono-label rounded-[var(--radius-xs)] border border-[var(--color-border)] px-1.5 py-0.5">
              ⌘K
            </kbd>
          </button>
          <DocsSidebar />
        </aside>

        {/* article */}
        <main className="min-w-0 py-12">
          <article className="max-w-[760px]">
            <DocMarkdown md={page.md} />
            <DocsPager slug={page.slug} />
          </article>
        </main>

        {/* TOC rail */}
        <aside className="sticky top-16 hidden h-[calc(100vh-4rem)] self-start overflow-y-auto py-12 xl:block">
          <DocsToc md={page.md} key={page.slug} />
        </aside>
      </div>

      <Footer />
      <DocsSearch open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  )
}
