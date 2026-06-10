import { useEffect, useState } from 'react'
import { cn } from '../lib/cn'
import { Button } from './primitives/Button'
import { ThemeToggle } from './primitives/ThemeToggle'
import { BrandMark } from './BrandMark'
import { GitHubIcon } from './GitHubIcon'

const LINKS = [
  { label: 'Features', href: '#features' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Docs', href: '/docs' },
]

export function Nav() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={cn(
        'fixed inset-x-0 top-0 z-50 transition-[background-color,border-color,backdrop-filter] duration-[var(--duration-base)]',
        scrolled ? 'glass border-b border-[var(--color-border)]' : 'border-b border-transparent',
      )}
    >
      <div className="mx-auto flex h-16 max-w-[1200px] items-center gap-3 px-6 md:px-8">
        <a href="#top" className="flex items-center gap-2.5">
          <BrandMark size={22} />
          <span className="text-[19px] font-semibold tracking-tight [font-family:var(--font-display)] text-[var(--color-text)]">
            Veda
          </span>
        </a>
        <nav className="ml-7 hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.label}
              href={l.href}
              className="rounded-[var(--radius-sm)] px-3 py-1.5 text-[14px] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text)]"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1.5">
          <a
            href="https://github.com/kshitij2605/Veda"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
            className="inline-flex h-9 w-9 items-center justify-center rounded-[var(--radius-md)] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text)]"
          >
            <GitHubIcon size={18} />
          </a>
          <ThemeToggle />
          <Button
            variant="primary"
            size="md"
            onClick={() => {
              document.getElementById('get-started')?.scrollIntoView({ behavior: 'smooth' })
            }}
          >
            Get started
          </Button>
        </div>
      </div>
    </header>
  )
}
