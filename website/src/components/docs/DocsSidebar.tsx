import { NavLink } from 'react-router-dom'
import { cn } from '../../lib/cn'
import { DOCS } from '../../docs/registry'

export function DocsSidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="space-y-7">
      {DOCS.map((g) => (
        <div key={g.label}>
          <p className="mono-label text-[var(--color-text-subtle)]">{g.label}</p>
          <ul className="mt-2.5 space-y-0.5">
            {g.items.map((p) => (
              <li key={p.slug}>
                <NavLink
                  to={`/docs/${p.slug}`}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      'relative block rounded-[var(--radius-sm)] px-3 py-1.5 text-[14px] transition-colors',
                      isActive
                        ? 'bg-[var(--color-surface)] font-medium text-[var(--color-accent)]'
                        : 'text-[var(--color-text-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]',
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <span
                          aria-hidden
                          className="absolute bottom-1.5 left-0 top-1.5 w-[2px] rounded-r bg-[var(--color-accent)]"
                        />
                      )}
                      {p.title}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}
