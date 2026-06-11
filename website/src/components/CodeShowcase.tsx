import { useState } from 'react'
import { cn } from '../lib/cn'
import { CODE_TABS } from '../content/code'

export function CodeShowcase() {
  const [active, setActive] = useState(0)
  return (
    <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)]">
      {/* terminal titlebar */}
      <div className="flex items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-4">
        <span className="mr-3 flex items-center gap-1.5" aria-hidden>
          <span className="h-2.5 w-2.5 rounded-full bg-[rgba(255,255,255,0.14)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[rgba(255,255,255,0.14)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[rgba(255,255,255,0.14)]" />
        </span>
        {CODE_TABS.map((t, i) => (
          <button
            key={t.label}
            type="button"
            onClick={() => setActive(i)}
            className={cn(
              'mono-label cursor-pointer px-3 py-3.5 transition-colors',
              i === active
                ? 'border-b-2 border-[var(--color-accent)] text-[var(--color-accent)]'
                : 'border-b-2 border-transparent text-[var(--color-text-subtle)] hover:text-[var(--color-text)]',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <pre className="overflow-x-auto bg-[var(--color-code)] px-6 py-6 text-[13px] leading-relaxed [font-family:var(--font-mono)] text-[var(--color-text)]">
        <code>{CODE_TABS[active]!.code}</code>
      </pre>
    </div>
  )
}
