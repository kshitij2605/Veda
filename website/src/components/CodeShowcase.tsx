import { useState } from 'react'
import { cn } from '../lib/cn'
import { CODE_TABS } from '../content/code'

export function CodeShowcase() {
  const [active, setActive] = useState(0)
  return (
    <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] shadow-[var(--shadow-md)]">
      <div className="flex items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-base-2)] px-3">
        <span className="mr-2 flex items-center gap-1.5" aria-hidden>
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-modality-video)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-modality-audio)]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-modality-image)]" />
        </span>
        {CODE_TABS.map((t, i) => (
          <button
            key={t.label}
            type="button"
            onClick={() => setActive(i)}
            className={cn(
              'px-3 py-3 text-[13px] font-medium transition-colors',
              i === active
                ? 'border-b-2 border-[var(--color-accent)] text-[var(--color-accent)]'
                : 'border-b-2 border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <pre className="overflow-x-auto bg-[var(--color-base-1)] px-5 py-5 text-[13px] leading-relaxed [font-family:var(--font-mono)] text-[var(--color-text)]">
        <code>{CODE_TABS[active]!.code}</code>
      </pre>
    </div>
  )
}
