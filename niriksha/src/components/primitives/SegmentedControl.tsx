import { cn } from '../../lib/cn'

export interface SegmentedOption {
  value: string
  label: string
}

/** A compact button group for mutually-exclusive choices (e.g. query type). */
export function SegmentedControl({
  options,
  value,
  onChange,
}: {
  options: SegmentedOption[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div
      role="group"
      className="inline-flex items-center gap-0.5 p-0.5 rounded-[var(--radius-sm)] bg-[var(--color-bg-overlay)] border border-[var(--color-border)]"
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(o.value)}
            className={cn(
              'px-2.5 h-7 text-[12px] rounded-[var(--radius-xs)] transition-colors',
              active
                ? 'bg-[var(--color-bg-active)] text-[var(--color-text)]'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
            )}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
