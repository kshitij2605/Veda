import { X } from 'lucide-react'
import { type MouseEventHandler, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface FilterChipProps {
  /** Visible label, e.g. `parent_doc_id = "doc-1234"`. */
  children: ReactNode
  /** If set, chip is in active state with × to remove. */
  onRemove?: () => void
  /** Click handler for editing the chip (opens a popover, etc.). */
  onClick?: MouseEventHandler<HTMLButtonElement>
  /** "Suggested" (`+ name`) state vs active. */
  suggested?: boolean
  className?: string
}

/**
 * FilterChip — Stripe pattern. Two states:
 *   - suggested: `+ name` (click to add a value)
 *   - active:    `name = value ×`
 *
 * Always editable on click; remove with the ×.
 */
export function FilterChip({
  children,
  onRemove,
  onClick,
  suggested,
  className,
}: FilterChipProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center h-6 rounded-[var(--radius-sm)]',
        'border text-[11.5px] font-mono whitespace-nowrap select-none',
        'transition-[background,border-color] duration-[var(--duration-fast)]',
        suggested
          ? 'pl-1.5 pr-1.5 gap-1 border-dashed border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-border-strong)]'
          : 'pl-2 pr-1 gap-1 bg-[var(--color-base-2)] border-[var(--color-border-strong)] text-[var(--color-text)]',
        className,
      )}
    >
      <button
        type="button"
        onClick={onClick}
        className="inline-flex items-center gap-1 hover:opacity-90"
      >
        {suggested && <span className="text-[var(--color-text-subtle)]">+</span>}
        {children}
      </button>
      {!suggested && onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          aria-label="Remove filter"
          className="p-0.5 rounded-[2px] text-[var(--color-text-subtle)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)]"
        >
          <X size={11} strokeWidth={2.25} />
        </button>
      )}
    </span>
  )
}
