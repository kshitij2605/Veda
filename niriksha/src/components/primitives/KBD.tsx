import { type HTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

/**
 * KBD — a compact keyboard-hint pill. Pass keys as children, separated
 * by `+`. Renders ⌘ on macOS for `Meta`.
 */
export function KBD({
  children,
  className,
  ...rest
}: HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'inline-flex items-center gap-0.5 px-1 h-[18px] rounded-[var(--radius-xs)]',
        'bg-[var(--color-base-2)] border border-[var(--color-border)]',
        'text-[10px] font-mono text-[var(--color-text-muted)]',
        'shadow-[inset_0_-1px_0_var(--color-border)]',
        className,
      )}
      {...rest}
    >
      {children}
    </kbd>
  )
}

const isMac =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

export function modKey() {
  return isMac ? '⌘' : 'Ctrl'
}
