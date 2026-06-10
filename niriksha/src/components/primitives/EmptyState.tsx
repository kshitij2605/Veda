import { type ReactNode } from 'react'

/** Reusable empty state: dot-grid surface, tinted icon, title/description, action. */
export function EmptyState({
  title,
  description,
  action,
  icon,
}: {
  title: string
  description?: string
  action?: ReactNode
  icon?: ReactNode
}) {
  return (
    <div className="dot-grid rounded-[var(--radius-lg)] border border-[var(--color-border)] flex flex-col items-center justify-center text-center py-16 px-6">
      {icon && (
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--color-accent-subtle)] text-[var(--color-accent)]">
          {icon}
        </div>
      )}
      <div className="text-[20px] font-semibold text-[var(--color-text)]">{title}</div>
      {description && (
        <div className="mt-1.5 text-[15px] text-[var(--color-text-muted)] max-w-md leading-relaxed">
          {description}
        </div>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}
