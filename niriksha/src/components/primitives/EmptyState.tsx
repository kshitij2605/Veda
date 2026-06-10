import { type ReactNode } from 'react'

/** Reusable empty state: dot-grid surface, icon, title/description, action. */
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
    <div className="dot-grid rounded-[var(--radius-lg)] border border-[var(--color-border)] flex flex-col items-center justify-center text-center py-12 px-6">
      {icon && <div className="mb-3 text-[var(--color-text-subtle)]">{icon}</div>}
      <div className="text-[14px] font-medium text-[var(--color-text)]">{title}</div>
      {description && (
        <div className="mt-1 text-[12px] text-[var(--color-text-muted)] max-w-sm">{description}</div>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
