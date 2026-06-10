import { type ReactNode } from 'react'

/** Standard page header: large display title / description / actions / optional tabs row. */
export function PageHeader({
  title,
  description,
  actions,
  tabs,
}: {
  title: string
  description?: string
  actions?: ReactNode
  tabs?: ReactNode
}) {
  return (
    <div className="mb-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-[30px] [font-family:var(--font-display)] font-semibold tracking-[-0.015em] text-[var(--color-text)]">
            {title}
          </h1>
          {description && (
            <p className="mt-1.5 text-[15px] text-[var(--color-text-muted)]">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {tabs && <div className="mt-5">{tabs}</div>}
    </div>
  )
}
