import { type ReactNode } from 'react'

/** Standard page header: title / description / actions / optional tabs row. */
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
    <div className="mb-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-[var(--color-text)]">{title}</h1>
          {description && (
            <p className="mt-1 text-[13px] text-[var(--color-text-muted)]">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {tabs && <div className="mt-4">{tabs}</div>}
    </div>
  )
}
