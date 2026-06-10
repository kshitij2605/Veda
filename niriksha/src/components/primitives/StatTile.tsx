import { type ReactNode } from 'react'

/** A single KPI tile: label, large value, optional icon + footnote. */
export function StatTile({
  label,
  value,
  icon,
  foot,
}: {
  label: string
  value: ReactNode
  icon?: ReactNode
  foot?: ReactNode
}) {
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 text-[11px] text-[var(--color-text-muted)] uppercase tracking-[0.06em]">
        {icon}
        <span>{label}</span>
      </div>
      <div className="mt-2 text-[32px] leading-none font-semibold text-[var(--color-text)] tabular-nums">
        {value}
      </div>
      {foot && <div className="mt-1.5 text-[11px] text-[var(--color-text-subtle)]">{foot}</div>}
    </div>
  )
}
