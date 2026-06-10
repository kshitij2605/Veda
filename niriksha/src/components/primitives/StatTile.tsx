import { type ReactNode } from 'react'

/** A single KPI tile: overline label, large value, optional icon + footnote. */
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
    <div className="p-5">
      <div className="flex items-center gap-2 text-[11px] text-[var(--color-text-subtle)] uppercase tracking-[0.08em] font-semibold">
        {icon}
        <span>{label}</span>
      </div>
      <div className="mt-2.5 text-[30px] leading-none font-semibold text-[var(--color-text)] tabular-nums [font-family:var(--font-mono)]">
        {value}
      </div>
      {foot && <div className="mt-2 text-[12px] text-[var(--color-text-subtle)]">{foot}</div>}
    </div>
  )
}
