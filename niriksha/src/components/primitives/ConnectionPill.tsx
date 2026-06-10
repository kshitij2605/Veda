/** Connection / health status pill: green dot when ok, red otherwise. */
export function ConnectionPill({ ok, label }: { ok: boolean; label: string }) {
  const c = ok ? 'var(--color-success)' : 'var(--color-error)'
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] text-[var(--color-text-muted)] border border-[var(--color-border)]">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
      {label}
    </span>
  )
}
