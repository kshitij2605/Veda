import { CountUp } from '../components/reactbits/CountUp'
import { useReveal } from '../lib/useReveal'
import { STATS } from '../content/stats'

export function Stats() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <div className="border-b border-[var(--color-border)] bg-[var(--color-bg-raised)]">
      <div
        ref={ref}
        className="mx-auto grid max-w-[1320px] grid-cols-2 divide-x divide-[var(--color-border)] px-6 md:grid-cols-4 md:px-10"
      >
        {STATS.map((s) => (
          <div key={s.label} className="px-6 py-12 md:px-10">
            <div className="text-[52px] [font-family:var(--font-display)] font-bold leading-none tabular-nums text-[var(--color-text)]">
              <CountUp to={s.value} />
            </div>
            <div className="mono-label mt-3 text-[var(--color-text-subtle)]">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
