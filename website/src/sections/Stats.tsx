import { CountUp } from '../components/reactbits/CountUp'
import { useReveal } from '../lib/useReveal'
import { STATS } from '../content/stats'

export function Stats() {
  const ref = useReveal<HTMLDivElement>()
  return (
    <div className="border-y border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
      <div
        ref={ref}
        className="mx-auto grid max-w-[1200px] grid-cols-2 gap-8 px-6 md:px-8 py-12 md:grid-cols-4"
      >
        {STATS.map((s) => (
          <div key={s.label} className="text-center md:text-left">
            <div className="text-[42px] [font-family:var(--font-display)] font-semibold tabular-nums leading-none text-[var(--color-text)]">
              <CountUp to={s.value} />
            </div>
            <div className="mt-2 text-[14px] text-[var(--color-text-muted)]">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
