import { useCallback, useEffect, useRef } from 'react'
import { useMotionValue, useReducedMotion, useSpring } from 'framer-motion'

interface CountUpProps {
  to: number
  from?: number
  duration?: number
  separator?: string
  className?: string
}

function decimalsOf(n: number): number {
  const s = n.toString()
  if (!s.includes('.')) return 0
  const d = s.split('.')[1]
  return parseInt(d, 10) !== 0 ? d.length : 0
}

/**
 * Vendored from ReactBits (TextAnimations/CountUp), adapted:
 * - imports from `framer-motion` (the package this repo ships) not `motion/react`
 * - animates on mount (the KPI band is above the fold; no IntersectionObserver)
 * - jumps straight to the final value under `prefers-reduced-motion`
 */
export function CountUp({ to, from = 0, duration = 1.4, separator = ',', className = '' }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const reduced = useReducedMotion()
  const motionValue = useMotionValue(from)
  const damping = 20 + 40 * (1 / duration)
  const stiffness = 100 * (1 / duration)
  const spring = useSpring(motionValue, { damping, stiffness })
  const decimals = Math.max(decimalsOf(from), decimalsOf(to))

  const format = useCallback(
    (n: number) =>
      new Intl.NumberFormat('en-US', {
        useGrouping: !!separator,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
        .format(n)
        .replace(/,/g, separator),
    [decimals, separator],
  )

  useEffect(() => {
    if (!ref.current) return
    if (reduced) {
      ref.current.textContent = format(to)
      return
    }
    ref.current.textContent = format(from)
    motionValue.set(to)
  }, [to, from, reduced, format, motionValue])

  useEffect(() => {
    if (reduced) return
    const unsub = spring.on('change', (v: number) => {
      if (ref.current) ref.current.textContent = format(v)
    })
    return () => unsub()
  }, [spring, format, reduced])

  return <span className={className} ref={ref} />
}
