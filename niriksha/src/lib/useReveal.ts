import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'

/**
 * Staggered entrance for the direct children of the returned ref, on mount:
 * fade + rise. Runs once (deps default to []). No-ops under
 * `prefers-reduced-motion` (children stay at their final state).
 *
 * Deliberately not ScrollTrigger-based: the app scrolls inside an inner
 * `<main>` (overflow-auto), not the window, so a mount reveal is reliable
 * and never leaves content stuck invisible.
 */
export function useReveal<T extends HTMLElement = HTMLDivElement>(deps: unknown[] = []) {
  const ref = useRef<T>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const items = Array.from(el.children) as HTMLElement[]
    if (items.length === 0) return
    const ctx = gsap.context(() => {
      gsap.from(items, {
        opacity: 0,
        y: 14,
        duration: 0.5,
        ease: 'power2.out',
        stagger: 0.07,
      })
    }, el)
    return () => ctx.revert()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}
