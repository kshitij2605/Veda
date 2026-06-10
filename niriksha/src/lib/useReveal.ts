import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * Reveal the direct children of the returned ref on scroll: fade + rise, staggered.
 * No-ops under `prefers-reduced-motion` (elements stay at their final state).
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
        scrollTrigger: { trigger: el, start: 'top 85%' },
      })
    }, el)
    return () => ctx.revert()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}
