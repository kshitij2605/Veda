import { useEffect, useRef } from 'react'

interface Pt {
  x: number
  y: number
  vx: number
  vy: number
  r: number
  color: string
}

/**
 * Animated vector scatter — a slowly drifting embedding point cloud with faint
 * similarity links, colored from the modality palette. Decorative hero
 * centerpiece; renders a single static frame under prefers-reduced-motion.
 */
export function HeroScatter({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cs = getComputedStyle(document.documentElement)
    const modal = ['text', 'image', 'audio', 'video', 'doc', 'mixed']
      .map((m) => cs.getPropertyValue(`--color-modality-${m}`).trim())
      .filter(Boolean)
    const colors = modal.length
      ? modal
      : ['#7c6af0', '#4ade80', '#fbbf24', '#f87171', '#38bdf8', '#a78bfa']

    let w = 0
    let h = 0
    const pts: Pt[] = []
    const N = 120
    const LINK = 116

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      w = rect.width
      h = rect.height
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const seed = () => {
      pts.length = 0
      for (let i = 0; i < N; i++) {
        pts.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.18,
          vy: (Math.random() - 0.5) * 0.18,
          r: 1.2 + Math.random() * 2.4,
          color: colors[Math.floor(Math.random() * colors.length)]!,
        })
      }
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      ctx.lineWidth = 1
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i]!
          const b = pts[j]!
          const dx = a.x - b.x
          const dy = a.y - b.y
          const d2 = dx * dx + dy * dy
          if (d2 < LINK * LINK) {
            const alpha = (1 - Math.sqrt(d2) / LINK) * 0.16
            ctx.strokeStyle = `oklch(0.62 0.2 282 / ${alpha.toFixed(3)})`
            ctx.beginPath()
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)
            ctx.stroke()
          }
        }
      }
      for (const p of pts) {
        ctx.globalAlpha = 0.9
        ctx.fillStyle = p.color
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    let raf = 0
    const step = () => {
      for (const p of pts) {
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0 || p.x > w) p.vx *= -1
        if (p.y < 0 || p.y > h) p.vy *= -1
      }
      draw()
      raf = requestAnimationFrame(step)
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resize()
    seed()
    if (reduced) draw()
    else raf = requestAnimationFrame(step)

    const ro = new ResizeObserver(() => {
      resize()
      seed()
      if (reduced) draw()
    })
    ro.observe(canvas)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return <canvas ref={canvasRef} className={className} aria-hidden />
}
