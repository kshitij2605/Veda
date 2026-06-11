import { useEffect, useRef } from 'react'

interface Pt {
  x: number
  y: number
  vx: number
  vy: number
  ix: number
  iy: number
  r: number
  color: string
}

/**
 * Interactive vector graph — a drifting embedding point cloud where the
 * cursor acts as a QUERY VECTOR: the k nearest points light up saffron and
 * link to it (nearest-neighbor search, visualized), and pointer movement
 * stirs nearby points. Renders a static frame under prefers-reduced-motion.
 */
export function HeroScatter({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cs = getComputedStyle(document.documentElement)
    const palette = [1, 2, 3, 4, 5, 6]
      .map((i) => cs.getPropertyValue(`--color-node-${i}`).trim())
      .filter(Boolean)
    const colors = palette.length
      ? palette
      : ['#FFB224', '#5AD8FF', '#B886FF', '#3DD68C', '#FF7A59', '#F2F4F8']
    const SAFFRON = colors[0]!

    let w = 0
    let h = 0
    const pts: Pt[] = []
    const N = 140
    const LINK = 110
    const QUERY_R = 260
    const K = 7

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
          ix: 0,
          iy: 0,
          r: 1.2 + Math.random() * 2.2,
          color: colors[Math.floor(Math.random() * colors.length)]!,
        })
      }
    }

    const pointer = { x: -9999, y: -9999, px: -9999, py: -9999, active: false }

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      pointer.px = pointer.x
      pointer.py = pointer.y
      pointer.x = e.clientX - r.left
      pointer.y = e.clientY - r.top
      pointer.active =
        pointer.x >= 0 && pointer.x <= r.width && pointer.y >= 0 && pointer.y <= r.height
      if (!pointer.active || pointer.px < -999) return
      const dxv = pointer.x - pointer.px
      const dyv = pointer.y - pointer.py
      for (const p of pts) {
        const dx = p.x - pointer.x
        const dy = p.y - pointer.y
        const d2 = dx * dx + dy * dy
        if (d2 < 120 * 120) {
          const f = (1 - Math.sqrt(d2) / 120) * 0.05
          p.ix = Math.max(-1.6, Math.min(1.6, p.ix + dxv * f))
          p.iy = Math.max(-1.6, Math.min(1.6, p.iy + dyv * f))
        }
      }
    }
    const onLeave = () => {
      pointer.active = false
      pointer.x = -9999
      pointer.y = -9999
      pointer.px = -9999
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h)

      // idle similarity links (neutral, faint)
      ctx.lineWidth = 1
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i]!
          const b = pts[j]!
          const dx = a.x - b.x
          const dy = a.y - b.y
          const d2 = dx * dx + dy * dy
          if (d2 < LINK * LINK) {
            const alpha = (1 - Math.sqrt(d2) / LINK) * 0.1
            ctx.strokeStyle = `rgba(242,244,248,${alpha.toFixed(3)})`
            ctx.beginPath()
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)
            ctx.stroke()
          }
        }
      }

      // nodes
      for (const p of pts) {
        ctx.globalAlpha = 0.88
        ctx.fillStyle = p.color
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1

      // query-vector pass: cursor links to its k nearest neighbors
      if (pointer.active) {
        const near = pts
          .map((p) => ({ p, d: Math.hypot(p.x - pointer.x, p.y - pointer.y) }))
          .filter((o) => o.d < QUERY_R)
          .sort((a, b) => a.d - b.d)
          .slice(0, K)
        ctx.lineWidth = 1.2
        for (const { p, d } of near) {
          const alpha = (1 - d / QUERY_R) * 0.6
          ctx.strokeStyle = `rgba(255,178,36,${alpha.toFixed(3)})`
          ctx.beginPath()
          ctx.moveTo(pointer.x, pointer.y)
          ctx.lineTo(p.x, p.y)
          ctx.stroke()
          ctx.fillStyle = SAFFRON
          ctx.beginPath()
          ctx.arc(p.x, p.y, p.r * 1.6, 0, Math.PI * 2)
          ctx.fill()
        }
        // the query point: saffron ring + crosshair ticks
        ctx.strokeStyle = 'rgba(255,178,36,0.9)'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.arc(pointer.x, pointer.y, 9, 0, Math.PI * 2)
        ctx.stroke()
        ctx.beginPath()
        for (const [tx, ty] of [
          [0, -1],
          [0, 1],
          [-1, 0],
          [1, 0],
        ] as const) {
          ctx.moveTo(pointer.x + tx * 12, pointer.y + ty * 12)
          ctx.lineTo(pointer.x + tx * 17, pointer.y + ty * 17)
        }
        ctx.stroke()
      }
    }

    let raf = 0
    const step = () => {
      for (const p of pts) {
        p.x += p.vx + p.ix
        p.y += p.vy + p.iy
        p.ix *= 0.94
        p.iy *= 0.94
        if (p.x < 0 || p.x > w) p.vx *= -1
        if (p.y < 0 || p.y > h) p.vy *= -1
        p.x = Math.max(0, Math.min(w, p.x))
        p.y = Math.max(0, Math.min(h, p.y))
      }
      draw()
      raf = requestAnimationFrame(step)
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resize()
    seed()
    if (reduced) {
      draw()
    } else {
      raf = requestAnimationFrame(step)
      window.addEventListener('pointermove', onMove, { passive: true })
      window.addEventListener('pointerleave', onLeave)
    }

    const ro = new ResizeObserver(() => {
      resize()
      seed()
      if (reduced) draw()
    })
    ro.observe(canvas)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerleave', onLeave)
    }
  }, [])

  return <canvas ref={canvasRef} className={className} aria-hidden />
}
