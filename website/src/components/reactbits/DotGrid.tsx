import { useCallback, useEffect, useMemo, useRef } from 'react'
import { gsap } from 'gsap'

/**
 * Vendored from ReactBits (Backgrounds/DotGrid), adapted:
 * - InertiaPlugin dependency removed — plain tween push + elastic return
 * - re-tokenized defaults for the obsidian/saffron theme
 * - plain absolute-fill wrapper (no section/padding), named export
 * - prefers-reduced-motion: single static draw, no listeners
 */

interface Dot {
  cx: number
  cy: number
  xOffset: number
  yOffset: number
  _pushed: boolean
}

export interface DotGridProps {
  dotSize?: number
  gap?: number
  baseColor?: string
  activeColor?: string
  proximity?: number
  speedTrigger?: number
  shockRadius?: number
  shockStrength?: number
  className?: string
}

function hexToRgb(hex: string) {
  const m = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)
  if (!m) return { r: 0, g: 0, b: 0 }
  return { r: parseInt(m[1]!, 16), g: parseInt(m[2]!, 16), b: parseInt(m[3]!, 16) }
}

export function DotGrid({
  dotSize = 2.5,
  gap = 26,
  baseColor = '#1A2030',
  activeColor = '#FFB224',
  proximity = 130,
  speedTrigger = 120,
  shockRadius = 220,
  shockStrength = 4,
  className = '',
}: DotGridProps) {
  const wrapperRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dotsRef = useRef<Dot[]>([])
  const pointerRef = useRef({ x: -9999, y: -9999, lastX: 0, lastY: 0, lastTime: 0, speed: 0 })

  const baseRgb = useMemo(() => hexToRgb(baseColor), [baseColor])
  const activeRgb = useMemo(() => hexToRgb(activeColor), [activeColor])

  const buildGrid = useCallback(() => {
    const wrap = wrapperRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return
    const { width, height } = wrap.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    canvas.getContext('2d')?.setTransform(dpr, 0, 0, dpr, 0, 0)

    const cell = dotSize + gap
    const cols = Math.floor((width + gap) / cell)
    const rows = Math.floor((height + gap) / cell)
    const startX = (width - (cell * cols - gap)) / 2 + dotSize / 2
    const startY = (height - (cell * rows - gap)) / 2 + dotSize / 2

    const dots: Dot[] = []
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        dots.push({ cx: startX + x * cell, cy: startY + y * cell, xOffset: 0, yOffset: 0, _pushed: false })
      }
    }
    dotsRef.current = dots
  }, [dotSize, gap])

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const proxSq = proximity * proximity
    let raf = 0

    const draw = () => {
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (!canvas || !ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const { x: px, y: py } = pointerRef.current
      for (const dot of dotsRef.current) {
        const dx = dot.cx - px
        const dy = dot.cy - py
        const dsq = dx * dx + dy * dy
        let fill = baseColor
        if (dsq <= proxSq) {
          const t = 1 - Math.sqrt(dsq) / proximity
          const r = Math.round(baseRgb.r + (activeRgb.r - baseRgb.r) * t)
          const g = Math.round(baseRgb.g + (activeRgb.g - baseRgb.g) * t)
          const b = Math.round(baseRgb.b + (activeRgb.b - baseRgb.b) * t)
          fill = `rgb(${r},${g},${b})`
        }
        ctx.fillStyle = fill
        ctx.beginPath()
        ctx.arc(dot.cx + dot.xOffset, dot.cy + dot.yOffset, dotSize / 2, 0, Math.PI * 2)
        ctx.fill()
      }
      if (!reduced) raf = requestAnimationFrame(draw)
    }

    buildGrid()
    draw()

    const ro = new ResizeObserver(() => {
      buildGrid()
      if (reduced) draw()
    })
    if (wrapperRef.current) ro.observe(wrapperRef.current)

    if (reduced) {
      return () => ro.disconnect()
    }

    const push = (dot: Dot, pushX: number, pushY: number) => {
      dot._pushed = true
      gsap.killTweensOf(dot)
      gsap.to(dot, {
        xOffset: pushX,
        yOffset: pushY,
        duration: 0.35,
        ease: 'power2.out',
        onComplete: () => {
          gsap.to(dot, {
            xOffset: 0,
            yOffset: 0,
            duration: 1.2,
            ease: 'elastic.out(1,0.6)',
          })
          dot._pushed = false
        },
      })
    }

    const onMove = (e: MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const now = performance.now()
      const pr = pointerRef.current
      const dt = pr.lastTime ? now - pr.lastTime : 16
      const vx = ((e.clientX - pr.lastX) / dt) * 1000
      const vy = ((e.clientY - pr.lastY) / dt) * 1000
      pr.speed = Math.hypot(vx, vy)
      pr.lastTime = now
      pr.lastX = e.clientX
      pr.lastY = e.clientY
      const rect = canvas.getBoundingClientRect()
      pr.x = e.clientX - rect.left
      pr.y = e.clientY - rect.top
      if (pr.speed > speedTrigger) {
        for (const dot of dotsRef.current) {
          const d = Math.hypot(dot.cx - pr.x, dot.cy - pr.y)
          if (d < proximity && !dot._pushed) {
            push(dot, (dot.cx - pr.x) * 0.4 + vx * 0.004, (dot.cy - pr.y) * 0.4 + vy * 0.004)
          }
        }
      }
    }

    const onClick = (e: MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const cx = e.clientX - rect.left
      const cy = e.clientY - rect.top
      for (const dot of dotsRef.current) {
        const d = Math.hypot(dot.cx - cx, dot.cy - cy)
        if (d < shockRadius && !dot._pushed) {
          const falloff = Math.max(0, 1 - d / shockRadius)
          push(dot, (dot.cx - cx) * shockStrength * falloff * 0.25, (dot.cy - cy) * shockStrength * falloff * 0.25)
        }
      }
    }

    window.addEventListener('mousemove', onMove, { passive: true })
    window.addEventListener('click', onClick)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('click', onClick)
    }
  }, [buildGrid, baseColor, baseRgb, activeRgb, proximity, speedTrigger, shockRadius, shockStrength, dotSize])

  return (
    <div ref={wrapperRef} className={`pointer-events-none absolute inset-0 ${className}`} aria-hidden>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
    </div>
  )
}
