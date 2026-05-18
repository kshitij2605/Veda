import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { cn } from '../../lib/cn'

export interface AtlasPoint {
  id: string
  x: number
  y: number
  color: string
  label?: string
}

interface Camera {
  x: number
  y: number
  zoom: number
}

interface AtlasCanvasProps {
  points: AtlasPoint[]
  width: number
  height: number
  onHover?: (p: AtlasPoint | null) => void
  onSelect?: (p: AtlasPoint | null) => void
  /** Lasso mode: when true the cursor draws a polygon; otherwise pans. */
  lasso?: boolean
  onLasso?: (selectedIds: Set<string>) => void
  /** External selection overlay (e.g. from BubbleUp lasso). */
  selectedIds?: Set<string>
}

const POINT_RADIUS = 2.4
const HOVER_RADIUS = 5
const HIT_RADIUS = 8

/**
 * Canvas2D scatter — pan / zoom / hover / click / lasso. Sized in CSS
 * pixels by props (parent uses ResizeObserver). Renders at devicePixelRatio.
 *
 * Lasso: when `lasso` is true, click+drag draws a polygon. On release,
 * `onLasso(ids)` fires with the point ids whose screen coords fell
 * inside the polygon.
 */
export function AtlasCanvas({
  points,
  width,
  height,
  onHover,
  onSelect,
  lasso,
  onLasso,
  selectedIds,
}: AtlasCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [camera, setCamera] = useState<Camera>({ x: 0, y: 0, zoom: 1 })
  const cameraRef = useRef(camera)
  cameraRef.current = camera

  const [hoverId, setHoverId] = useState<string | null>(null)
  const [lassoPts, setLassoPts] = useState<[number, number][]>([])
  const lassoActive = useRef(false)
  const drag = useRef<{ ox: number; oy: number; cx: number; cy: number } | null>(null)

  // Compute bounds + auto-fit on first render or when points change.
  const bounds = useMemo(() => {
    if (points.length === 0)
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 }
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity
    for (const p of points) {
      if (p.x < minX) minX = p.x
      if (p.x > maxX) maxX = p.x
      if (p.y < minY) minY = p.y
      if (p.y > maxY) maxY = p.y
    }
    return { minX, maxX, minY, maxY }
  }, [points])

  // World → screen helpers, captured per-render based on the live camera.
  const toScreen = useCallback(
    (px: number, py: number): [number, number] => {
      const cam = cameraRef.current
      const w = bounds.maxX - bounds.minX || 1
      const h = bounds.maxY - bounds.minY || 1
      const scale = (Math.min(width, height) * 0.45 * cam.zoom)
      const cx = width / 2 + cam.x
      const cy = height / 2 + cam.y
      const nx = (px - (bounds.minX + bounds.maxX) / 2) / w
      const ny = (py - (bounds.minY + bounds.maxY) / 2) / h
      return [cx + nx * scale * 2, cy - ny * scale * 2]
    },
    [bounds, width, height],
  )

  // Repaint when anything visual changes.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    canvas.style.width = width + 'px'
    canvas.style.height = height + 'px'
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // Subtle background grid
    ctx.clearRect(0, 0, width, height)
    drawGrid(ctx, width, height)

    // Faded backdrop for non-selected points if a selection is active.
    const hasSel = selectedIds && selectedIds.size > 0

    for (const p of points) {
      const [sx, sy] = toScreen(p.x, p.y)
      if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue
      const isSel = selectedIds?.has(p.id)
      const isHover = hoverId === p.id
      ctx.globalAlpha = hasSel ? (isSel ? 0.95 : 0.15) : 0.85
      ctx.fillStyle = isHover ? '#ffffff' : p.color
      ctx.beginPath()
      ctx.arc(sx, sy, isHover ? HOVER_RADIUS : POINT_RADIUS, 0, Math.PI * 2)
      ctx.fill()
      if (isSel) {
        ctx.globalAlpha = 1
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1

    // Lasso polygon
    if (lassoPts.length >= 2) {
      ctx.strokeStyle = 'oklch(0.62 0.22 282)'
      ctx.fillStyle = 'color-mix(in oklch, oklch(0.62 0.22 282) 12%, transparent)'
      ctx.lineWidth = 1.25
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      const [sx0, sy0] = lassoPts[0]!
      ctx.moveTo(sx0, sy0)
      for (let i = 1; i < lassoPts.length; i++) {
        ctx.lineTo(lassoPts[i]![0], lassoPts[i]![1])
      }
      if (!lassoActive.current) ctx.closePath()
      ctx.stroke()
      ctx.fill()
      ctx.setLineDash([])
    }
  }, [points, width, height, toScreen, hoverId, lassoPts, selectedIds])

  /* Mouse handlers — switched by `lasso` mode */

  const findHit = (sx: number, sy: number): AtlasPoint | null => {
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i]!
      const [px, py] = toScreen(p.x, p.y)
      if (Math.hypot(px - sx, py - sy) < HIT_RADIUS) return p
    }
    return null
  }

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top

    if (drag.current) {
      setCamera({
        x: drag.current.cx + (sx - drag.current.ox),
        y: drag.current.cy + (sy - drag.current.oy),
        zoom: cameraRef.current.zoom,
      })
      return
    }
    if (lassoActive.current) {
      setLassoPts((pts) => [...pts, [sx, sy]])
      return
    }

    const hit = findHit(sx, sy)
    setHoverId(hit?.id ?? null)
    onHover?.(hit)
  }

  const onDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    if (lasso) {
      lassoActive.current = true
      setLassoPts([[sx, sy]])
    } else {
      drag.current = { ox: sx, oy: sy, cx: cameraRef.current.x, cy: cameraRef.current.y }
    }
  }

  const onUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (drag.current) {
      const rect = e.currentTarget.getBoundingClientRect()
      const sx = e.clientX - rect.left
      const sy = e.clientY - rect.top
      const moved =
        Math.hypot(sx - drag.current.ox, sy - drag.current.oy) > 4
      drag.current = null
      if (!moved) {
        const hit = findHit(sx, sy)
        onSelect?.(hit)
      }
    }
    if (lassoActive.current) {
      lassoActive.current = false
      if (lassoPts.length >= 3 && onLasso) {
        const inside = new Set<string>()
        for (const p of points) {
          const [px, py] = toScreen(p.x, p.y)
          if (pointInPolygon(px, py, lassoPts)) inside.add(p.id)
        }
        onLasso(inside)
      }
      // keep the polygon visible until next interaction
    }
  }

  const onLeave = () => {
    setHoverId(null)
    onHover?.(null)
    drag.current = null
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const delta = -e.deltaY * 0.001
    setCamera((c) => ({
      ...c,
      zoom: Math.min(20, Math.max(0.2, c.zoom * (1 + delta))),
    }))
  }

  const clearLasso = () => setLassoPts([])

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        onMouseMove={onMove}
        onMouseDown={onDown}
        onMouseUp={onUp}
        onMouseLeave={onLeave}
        onWheel={onWheel}
        className={cn(
          'block w-full h-full',
          lasso ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing',
        )}
      />
      {lassoPts.length > 0 && !lassoActive.current && (
        <button
          type="button"
          onClick={clearLasso}
          className="absolute top-2 right-2 px-2 h-6 rounded-[var(--radius-xs)] bg-[var(--color-bg-overlay)] border border-[var(--color-border-strong)] text-[10.5px] text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        >
          Clear lasso
        </button>
      )}
    </div>
  )
}

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.strokeStyle = 'oklch(0.21 0.005 252)'
  ctx.lineWidth = 0.5
  const step = 40
  ctx.beginPath()
  for (let x = 0; x < w; x += step) {
    ctx.moveTo(x, 0)
    ctx.lineTo(x, h)
  }
  for (let y = 0; y < h; y += step) {
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
  }
  ctx.stroke()
}

function pointInPolygon(
  x: number,
  y: number,
  poly: [number, number][],
): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i]![0]
    const yi = poly[i]![1]
    const xj = poly[j]![0]
    const yj = poly[j]![1]
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-9) + xi
    if (intersect) inside = !inside
  }
  return inside
}
