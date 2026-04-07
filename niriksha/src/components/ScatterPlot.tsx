import { useRef, useEffect, useCallback, useState } from 'react'

export interface ScatterPoint {
  id: string
  x: number
  y: number
  color: string
  label?: string
  data: Record<string, unknown>
}

interface ScatterPlotProps {
  points: ScatterPoint[]
  onSelect: (point: ScatterPoint | null) => void
  onHover: (point: ScatterPoint | null) => void
  width: number
  height: number
}

interface Camera {
  x: number
  y: number
  zoom: number
}

const POINT_RADIUS = 3
const HOVER_RADIUS = 6
const HIT_RADIUS = 8

export function ScatterPlot({ points, onSelect, onHover, width, height }: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cameraRef = useRef<Camera>({ x: 0, y: 0, zoom: 1 })
  const draggingRef = useRef(false)
  const lastMouseRef = useRef({ x: 0, y: 0 })
  const hoveredRef = useRef<number | null>(null)
  const [, forceRender] = useState(0)

  // Compute data bounds once
  const bounds = useRef({ minX: 0, maxX: 1, minY: 0, maxY: 1 })
  useEffect(() => {
    if (points.length === 0) return
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const p of points) {
      if (p.x < minX) minX = p.x
      if (p.x > maxX) maxX = p.x
      if (p.y < minY) minY = p.y
      if (p.y > maxY) maxY = p.y
    }
    const padX = (maxX - minX) * 0.05 || 1
    const padY = (maxY - minY) * 0.05 || 1
    bounds.current = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }
    cameraRef.current = { x: 0, y: 0, zoom: 1 }
  }, [points])

  const toScreen = useCallback((px: number, py: number): [number, number] => {
    const { minX, maxX, minY, maxY } = bounds.current
    const cam = cameraRef.current
    const sx = ((px - minX) / (maxX - minX)) * width
    const sy = ((py - minY) / (maxY - minY)) * height
    return [
      (sx - width / 2) * cam.zoom + width / 2 + cam.x,
      (sy - height / 2) * cam.zoom + height / 2 + cam.y,
    ]
  }, [width, height])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)

    // Grid lines
    ctx.strokeStyle = 'rgba(42, 42, 54, 0.5)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const gx = (i / 4) * width
      const gy = (i / 4) * height
      ctx.beginPath()
      ctx.moveTo(gx, 0)
      ctx.lineTo(gx, height)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, gy)
      ctx.lineTo(width, gy)
      ctx.stroke()
    }

    // Draw points
    const hovered = hoveredRef.current
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      const [sx, sy] = toScreen(p.x, p.y)
      if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue

      const isHovered = hovered === i
      ctx.beginPath()
      ctx.arc(sx, sy, isHovered ? HOVER_RADIUS : POINT_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = isHovered ? '#fff' : p.color
      ctx.globalAlpha = isHovered ? 1 : 0.7
      ctx.fill()
      ctx.globalAlpha = 1

      if (isHovered) {
        ctx.strokeStyle = p.color
        ctx.lineWidth = 2
        ctx.stroke()
      }
    }
  }, [points, width, height, toScreen])

  useEffect(() => {
    draw()
  }, [draw])

  const findPoint = useCallback((mx: number, my: number): number | null => {
    let closest: number | null = null
    let closestDist = HIT_RADIUS * HIT_RADIUS
    for (let i = 0; i < points.length; i++) {
      const [sx, sy] = toScreen(points[i].x, points[i].y)
      const dx = mx - sx
      const dy = my - sy
      const dist = dx * dx + dy * dy
      if (dist < closestDist) {
        closest = i
        closestDist = dist
      }
    }
    return closest
  }, [points, toScreen])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    draggingRef.current = true
    lastMouseRef.current = { x: e.clientX, y: e.clientY }
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return

    if (draggingRef.current) {
      const dx = e.clientX - lastMouseRef.current.x
      const dy = e.clientY - lastMouseRef.current.y
      cameraRef.current.x += dx
      cameraRef.current.y += dy
      lastMouseRef.current = { x: e.clientX, y: e.clientY }
      draw()
      return
    }

    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const idx = findPoint(mx, my)

    if (idx !== hoveredRef.current) {
      hoveredRef.current = idx
      onHover(idx !== null ? points[idx] : null)
      draw()
    }
  }, [draw, findPoint, onHover, points])

  const handleMouseUp = useCallback(() => {
    draggingRef.current = false
  }, [])

  const handleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const idx = findPoint(mx, my)
    onSelect(idx !== null ? points[idx] : null)
  }, [findPoint, onSelect, points])

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 0.9 : 1.1
    const cam = cameraRef.current
    const newZoom = Math.max(0.1, Math.min(20, cam.zoom * factor))

    const rect = canvasRef.current?.getBoundingClientRect()
    if (rect) {
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      cam.x = mx - (mx - cam.x) * (newZoom / cam.zoom)
      cam.y = my - (my - cam.y) * (newZoom / cam.zoom)
    }

    cam.zoom = newZoom
    forceRender((v) => v + 1)
    draw()
  }, [draw])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="bg-bg-primary rounded-lg cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleClick}
      onWheel={handleWheel}
    />
  )
}
