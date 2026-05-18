import { useMemo } from 'react'
import { cn } from '../../lib/cn'

/**
 * Sparkline — compact inline visualisation of a numeric series.
 * Designed to render in a single table cell as a preview of a vector
 * (instead of dumping the raw float array). Default 80×16, no axes.
 *
 * For vector cells: pass the first ~64 dims; we'll normalise. The point
 * is *recognition* (which vector is which) not measurement.
 */
export interface SparklineProps {
  values: ArrayLike<number>
  width?: number
  height?: number
  stroke?: string
  fill?: string
  /** Render as blocks (▁▂▄▆▇) rather than path. Cheaper, more terminal-y. */
  glyph?: boolean
  className?: string
}

const BLOCKS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇']

export function Sparkline({
  values,
  width = 80,
  height = 16,
  stroke = 'var(--color-data-5)',
  fill = 'transparent',
  glyph = false,
  className,
}: SparklineProps) {
  const { path, area, bars } = useMemo(() => {
    const arr = Array.from(values as number[])
    if (arr.length === 0) return { path: '', area: '', bars: '' }
    let min = Infinity
    let max = -Infinity
    for (const v of arr) {
      if (v < min) min = v
      if (v > max) max = v
    }
    if (min === max) {
      // flat → midline
      min = -1
      max = 1
    }
    const range = max - min
    const stepX = arr.length > 1 ? width / (arr.length - 1) : width
    const pts = arr.map((v, i) => {
      const x = i * stepX
      const y = height - ((v - min) / range) * height
      return [x, y] as const
    })
    const path =
      'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ')
    const area =
      `M ${pts[0]![0].toFixed(1)} ${height} ` +
      pts.map(([x, y]) => `L ${x.toFixed(1)} ${y.toFixed(1)}`).join(' ') +
      ` L ${pts[pts.length - 1]![0].toFixed(1)} ${height} Z`
    let bars = ''
    if (glyph) {
      const compressed: string[] = []
      // Compress to 16 cells max for glyph rendering
      const target = Math.min(16, arr.length)
      const stride = arr.length / target
      for (let i = 0; i < target; i++) {
        const idx = Math.floor(i * stride)
        const norm = (arr[idx]! - min) / range
        compressed.push(BLOCKS[Math.min(BLOCKS.length - 1, Math.floor(norm * BLOCKS.length))]!)
      }
      bars = compressed.join('')
    }
    return { path, area, bars }
  }, [values, width, height, glyph])

  if (glyph) {
    return (
      <span
        className={cn(
          'font-mono leading-none tracking-[0.05em] text-[var(--color-data-5)]',
          className,
        )}
        style={{ fontSize: height }}
      >
        {bars}
      </span>
    )
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={cn('inline-block align-middle overflow-visible', className)}
      role="img"
      aria-hidden
    >
      {fill !== 'transparent' && <path d={area} fill={fill} opacity={0.35} />}
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.25} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
