import { type MouseEventHandler, type PropsWithChildren, useRef, useState } from 'react'
import { cn } from '../../lib/cn'

interface SpotlightCardProps extends PropsWithChildren {
  className?: string
  /** Any CSS color; defaults to a soft accent. */
  spotlightColor?: string
}

/**
 * Vendored from ReactBits (Components/SpotlightCard), re-tokenized: the
 * surface uses our elevated/border tokens and the cursor spotlight uses the
 * accent instead of the hardcoded neutral-900 / white. Padding is left to
 * the child so it can wrap a StatTile cleanly.
 */
export function SpotlightCard({
  children,
  className,
  spotlightColor = 'color-mix(in oklch, var(--color-accent) 22%, transparent)',
}: SpotlightCardProps) {
  const divRef = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [opacity, setOpacity] = useState(0)

  const handleMouseMove: MouseEventHandler<HTMLDivElement> = (e) => {
    if (!divRef.current) return
    const rect = divRef.current.getBoundingClientRect()
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setOpacity(0.55)}
      onMouseLeave={() => setOpacity(0)}
      className={cn(
        'relative overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]',
        className,
      )}
    >
      <div
        className="pointer-events-none absolute inset-0 transition-opacity duration-500"
        style={{
          opacity,
          background: `radial-gradient(circle at ${position.x}px ${position.y}px, ${spotlightColor}, transparent 80%)`,
        }}
      />
      <div className="relative">{children}</div>
    </div>
  )
}
