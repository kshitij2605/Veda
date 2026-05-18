import { type HTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

/**
 * Skeleton — pulsing placeholder. Match the geometry of the real element
 * so the layout never jumps when content arrives.
 */
export function Skeleton({
  className,
  ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'animate-pulse rounded-[var(--radius-xs)]',
        'bg-[color-mix(in_oklch,var(--color-base-3)_70%,transparent)]',
        className,
      )}
      style={{ animationDuration: '1.6s' }}
      {...rest}
    />
  )
}

/** Stack of skeleton rows that match a table — used for Records loading. */
export function SkeletonRows({
  rows = 8,
  className,
}: {
  rows?: number
  className?: string
}) {
  return (
    <div className={cn('flex flex-col', className)}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-3 h-7 px-3 border-b border-[var(--color-border)]"
        >
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-3 flex-1 max-w-md" />
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-12" />
        </div>
      ))}
    </div>
  )
}
