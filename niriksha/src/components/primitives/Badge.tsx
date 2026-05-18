import { forwardRef, type HTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/cn'

const badge = cva(
  [
    'inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)]',
    'text-[10.5px] font-medium tracking-wide tabular-nums',
    'border whitespace-nowrap select-none',
  ],
  {
    variants: {
      variant: {
        default: 'bg-[var(--color-base-2)] text-[var(--color-text-muted)] border-[var(--color-border)]',
        accent: 'bg-[color-mix(in_oklch,var(--color-accent)_15%,transparent)] text-[var(--color-accent-9)] border-[color-mix(in_oklch,var(--color-accent)_25%,transparent)]',
        success: 'bg-[color-mix(in_oklch,var(--color-success)_15%,transparent)] text-[var(--color-success)] border-[color-mix(in_oklch,var(--color-success)_30%,transparent)]',
        warning: 'bg-[color-mix(in_oklch,var(--color-warning)_15%,transparent)] text-[var(--color-warning)] border-[color-mix(in_oklch,var(--color-warning)_30%,transparent)]',
        error:   'bg-[color-mix(in_oklch,var(--color-error)_15%,transparent)] text-[var(--color-error)] border-[color-mix(in_oklch,var(--color-error)_30%,transparent)]',
        ghost:   'bg-transparent text-[var(--color-text-muted)] border-transparent',
        outline: 'bg-transparent text-[var(--color-text)] border-[var(--color-border-strong)]',
      },
      mono: { true: 'font-mono lowercase' },
    },
    defaultVariants: { variant: 'default' },
  },
)

type BadgeVariants = VariantProps<typeof badge>

export interface BadgeProps
  extends Omit<HTMLAttributes<HTMLSpanElement>, 'children'>,
    BadgeVariants {
  label: string
  dot?: boolean
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ label, dot, className, variant, mono, ...rest }, ref) => (
    <span
      ref={ref}
      className={cn(badge({ variant, mono }), className)}
      {...rest}
    >
      {dot && (
        <span
          className="w-1 h-1 rounded-full bg-current"
          aria-hidden
        />
      )}
      {label}
    </span>
  ),
)
Badge.displayName = 'Badge'
