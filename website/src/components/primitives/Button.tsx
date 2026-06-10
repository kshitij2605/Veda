import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/cn'

const button = cva(
  [
    'inline-flex items-center justify-center gap-1.5 select-none whitespace-nowrap',
    'rounded-[var(--radius-sm)] font-medium tracking-[-0.005em]',
    'transition-[background,color,border-color,box-shadow,transform] duration-[var(--duration-fast)] ease-[var(--ease-out)]',
    'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent)]',
    'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
    'active:scale-[0.98]',
  ],
  {
    variants: {
      variant: {
        primary: [
          'bg-[var(--color-accent)] text-[var(--color-text-on-accent)]',
          'hover:bg-[var(--color-accent-hover)]',
          'shadow-[inset_0_1px_0_oklch(1_0_0/0.12)]',
        ],
        secondary: [
          'bg-[var(--color-base-2)] text-[var(--color-text)]',
          'border border-[var(--color-border)]',
          'hover:bg-[var(--color-bg-hover)] hover:border-[var(--color-border-strong)]',
        ],
        ghost: [
          'bg-transparent text-[var(--color-text-muted)]',
          'hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text)]',
        ],
        destructive: [
          'bg-[color-mix(in_oklch,var(--color-error)_18%,transparent)]',
          'text-[var(--color-error)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)]',
          'hover:bg-[color-mix(in_oklch,var(--color-error)_28%,transparent)]',
        ],
        link: [
          'bg-transparent text-[var(--color-accent)]',
          'hover:underline underline-offset-4',
        ],
      },
      size: {
        xs: 'h-7 px-2.5 text-[12px]',
        sm: 'h-8 px-3 text-[13px]',
        md: 'h-9 px-3.5 text-[14px]',
        lg: 'h-10 px-5 text-[15px]',
      },
      block: {
        true: 'w-full',
      },
    },
    defaultVariants: {
      variant: 'secondary',
      size: 'md',
    },
  },
)

type ButtonVariants = VariantProps<typeof button>

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    ButtonVariants {
  leading?: ReactNode
  trailing?: ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, block, leading, trailing, children, ...rest },
    ref,
  ) => (
    <button
      ref={ref}
      className={cn(button({ variant, size, block }), className)}
      {...rest}
    >
      {leading}
      {children}
      {trailing}
    </button>
  ),
)
Button.displayName = 'Button'
