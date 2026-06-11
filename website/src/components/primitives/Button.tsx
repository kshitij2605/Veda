import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../lib/cn'

const button = cva(
  [
    'inline-flex items-center justify-center gap-2 select-none whitespace-nowrap cursor-pointer',
    'rounded-[var(--radius-md)] font-semibold tracking-[-0.005em]',
    'transition-[background,color,border-color,box-shadow,transform] duration-[var(--duration-fast)] ease-[var(--ease-out)]',
    'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent)]',
    'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
    'active:scale-[0.98]',
  ],
  {
    variants: {
      variant: {
        primary: [
          'bg-[var(--color-accent)] text-[var(--color-accent-ink)]',
          'hover:bg-[var(--color-accent-hot)] hover:shadow-[var(--shadow-glow)]',
        ],
        secondary: [
          'bg-transparent text-[var(--color-text)]',
          'border border-[var(--color-border-strong)]',
          'hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]',
        ],
        ghost: [
          'bg-transparent text-[var(--color-text-muted)]',
          'hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text)]',
        ],
        destructive: [
          'bg-transparent text-[var(--color-danger)]',
          'border border-[color-mix(in_srgb,var(--color-danger)_40%,transparent)]',
          'hover:bg-[color-mix(in_srgb,var(--color-danger)_14%,transparent)]',
        ],
        link: [
          'bg-transparent text-[var(--color-accent)]',
          'hover:underline underline-offset-4',
        ],
      },
      size: {
        xs: 'h-7 px-2.5 text-[12px]',
        sm: 'h-8 px-3 text-[13px]',
        md: 'h-9 px-4 text-[14px]',
        lg: 'h-11 px-5 text-[15px]',
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
