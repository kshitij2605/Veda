import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  type TextareaHTMLAttributes,
} from 'react'
import { cn } from '../../lib/cn'

const base = [
  'w-full rounded-[var(--radius-sm)] bg-[var(--color-base-1)]',
  'border border-[var(--color-border)] text-[var(--color-text)]',
  'placeholder:text-[var(--color-text-subtle)]',
  'transition-[border-color,box-shadow,background] duration-[var(--duration-fast)]',
  'focus:border-[var(--color-accent)] focus:outline-none focus:shadow-[var(--shadow-glow)]',
  'disabled:opacity-50 disabled:cursor-not-allowed',
].join(' ')

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  leading?: ReactNode
  trailing?: ReactNode
  invalid?: boolean
  mono?: boolean
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, leading, trailing, invalid, mono, ...rest }, ref) => {
    if (leading || trailing) {
      return (
        <div
          className={cn(
            base,
            'flex items-center gap-2 px-2.5 h-8 focus-within:border-[var(--color-accent)] focus-within:shadow-[var(--shadow-glow)]',
            invalid &&
              'border-[var(--color-error)] focus-within:border-[var(--color-error)] focus-within:shadow-[0_0_0_3px_color-mix(in_oklch,var(--color-error)_25%,transparent)]',
            className,
          )}
        >
          {leading && (
            <span className="text-[var(--color-text-subtle)] flex-none">
              {leading}
            </span>
          )}
          <input
            ref={ref}
            className={cn(
              'flex-1 bg-transparent border-none outline-none text-[13px] min-w-0',
              mono && 'font-mono',
            )}
            {...rest}
          />
          {trailing && (
            <span className="text-[var(--color-text-subtle)] flex-none">
              {trailing}
            </span>
          )}
        </div>
      )
    }
    return (
      <input
        ref={ref}
        className={cn(
          base,
          'h-8 px-2.5 text-[13px]',
          mono && 'font-mono',
          invalid &&
            'border-[var(--color-error)] focus:border-[var(--color-error)] focus:shadow-[0_0_0_3px_color-mix(in_oklch,var(--color-error)_25%,transparent)]',
          className,
        )}
        {...rest}
      />
    )
  },
)
Input.displayName = 'Input'

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  mono?: boolean
  invalid?: boolean
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, mono, invalid, ...rest }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        base,
        'min-h-[80px] py-2 px-2.5 text-[13px] leading-relaxed resize-y',
        mono && 'font-mono',
        invalid &&
          'border-[var(--color-error)] focus:border-[var(--color-error)] focus:shadow-[0_0_0_3px_color-mix(in_oklch,var(--color-error)_25%,transparent)]',
        className,
      )}
      {...rest}
    />
  ),
)
Textarea.displayName = 'Textarea'
