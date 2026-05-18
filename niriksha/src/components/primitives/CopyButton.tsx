import { useState, type ButtonHTMLAttributes } from 'react'
import { Check, Copy } from 'lucide-react'
import { cn } from '../../lib/cn'

export interface CopyButtonProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  value: string
  size?: 'sm' | 'md'
  label?: string
}

/**
 * CopyButton — copies `value` to clipboard, flashes a checkmark for 1.4s.
 * Used everywhere an ID, vector, or URL appears in the UI.
 */
export function CopyButton({
  value,
  size = 'sm',
  label,
  className,
  ...rest
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false)
  const handle = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 1400)
    } catch {
      // clipboard blocked — silently fail
    }
  }
  const Icon = copied ? Check : Copy
  return (
    <button
      type="button"
      onClick={handle}
      title={copied ? 'Copied' : `Copy${label ? ' ' + label : ''}`}
      className={cn(
        'inline-flex items-center gap-1 rounded-[var(--radius-xs)]',
        'text-[var(--color-text-subtle)] hover:text-[var(--color-text)]',
        'hover:bg-[var(--color-bg-hover)] transition-colors duration-[var(--duration-fast)]',
        size === 'sm' ? 'p-1' : 'p-1.5',
        copied && 'text-[var(--color-success)] hover:text-[var(--color-success)]',
        className,
      )}
      {...rest}
    >
      <Icon size={size === 'sm' ? 12 : 14} strokeWidth={2.25} />
      {label && <span className="text-[11px]">{label}</span>}
    </button>
  )
}
