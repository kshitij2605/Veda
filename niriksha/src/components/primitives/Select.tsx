import * as RSelect from '@radix-ui/react-select'
import { Check, ChevronDown } from 'lucide-react'
import { forwardRef, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface SelectOption<V extends string = string> {
  value: V
  label: ReactNode
  description?: ReactNode
  disabled?: boolean
}

export interface SelectProps<V extends string = string> {
  value: V
  onValueChange: (v: V) => void
  options: SelectOption<V>[]
  placeholder?: string
  size?: 'sm' | 'md'
  disabled?: boolean
  className?: string
}

/**
 * Select — accessible single-value combobox built on Radix Select.
 * For multi-select / typeahead, use Combobox (next iteration).
 */
export function Select<V extends string = string>({
  value,
  onValueChange,
  options,
  placeholder,
  size = 'md',
  disabled,
  className,
}: SelectProps<V>) {
  return (
    <RSelect.Root value={value} onValueChange={(v) => onValueChange(v as V)} disabled={disabled}>
      <RSelect.Trigger
        className={cn(
          'inline-flex items-center gap-2 rounded-[var(--radius-sm)]',
          'bg-[var(--color-base-1)] border border-[var(--color-border)] text-[var(--color-text)]',
          'hover:border-[var(--color-border-strong)]',
          'data-[state=open]:border-[var(--color-accent)] data-[state=open]:shadow-[var(--shadow-glow)]',
          'data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed',
          size === 'sm' ? 'h-7 px-2 text-[12px]' : 'h-8 px-2.5 text-[13px]',
          'transition-[border-color,box-shadow] duration-[var(--duration-fast)]',
          className,
        )}
        aria-label={placeholder}
      >
        <RSelect.Value placeholder={placeholder} />
        <RSelect.Icon className="text-[var(--color-text-subtle)] ml-auto">
          <ChevronDown size={12} strokeWidth={2.5} />
        </RSelect.Icon>
      </RSelect.Trigger>
      <RSelect.Portal>
        <RSelect.Content
          position="popper"
          sideOffset={4}
          className={cn(
            'z-50 overflow-hidden min-w-[var(--radix-select-trigger-width)]',
            'bg-[var(--color-bg-overlay)] border border-[var(--color-border-strong)]',
            'rounded-[var(--radius-md)] shadow-[var(--shadow-md)]',
          )}
        >
          <RSelect.Viewport className="p-1">
            {options.map((opt) => (
              <SelectItemRow key={opt.value} {...opt} />
            ))}
          </RSelect.Viewport>
        </RSelect.Content>
      </RSelect.Portal>
    </RSelect.Root>
  )
}

const SelectItemRow = forwardRef<HTMLDivElement, SelectOption>(
  ({ value, label, description, disabled }, ref) => (
    <RSelect.Item
      ref={ref}
      value={value}
      disabled={disabled}
      className={cn(
        'flex items-start gap-2 px-2 py-1.5 rounded-[var(--radius-xs)]',
        'text-[12.5px] text-[var(--color-text)] cursor-pointer outline-none',
        'data-[highlighted]:bg-[var(--color-bg-hover)]',
        'data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed',
      )}
    >
      <span className="w-3 h-3 flex-none mt-0.5">
        <RSelect.ItemIndicator>
          <Check size={12} strokeWidth={2.5} className="text-[var(--color-accent)]" />
        </RSelect.ItemIndicator>
      </span>
      <span className="flex-1 min-w-0">
        <RSelect.ItemText>{label}</RSelect.ItemText>
        {description && (
          <span className="block text-[11px] text-[var(--color-text-subtle)] mt-0.5">
            {description}
          </span>
        )}
      </span>
    </RSelect.Item>
  ),
)
SelectItemRow.displayName = 'SelectItemRow'
