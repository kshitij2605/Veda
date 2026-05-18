import * as RTooltip from '@radix-ui/react-tooltip'
import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export function TooltipProvider({ children }: { children: ReactNode }) {
  return (
    <RTooltip.Provider delayDuration={350} skipDelayDuration={120}>
      {children}
    </RTooltip.Provider>
  )
}

export interface TooltipProps {
  label: ReactNode
  side?: 'top' | 'right' | 'bottom' | 'left'
  shortcut?: ReactNode
  children: ReactNode
  className?: string
}

export function Tooltip({
  label,
  side = 'top',
  shortcut,
  children,
  className,
}: TooltipProps) {
  return (
    <RTooltip.Root>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content
          side={side}
          sideOffset={6}
          className={cn(
            'z-50 inline-flex items-center gap-2 px-2 py-1 rounded-[var(--radius-xs)]',
            'bg-[var(--color-base-3)] border border-[var(--color-border-strong)]',
            'text-[11px] text-[var(--color-text)] shadow-[var(--shadow-md)]',
            'data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0',
            className,
          )}
          style={{ animationDuration: '90ms' }}
        >
          <span>{label}</span>
          {shortcut && (
            <span className="text-[var(--color-text-subtle)] font-mono">
              {shortcut}
            </span>
          )}
          <RTooltip.Arrow className="fill-[var(--color-base-3)]" />
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  )
}
