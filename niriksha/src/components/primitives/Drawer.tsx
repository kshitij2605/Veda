import * as Dialog from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface DrawerProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  subtitle?: ReactNode
  /** Width of the drawer in px. Default 480. */
  width?: number
  /** Optional content rendered in the header to the right of the title. */
  actions?: ReactNode
  /** Render full-bleed (no padding) for content like editors. */
  bleed?: boolean
  children: ReactNode
  className?: string
}

/**
 * Drawer — right-side slide-over for record/chunk detail, settings panels,
 * inspector views. Built on Radix Dialog for a11y. Stacks naturally if
 * multiple drawers are open (each gets its own portal).
 *
 * NOT for destructive confirms — use Modal for that.
 */
export function Drawer({
  open,
  onClose,
  title,
  subtitle,
  width = 480,
  actions,
  bleed,
  children,
  className,
}: DrawerProps) {
  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay
          className={cn(
            'fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px]',
            'data-[state=open]:animate-in data-[state=open]:fade-in-0',
            'data-[state=closed]:animate-out data-[state=closed]:fade-out-0',
          )}
          style={{ animationDuration: '150ms' }}
        />
        <Dialog.Content
          aria-describedby={subtitle ? undefined : undefined}
          className={cn(
            'fixed inset-y-0 right-0 z-50 flex flex-col',
            'bg-[var(--color-bg-elevated)] border-l border-[var(--color-border)]',
            'shadow-[var(--shadow-lg)]',
            'data-[state=open]:animate-in data-[state=open]:slide-in-from-right',
            'data-[state=closed]:animate-out data-[state=closed]:slide-out-to-right',
            className,
          )}
          style={{ width, animationDuration: '180ms' }}
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          {(title || actions) && (
            <header className="flex items-start justify-between gap-3 px-4 h-12 border-b border-[var(--color-border)] flex-none">
              <div className="min-w-0 flex-1">
                {title && (
                  <Dialog.Title className="text-[13px] font-medium text-[var(--color-text)] truncate leading-[44px]">
                    {title}
                  </Dialog.Title>
                )}
                {subtitle && (
                  <Dialog.Description className="text-[11px] text-[var(--color-text-subtle)] truncate -mt-2">
                    {subtitle}
                  </Dialog.Description>
                )}
              </div>
              <div className="flex items-center gap-1 flex-none mt-2">
                {actions}
                <Dialog.Close asChild>
                  <button
                    aria-label="Close"
                    className="p-1.5 rounded-[var(--radius-xs)] text-[var(--color-text-subtle)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)]"
                  >
                    <X size={14} strokeWidth={2.25} />
                  </button>
                </Dialog.Close>
              </div>
            </header>
          )}
          <div
            className={cn(
              'flex-1 overflow-auto',
              !bleed && 'p-4',
            )}
          >
            {children}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
