import * as Dialog from '@radix-ui/react-dialog'
import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface ModalProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  description?: ReactNode
  /** Default 440. */
  width?: number
  children: ReactNode
  className?: string
}

/**
 * Modal — centered dialog. Reserve for **destructive confirms** and
 * synchronous flows that block all other actions. Everything else
 * should be a Drawer (right-side slide-over).
 */
export function Modal({
  open,
  onClose,
  title,
  description,
  width = 440,
  children,
  className,
}: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay
          className="fixed inset-0 z-40 bg-black/55 backdrop-blur-[2px] data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=closed]:animate-out data-[state=closed]:fade-out-0"
          style={{ animationDuration: '120ms' }}
        />
        <Dialog.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2',
            'bg-[var(--color-bg-elevated)] border border-[var(--color-border-strong)]',
            'rounded-[var(--radius-lg)] shadow-[var(--shadow-lg)]',
            'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
            'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
            className,
          )}
          style={{ width, animationDuration: '140ms' }}
        >
          {(title || description) && (
            <header className="px-5 pt-4 pb-3 border-b border-[var(--color-border)]">
              {title && (
                <Dialog.Title className="text-[14px] font-medium text-[var(--color-text)]">
                  {title}
                </Dialog.Title>
              )}
              {description && (
                <Dialog.Description className="text-[12px] text-[var(--color-text-muted)] mt-1">
                  {description}
                </Dialog.Description>
              )}
            </header>
          )}
          <div className="p-5">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
