import { Toaster as Sonner, toast } from 'sonner'

export { toast }

/**
 * Toast host — drop once near the AppShell root. `toast()` from `sonner`
 * is the trigger API. Styled to match Linear-style dark surfaces.
 */
export function Toaster() {
  return (
    <Sonner
      theme="dark"
      position="bottom-right"
      duration={4000}
      gap={8}
      visibleToasts={4}
      toastOptions={{
        className: '!font-sans',
        style: {
          background: 'var(--color-bg-overlay)',
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--color-text)',
          fontSize: '12.5px',
          boxShadow: 'var(--shadow-md)',
        },
      }}
    />
  )
}
