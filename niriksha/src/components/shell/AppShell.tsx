import { type ReactNode, useEffect, useState } from 'react'
import { Sidebar } from './Sidebar'
import { Breadcrumbs } from './Breadcrumbs'
import { TrustBadges } from './TrustBadges'
import { CommandPalette } from '../palette/CommandPalette'
import { Toaster, TooltipProvider } from '../primitives'

export function AppShell({ children }: { children: ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false)

  // Cmd/Ctrl+K toggles the palette anywhere
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <TooltipProvider>
      <div className="flex h-screen bg-[var(--color-bg)] text-[var(--color-text)] overflow-hidden">
        <Sidebar onOpenPalette={() => setPaletteOpen(true)} />
        <div className="flex flex-col flex-1 min-w-0">
          <Header />
          <main className="flex-1 overflow-auto min-h-0">{children}</main>
        </div>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <Toaster />
    </TooltipProvider>
  )
}

function Header() {
  return (
    <header className="h-11 flex-none border-b border-[var(--color-border)] bg-[var(--color-bg)] flex items-center px-4 gap-3">
      <div className="flex-1 min-w-0">
        <Breadcrumbs />
      </div>
      <TrustBadges />
    </header>
  )
}
