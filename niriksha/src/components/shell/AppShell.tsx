import { type ReactNode, useEffect, useState } from 'react'
import { Sidebar } from './Sidebar'
import { Breadcrumbs } from './Breadcrumbs'
import { TrustBadges } from './TrustBadges'
import { CommandPalette } from '../palette/CommandPalette'
import { Toaster, TooltipProvider, ConnectionPill, ThemeToggle } from '../primitives'
import { useHydrateSavedObjects } from '../../stores/savedObjectStore'
import { useCollectionStore } from '../../stores/collectionStore'
import { applyTheme, getStoredTheme } from '../../lib/theme'

const COLLAPSE_KEY = 'veda_sidebar_collapsed'

export function AppShell({ children }: { children: ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1'
    } catch {
      return false
    }
  })
  useHydrateSavedObjects()

  // Apply the persisted theme once on mount.
  useEffect(() => {
    applyTheme(getStoredTheme())
  }, [])

  // Cmd/Ctrl+K toggles the palette anywhere.
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

  const toggleCollapsed = () => {
    setCollapsed((v) => {
      const nv = !v
      try {
        localStorage.setItem(COLLAPSE_KEY, nv ? '1' : '0')
      } catch {
        /* storage unavailable — non-fatal */
      }
      return nv
    })
  }

  return (
    <TooltipProvider>
      <div className="flex h-screen bg-[var(--color-bg)] text-[var(--color-text)] overflow-hidden">
        <Sidebar
          onOpenPalette={() => setPaletteOpen(true)}
          collapsed={collapsed}
          onToggle={toggleCollapsed}
        />
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
  const { health, fetchHealth } = useCollectionStore()
  useEffect(() => {
    fetchHealth()
  }, [fetchHealth])
  const healthy = health?.status === 'healthy'
  return (
    <header className="h-11 flex-none border-b border-[var(--color-border)] bg-[var(--color-bg)] flex items-center px-4 gap-3">
      <div className="flex-1 min-w-0">
        <Breadcrumbs />
      </div>
      <ConnectionPill ok={!!healthy} label={healthy ? 'healthy' : health ? 'degraded' : 'offline'} />
      <ThemeToggle />
      <TrustBadges />
    </header>
  )
}
