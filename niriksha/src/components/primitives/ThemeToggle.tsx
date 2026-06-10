import { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'
import { applyTheme, getStoredTheme, type Theme } from '../../lib/theme'

/** Sun/Moon button that toggles + persists the theme. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getStoredTheme())

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  const next: Theme = theme === 'dark' ? 'light' : 'dark'
  return (
    <button
      type="button"
      aria-label={`Switch to ${next} theme`}
      onClick={() => setTheme(next)}
      className="inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-hover)] transition-colors"
    >
      {theme === 'dark' ? <Moon size={14} /> : <Sun size={14} />}
    </button>
  )
}
