export type Theme = 'dark' | 'light'

const KEY = 'veda_theme'

/** Persisted theme, defaulting to dark (the canonical theme). */
export function getStoredTheme(): Theme {
  const v = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null
  return v === 'light' ? 'light' : 'dark'
}

/** Apply a theme by toggling `data-theme` on <html> (dark = no attribute). */
export function applyTheme(t: Theme): void {
  const el = document.documentElement
  if (t === 'light') el.dataset.theme = 'light'
  else delete el.dataset.theme
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* storage unavailable — non-fatal */
  }
}
