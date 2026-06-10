export type Theme = 'dark' | 'light'

const KEY = 'veda_theme'

/** Persisted theme, defaulting to light (the canonical theme). */
export function getStoredTheme(): Theme {
  const v = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null
  return v === 'dark' ? 'dark' : 'light'
}

/** Apply a theme via `data-theme` on <html> (light = canonical; dark sets the attribute). */
export function applyTheme(t: Theme): void {
  const el = document.documentElement
  el.dataset.theme = t
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* storage unavailable — non-fatal */
  }
}
