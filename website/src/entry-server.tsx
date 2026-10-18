import { StrictMode } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { AppRoutes } from './App'

// Server-render the app at a given route to an HTML string. Used by
// scripts/prerender.mjs to generate static per-route HTML. MemoryRouter is a
// top-level react-router-dom export (no /server subpath) and renders the route
// matching `url`. All browser-API work in the app is inside effects, which do
// not run during renderToString, so this is SSR-safe.
export function render(url: string): string {
  // The client mounts BrowserRouter with basename=BASE_URL. MemoryRouter must
  // be given the same basename or every <Link> renders a base-less href into
  // the prerendered HTML — "/docs/search" instead of "/Veda/docs/search" —
  // which 404s on a project Pages site. `initialEntries` is a real location, so
  // it carries the basename too; react-router strips it before matching.
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
  return renderToString(
    <StrictMode>
      <MemoryRouter basename={import.meta.env.BASE_URL} initialEntries={[base + url]}>
        <AppRoutes />
      </MemoryRouter>
    </StrictMode>,
  )
}
