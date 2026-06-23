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
  return renderToString(
    <StrictMode>
      <MemoryRouter initialEntries={[url]}>
        <AppRoutes />
      </MemoryRouter>
    </StrictMode>,
  )
}
