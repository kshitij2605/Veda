import { StrictMode } from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { AppRoutes } from './App'
import { FLAT } from './docs/registry'

// The prerender needs the canonical doc page list. Exporting it from the SSR
// bundle keeps ONE source of truth — docs/registry.ts, which also drives the
// sidebar and the router — instead of a second hand-maintained copy in
// scripts/prerender.mjs. That copy had silently drifted by four pages
// (auto-tuning, external-collections, multi-hop, retrieval-guardrails), which
// shipped as 404s while being linked from the docs body and the sidebar.
export const docPages: { slug: string; title: string }[] = FLAT.map(({ slug, title }) => ({
  slug,
  title,
}))

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
