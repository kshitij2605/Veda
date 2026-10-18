/**
 * Prefix a root-absolute path with the deploy base.
 *
 * GitHub Pages serves this site from a sub-path ("/Veda/"), so a bare
 * "/docs/quickstart" or "/videos/x.mp4" resolves against the ORIGIN, not the
 * site: it lands on kshitij2605.github.io/docs/quickstart and 404s. Vite
 * exposes the configured base as `import.meta.env.BASE_URL`, always with a
 * leading and trailing slash.
 *
 * Use this for plain <a href> and for asset URLs (video, img, poster). Do NOT
 * use it on react-router `Link to=` / `Navigate to=` props: the router applies
 * `basename` itself, so prefixing there would yield "/Veda/Veda/...".
 *
 * Idempotent, and a no-op when the site is served from the origin root.
 */
export function withBase(path: string): string {
  if (!path.startsWith('/')) return path // relative, "#anchor", or has a scheme
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
  if (!base) return path
  return path === base || path.startsWith(base + '/') ? path : base + path
}
