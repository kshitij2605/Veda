import { useEffect } from 'react'
import { Navigate, useLocation, useParams } from 'react-router-dom'
import { bySlug } from '../docs/registry'
import { DocsLayout } from '../components/docs/DocsLayout'

export function DocPage() {
  const { slug } = useParams()
  const page = bySlug(slug)
  const { hash } = useLocation()

  useEffect(() => {
    if (!page) return
    if (hash) {
      const el = document.getElementById(hash.slice(1))
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        return
      }
    }
    window.scrollTo(0, 0)
  }, [page, slug, hash])

  if (!page) return <Navigate to="/docs/introduction" replace />
  return <DocsLayout page={page} />
}
