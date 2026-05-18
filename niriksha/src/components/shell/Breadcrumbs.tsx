import { ChevronRight } from 'lucide-react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { cn } from '../../lib/cn'

interface Crumb {
  label: string
  to?: string
}

/**
 * Breadcrumbs — derived from the URL path. Knows about Veda's IA:
 *   /collections          → Collections
 *   /collections/:name    → Collections / :name
 *   /collections/:name/records → Collections / :name / Records
 *   /search               → Search
 *   /atlas                → Atlas
 *   /quality              → Quality Lab
 *   /pipelines            → Pipelines
 *   /health               → Health
 *   /console              → Dev Console
 *
 * Fallback for unknown paths is a Titlecased segment list.
 */
export function Breadcrumbs() {
  const location = useLocation()
  const params = useParams<{ name?: string; id?: string }>()
  const crumbs = pathToCrumbs(location.pathname, params)
  if (crumbs.length === 0) return null
  return (
    <nav aria-label="Breadcrumb" className="flex items-center min-w-0">
      <ol className="flex items-center gap-1 min-w-0">
        {crumbs.map((c, i) => {
          const last = i === crumbs.length - 1
          return (
            <li key={i} className="flex items-center gap-1 min-w-0">
              {i > 0 && (
                <ChevronRight
                  size={11}
                  strokeWidth={2.25}
                  className="text-[var(--color-text-subtle)] flex-none"
                />
              )}
              {c.to && !last ? (
                <Link
                  to={c.to}
                  className={cn(
                    'text-[12.5px] text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
                    'transition-colors duration-[var(--duration-fast)] truncate',
                  )}
                >
                  {c.label}
                </Link>
              ) : (
                <span
                  className={cn(
                    'text-[12.5px] truncate',
                    last
                      ? 'text-[var(--color-text)] font-medium'
                      : 'text-[var(--color-text-muted)]',
                  )}
                >
                  {c.label}
                </span>
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function pathToCrumbs(
  pathname: string,
  params: { name?: string; id?: string },
): Crumb[] {
  const parts = pathname.split('/').filter(Boolean)
  if (parts.length === 0) return [{ label: 'Home' }]

  const top = parts[0]!
  const TOP_LABEL: Record<string, string> = {
    collections: 'Collections',
    search: 'Search',
    atlas: 'Atlas',
    quality: 'Quality Lab',
    evaluation: 'Evaluation',
    pipelines: 'Pipelines',
    health: 'Health',
    indexes: 'Indexes',
    observability: 'Observability',
    console: 'Dev Console',
    settings: 'Settings',
  }
  const topLabel = TOP_LABEL[top] ?? titlecase(top)
  const out: Crumb[] = [{ label: topLabel, to: '/' + top }]

  // /collections/:name/...
  if (top === 'collections' && params.name) {
    out.push({ label: params.name, to: `/collections/${params.name}` })
    // Tab segment after collection name
    const tab = parts[2]
    if (tab) {
      const TAB_LABEL: Record<string, string> = {
        records: 'Records',
        schema: 'Schema',
        stats: 'Statistics',
        settings: 'Settings',
        indexes: 'Indexes',
      }
      out.push({ label: TAB_LABEL[tab] ?? titlecase(tab) })
    }
    return out
  }

  // /search/:collection
  if ((top === 'search' || top === 'atlas') && parts[1]) {
    out.push({ label: parts[1] })
  }

  return out
}

function titlecase(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1).replace(/-/g, ' ')
}
