import { useCallback, useEffect, useState } from 'react'

/**
 * useUrlState — every meaningful UI state should round-trip through the URL,
 * so back/forward and link-sharing both work.
 *
 * Stores values as JSON in `location.hash` under a `state` key, plus
 * individual query-string keys. We use the hash so the React Router
 * sees the path unchanged and doesn't re-mount.
 *
 * Usage:
 *   const [filter, setFilter] = useUrlState('filter', '')
 *
 * The serialiser/parser are JSON.stringify/parse by default; pass
 * `serialize`/`parse` for custom encoding.
 */
export function useUrlState<T>(
  key: string,
  initial: T,
  opts: {
    serialize?: (v: T) => string
    parse?: (s: string) => T
  } = {},
): [T, (value: T | ((prev: T) => T)) => void] {
  const serialize = opts.serialize ?? ((v: T) => JSON.stringify(v))
  const parse = opts.parse ?? ((s: string) => JSON.parse(s) as T)

  const readFromUrl = useCallback((): T => {
    if (typeof window === 'undefined') return initial
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ''))
    const raw = params.get(key)
    if (raw == null) return initial
    try {
      return parse(raw)
    } catch {
      return initial
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const [value, setLocal] = useState<T>(readFromUrl)

  // Push the new value into the URL hash without triggering a navigation.
  const push = useCallback(
    (next: T) => {
      const params = new URLSearchParams(window.location.hash.replace(/^#/, ''))
      if (
        next == null ||
        (typeof next === 'string' && next === '') ||
        (Array.isArray(next) && next.length === 0)
      ) {
        params.delete(key)
      } else {
        params.set(key, serialize(next))
      }
      const newHash = params.toString()
      const newUrl =
        window.location.pathname +
        window.location.search +
        (newHash ? '#' + newHash : '')
      window.history.replaceState(null, '', newUrl)
    },
    [key, serialize],
  )

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      setLocal((prev) => {
        const resolved =
          typeof next === 'function' ? (next as (p: T) => T)(prev) : next
        push(resolved)
        return resolved
      })
    },
    [push],
  )

  // Listen for back/forward — sync local state from URL.
  useEffect(() => {
    const handler = () => setLocal(readFromUrl())
    window.addEventListener('popstate', handler)
    window.addEventListener('hashchange', handler)
    return () => {
      window.removeEventListener('popstate', handler)
      window.removeEventListener('hashchange', handler)
    }
  }, [readFromUrl])

  return [value, set]
}
