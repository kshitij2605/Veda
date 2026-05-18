import { useMemo } from 'react'
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

/**
 * Saved Object — Kibana-style. Every search / atlas view / eval run /
 * console query / pipeline config is a first-class persistable thing
 * with a stable id. Backed today by localStorage; the abstraction is
 * shaped so we can swap to a gateway endpoint later by replacing one
 * adapter (no call-sites change).
 */
export type SavedObjectType =
  | 'search'
  | 'atlas'
  | 'eval'
  | 'eval_run'
  | 'console_query'
  | 'pipeline'

export interface SavedObject<T = unknown> {
  id: string
  type: SavedObjectType
  name: string
  body: T
  /** Owner is just the local user marker today. */
  owner?: string
  createdAt: number
  updatedAt: number
  /** Last-used timestamp powers the "recent" rail in the palette. */
  lastUsedAt?: number
}

interface SavedObjectState {
  objects: Record<string, SavedObject>
  upsert: <T>(o: Omit<SavedObject<T>, 'createdAt' | 'updatedAt'>) => SavedObject<T>
  remove: (id: string) => void
  touch: (id: string) => void
  list: (type?: SavedObjectType) => SavedObject[]
  get: <T>(id: string) => SavedObject<T> | undefined
}

const genId = (type: SavedObjectType) =>
  `${type}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

export const useSavedObjects = create<SavedObjectState>()(
  persist(
    (set, get) => ({
      objects: {},
      upsert: <T>(o: Omit<SavedObject<T>, 'createdAt' | 'updatedAt'>) => {
        const now = Date.now()
        const id = o.id || genId(o.type)
        const existing = get().objects[id]
        const next: SavedObject<T> = {
          ...o,
          id,
          createdAt: existing?.createdAt ?? now,
          updatedAt: now,
        }
        set((s) => ({ objects: { ...s.objects, [id]: next as SavedObject } }))
        return next
      },
      remove: (id) =>
        set((s) => {
          const { [id]: _, ...rest } = s.objects
          return { objects: rest }
        }),
      touch: (id) =>
        set((s) => {
          const obj = s.objects[id]
          if (!obj) return s
          return {
            objects: { ...s.objects, [id]: { ...obj, lastUsedAt: Date.now() } },
          }
        }),
      list: (type) => {
        const all = Object.values(get().objects)
        const filtered = type ? all.filter((o) => o.type === type) : all
        return filtered.sort(
          (a, b) =>
            (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
        )
      },
      get: <T>(id: string) => get().objects[id] as SavedObject<T> | undefined,
    }),
    {
      name: 'niriksha.savedObjects.v1',
      storage: createJSONStorage(() => localStorage),
    },
  ),
)

/**
 * Recent-items hook for the palette. Selects the raw `objects` map (a
 * stable reference managed by zustand+persist) and derives the sorted
 * recent list in a memo — selecting `s.list()` directly would return a
 * fresh array on every render and trip React's infinite-loop guard.
 */
export function useRecentObjects(limit = 8): SavedObject[] {
  const objects = useSavedObjects((s) => s.objects)
  return useMemo(() => {
    const all = Object.values(objects) as SavedObject[]
    return all
      .sort(
        (a, b) => (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
      )
      .slice(0, limit)
  }, [objects, limit])
}
