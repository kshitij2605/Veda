import { useEffect, useMemo } from 'react'
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import {
  deleteSavedObject,
  listSavedObjects,
  putSavedObject,
  type SavedObjectDTO,
} from '../api/collections'

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
  | 'dashboard'

/** Phase 6.5 — body shape for SavedObject<DashboardBody>. */
export interface DashboardBody {
  items: Array<{ savedObjectId: string }>
}

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

function toDto(o: SavedObject): SavedObjectDTO {
  return {
    id: o.id,
    type: o.type,
    name: o.name,
    body: o.body,
    owner: o.owner,
    created_at: o.createdAt,
    updated_at: o.updatedAt,
    last_used_at: o.lastUsedAt,
  }
}

function fromDto(d: SavedObjectDTO): SavedObject {
  return {
    id: d.id,
    type: d.type as SavedObjectType,
    name: d.name,
    body: d.body,
    owner: d.owner,
    createdAt: d.created_at,
    updatedAt: d.updated_at,
    lastUsedAt: d.last_used_at,
  }
}

/** Fire-and-forget server write. Failures are logged, never thrown —
 *  the in-memory + localStorage copy already reflects the change, so
 *  the user's action never blocks on the gateway. */
function pushUpsert(obj: SavedObject) {
  putSavedObject(obj.id, toDto(obj)).catch((e) => {
    console.warn('saved_objects: PUT failed', obj.id, e)
  })
}
function pushDelete(id: string) {
  deleteSavedObject(id).catch((e) => {
    console.warn('saved_objects: DELETE failed', id, e)
  })
}

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
        pushUpsert(next as SavedObject)
        return next
      },
      remove: (id) => {
        set((s) => {
          const { [id]: _, ...rest } = s.objects
          return { objects: rest }
        })
        pushDelete(id)
      },
      touch: (id) =>
        set((s) => {
          const obj = s.objects[id]
          if (!obj) return s
          const next = { ...obj, lastUsedAt: Date.now() }
          pushUpsert(next)
          return {
            objects: { ...s.objects, [id]: next },
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
 * One-shot hook: on app boot, fetch the server's SavedObject snapshot
 * and merge it into the local store. Server-side timestamps win when
 * they're newer than the localStorage cache (handles multi-device).
 * Failures are silent — the localStorage cache is still authoritative.
 *
 * Call once from AppShell. Cheap enough to skip a "loaded" gate: the
 * UI renders instantly off the cache and reconciles when the GET
 * returns.
 */
export function useHydrateSavedObjects() {
  useEffect(() => {
    let cancelled = false
    listSavedObjects()
      .then(({ data }) => {
        if (cancelled) return
        const incoming = data.objects.map(fromDto)
        useSavedObjects.setState((s) => {
          const merged = { ...s.objects }
          for (const o of incoming) {
            const local = merged[o.id]
            if (!local || o.updatedAt >= local.updatedAt) {
              merged[o.id] = o
            }
          }
          return { objects: merged }
        })
      })
      .catch((e) => {
        console.warn('saved_objects: hydrate failed, using local cache only', e)
      })
    return () => {
      cancelled = true
    }
  }, [])
}

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
