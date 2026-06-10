import { create } from 'zustand'
import { fetchCollections, fetchHealth, fetchCollection } from '../api/collections'
import type { CollectionSummary, HealthResponse, CollectionInfo } from '../api/collections'

interface CollectionState {
  collections: CollectionSummary[]
  details: CollectionInfo[]
  health: HealthResponse | null
  loading: boolean
  error: string | null
  fetchCollections: () => Promise<void>
  fetchHealth: () => Promise<void>
  fetchAllDetails: () => Promise<void>
}

export const useCollectionStore = create<CollectionState>((set, get) => ({
  collections: [],
  details: [],
  health: null,
  loading: false,
  error: null,

  fetchCollections: async () => {
    set({ loading: true, error: null })
    try {
      const { data } = await fetchCollections()
      set({ collections: data.collections, loading: false })
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to fetch collections',
        loading: false,
      })
    }
  },

  fetchHealth: async () => {
    try {
      const { data } = await fetchHealth()
      set({ health: data })
    } catch {
      set({ health: null })
    }
  },

  // Fetch every collection's full schema (needed for modality derivation).
  // Resilient: a failed detail is skipped, not fatal.
  fetchAllDetails: async () => {
    const names = get().collections.map((c) => c.name)
    const results = await Promise.allSettled(names.map((n) => fetchCollection(n)))
    const details = results
      .filter(
        (r): r is PromiseFulfilledResult<Awaited<ReturnType<typeof fetchCollection>>> =>
          r.status === 'fulfilled',
      )
      .map((r) => r.value.data)
    set({ details })
  },
}))
