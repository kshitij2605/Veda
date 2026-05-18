import { create } from 'zustand'
import { fetchCollections, fetchHealth } from '../api/collections'
import type { CollectionSummary, HealthResponse } from '../api/collections'

interface CollectionState {
  collections: CollectionSummary[]
  health: HealthResponse | null
  loading: boolean
  error: string | null
  fetchCollections: () => Promise<void>
  fetchHealth: () => Promise<void>
}

export const useCollectionStore = create<CollectionState>((set) => ({
  collections: [],
  health: null,
  loading: false,
  error: null,

  fetchCollections: async () => {
    set({ loading: true, error: null })
    try {
      const { data } = await fetchCollections()
      set({ collections: data.collections, loading: false })
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to fetch collections', loading: false })
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
}))
