import { create } from 'zustand'
import { searchCollection } from '../api/collections'
import type { SearchHit, SearchParams } from '../api/collections'

interface SearchState {
  collection: string | null
  results: SearchHit[]
  loading: boolean
  error: string | null
  tookMs: number
  setCollection: (name: string | null) => void
  search: (collection: string, params: SearchParams) => Promise<void>
  clearResults: () => void
}

export const useSearchStore = create<SearchState>((set) => ({
  collection: null,
  results: [],
  loading: false,
  error: null,
  tookMs: 0,

  setCollection: (name) => set({ collection: name }),

  search: async (collection, params) => {
    set({ loading: true, error: null, results: [] })
    try {
      const { data } = await searchCollection(collection, params)
      set({ results: data.results, tookMs: data.took_ms, loading: false })
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Search failed', loading: false })
    }
  },

  clearResults: () => set({ results: [], tookMs: 0, error: null }),
}))
