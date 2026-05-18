import { create } from 'zustand'
import { searchCollection } from '../api/collections'
import type {
  ExplainTiming,
  HybridExplain,
  SearchHit,
  SearchRequest,
} from '../api/collections'

interface SearchState {
  collection: string | null
  results: SearchHit[]
  loading: boolean
  error: string | null
  tookMs: number
  explain: ExplainTiming | null
  hybridExplain: HybridExplain | null
  setCollection: (name: string | null) => void
  search: (collection: string, body: SearchRequest) => Promise<void>
  clearResults: () => void
}

export const useSearchStore = create<SearchState>((set) => ({
  collection: null,
  results: [],
  loading: false,
  error: null,
  tookMs: 0,
  explain: null,
  hybridExplain: null,

  setCollection: (name) => set({ collection: name }),

  search: async (collection, body) => {
    set({
      loading: true,
      error: null,
      results: [],
      explain: null,
      hybridExplain: null,
    })
    try {
      const { data } = await searchCollection(collection, body)
      set({
        results: data.results,
        tookMs: data.took_ms,
        explain: data.explain ?? null,
        hybridExplain: data.hybrid_explain ?? null,
        loading: false,
      })
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Search failed',
        loading: false,
      })
    }
  },

  clearResults: () =>
    set({
      results: [],
      tookMs: 0,
      explain: null,
      hybridExplain: null,
      error: null,
    }),
}))
