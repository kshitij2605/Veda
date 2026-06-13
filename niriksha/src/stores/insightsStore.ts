import { create } from 'zustand'
import { fetchInsights, type OpsEvent, type Severity } from '../api/insights'

/**
 * Lightweight store backing the sidebar Insights badge. Polls
 * `GET /v1/admin/insights` on an interval to surface the open-event
 * count + the worst open severity, independent of whether the Insights
 * page is mounted. The page itself owns its richer state.
 */
interface InsightsState {
  openCount: number
  maxSeverity: Severity | null
  /** Refresh once from the open-events endpoint. Silent on error. */
  refresh: () => Promise<void>
  /** Update straight from a freshly-fetched events list (no extra GET). */
  setFromEvents: (events: OpsEvent[]) => void
}

const RANK: Record<Severity, number> = { low: 1, medium: 2, high: 3 }

function worst(events: OpsEvent[]): Severity | null {
  let max: Severity | null = null
  for (const e of events) {
    if (max === null || RANK[e.severity] > RANK[max]) max = e.severity
  }
  return max
}

export const useInsightsStore = create<InsightsState>((set) => ({
  openCount: 0,
  maxSeverity: null,

  refresh: async () => {
    try {
      const { data } = await fetchInsights()
      set({ openCount: data.events.length, maxSeverity: worst(data.events) })
    } catch {
      /* badge is best-effort — leave the last known value */
    }
  },

  setFromEvents: (events) =>
    set({ openCount: events.length, maxSeverity: worst(events) }),
}))
