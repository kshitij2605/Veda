/**
 * Data shapes for the Quality Lab.
 *
 * Both eval sets and eval runs are stored as `SavedObject<T>` via the
 * Phase 0 saved-object store — they're first-class persistable things
 * indexable from the Cmd-K palette.
 */

export interface EvalItem {
  query: string
  /** Gold record ids that *should* surface in top-K. Optional; absent items
   *  count toward latency stats only. */
  expected_doc_ids?: string[]
  meta?: Record<string, string | number | boolean>
}

export interface EvalSetBody {
  collection: string
  items: EvalItem[]
  source: 'upload' | 'saved-search' | 'manual'
  notes?: string
}

/** Per-query trace from an eval run. */
export interface PerQueryResult {
  query: string
  expected: string[]
  retrieved_ids: string[]
  hits: number
  recall: number
  precision: number
  /** Reciprocal rank of the first expected hit; 0 if none. */
  rr: number
  hit_at_1: boolean
  took_ms: number
  error?: string
}

/** Roll-up of per-query metrics. */
export interface EvalRunSummary {
  n: number
  failed: number
  recall_mean: number
  precision_mean: number
  mrr: number
  hit_at_1_pct: number
  p50_ms: number
  p95_ms: number
}

/** What gets persisted per run — snapshot of the retrieval config + results. */
export interface EvalRunBody {
  evalSetId: string
  evalSetName: string
  collection: string
  searchSavedId?: string
  searchSavedName?: string
  /** Snapshotted PlaygroundState (loose shape so we don't pin Phase 2 here). */
  searchConfig: unknown
  topK: number
  perQuery: PerQueryResult[]
  summary: EvalRunSummary
  startedAt: number
  finishedAt: number
}
