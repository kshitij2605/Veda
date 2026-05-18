import {
  embedQuery,
  searchCollection,
  type FieldSchema,
  type SearchRequest,
} from '../../api/collections'
import type {
  EvalItem,
  EvalRunSummary,
  PerQueryResult,
} from './types'

/**
 * Retrieval config the runner uses per-query. Mirrors the bits of the
 * Phase 2 PlaygroundState that drive a single search call. Loose typing
 * on purpose — we just need enough to build a SearchRequest.
 */
export interface RetrievalConfig {
  collection: string
  vectorField: string
  /** Optional: model id to pass to /v1/embed (overrides registry default). */
  model?: string
  topK: number
  filter?: string
  truncateTo?: number
  /** Whether to dispatch via the dense path. Only mode the v1 runner
   *  supports — sparse / BM25 / hybrid follow once those embedders ship
   *  client-callable endpoints. */
  mode: 'dense'
}

/**
 * Per-query event the runner emits while it executes. The page wires
 * this to a progress bar + a streaming results table.
 */
export type RunnerEvent =
  | { type: 'progress'; done: number; total: number }
  | { type: 'item'; result: PerQueryResult }
  | { type: 'done'; perQuery: PerQueryResult[]; summary: EvalRunSummary }
  | { type: 'error'; message: string }

/**
 * Execute one eval set against one retrieval config. Streams events
 * through `onEvent` and resolves when complete (or aborted).
 *
 * Sequential by default — the embedding service is usually the
 * bottleneck and parallelising the kNN inflates p95 noisily. If we
 * need parallel later, the right knob is a small fixed concurrency.
 */
export async function runEval(
  items: EvalItem[],
  config: RetrievalConfig,
  schema: FieldSchema[] | undefined,
  onEvent: (e: RunnerEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  void schema // reserved for future per-mode validation
  const perQuery: PerQueryResult[] = []

  for (let i = 0; i < items.length; i++) {
    if (signal?.aborted) {
      onEvent({ type: 'error', message: 'Aborted by user' })
      return
    }
    const item = items[i]!
    onEvent({ type: 'progress', done: i, total: items.length })
    const result = await runOne(item, config)
    perQuery.push(result)
    onEvent({ type: 'item', result })
  }

  onEvent({ type: 'progress', done: items.length, total: items.length })
  const summary = rollUp(perQuery)
  onEvent({ type: 'done', perQuery, summary })
}

async function runOne(
  item: EvalItem,
  cfg: RetrievalConfig,
): Promise<PerQueryResult> {
  const expected = item.expected_doc_ids ?? []
  const start = performance.now()
  try {
    // 1) Embed the query server-side.
    const { data: embed } = await embedQuery({
      text: item.query,
      model: cfg.model,
    })

    // 2) Issue the search.
    const body: SearchRequest = {
      query: { vector: { field: cfg.vectorField, values: embed.values } },
      limit: cfg.topK,
    }
    if (cfg.filter) body.filter = cfg.filter
    if (cfg.truncateTo) body.truncate_to = cfg.truncateTo

    const { data } = await searchCollection(cfg.collection, body)
    const took_ms = performance.now() - start

    const retrieved_ids = data.results.map((r) => r.id)
    const expectedSet = new Set(expected)
    const hits = retrieved_ids.filter((id) => expectedSet.has(id)).length
    const recall = expected.length === 0 ? 0 : hits / expected.length
    const precision =
      retrieved_ids.length === 0 ? 0 : hits / retrieved_ids.length
    const firstHitIdx = retrieved_ids.findIndex((id) => expectedSet.has(id))
    const rr = firstHitIdx >= 0 ? 1 / (firstHitIdx + 1) : 0
    return {
      query: item.query,
      expected,
      retrieved_ids,
      hits,
      recall,
      precision,
      rr,
      hit_at_1: firstHitIdx === 0,
      took_ms,
    }
  } catch (e) {
    const took_ms = performance.now() - start
    return {
      query: item.query,
      expected,
      retrieved_ids: [],
      hits: 0,
      recall: 0,
      precision: 0,
      rr: 0,
      hit_at_1: false,
      took_ms,
      error: extractError(e),
    }
  }
}

function rollUp(results: PerQueryResult[]): EvalRunSummary {
  const failed = results.filter((r) => r.error).length
  const ok = results.filter((r) => !r.error)
  const n = results.length
  const mean = (xs: number[]): number =>
    xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length
  const recall_mean = mean(ok.map((r) => r.recall))
  const precision_mean = mean(ok.map((r) => r.precision))
  const mrr = mean(ok.map((r) => r.rr))
  const hit_at_1_pct =
    ok.length === 0 ? 0 : ok.filter((r) => r.hit_at_1).length / ok.length
  const sortedMs = ok.map((r) => r.took_ms).sort((a, b) => a - b)
  const p50_ms = pct(sortedMs, 0.5)
  const p95_ms = pct(sortedMs, 0.95)
  return {
    n,
    failed,
    recall_mean,
    precision_mean,
    mrr,
    hit_at_1_pct,
    p50_ms,
    p95_ms,
  }
}

function pct(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((sorted.length - 1) * p)),
  )
  return sorted[idx]!
}

function extractError(e: unknown): string {
  const ax = e as { response?: { data?: { error?: string | { message?: string } } } }
  const er = ax?.response?.data?.error
  if (typeof er === 'string') return er
  if (er && typeof er === 'object' && 'message' in er && typeof er.message === 'string') {
    return er.message
  }
  if (e instanceof Error) return e.message
  return String(e)
}
