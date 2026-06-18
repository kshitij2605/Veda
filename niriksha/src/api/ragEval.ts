import api from './client'

/**
 * Typed client + DTOs for the RAG/agent eval harness.
 * Mirrors `crates/veda-gateway/src/rest/rag_eval.rs`. Tenant/RLS-scoped
 * like `/converse` (not admin-only); the bearer token is attached by the
 * shared axios instance.
 *
 * NOTE: `recall_at_k` is a retrieval *proxy* — the fraction of a case's
 * known-good `expected_chunk_ids` that the agent's answer cited. Surface
 * it labeled accordingly in the UI.
 */

export interface EvalCase {
  id: string
  query: string
  /** Known-good retrieval; recall@k is scored against these. */
  expected_chunk_ids: string[]
  gold_answer?: string | null
}

export interface EvalSet {
  id: string
  name: string
  agent_id?: string | null
  cases: EvalCase[]
}

export interface CaseResult {
  case_id: string
  query: string
  response: string
  retrieved_chunk_ids: string[]
  /** recall@k proxy; null when the case had no expected ids. */
  recall_at_k?: number | null
  faithfulness?: number | null
  answer_relevancy?: number | null
  passed: boolean
}

export interface RunAggregate {
  case_count: number
  mean_recall_at_k?: number | null
  mean_faithfulness?: number | null
  mean_answer_relevancy?: number | null
  pass_rate: number
}

export interface EvalRun {
  id: string
  eval_set_id: string
  agent_id: string
  created_at: number
  recall_bar: number
  faithfulness_bar: number
  aggregate: RunAggregate
  cases: CaseResult[]
}

export interface ListSetsResponse {
  eval_sets: EvalSet[]
}
export interface ListRunsResponse {
  runs: EvalRun[]
}

/** Create body — `id` optional (server assigns when empty/absent). */
export type EvalSetCreate = Omit<EvalSet, 'id'> & { id?: string }

export interface RunRequest {
  agent_id?: string
  recall_bar?: number
  faithfulness_bar?: number
}

export interface FromConversationRequest {
  conversation_id?: string
  query?: string
  retrieved_chunk_ids?: string[]
  answer?: string
}

export const listEvalSets = () => api.get<ListSetsResponse>('/eval-sets')

export const createEvalSet = (set: EvalSetCreate) =>
  api.post<EvalSet>('/eval-sets', set)

export const getEvalSet = (id: string) =>
  api.get<EvalSet>(`/eval-sets/${encodeURIComponent(id)}`)

export const updateEvalSet = (id: string, set: EvalSetCreate) =>
  api.put<EvalSet>(`/eval-sets/${encodeURIComponent(id)}`, set)

export const deleteEvalSet = (id: string) =>
  api.delete(`/eval-sets/${encodeURIComponent(id)}`)

export const runEvalSet = (id: string, body: RunRequest = {}) =>
  api.post<EvalRun>(`/eval-sets/${encodeURIComponent(id)}/run`, body)

export const listRuns = (id: string) =>
  api.get<ListRunsResponse>(`/eval-sets/${encodeURIComponent(id)}/runs`)

export const caseFromConversation = (
  id: string,
  body: FromConversationRequest,
) =>
  api.post<EvalSet>(
    `/eval-sets/${encodeURIComponent(id)}/cases/from-conversation`,
    body,
  )

/** Format a 0–1 metric as a percentage, null-safe. */
export function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${(v * 100).toFixed(0)}%`
}
