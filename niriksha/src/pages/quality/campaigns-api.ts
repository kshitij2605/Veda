export interface TrialObjectives {
  recall: number
  faithfulness: number
  latency_p50_ms: number
}

export interface PipelineConfig {
  tier1: {
    ef_search: number
    fusion_mode: 'Rrf' | 'Weighted'
    dense_weight: number
    rerank: boolean
    top_k: number
    truncate_to: number | null
  }
  tier2: {
    index_type: 'Hnsw' | 'DiskAnn' | 'IvfPq' | 'Flat' | 'RaBitQ' | 'Sq8'
    ef_construction: number
    m: number
  }
  tier3: {
    chunker_strategy:
      | 'MarkdownRecursive'
      | 'Semantic'
      | 'Page'
      | 'FixedWindow'
      | 'SlidingWindow'
      | 'Section'
      | 'RecursiveCharacter'
    chunk_size: number
    overlap: number
  }
}

export interface Trial {
  id: string
  config: PipelineConfig
  objectives: TrialObjectives
  is_pareto_optimal: boolean
  started_at: number
  duration_ms: number
  sibling_collection: string | null
}

export type CampaignStatus = 'Running' | 'Converged' | 'BudgetExhausted' | 'Failed' | 'Cancelled'

export interface TuningCampaign {
  id: string
  collection: string
  eval_set_id: string
  status: CampaignStatus
  budget: { max_trials: number; max_duration_secs: number }
  trials: Trial[]
  pareto_front: string[]
  created_at: number
  completed_at: number | null
}

export interface StartCampaignRequest {
  eval_set_id: string
  max_trials?: number
  max_duration_secs?: number
  num_bases?: number
  mode?: 'Quality' | 'Performance'
}

export interface ApplyResponse {
  applied: boolean
  kind: 'Tier1Instant' | 'ReindexAndSwap'
  previous_alias_target: string | null
}

const base = (path: string) => `/v1${path}`

export async function listCampaigns(collection: string): Promise<TuningCampaign[]> {
  const res = await fetch(base(`/collections/${encodeURIComponent(collection)}/tuning-campaigns`))
  if (!res.ok) throw new Error(`listCampaigns failed: ${res.status}`)
  const body = (await res.json()) as { campaigns: TuningCampaign[] }
  return body.campaigns
}

export async function startCampaign(
  collection: string,
  req: StartCampaignRequest,
): Promise<TuningCampaign> {
  const res = await fetch(base(`/collections/${encodeURIComponent(collection)}/tuning-campaigns`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`startCampaign failed: ${res.status}`)
  return (await res.json()) as TuningCampaign
}

export async function applyTrial(campaignId: string, trialId: string): Promise<ApplyResponse> {
  const res = await fetch(
    base(`/tuning-campaigns/${encodeURIComponent(campaignId)}/trials/${encodeURIComponent(trialId)}/apply`),
    { method: 'POST' },
  )
  if (!res.ok) throw new Error(`applyTrial failed: ${res.status}`)
  return (await res.json()) as ApplyResponse
}
