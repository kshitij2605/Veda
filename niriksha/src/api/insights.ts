import api from './client'

/**
 * Typed client + DTOs for the ops-advisor ("Insights") admin API.
 * Mirrors `crates/veda-gateway/src/rest/insights.rs` and the advisor
 * model in `advisor/{event.rs,config.rs}`. All endpoints are admin-only;
 * the bearer token is attached by the shared axios instance.
 */

export type Severity = 'low' | 'medium' | 'high'
export type EventStatus = 'open' | 'closed'

/**
 * Event scope — adjacently-tagged on the wire as `{ kind, target }`.
 *   global      → no target
 *   collection  → target: string
 *   shard       → target: { collection, shard }
 *   node        → target: string
 */
export type Scope =
  | { kind: 'global' }
  | { kind: 'collection'; target: string }
  | { kind: 'shard'; target: { collection: string; shard: number } }
  | { kind: 'node'; target: string }

/**
 * Remediation marker — adjacently-tagged `{ action, params }`. Only the
 * advisor's auto-remediation loop executes these; there is no per-event
 * manual-trigger endpoint.
 */
export type RemediationAction =
  | { action: 'none' }
  | { action: 'compact'; params: { collection: string } }
  | { action: 'reindex'; params: { collection: string } }

export interface OpsEvent {
  id: string
  rule_id: string
  scope: Scope
  severity: Severity
  status: EventStatus
  title: string
  /** Root cause — what tripped the rule. */
  detail: string
  /** What the operator should do about it. */
  recommendation: string
  /** Optional runbook pointer (command / endpoint hint). */
  runbook?: string
  remediation: RemediationAction
  opened_at_ms: number
  closed_at_ms?: number
  acknowledged: boolean
  /** Last auto-remediation outcome note, if any. */
  remediation_note?: string | null
}

export interface EventsResponse {
  events: OpsEvent[]
}

/** Per-rule config as stored/returned by the gateway (`RuleConfig`). */
export interface RuleConfig {
  enabled: boolean
  muted: boolean
  severity_override?: Severity | null
  /** Opaque per-rule threshold knobs. */
  thresholds: unknown
  auto_remediate: boolean
}

export interface RuleInfo {
  id: string
  /** Natural severity for threshold rules; absent for anomaly detectors. */
  default_severity?: Severity
  /** True for the statistical anomaly detectors. */
  anomaly: boolean
  effective_config: RuleConfig
}

export interface RulesResponse {
  rules: RuleInfo[]
}

/** Partial patch for `PUT /rules/{id}` — absent fields are unchanged. */
export interface RuleConfigPatch {
  enabled?: boolean
  muted?: boolean
  /** `null` clears an override; a severity sets one. */
  severity_override?: Severity | null
  thresholds?: unknown
  auto_remediate?: boolean
}

/** Non-secret SMTP addressing returned for the email channel. */
export interface EmailView {
  smtp_host: string
  smtp_port: number
  from: string
  recipients: number
}

/** Redacted per-channel view (no URLs/keys/passwords). */
export interface ChannelView {
  configured: boolean
  enabled: boolean
  min_severity?: Severity
  email?: EmailView
}

export interface ChannelsView {
  webhook: ChannelView
  slack: ChannelView
  pagerduty: ChannelView
  email: ChannelView
}

// ── Write-side channel config (secrets are write-only) ────────────────
export interface WebhookChannelConfig {
  enabled: boolean
  url: string
  min_severity: Severity
}
export interface SlackChannelConfig {
  enabled: boolean
  webhook_url: string
  min_severity: Severity
}
export interface PagerDutyChannelConfig {
  enabled: boolean
  routing_key: string
  min_severity: Severity
}
export interface EmailChannelConfig {
  enabled: boolean
  smtp_host: string
  smtp_port: number
  username: string
  password: string
  from: string
  to: string[]
  min_severity: Severity
}

/** Full channels config accepted by `PUT /channels`. */
export interface ChannelsConfig {
  webhook?: WebhookChannelConfig | null
  slack?: SlackChannelConfig | null
  pagerduty?: PagerDutyChannelConfig | null
  email?: EmailChannelConfig | null
}

export interface RunResponse {
  opened: number
  closed: number
}

export interface AckResponse {
  acknowledged: boolean
}

export interface InsightsQuery {
  severity?: Severity
  scope?: Scope['kind']
}

export const fetchInsights = (params?: InsightsQuery) =>
  api.get<EventsResponse>('/admin/insights', { params })

export const fetchInsightsHistory = () =>
  api.get<EventsResponse>('/admin/insights/history')

export const fetchRules = () =>
  api.get<RulesResponse>('/admin/insights/rules')

export const putRule = (id: string, patch: RuleConfigPatch) =>
  api.put<RuleConfig>(
    `/admin/insights/rules/${encodeURIComponent(id)}`,
    patch,
  )

export const fetchChannels = () =>
  api.get<ChannelsView>('/admin/insights/channels')

export const putChannels = (channels: ChannelsConfig) =>
  api.put<ChannelsView>('/admin/insights/channels', channels)

export const runInsights = () =>
  api.post<RunResponse>('/admin/insights/run')

export const ackInsight = (eventId: string) =>
  api.post<AckResponse>(
    `/admin/insights/${encodeURIComponent(eventId)}/ack`,
  )

/**
 * One step in the triage agent's reasoning/acting trace — internally
 * tagged by `kind` (`AgentStep` in `agent/model.rs`):
 *   reasoning    → { text }
 *   tool_call    → { tool_call_id, tool_id, args }
 *   tool_result  → { tool_call_id, result, is_error }
 */
export type AgentStep =
  | { kind: 'reasoning'; text: string }
  | {
      kind: 'tool_call'
      tool_call_id: string
      tool_id: string
      args: unknown
    }
  | {
      kind: 'tool_result'
      tool_call_id: string
      result: unknown
      is_error: boolean
    }

/** Result of `POST /insights/{id}/triage` — the agent's explanation plus
 * the step trace it produced. 503 when no LLM is configured. */
export interface TriageResponse {
  event_id: string
  explanation: string
  steps: AgentStep[]
}

export const triageEvent = (eventId: string) =>
  api.post<TriageResponse>(
    `/admin/insights/${encodeURIComponent(eventId)}/triage`,
  )

/**
 * Cross-signal root-cause bundle for one event. Mirrors
 * `advisor::rca::RcaBundle` — the correlated logs + slow/errored queries +
 * nearest resource snapshot + related incident joined in the incident
 * window. Read from `GET /admin/insights/{id}/rca`.
 */
export interface RcaLogLine {
  ts_ms: number
  level: string
  target: string
  message: string
  trace_id?: string | null
}

export interface RcaQueryLine {
  ts_ms: number
  took_ms: number
  status: string
  error_kind?: string | null
  query_type: string
  collection: string
  index_type?: string | null
  trace_id?: string | null
}

export interface RcaResourceState {
  taken_at_ms: number
  disk_used_pct?: number | null
  rss_bytes?: number | null
  query_p95_ms: number
  error_log_count: number
}

export interface RcaIncidentRef {
  id: string
  representative: string
  count: number
}

export interface RcaBundle {
  event_id: string
  /** Half-window Δ (ms) applied around opened_at_ms. */
  window_ms: number
  logs: RcaLogLine[]
  slow_queries: RcaQueryLine[]
  /** Distinct trace ids across the correlated logs + queries. */
  trace_ids: string[]
  resource_snapshot?: RcaResourceState | null
  related_incident?: RcaIncidentRef | null
}

export const fetchEventRca = (eventId: string) =>
  api.get<RcaBundle>(
    `/admin/insights/${encodeURIComponent(eventId)}/rca`,
  )

/**
 * A semantic incident cluster — recurring/duplicate ops events collapsed
 * into one row. Mirrors `advisor::incidents::Incident`. Produced by the
 * off-tick incident correlator and read from `GET /admin/incidents`.
 */
export interface Incident {
  id: string
  /** Representative title (highest-severity / earliest member). */
  representative: string
  /** Member event ids collapsed into this incident. */
  member_event_ids: string[]
  /** Distinct rule ids contributing to this incident. */
  rule_ids: string[]
  /** Number of member events (alert-noise collapsed). */
  count: number
  max_severity: Severity
  first_seen_ms: number
  last_seen_ms: number
}

export interface IncidentsResponse {
  incidents: Incident[]
}

export const fetchIncidents = () =>
  api.get<IncidentsResponse>('/admin/incidents')

/** Human-readable label for a scope. */
export function scopeLabel(scope: Scope): string {
  switch (scope.kind) {
    case 'global':
      return 'Global'
    case 'collection':
      return `Collection · ${scope.target}`
    case 'node':
      return `Node · ${scope.target}`
    case 'shard':
      return `Shard · ${scope.target.collection}/${scope.target.shard}`
  }
}
