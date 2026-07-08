import { useEffect, useState } from 'react'
import {
  applyTrial,
  listCampaigns,
  startCampaign,
  type StartCampaignRequest,
  type Trial,
  type TuningCampaign,
} from './campaigns-api'

export function Campaigns({ collection, evalSetIds }: { collection: string; evalSetIds: string[] }) {
  const [campaigns, setCampaigns] = useState<TuningCampaign[]>([])
  const [active, setActive] = useState<TuningCampaign | null>(null)
  const [showStart, setShowStart] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = () => {
    listCampaigns(collection)
      .then(setCampaigns)
      .catch((e) => setError(String(e)))
  }
  useEffect(refresh, [collection])

  const onStart = async (req: StartCampaignRequest) => {
    setShowStart(false)
    try {
      const campaign = await startCampaign(collection, req)
      setCampaigns((prev) => [campaign, ...prev])
    } catch (e) {
      setError(String(e))
    }
  }

  const onApply = async (trial: Trial) => {
    if (!active) return
    const confirmMsg =
      trial.sibling_collection || trial.config.tier2 !== active.trials[0]?.config.tier2
        ? 'This will trigger a reindex into a new collection version and swap traffic to it. Continue?'
        : 'This applies query-time defaults instantly and is reversible. Continue?'
    if (!window.confirm(confirmMsg)) return
    try {
      const resp = await applyTrial(active.id, trial.id)
      window.alert(`Applied (${resp.kind}). Previous alias target: ${resp.previous_alias_target ?? '(none)'}`)
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="flex h-full">
      <div className="w-72 shrink-0 border-r border-[var(--color-border)] overflow-y-auto">
        <button
          type="button"
          className="w-full px-4 py-2 text-left text-sm font-medium hover:bg-[var(--color-bg-hover)]"
          onClick={() => setShowStart(true)}
        >
          + New campaign
        </button>
        {campaigns.map((c) => (
          <button
            key={c.id}
            type="button"
            className="block w-full px-4 py-2 text-left text-xs hover:bg-[var(--color-bg-hover)]"
            onClick={() => setActive(c)}
          >
            <span className="font-mono">{c.id.slice(0, 12)}</span>
            <span className="ml-2 rounded px-1.5 py-0.5 text-[10px] uppercase">{c.status}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {error && <div className="mb-3 text-sm text-red-500">{error}</div>}
        {showStart && (
          <StartCampaignForm evalSetIds={evalSetIds} onSubmit={onStart} onCancel={() => setShowStart(false)} />
        )}
        {active && <CampaignDetail campaign={active} onApply={onApply} />}
        {!active && !showStart && (
          <div className="text-sm text-[var(--color-text-muted)]">Select a campaign, or start a new one.</div>
        )}
      </div>
    </div>
  )
}

function StartCampaignForm({
  evalSetIds,
  onSubmit,
  onCancel,
}: {
  evalSetIds: string[]
  onSubmit: (req: StartCampaignRequest) => void
  onCancel: () => void
}) {
  const [evalSetId, setEvalSetId] = useState(evalSetIds[0] ?? '')
  const [mode, setMode] = useState<'Quality' | 'Performance'>('Quality')
  const [maxTrials, setMaxTrials] = useState(40)
  return (
    <div className="mb-4 rounded border border-[var(--color-border)] p-4">
      <label className="mb-2 block text-xs font-medium">Eval set</label>
      <select
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={evalSetId}
        onChange={(e) => setEvalSetId(e.target.value)}
      >
        {evalSetIds.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
      <label className="mb-2 block text-xs font-medium">Optimization mode</label>
      <select
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={mode}
        onChange={(e) => setMode(e.target.value as 'Quality' | 'Performance')}
      >
        <option value="Quality">Quality (recall + faithfulness + latency, full Pareto front)</option>
        <option value="Performance">Performance (fastest, without dropping below today's recall)</option>
      </select>
      <label className="mb-2 block text-xs font-medium">Max trials</label>
      <input
        type="number"
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={maxTrials}
        onChange={(e) => setMaxTrials(Number(e.target.value))}
      />
      <div className="flex gap-2">
        <button
          type="button"
          className="rounded bg-[var(--color-accent)] px-3 py-1.5 text-sm text-white"
          onClick={() => onSubmit({ eval_set_id: evalSetId, max_trials: maxTrials, mode })}
        >
          Start
        </button>
        <button type="button" className="rounded border px-3 py-1.5 text-sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}

function CampaignDetail({ campaign, onApply }: { campaign: TuningCampaign; onApply: (t: Trial) => void }) {
  const frontTrials = campaign.trials.filter((t) => campaign.pareto_front.includes(t.id))
  const sorted = [...frontTrials].sort((a, b) => a.objectives.latency_p50_ms - b.objectives.latency_p50_ms)
  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold">
        Pareto front ({sorted.length} of {campaign.trials.length} trials)
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[var(--color-border)] text-left">
            <th className="p-2">recall</th>
            <th className="p-2">faithfulness</th>
            <th className="p-2">p50 latency (ms)</th>
            <th className="p-2" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((t) => (
            <tr key={t.id} className="border-b border-[var(--color-border)]">
              <td className="p-2">{t.objectives.recall.toFixed(3)}</td>
              <td className="p-2">{t.objectives.faithfulness.toFixed(3)}</td>
              <td className="p-2">{t.objectives.latency_p50_ms.toFixed(1)}</td>
              <td className="p-2">
                <button
                  type="button"
                  className="rounded border px-2 py-1 text-[11px] hover:bg-[var(--color-bg-hover)]"
                  onClick={() => onApply(t)}
                >
                  Apply
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
