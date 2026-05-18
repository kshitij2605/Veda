import { useCallback, useEffect, useState } from 'react'
import { fetchCoverage, type CoverageResponse } from '../../api/collections'
import { Badge, Button, Select } from '../../components/primitives'
import { RefreshCw } from 'lucide-react'

interface CoverageViewProps {
  collection: string
}

/**
 * Phase 5.7 — orphan-chunk + monoculture detector. Fetches the
 * rollup from /v1/collections/{name}/coverage?window=… and shows:
 *   - total queries / chunks / retrieved / orphan counts
 *   - top-10 most-retrieved chunks (monoculture risk)
 *   - sample of chunks never retrieved in the window (orphans)
 * Empty windows render an explainer instead of zeros.
 */
export function CoverageView({ collection }: CoverageViewProps) {
  const [data, setData] = useState<CoverageResponse | null>(null)
  const [window, setWindow] = useState('7d')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await fetchCoverage(collection, window)
      setData(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed')
    } finally {
      setLoading(false)
    }
  }, [collection, window])

  useEffect(() => {
    void load()
  }, [load])

  return (
    <div className="px-6 py-5 space-y-5">
      <header className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <h2 className="text-[14px] font-semibold text-[var(--color-text)]">
            Coverage
          </h2>
          <p className="text-[11.5px] text-[var(--color-text-subtle)]">
            Retrieval frequency from the per-collection query log.
            Surfaces chunks that never come back in search results
            (orphans) and chunks that dominate top-k (monoculture).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            value={window}
            onValueChange={setWindow}
            options={[
              { value: '1h', label: '1 hour' },
              { value: '24h', label: '24 hours' },
              { value: '7d', label: '7 days' },
              { value: '30d', label: '30 days' },
            ]}
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void load()}
            leading={<RefreshCw size={11} strokeWidth={2.25} />}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </header>

      {error && (
        <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
          {error}
        </div>
      )}

      {data && (
        <>
          <section className="grid grid-cols-4 gap-3">
            <Stat
              label="Total queries"
              value={data.total_queries.toLocaleString()}
              hint={`in last ${window}`}
            />
            <Stat
              label="Retrieved chunks"
              value={data.retrieved_chunks.toLocaleString()}
              hint={`of ${data.total_chunks.toLocaleString()} total`}
            />
            <Stat
              label="Orphan chunks"
              value={data.orphan_count.toLocaleString()}
              hint="never returned in window"
              tone={
                data.total_chunks > 0 &&
                data.orphan_count / data.total_chunks > 0.5
                  ? 'warn'
                  : undefined
              }
            />
            <Stat
              label="Coverage %"
              value={
                data.total_chunks > 0
                  ? `${(
                      (data.retrieved_chunks / data.total_chunks) *
                      100
                    ).toFixed(1)}%`
                  : '—'
              }
              hint="retrieved ÷ total"
            />
          </section>

          {data.total_queries === 0 ? (
            <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-4 py-6 text-center">
              <p className="text-[12.5px] text-[var(--color-text-muted)]">
                No queries logged in the last {window}.
              </p>
              <p className="text-[11px] text-[var(--color-text-subtle)] mt-1">
                Coverage updates after each /search on this
                collection. Run a few queries in the Search
                Playground to populate the rollup.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-5">
              <section>
                <h3 className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-2">
                  Top hits · monoculture risk
                </h3>
                <ul className="space-y-1">
                  {data.top_hits.length === 0 && (
                    <li className="text-[11.5px] text-[var(--color-text-subtle)] italic">
                      No hits yet.
                    </li>
                  )}
                  {data.top_hits.map((h) => (
                    <li
                      key={h.doc_id}
                      className="flex items-center gap-2 text-[11.5px] font-mono tabular-nums"
                    >
                      <span className="flex-1 truncate text-[var(--color-text-muted)]">
                        {h.doc_id}
                      </span>
                      <span className="text-[var(--color-text)]">
                        {h.count}×
                      </span>
                      <Badge
                        variant="default"
                        label={`${(h.fraction * 100).toFixed(1)}%`}
                        className="w-12 text-right justify-end"
                      />
                    </li>
                  ))}
                </ul>
              </section>
              <section>
                <h3 className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] mb-2">
                  Orphan sample ({Math.min(50, data.orphan_count)} of{' '}
                  {data.orphan_count.toLocaleString()})
                </h3>
                <ul className="space-y-1 max-h-[260px] overflow-y-auto">
                  {data.orphan_sample.length === 0 && (
                    <li className="text-[11.5px] text-[var(--color-text-subtle)] italic">
                      No orphans — all chunks have been retrieved at
                      least once.
                    </li>
                  )}
                  {data.orphan_sample.map((id) => (
                    <li
                      key={id}
                      className="text-[11.5px] font-mono text-[var(--color-text-muted)] truncate"
                    >
                      {id}
                    </li>
                  ))}
                </ul>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: string
  hint?: string
  tone?: 'warn'
}) {
  return (
    <div className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
        {label}
      </div>
      <div
        className={
          'text-[20px] font-semibold tabular-nums leading-tight mt-1 ' +
          (tone === 'warn'
            ? 'text-[var(--color-warning)]'
            : 'text-[var(--color-text)]')
        }
      >
        {value}
      </div>
      {hint && (
        <div className="text-[10.5px] text-[var(--color-text-subtle)] mt-0.5">
          {hint}
        </div>
      )}
    </div>
  )
}
