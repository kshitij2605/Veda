import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ChevronLeft } from 'lucide-react'
import * as Tabs from '@radix-ui/react-tabs'
import api from '../../api/client'
import { fetchCollection } from '../../api/collections'
import type { CollectionInfo, RecordRow } from '../../api/collections'
import {
  Badge,
  Button,
  CopyButton,
  Skeleton,
} from '../../components/primitives'
import { cn } from '../../lib/cn'
import { isVectorField } from './types'
import { Sparkline } from '../../components/primitives/Sparkline'

/**
 * RecordPage — permalink view for a single record. Same content as the
 * record drawer but rendered as a full page so the URL is shareable.
 */
export function RecordPage() {
  const { name, id } = useParams<{ name: string; id: string }>()
  const [collection, setCollection] = useState<CollectionInfo | null>(null)
  const [record, setRecord] = useState<RecordRow | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!name || !id) return
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([
      fetchCollection(name),
      api.get<RecordRow>(`/collections/${name}/records/${id}`),
    ])
      .then(([col, rec]) => {
        if (cancelled) return
        setCollection(col.data)
        setRecord(rec.data)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to load record')
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [name, id])

  if (!name || !id) return null

  return (
    <div className="px-6 py-5 max-w-4xl">
      <Link
        to={`/collections/${name}`}
        className="inline-flex items-center gap-1 text-[11.5px] text-[var(--color-text-muted)] hover:text-[var(--color-text)] mb-3"
      >
        <ChevronLeft size={11} strokeWidth={2.25} />
        Back to {name}
      </Link>

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-5 w-72" />
          <Skeleton className="h-4 w-44" />
          <Skeleton className="h-[300px] w-full" />
        </div>
      )}

      {error && (
        <div className="rounded-[var(--radius-sm)] border border-[color-mix(in_oklch,var(--color-error)_30%,transparent)] bg-[color-mix(in_oklch,var(--color-error)_8%,transparent)] px-3 py-2 text-[12px] text-[var(--color-error)]">
          {error}
        </div>
      )}

      {record && collection && (
        <>
          <header className="flex items-center gap-3 mb-4">
            <Badge label="record" variant="accent" />
            <span className="font-mono text-[15px] font-semibold text-[var(--color-text)] truncate">
              {record.id}
            </span>
            <CopyButton value={record.id} label="id" />
          </header>

          <Tabs.Root defaultValue="table">
            <Tabs.List className="flex items-center gap-1 border-b border-[var(--color-border)] mb-4">
              <FullPageTab value="table">Table</FullPageTab>
              <FullPageTab value="json">JSON</FullPageTab>
            </Tabs.List>
            <Tabs.Content value="table">
              <RecordTable record={record} collection={collection} />
            </Tabs.Content>
            <Tabs.Content value="json">
              <pre className="font-mono text-[12px] leading-relaxed text-[var(--color-text)] bg-[var(--color-bg-elevated)] rounded-[var(--radius-md)] border border-[var(--color-border)] p-4 whitespace-pre-wrap break-words">
                {JSON.stringify(record, null, 2)}
              </pre>
            </Tabs.Content>
          </Tabs.Root>
        </>
      )}
    </div>
  )
}

function FullPageTab({
  value,
  children,
}: {
  value: string
  children: React.ReactNode
}) {
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        'relative px-3 h-8 text-[12.5px] rounded-t-[var(--radius-xs)]',
        'transition-colors',
        'data-[state=active]:text-[var(--color-text)] data-[state=active]:font-medium',
        'data-[state=inactive]:text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
        'data-[state=active]:after:content-[""] data-[state=active]:after:absolute',
        'data-[state=active]:after:left-1.5 data-[state=active]:after:right-1.5',
        'data-[state=active]:after:-bottom-px data-[state=active]:after:h-[2px]',
        'data-[state=active]:after:bg-[var(--color-accent)]',
      )}
    >
      {children}
    </Tabs.Trigger>
  )
}

function RecordTable({
  record,
  collection,
}: {
  record: RecordRow
  collection: CollectionInfo
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] overflow-hidden">
      <table className="w-full text-[12.5px]">
        <tbody>
          {collection.fields.map((f) => {
            const v = record.fields[f.name]
            return (
              <tr
                key={f.name}
                className="border-b border-[var(--color-border)] last:border-b-0"
              >
                <td className="px-4 py-2.5 align-top w-48 bg-[var(--color-bg-elevated)] font-mono text-[var(--color-text-subtle)]">
                  <div className="flex items-center gap-1.5">
                    <span className="truncate">{f.name}</span>
                    <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-disabled)]">
                      {f.type}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2.5 align-top">
                  {v == null ? (
                    <span className="text-[var(--color-text-subtle)] italic">null</span>
                  ) : isVectorField(f) && Array.isArray(v) ? (
                    <span className="inline-flex items-center gap-2">
                      <Sparkline values={(v as number[]).slice(0, 128)} width={180} height={20} />
                      <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
                        {(v as number[]).length}d
                      </span>
                      <Button
                        size="xs"
                        variant="ghost"
                        onClick={() => {
                          navigator.clipboard.writeText(JSON.stringify(v))
                        }}
                      >
                        Copy
                      </Button>
                    </span>
                  ) : typeof v === 'string' ? (
                    <div className="text-[var(--color-text)] whitespace-pre-wrap break-words font-sans">
                      {v}
                    </div>
                  ) : typeof v === 'object' ? (
                    <pre className="font-mono text-[11.5px] text-[var(--color-text-muted)] bg-[var(--color-bg-elevated)] rounded p-2 overflow-auto">
                      {JSON.stringify(v, null, 2)}
                    </pre>
                  ) : (
                    <span className="font-mono tabular-nums text-[var(--color-text)]">
                      {String(v)}
                    </span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
