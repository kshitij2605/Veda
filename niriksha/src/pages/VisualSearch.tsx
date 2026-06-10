import { type FormEvent, useEffect, useRef, useState } from 'react'
import { Search, Upload } from 'lucide-react'
import { Button, Card, Input, PageHeader } from '../components/primitives'
import {
  embedQuery,
  fetchCollections,
  searchCollection,
} from '../api/collections'
import type { CollectionSummary, SearchHit } from '../api/collections'

const CLIP_MODEL = 'clip-vit-base-patch32'
const VECTOR_FIELD = 'image_vec'

/**
 * Cross-modal visual search. A text query (or an uploaded image) is embedded
 * by CLIP into the shared text+image space, then matched against a collection's
 * image vectors — so "a mountain landscape" finds mountain photos.
 */
export function VisualSearch() {
  const [collection, setCollection] = useState('photos')
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [text, setText] = useState('')
  const [results, setResults] = useState<SearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tookMs, setTookMs] = useState<number | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchCollections()
      .then(({ data }) => setCollections(data.collections))
      .catch(() => {})
  }, [])

  async function runSearch(values: number[]) {
    const { data } = await searchCollection(collection, {
      query: { vector: { field: VECTOR_FIELD, values } },
      limit: 18,
      explain: true,
    })
    setResults(data.results)
    setTookMs(data.took_ms)
  }

  const onText = async (e: FormEvent) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setTookMs(null)
    try {
      const { data } = await embedQuery({ text: text.trim(), model: CLIP_MODEL })
      await runSearch(data.values)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setLoading(false)
    }
  }

  const onImage = async (file: File) => {
    setLoading(true)
    setError(null)
    setTookMs(null)
    setText('')
    try {
      const b64 = await fileToBase64(file)
      const { data } = await embedQuery({ image_b64: b64, model: CLIP_MODEL })
      await runSearch(data.values)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="Visual search"
        description="Search images by meaning — type a description or drop an image. Powered by CLIP in a shared text + image space."
      />

      <form onSubmit={onText} className="flex items-center gap-2 max-w-2xl">
        <Input
          leading={<Search size={13} strokeWidth={2.25} />}
          placeholder='Try "a mountain landscape", "city at night", "something blue"…'
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1"
        />
        <Button type="submit" variant="primary" disabled={loading || !text.trim()}>
          {loading ? 'Searching…' : 'Search'}
        </Button>
        <Button
          type="button"
          variant="secondary"
          leading={<Upload size={13} strokeWidth={2.25} />}
          onClick={() => fileRef.current?.click()}
        >
          Image
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void onImage(f)
          }}
        />
      </form>

      <div className="mt-2 flex items-center gap-1.5 text-[11px] text-[var(--color-text-subtle)]">
        <span>in</span>
        <select
          value={collection}
          onChange={(e) => setCollection(e.target.value)}
          className="bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-[var(--radius-xs)] px-1.5 py-0.5 text-[var(--color-text)]"
        >
          {collections.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
        {tookMs != null && (
          <span className="ml-2 tabular-nums">
            · {results.length} results in {tookMs.toFixed(1)} ms
          </span>
        )}
      </div>

      {error && <div className="mt-4 text-[12px] text-[var(--color-error)]">{error}</div>}

      <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {results.map((r) => (
          <ResultCard key={r.id} hit={r} />
        ))}
      </div>

      {!loading && results.length === 0 && !error && (
        <div className="mt-12 text-center text-[13px] text-[var(--color-text-muted)]">
          Type a description or drop an image to search across the embedding space.
        </div>
      )}
    </div>
  )
}

function ResultCard({ hit }: { hit: SearchHit }) {
  const img = typeof hit.fields.image === 'string' ? (hit.fields.image as string) : null
  const caption =
    typeof hit.fields.caption === 'string' ? (hit.fields.caption as string) : hit.id
  const sim = Math.max(0, Math.min(100, (1 - hit.distance) * 100))
  return (
    <Card className="overflow-hidden group">
      <div className="aspect-square bg-[var(--color-bg-overlay)] overflow-hidden">
        {img ? (
          <img
            src={`data:image/jpeg;base64,${img}`}
            alt={caption}
            loading="lazy"
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
        ) : (
          <div className="h-full w-full flex items-center justify-center text-[11px] text-[var(--color-text-subtle)]">
            no image
          </div>
        )}
      </div>
      <div className="p-2">
        <div className="text-[11px] text-[var(--color-text)] truncate" title={caption}>
          {caption}
        </div>
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="flex-1 h-1 rounded-full bg-[var(--color-bg-active)] overflow-hidden">
            <div className="h-full bg-[var(--color-accent)]" style={{ width: `${sim}%` }} />
          </div>
          <span className="text-[9.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
            {sim.toFixed(0)}
          </span>
        </div>
      </div>
    </Card>
  )
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const r = reader.result as string
      resolve(r.split(',')[1] ?? '')
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function errMsg(e: unknown): string {
  const ax = e as { response?: { data?: { error?: string } } }
  return ax?.response?.data?.error ?? (e instanceof Error ? e.message : 'search failed')
}
