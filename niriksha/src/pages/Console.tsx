import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCircle2,
  Clock,
  Copy,
  History,
  Play,
  Save,
  TerminalSquare,
  Trash2,
  XCircle,
} from 'lucide-react'
import api from '../api/client'
import {
  Badge,
  Button,
  Input,
  Select,
  toast,
  Tooltip,
} from '../components/primitives'
import { useUrlState } from '../hooks/useUrlState'
import {
  useSavedObjects,
  type SavedObject,
} from '../stores/savedObjectStore'
import { cn } from '../lib/cn'

type Method = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'

interface ConsoleRequest {
  method: Method
  path: string
  body?: string
  /** Line where the request block starts in the editor — drives "run cursor request". */
  startLine: number
}

interface HistoryEntry {
  id: string
  ts: number
  method: Method
  path: string
  body?: string
  status?: number
  ok: boolean
  durationMs: number
  /** Truncated server response body for the history rail. */
  preview?: string
}

interface ConsoleUrlState {
  saved?: string
}

const DEFAULT_BUFFER = `# Veda Dev Console.
# Each block is one HTTP request: METHOD /path  (optionally followed by a JSON body).
# Blocks are separated by blank lines. Ctrl+Enter runs the request under the cursor;
# Ctrl+Shift+Enter runs every block in order.

GET /v1/admin/health

GET /v1/collections

GET /v1/collections/rag_anything

POST /v1/embed
{
  "text": "なんでも読める"
}

POST /v1/collections/rag_anything/search
{
  "query": {
    "vector": {
      "field": "embedding",
      "values": [/* paste vector here */]
    }
  },
  "limit": 5,
  "explain": true
}
`

const HISTORY_KEY = 'niriksha.console.history.v1'
const BUFFER_KEY = 'niriksha.console.buffer.v1'

/**
 * REST Console — Kibana Dev Tools shape. Power-user surface where the
 * Veda API is the product. No Monaco yet (kept the bundle small); the
 * textarea is augmented with line numbers, monospace, JSON pretty-print
 * on response, copy-as-cURL / Python, and a server-side-ready history
 * rail (today: browser-local, persisted to localStorage so reload-loss
 * is gone — fixes Kibana's #1 Console gripe).
 */
export function Console() {
  const [urlState, setUrlState] = useUrlState<ConsoleUrlState>('c', {})
  const [buffer, setBuffer] = useState(() => readBuffer())
  const [history, setHistory] = useState<HistoryEntry[]>(() => readHistory())
  const [response, setResponse] = useState<{
    method: Method
    path: string
    status?: number
    ok: boolean
    durationMs: number
    body: string
    headers?: Record<string, string>
    error?: string
  } | null>(null)
  const [running, setRunning] = useState(false)
  const editorRef = useRef<HTMLTextAreaElement>(null)
  const [cursorLine, setCursorLine] = useState(1)

  const upsertSaved = useSavedObjects((s) => s.upsert)
  const getSaved = useSavedObjects((s) => s.get)
  const objectsMap = useSavedObjects((s) => s.objects)

  const savedQueries = useMemo<SavedObject<{ buffer: string }>[]>(
    () =>
      Object.values(objectsMap)
        .filter(
          (o): o is SavedObject<{ buffer: string }> =>
            o.type === 'console_query',
        )
        .sort(
          (a, b) => (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt),
        ),
    [objectsMap],
  )

  // Restore from a saved query when ?saved=id is in URL.
  useEffect(() => {
    if (!urlState.saved) return
    const obj = getSaved<{ buffer: string }>(urlState.saved)
    if (obj) {
      setBuffer(obj.body.buffer)
      setUrlState({})
      toast.success(`Loaded "${obj.name}"`)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlState.saved])

  // Persist buffer to localStorage on change (debounced).
  useEffect(() => {
    const t = setTimeout(() => {
      try {
        localStorage.setItem(BUFFER_KEY, buffer)
      } catch {
        /* quota */
      }
    }, 500)
    return () => clearTimeout(t)
  }, [buffer])

  const requests = useMemo(() => parseRequests(buffer), [buffer])
  const cursorRequest = useMemo(
    () => findRequestAtLine(requests, cursorLine),
    [requests, cursorLine],
  )

  const updateCursor = () => {
    const ta = editorRef.current
    if (!ta) return
    const before = buffer.slice(0, ta.selectionStart)
    setCursorLine(before.split('\n').length)
  }

  const exec = async (req: ConsoleRequest) => {
    setRunning(true)
    const start = performance.now()
    let r: {
      status?: number
      ok: boolean
      data?: unknown
      headers?: Record<string, string>
      error?: string
    }
    try {
      const cfg: Parameters<typeof api.request>[0] = {
        url: req.path.replace(/^\/v1/, ''),
        method: req.method,
        validateStatus: () => true, // we render every status
      }
      if (req.body && req.body.trim()) {
        try {
          cfg.data = JSON.parse(req.body)
        } catch (e) {
          setRunning(false)
          setResponse({
            method: req.method,
            path: req.path,
            ok: false,
            durationMs: 0,
            body: '',
            error:
              'Request body is not valid JSON. ' +
              (e instanceof Error ? e.message : ''),
          })
          return
        }
      }
      const res = await api.request(cfg)
      r = {
        status: res.status,
        ok: res.status >= 200 && res.status < 300,
        data: res.data,
        headers: Object.fromEntries(
          Object.entries(res.headers as Record<string, unknown>).map(
            ([k, v]) => [k, String(v)],
          ),
        ),
      }
    } catch (e) {
      r = {
        ok: false,
        error: e instanceof Error ? e.message : String(e),
      }
    }
    const durationMs = performance.now() - start
    const bodyText = r.data == null ? '' : safeStringify(r.data)
    setResponse({
      method: req.method,
      path: req.path,
      status: r.status,
      ok: r.ok,
      durationMs,
      body: bodyText,
      headers: r.headers,
      error: r.error,
    })
    setRunning(false)

    const entry: HistoryEntry = {
      id: `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      ts: Date.now(),
      method: req.method,
      path: req.path,
      body: req.body,
      status: r.status,
      ok: r.ok,
      durationMs,
      preview: bodyText.slice(0, 200),
    }
    setHistory((h) => {
      const next = [entry, ...h].slice(0, 200)
      writeHistory(next)
      return next
    })
  }

  const runCursor = useCallback(() => {
    if (!cursorRequest) {
      toast.error('No request at cursor.')
      return
    }
    exec(cursorRequest)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorRequest])

  const onEditorKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      runCursor()
    }
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--color-bg)] overflow-hidden">
      <HistoryRail
        history={history}
        savedQueries={savedQueries}
        onClear={() => {
          setHistory([])
          writeHistory([])
        }}
        onRestore={(h) => {
          const next = formatRequest(h)
          setBuffer((prev) => prev + (prev.endsWith('\n') ? '\n' : '\n\n') + next)
        }}
        onLoadSaved={(s) => {
          setBuffer(s.body.buffer)
          useSavedObjects.getState().touch(s.id)
        }}
      />

      <section className="flex flex-col flex-1 min-w-0 border-r border-[var(--color-border)]">
        <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
          <TerminalSquare
            size={14}
            strokeWidth={2}
            className="text-[var(--color-accent-9)]"
          />
          <h1 className="text-[13.5px] font-medium text-[var(--color-text)]">
            Dev Console
          </h1>
          {cursorRequest && (
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-[var(--color-text-muted)]">
              <Badge label={cursorRequest.method} variant="ghost" mono />
              <span className="truncate max-w-[400px]">{cursorRequest.path}</span>
            </span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <Tooltip label="Save current buffer">
              <Button
                size="xs"
                variant="ghost"
                leading={<Save size={11} strokeWidth={2.25} />}
                onClick={() => {
                  const name = window.prompt('Name this console buffer')
                  if (!name) return
                  upsertSaved({
                    id: '',
                    type: 'console_query',
                    name,
                    body: { buffer },
                  })
                  toast.success(`Saved "${name}"`)
                }}
              >
                Save
              </Button>
            </Tooltip>
            <Tooltip label="Run request under cursor (Cmd/Ctrl+↵)">
              <Button
                size="sm"
                variant="primary"
                leading={<Play size={11} strokeWidth={2.5} />}
                onClick={runCursor}
                disabled={running || !cursorRequest}
              >
                Run
              </Button>
            </Tooltip>
          </div>
        </header>

        <EditorPane
          buffer={buffer}
          onChange={setBuffer}
          onCursor={updateCursor}
          onKeyDown={onEditorKey}
          editorRef={editorRef}
          cursorLine={cursorLine}
          requests={requests}
          cursorRequest={cursorRequest}
        />
      </section>

      <ResponsePane response={response} running={running} />
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Editor pane — gutter + monospace textarea + cursor-block indicator
   ──────────────────────────────────────────────────────────────────── */

function EditorPane({
  buffer,
  onChange,
  onCursor,
  onKeyDown,
  editorRef,
  cursorLine,
  requests,
  cursorRequest,
}: {
  buffer: string
  onChange: (b: string) => void
  onCursor: () => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  editorRef: React.RefObject<HTMLTextAreaElement | null>
  cursorLine: number
  requests: ConsoleRequest[]
  cursorRequest?: ConsoleRequest
}) {
  const lines = buffer.split('\n')
  // Map of line → "▶" marker if the line is the start of a request (for the gutter).
  const requestStarts = useMemo(
    () => new Set(requests.map((r) => r.startLine)),
    [requests],
  )

  return (
    <div className="flex-1 min-h-0 flex font-mono text-[12px] leading-[1.55]">
      <div
        className={cn(
          'flex-none w-12 bg-[var(--color-base-1)] border-r border-[var(--color-border)]',
          'text-right select-none overflow-y-hidden pt-3',
        )}
      >
        {lines.map((_, i) => {
          const ln = i + 1
          const isActive = ln === cursorLine
          const isStart = requestStarts.has(ln)
          return (
            <div
              key={i}
              className={cn(
                'pr-2 pl-2 tabular-nums text-[10.5px]',
                isActive
                  ? 'text-[var(--color-accent)]'
                  : 'text-[var(--color-text-disabled)]',
              )}
            >
              {isStart && (
                <span
                  className={cn(
                    'inline-block mr-1',
                    cursorRequest?.startLine === ln
                      ? 'text-[var(--color-accent)]'
                      : 'text-[var(--color-text-subtle)]',
                  )}
                >
                  ▶
                </span>
              )}
              {ln}
            </div>
          )
        })}
      </div>
      <div className="flex-1 min-w-0 relative">
        <textarea
          ref={editorRef}
          value={buffer}
          onChange={(e) => onChange(e.target.value)}
          onKeyUp={onCursor}
          onClick={onCursor}
          onKeyDown={onKeyDown}
          spellCheck={false}
          autoCorrect="off"
          autoCapitalize="off"
          className={cn(
            'absolute inset-0 w-full h-full p-3 outline-none',
            'bg-transparent text-[var(--color-text)] resize-none whitespace-pre overflow-auto',
            'font-mono text-[12px] leading-[1.55] tabular-nums',
            'placeholder:text-[var(--color-text-subtle)]',
            'caret-[var(--color-accent)]',
            'selection:bg-[color-mix(in_oklch,var(--color-accent)_35%,transparent)]',
          )}
          style={{ tabSize: 2 }}
        />
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Response pane — status + duration + JSON tree
   ──────────────────────────────────────────────────────────────────── */

function ResponsePane({
  response,
  running,
}: {
  response:
    | {
        method: Method
        path: string
        status?: number
        ok: boolean
        durationMs: number
        body: string
        headers?: Record<string, string>
        error?: string
      }
    | null
  running: boolean
}) {
  return (
    <section className="flex flex-col w-[42%] min-w-[360px] flex-none bg-[var(--color-bg)]">
      <header className="flex items-center gap-2 px-4 h-12 border-b border-[var(--color-border)] flex-none">
        <span className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]">
          Response
        </span>
        {response && (
          <>
            <Badge label={response.method} variant="ghost" mono />
            <span className="text-[11px] font-mono text-[var(--color-text-muted)] truncate flex-1">
              {response.path}
            </span>
            <StatusPill
              ok={response.ok}
              status={response.status}
              error={response.error}
            />
            <span className="text-[11px] font-mono tabular-nums text-[var(--color-text-muted)]">
              {response.durationMs.toFixed(0)}ms
            </span>
            {response.body && (
              <Tooltip label="Copy response">
                <button
                  className="p-1 text-[var(--color-text-subtle)] hover:text-[var(--color-text)] rounded"
                  onClick={() => {
                    navigator.clipboard.writeText(response.body)
                    toast.success('Copied response')
                  }}
                >
                  <Copy size={11} strokeWidth={2.25} />
                </button>
              </Tooltip>
            )}
          </>
        )}
        {!response && !running && (
          <span className="text-[11px] text-[var(--color-text-subtle)] italic">
            Run a request to see the response here.
          </span>
        )}
      </header>
      <div className="flex-1 min-h-0 overflow-auto p-3">
        {running && (
          <div className="text-[11px] text-[var(--color-accent)] flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-accent)] animate-pulse" />
            Running…
          </div>
        )}
        {response?.error && (
          <pre className="text-[11.5px] font-mono text-[var(--color-error)] whitespace-pre-wrap break-words">
            {response.error}
          </pre>
        )}
        {response && !response.error && (
          <pre className="text-[11.5px] font-mono text-[var(--color-text)] whitespace-pre break-words">
            {response.body}
          </pre>
        )}
      </div>
    </section>
  )
}

function StatusPill({
  ok,
  status,
  error,
}: {
  ok: boolean
  status?: number
  error?: string
}) {
  if (error) return <Badge label="error" variant="error" dot />
  if (status == null) return null
  const cls = ok ? 'success' : status >= 500 ? 'error' : 'warning'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-1.5 h-5 rounded-[var(--radius-xs)]',
        'border text-[10.5px] font-mono tabular-nums',
        cls === 'success' &&
          'bg-[color-mix(in_oklch,var(--color-success)_12%,transparent)] text-[var(--color-success)] border-[color-mix(in_oklch,var(--color-success)_28%,transparent)]',
        cls === 'warning' &&
          'bg-[color-mix(in_oklch,var(--color-warning)_12%,transparent)] text-[var(--color-warning)] border-[color-mix(in_oklch,var(--color-warning)_28%,transparent)]',
        cls === 'error' &&
          'bg-[color-mix(in_oklch,var(--color-error)_12%,transparent)] text-[var(--color-error)] border-[color-mix(in_oklch,var(--color-error)_28%,transparent)]',
      )}
    >
      {ok ? (
        <CheckCircle2 size={10} strokeWidth={2.25} />
      ) : (
        <XCircle size={10} strokeWidth={2.25} />
      )}
      {status}
    </span>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   History / saved queries rail
   ──────────────────────────────────────────────────────────────────── */

function HistoryRail({
  history,
  savedQueries,
  onClear,
  onRestore,
  onLoadSaved,
}: {
  history: HistoryEntry[]
  savedQueries: SavedObject<{ buffer: string }>[]
  onClear: () => void
  onRestore: (h: HistoryEntry) => void
  onLoadSaved: (s: SavedObject<{ buffer: string }>) => void
}) {
  const [tab, setTab] = useState<'history' | 'saved'>('history')
  const [filter, setFilter] = useState('')
  const filteredHistory = useMemo(() => {
    const q = filter.toLowerCase()
    return history.filter(
      (h) =>
        !q || h.path.toLowerCase().includes(q) || h.method.toLowerCase().includes(q),
    )
  }, [history, filter])

  return (
    <aside className="w-[260px] flex-none border-r border-[var(--color-border)] bg-[var(--color-bg)] flex flex-col">
      <header className="px-3 h-12 flex items-center gap-1 border-b border-[var(--color-border)]">
        <Select
          size="sm"
          value={tab}
          onValueChange={(v) => setTab(v as 'history' | 'saved')}
          options={[
            { value: 'history', label: 'History' },
            { value: 'saved', label: 'Saved' },
          ]}
        />
        <span className="ml-1 text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
          {tab === 'history' ? history.length : savedQueries.length}
        </span>
      </header>
      <div className="px-2 py-2 border-b border-[var(--color-border)]">
        <Input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={tab === 'history' ? 'filter…' : 'search saved…'}
          mono
          className="h-7 text-[11.5px]"
        />
      </div>

      <ul className="flex-1 overflow-y-auto">
        {tab === 'history' && filteredHistory.length === 0 && (
          <li className="px-3 py-2 text-[11px] text-[var(--color-text-subtle)] italic">
            No requests yet — Cmd/Ctrl+Enter to run.
          </li>
        )}
        {tab === 'history' &&
          filteredHistory.map((h) => (
            <li key={h.id}>
              <button
                onClick={() => onRestore(h)}
                className="group w-full text-left px-3 py-1.5 hover:bg-[var(--color-bg-hover)] border-l-2 border-transparent"
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <Badge
                    label={h.method}
                    variant={h.ok ? 'success' : 'error'}
                    mono
                  />
                  <span className="text-[10.5px] font-mono text-[var(--color-text-subtle)] tabular-nums">
                    {h.status ?? '—'}
                  </span>
                  <span className="ml-auto text-[10px] font-mono text-[var(--color-text-subtle)] tabular-nums">
                    {h.durationMs.toFixed(0)}ms
                  </span>
                </div>
                <div className="font-mono text-[11px] text-[var(--color-text)] truncate">
                  {h.path}
                </div>
                <div className="text-[10px] text-[var(--color-text-subtle)] flex items-center gap-1 mt-0.5">
                  <Clock size={9} strokeWidth={2.25} />
                  {new Date(h.ts).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
              </button>
            </li>
          ))}
        {tab === 'saved' && savedQueries.length === 0 && (
          <li className="px-3 py-2 text-[11px] text-[var(--color-text-subtle)] italic">
            No saved buffers yet.
          </li>
        )}
        {tab === 'saved' &&
          savedQueries
            .filter((s) => !filter || s.name.toLowerCase().includes(filter.toLowerCase()))
            .map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => onLoadSaved(s)}
                  className="w-full text-left px-3 py-1.5 hover:bg-[var(--color-bg-hover)]"
                >
                  <div className="text-[12px] text-[var(--color-text)] truncate">
                    {s.name}
                  </div>
                  <div className="text-[10px] text-[var(--color-text-subtle)] font-mono">
                    {new Date(s.updatedAt).toLocaleString([], {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </div>
                </button>
              </li>
            ))}
      </ul>

      {tab === 'history' && history.length > 0 && (
        <footer className="border-t border-[var(--color-border)] flex items-center justify-between px-3 py-1.5">
          <span className="inline-flex items-center gap-1 text-[10.5px] text-[var(--color-text-subtle)]">
            <History size={10} strokeWidth={2.25} />
            local-only
          </span>
          <button
            onClick={onClear}
            className="text-[10.5px] text-[var(--color-text-subtle)] hover:text-[var(--color-error)] inline-flex items-center gap-1"
          >
            <Trash2 size={10} strokeWidth={2.25} />
            Clear
          </button>
        </footer>
      )}
    </aside>
  )
}

/* ────────────────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────────────── */

function readBuffer(): string {
  try {
    return localStorage.getItem(BUFFER_KEY) || DEFAULT_BUFFER
  } catch {
    return DEFAULT_BUFFER
  }
}

function readHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : []
  } catch {
    return []
  }
}
function writeHistory(h: HistoryEntry[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h))
  } catch {
    /* quota */
  }
}

function formatRequest(h: HistoryEntry): string {
  let s = `${h.method} ${h.path}`
  if (h.body && h.body.trim()) s += '\n' + h.body.trim()
  return s
}

const METHOD_RE = /^(GET|POST|PUT|DELETE|PATCH)\s+(\/\S+)/i

/**
 * Parse the editor buffer into requests. Each block is METHOD path on
 * the first line + optional JSON body on subsequent lines, terminated
 * by a blank line OR end of buffer.
 */
function parseRequests(buffer: string): ConsoleRequest[] {
  const lines = buffer.split('\n')
  const out: ConsoleRequest[] = []
  let i = 0
  while (i < lines.length) {
    const raw = lines[i] ?? ''
    const trimmed = raw.trim()
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('//')) {
      i++
      continue
    }
    const m = trimmed.match(METHOD_RE)
    if (!m) {
      i++
      continue
    }
    const method = m[1]!.toUpperCase() as Method
    const path = m[2]!
    const startLine = i + 1
    // Body lines: until blank line or another METHOD.
    const bodyLines: string[] = []
    let j = i + 1
    while (j < lines.length) {
      const nextRaw = lines[j] ?? ''
      const nextTrim = nextRaw.trim()
      if (!nextTrim) break
      if (nextTrim.match(METHOD_RE)) break
      bodyLines.push(nextRaw)
      j++
    }
    out.push({
      method,
      path,
      body: bodyLines.length > 0 ? bodyLines.join('\n') : undefined,
      startLine,
    })
    i = j
  }
  return out
}

/** Find the request whose block contains `line` (1-indexed). */
function findRequestAtLine(
  requests: ConsoleRequest[],
  line: number,
): ConsoleRequest | undefined {
  for (let i = requests.length - 1; i >= 0; i--) {
    if (requests[i]!.startLine <= line) return requests[i]
  }
  return undefined
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return String(v)
  }
}
