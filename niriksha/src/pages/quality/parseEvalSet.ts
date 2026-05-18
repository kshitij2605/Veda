import type { EvalItem } from './types'

/**
 * Parse uploaded eval-set content. Accepts:
 *
 *   - **JSON**: top-level array of items
 *     `[{ "query": "...", "expected_doc_ids": ["..."], ... }]`
 *   - **JSONL**: one JSON object per line
 *   - **CSV/TSV**: header row with `query` (required) and optional
 *     `expected_doc_ids` column. Multi-value cells are split by `|`
 *     (or `,` if the column is JSON-wrapped `[...]`).
 *
 * Throws on completely-unparseable input. Skips empty rows quietly.
 */
export function parseEvalSet(text: string, filename = ''): EvalItem[] {
  const trimmed = text.trim()
  if (!trimmed) return []
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''

  // JSON / JSONL detection — JSON files start with [ or {, JSONL with {.
  if (trimmed.startsWith('[')) return parseJson(trimmed)
  if (trimmed.startsWith('{') || ext === 'jsonl' || ext === 'ndjson') {
    return parseJsonl(trimmed)
  }

  // Fall through to CSV / TSV.
  return parseDelimited(trimmed)
}

function parseJson(text: string): EvalItem[] {
  const data = JSON.parse(text) as unknown
  if (!Array.isArray(data)) throw new Error('JSON must be an array of items.')
  return data.map(normalize)
}

function parseJsonl(text: string): EvalItem[] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => normalize(JSON.parse(l) as unknown))
}

function parseDelimited(text: string): EvalItem[] {
  const sep = text.split('\n', 1)[0]!.includes('\t') ? '\t' : ','
  const lines = text.split('\n').map((l) => l.replace(/\r$/, ''))
  // Skip BOM if present.
  if (lines[0]) lines[0] = lines[0].replace(/^﻿/, '')
  if (lines.length === 0) return []
  const header = parseRow(lines[0]!, sep).map((h) => h.toLowerCase())
  const qIdx = header.findIndex((h) => h === 'query' || h === 'question')
  if (qIdx < 0)
    throw new Error("CSV must have a 'query' (or 'question') column.")
  const idsIdx = header.findIndex(
    (h) =>
      h === 'expected_doc_ids' ||
      h === 'expected' ||
      h === 'gold' ||
      h === 'doc_ids',
  )

  const out: EvalItem[] = []
  for (let i = 1; i < lines.length; i++) {
    const raw = lines[i]
    if (!raw || !raw.trim()) continue
    const cells = parseRow(raw, sep)
    const query = (cells[qIdx] ?? '').trim()
    if (!query) continue
    const item: EvalItem = { query }
    if (idsIdx >= 0) {
      const cell = (cells[idsIdx] ?? '').trim()
      if (cell) item.expected_doc_ids = parseIdList(cell)
    }
    out.push(item)
  }
  return out
}

/** Minimal RFC 4180-ish row parser — supports quoted fields with escaped quotes. */
function parseRow(row: string, sep: string): string[] {
  const out: string[] = []
  let i = 0
  while (i < row.length) {
    let cell = ''
    if (row[i] === '"') {
      i++
      while (i < row.length) {
        if (row[i] === '"' && row[i + 1] === '"') {
          cell += '"'
          i += 2
        } else if (row[i] === '"') {
          i++
          break
        } else {
          cell += row[i]
          i++
        }
      }
    } else {
      while (i < row.length && row[i] !== sep) {
        cell += row[i]
        i++
      }
    }
    out.push(cell)
    if (row[i] === sep) i++
  }
  return out
}

function parseIdList(cell: string): string[] {
  if (cell.startsWith('[')) {
    try {
      const parsed = JSON.parse(cell) as unknown
      if (Array.isArray(parsed)) return parsed.map(String)
    } catch {
      // fall through
    }
  }
  return cell
    .split('|')
    .map((s) => s.trim())
    .filter(Boolean)
}

function normalize(raw: unknown): EvalItem {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Each item must be an object with a `query` field.')
  }
  const o = raw as Record<string, unknown>
  const query = String(o.query ?? o.question ?? '').trim()
  if (!query) throw new Error('Item is missing a `query` field.')
  const exp = (o.expected_doc_ids ?? o.expected ?? o.gold ?? o.doc_ids) as unknown
  let expected_doc_ids: string[] | undefined
  if (Array.isArray(exp)) expected_doc_ids = exp.map(String)
  else if (typeof exp === 'string' && exp) expected_doc_ids = parseIdList(exp)
  const item: EvalItem = { query }
  if (expected_doc_ids) item.expected_doc_ids = expected_doc_ids
  if (o.meta && typeof o.meta === 'object') {
    item.meta = o.meta as EvalItem['meta']
  }
  return item
}
