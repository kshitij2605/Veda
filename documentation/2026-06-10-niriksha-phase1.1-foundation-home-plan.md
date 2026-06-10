# Niriksha Phase 1.1 — Foundation Tokens + Dashboard Home Showcase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a premium, verified Dashboard Home for Niriksha running on the evolved OKLCH token system with a regrouped nav, backed by real seeded data on gpu-host-a.

**Architecture:** Evolve the existing Tailwind-4 `@theme` tokens in `niriksha/src/styles/global.css` (no rip-out); add a small set of token-driven primitives + two re-tokenized reactbits components (CountUp, SpotlightCard) + a Recharts donut; rebuild `DashboardHome` to compose them against the existing Zustand stores and the live REST API. Stand up a dev `veda-server` + seed multimodal-shaped collections so the page shows real data. Verify with Vitest unit tests, a green build/lint, and Playwright screenshots.

**Tech Stack:** React 19, Vite 8, Tailwind 4 (CSS-first `@theme`), Zustand, framer-motion, Recharts (new), Vitest + Testing Library (new), reactbits (vendored via MCP), Playwright (MCP, verification).

**Environment (verified 2026-06-10):**
- Remote: `gpu-host-a.internal.example:/home/mac/test/project-mdb`. Edit locally → `scp` → run over SSH.
- `veda-server` release binary at `target/release/veda-server`; runs standalone (local blob store, real BGE-M3 text embed endpoint). Already running on `./veda-data-dev` at `0.0.0.0:6340`.
- node via nvm: `export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 24.3.0` → node v24.3.0, npm 11.4.2. `niriksha/node_modules` present.
- Niriksha dev server: `npm run dev -- --host 0.0.0.0 --port 5174` → `http://gpu-host-a.internal.example:5174/niriksha` (Vite host-allowlist already set; proxies `/v1` → 6340).

**Verified REST contract:**
- Create: `POST /v1/collections` `{name, fields:[{name,type,dimension,metric,filterable,model?}], shard_count}` → `{name,status:"created"}` (201).
- List: `GET /v1/collections` → `{collections:[{name,field_count,shard_count}]}`.
- Detail: `GET /v1/collections/{name}` → `{name, fields:[{name,type,filterable,dimension?,metric?}], shard_count, replication_factor}`.
- Insert: `POST /v1/collections/{name}/insert` `{records:[ {<vectorField>:[..], <scalar>:.., id?} ]}` (records are FLAT maps; id optional) → `{inserted:N}`.
- Search: `POST /v1/collections/{name}/search` `{query:{vector:{field,values}}, limit, explain?}` → `{results, count, took_ms, explain:{filter_ms,ann_search_ms,scoring_ms,fusion_ms,total_ms}}`.
- Records: `GET /v1/collections/{name}/records?limit&offset` → `{records:[{id,fields:{...}}], total}`.
- Health: `GET /v1/admin/health` → `{status,version,checks:{...}}`. Metrics: `GET /v1/admin/metrics` (Prometheus text; includes `veda_records_total{collection=...}`, `veda_active_collections`).

**Note on modality:** the schema does NOT tag a vector field's modality — type is just `vector`/`sparse_vector`/`multi_vector`. Modality is therefore derived heuristically from vector-field NAMES (e.g. `image_embedding`→image), with the `model` hint as secondary signal. The seed sets modality-indicative field names so derivation is deterministic.

---

## File map

| File | New? | Responsibility |
|---|---|---|
| `scripts/seed-niriksha-dev.py` | new (remote) | seed multimodal-shaped collections + synthetic records |
| `niriksha/vitest.config.ts` | new | Vitest + jsdom config |
| `niriksha/src/test/setup.ts` | new | Testing-Library matchers |
| `niriksha/src/styles/global.css` | modify | token additions (accent ramp, modality, chart, premium) |
| `niriksha/src/lib/modality.ts` | new | modality types, `modalityColorVar`, `deriveModality` |
| `niriksha/src/lib/format.ts` | new | `compactNumber`, `formatInt` |
| `niriksha/src/lib/metrics.ts` | new | `sumPromMetric` (Prometheus text → number) |
| `niriksha/src/lib/modalityMix.ts` | new | `deriveModalityMix(details)` → counts |
| `niriksha/src/components/primitives/Card.tsx` | new | surface panel |
| `niriksha/src/components/primitives/StatTile.tsx` | new | KPI tile (label/value/icon) |
| `niriksha/src/components/primitives/ModalityChip.tsx` | new | modality badge |
| `niriksha/src/components/primitives/ConnectionPill.tsx` | new | health/connection pill |
| `niriksha/src/components/primitives/EmptyState.tsx` | new (replaces `components/EmptyState.tsx`) | upgraded empty state |
| `niriksha/src/components/reactbits/CountUp.tsx` | new (vendored) | animated number |
| `niriksha/src/components/reactbits/SpotlightCard.tsx` | new (vendored) | cursor-spotlight card |
| `niriksha/src/components/charts/DonutChart.tsx` | new | themed Recharts donut |
| `niriksha/src/stores/collectionStore.ts` | modify | add `details` + `fetchAllDetails` |
| `niriksha/src/components/shell/Sidebar.tsx` | modify | regroup nav + add Home |
| `niriksha/src/App.tsx` | modify | default route `/` → `/home` |
| `niriksha/src/pages/DashboardHome.tsx` | rewrite | the showcase |

---

## Task 0: Environment up + seed real data

**Files:** Create `scripts/seed-niriksha-dev.py` (on remote).

- [ ] **Step 1: Ensure dev server running**

Run (SSH):
```bash
ssh gpu-host-a.internal.example 'cd /home/mac/test/project-mdb && curl -s -m3 http://localhost:6340/v1/admin/health || (setsid ./target/release/veda-server ./veda-data-dev 0.0.0.0:6340 >/tmp/veda-dev.log 2>&1 </dev/null & sleep 4; curl -s http://localhost:6340/v1/admin/health)'
```
Expected: `{"status":"healthy",...}`.

- [ ] **Step 2: Write the seed script**

Create `scripts/seed-niriksha-dev.py` (stdlib only — no pip deps):
```python
#!/usr/bin/env python3
"""Seed Niriksha dev data: multimodal-shaped collections + synthetic vectors."""
import json, random, urllib.request, urllib.error
BASE = "http://localhost:6340"
random.seed(7)

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

# (collection, [(field, dim, model)], [scalar fields], n_records)
COLLECTIONS = [
    ("articles",        [("text_embedding", 1024, "BAAI/bge-m3")],
        [("title","string",False),("category","string",True)], 42),
    ("product_images",  [("image_embedding", 768, "Qwen3-VL-Embedding-2B")],
        [("caption","string",False),("sku","string",True)], 36),
    ("voice_notes",     [("audio_embedding", 512, "laion/clap")],
        [("transcript","string",False)], 21),
    ("video_clips",     [("video_embedding", 1024, "videoclip")],
        [("title","string",False)], 15),
    ("contracts",       [("doc_embedding", 1024, "vidore/colpali")],
        [("filename","string",True)], 27),
    ("product_catalog", [("text_embedding",1024,"BAAI/bge-m3"),("image_embedding",768,"Qwen3-VL-Embedding-2B")],
        [("name","string",False),("price","float",True),("in_stock","bool",True)], 50),
]

def main():
    for name, vecs, scalars, n in COLLECTIONS:
        req("DELETE", f"/v1/collections/{name}")
        fields = [{"name": f, "type": "vector", "dimension": d, "metric": "cosine", "model": m} for f,d,m in vecs]
        for f,t,filt in scalars:
            fields.append({"name": f, "type": t, "filterable": filt})
        st, body = req("POST", "/v1/collections", {"name": name, "fields": fields, "shard_count": 1})
        print("create", name, st, body)
        recs = []
        for i in range(n):
            rec = {}
            for f,d,_ in vecs:
                rec[f] = [round(random.uniform(-1,1),4) for _ in range(d)]
            for f,t,_ in scalars:
                rec[f] = (f"{name}-{f}-{i}" if t=="string" else
                          round(random.uniform(1,999),2) if t=="float" else bool(i%2))
            recs.append(rec)
        st, body = req("POST", f"/v1/collections/{name}/insert", {"records": recs})
        print("insert", name, st, body)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the seed + verify**

```bash
scp scripts/seed-niriksha-dev.py gpu-host-a...:/home/mac/test/project-mdb/scripts/
ssh gpu-host-a... 'cd /home/mac/test/project-mdb && python3 scripts/seed-niriksha-dev.py && curl -s http://localhost:6340/v1/collections'
```
Expected: 6 `create ... 201 {'status':'created'}` + `insert ... {'inserted':N}`; list shows 6 collections. Confirm a detail echoes `model` (note for Step modality wiring; derivation falls back to field-name regardless): `curl -s http://localhost:6340/v1/collections/product_images`.

- [ ] **Step 4: Start the niriksha dev server (background)**

```bash
ssh gpu-host-a... 'cd /home/mac/test/project-mdb/niriksha && export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 24.3.0; setsid npm run dev -- --host 0.0.0.0 --port 5174 >/tmp/niriksha-dev.log 2>&1 </dev/null & sleep 6; tail -5 /tmp/niriksha-dev.log'
```
Expected: Vite ready on :5174.

- [ ] **Step 5: Playwright baseline screenshot (current Home)**

Navigate Playwright (local) to `http://gpu-host-a.internal.example:5174/niriksha/home` and screenshot to `baseline-home.png`. No commit (ops task).

---

## Task 1: Vitest test infrastructure

**Files:** Create `niriksha/vitest.config.ts`, `niriksha/src/test/setup.ts`; modify `niriksha/package.json`.

- [ ] **Step 1: Install dev deps** — `npm i -D vitest@^3 @testing-library/react@^16 @testing-library/jest-dom@^6 @testing-library/user-event@^14 jsdom@^25` (run in `niriksha/` with nvm active).

- [ ] **Step 2: Config** — `vitest.config.ts`:
```ts
/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  test: { environment: 'jsdom', globals: true, setupFiles: ['./src/test/setup.ts'], css: true },
})
```
`src/test/setup.ts`: `import '@testing-library/jest-dom'`

- [ ] **Step 3: Add scripts** to `package.json`: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 4: Sanity test** — `src/lib/__tests__/sanity.test.ts`: `test('math', () => { expect(1+1).toBe(2) })`. Run `npm test` → PASS.

- [ ] **Step 5: Commit** — `git add niriksha/vitest.config.ts niriksha/src/test/setup.ts niriksha/package.json niriksha/package-lock.json niriksha/src/lib/__tests__/sanity.test.ts && git commit -m "test(niriksha): add vitest + testing-library infra"`

---

## Task 2: Token system evolution + modality util

**Files:** Modify `niriksha/src/styles/global.css`; create `niriksha/src/lib/modality.ts` + test.

- [ ] **Step 1: Write the failing test** — `src/lib/__tests__/modality.test.ts`:
```ts
import { deriveModality, modalityColorVar } from '../modality'
test('image field name → image', () => {
  expect(deriveModality([{ name: 'image_embedding', type: 'vector' }])).toBe('image')
})
test('text default', () => {
  expect(deriveModality([{ name: 'text_embedding', type: 'vector' }])).toBe('text')
})
test('two distinct modalities → mixed', () => {
  expect(deriveModality([
    { name: 'text_embedding', type: 'vector' },
    { name: 'image_embedding', type: 'vector' },
  ])).toBe('mixed')
})
test('color var', () => {
  expect(modalityColorVar('audio')).toBe('var(--color-modality-audio)')
})
```
Run `npm test src/lib/__tests__/modality.test.ts` → FAIL (module missing).

- [ ] **Step 2: Implement `src/lib/modality.ts`**:
```ts
export type Modality = 'text' | 'image' | 'audio' | 'video' | 'doc' | 'mixed'
export interface SchemaField { name: string; type: string }
const VECTORY = new Set(['vector', 'sparse_vector', 'multi_vector'])
const RULES: [RegExp, Modality][] = [
  [/imag|img|clip|visual|photo|vision/i, 'image'],
  [/audio|sound|voice|speech|clap/i, 'audio'],
  [/video|motion/i, 'video'],
  [/doc|page|pdf|colpali|ocr/i, 'doc'],
  [/text|bge|e5|title|caption|body|content|embedding/i, 'text'],
]
function classify(name: string): Modality {
  for (const [re, m] of RULES) if (re.test(name)) return m
  return 'text'
}
export function deriveModality(fields: SchemaField[]): Modality {
  const found = new Set<Modality>()
  for (const f of fields) if (VECTORY.has(f.type)) found.add(classify(f.name))
  if (found.size === 0) return 'text'
  if (found.size === 1) return [...found][0]
  return 'mixed'
}
export function modalityColorVar(m: Modality): string { return `var(--color-modality-${m})` }
export const MODALITY_LABEL: Record<Modality, string> = {
  text: 'Text', image: 'Image', audio: 'Audio', video: 'Video', doc: 'Document', mixed: 'Multimodal',
}
```
Run test → PASS.

- [ ] **Step 3: Add tokens to `global.css`** (inside `@theme`, after the existing accent block). Fill the accent ramp and add modality + chart + premium tokens:
```css
  /* Accent ramp completion */
  --color-accent-1:  oklch(0.220 0.060 282);
  --color-accent-3:  oklch(0.320 0.130 282);
  --color-accent-6:  oklch(0.560 0.215 282);
  --color-accent-8:  oklch(0.690 0.195 282);
  --color-accent-10: oklch(0.830 0.120 282);
  --color-accent-12: oklch(0.960 0.030 282);

  /* Modality palette (categorical, colorblind-aware) */
  --color-modality-text:  oklch(0.70 0.15 250);
  --color-modality-image: oklch(0.72 0.18 320);
  --color-modality-audio: oklch(0.74 0.16 160);
  --color-modality-video: oklch(0.76 0.15 60);
  --color-modality-doc:   oklch(0.70 0.14 25);
  --color-modality-mixed: oklch(0.78 0.10 282);

  /* Categorical chart ramp (8) */
  --color-chart-1: oklch(0.70 0.13 195);
  --color-chart-2: oklch(0.72 0.16 282);
  --color-chart-3: oklch(0.74 0.16 160);
  --color-chart-4: oklch(0.76 0.15 60);
  --color-chart-5: oklch(0.70 0.17 320);
  --color-chart-6: oklch(0.72 0.14 30);
  --color-chart-7: oklch(0.74 0.12 230);
  --color-chart-8: oklch(0.80 0.13 110);

  /* Premium showcase surfaces (gated to hero/empty/login) */
  --glow-accent: radial-gradient(60% 60% at 50% 0%, oklch(0.62 0.22 282 / 0.18), transparent 70%);
  --gradient-hero: linear-gradient(135deg, oklch(0.62 0.22 282), oklch(0.66 0.21 320));
```

- [ ] **Step 4: Verify build** — `npm run build` → succeeds (tsc + vite). Run `npm test` → all green.

- [ ] **Step 5: Commit** — `git add niriksha/src/styles/global.css niriksha/src/lib/modality.ts niriksha/src/lib/__tests__/modality.test.ts && git commit -m "feat(niriksha): evolve tokens (accent ramp, modality, chart, premium) + modality util"`

---

## Task 3: Formatting + Prometheus utils

**Files:** Create `src/lib/format.ts`, `src/lib/metrics.ts` + tests.

- [ ] **Step 1: Failing tests** — `src/lib/__tests__/format.test.ts`:
```ts
import { compactNumber, formatInt } from '../format'
test('compact', () => { expect(compactNumber(3200000)).toBe('3.2M'); expect(compactNumber(950)).toBe('950') })
test('int', () => { expect(formatInt(12345)).toBe('12,345') })
```
`src/lib/__tests__/metrics.test.ts`:
```ts
import { sumPromMetric } from '../metrics'
const TXT = `# TYPE veda_records_total gauge
veda_records_total{collection="a"} 42
veda_records_total{collection="b"} 8`
test('sum', () => { expect(sumPromMetric(TXT, 'veda_records_total')).toBe(50) })
```
Run → FAIL.

- [ ] **Step 2: Implement** — `src/lib/format.ts`:
```ts
export function compactNumber(n: number): string {
  if (n < 1000) return String(n)
  const u = ['', 'K', 'M', 'B', 'T']; let i = 0; let v = n
  while (v >= 1000 && i < u.length - 1) { v /= 1000; i++ }
  return `${(Math.round(v * 10) / 10).toString()}${u[i]}`
}
export function formatInt(n: number): string { return n.toLocaleString('en-US') }
```
`src/lib/metrics.ts`:
```ts
export function sumPromMetric(text: string, metric: string): number {
  let sum = 0
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.startsWith(metric)) continue
    const head = line.slice(metric.length)
    if (head && !head.startsWith(' ') && !head.startsWith('{')) continue
    const v = parseFloat(line.trim().split(/\s+/).pop() || '')
    if (!Number.isNaN(v)) sum += v
  }
  return sum
}
```
Run → PASS.

- [ ] **Step 3: Commit** — `git add niriksha/src/lib/format.ts niriksha/src/lib/metrics.ts niriksha/src/lib/__tests__/format.test.ts niriksha/src/lib/__tests__/metrics.test.ts && git commit -m "feat(niriksha): compact/int formatters + prometheus sum util"`

---

## Task 4: Modality mix derivation

**Files:** Create `src/lib/modalityMix.ts` + test.

- [ ] **Step 1: Failing test** — `src/lib/__tests__/modalityMix.test.ts`:
```ts
import { deriveModalityMix } from '../modalityMix'
test('counts by modality', () => {
  const mix = deriveModalityMix([
    { name: 'a', fields: [{ name: 'text_embedding', type: 'vector' }] },
    { name: 'b', fields: [{ name: 'image_embedding', type: 'vector' }] },
    { name: 'c', fields: [{ name: 'text_embedding', type: 'vector' }, { name: 'image_embedding', type: 'vector' }] },
  ])
  expect(mix).toEqual([
    { modality: 'text', count: 1 },
    { modality: 'image', count: 1 },
    { modality: 'mixed', count: 1 },
  ])
})
```
Run → FAIL.

- [ ] **Step 2: Implement** — `src/lib/modalityMix.ts`:
```ts
import { deriveModality, type Modality, type SchemaField } from './modality'
export interface CollectionDetail { name: string; fields: SchemaField[] }
export interface ModalitySlice { modality: Modality; count: number }
const ORDER: Modality[] = ['text', 'image', 'audio', 'video', 'doc', 'mixed']
export function deriveModalityMix(details: CollectionDetail[]): ModalitySlice[] {
  const counts = new Map<Modality, number>()
  for (const d of details) {
    const m = deriveModality(d.fields)
    counts.set(m, (counts.get(m) ?? 0) + 1)
  }
  return ORDER.filter((m) => counts.has(m)).map((m) => ({ modality: m, count: counts.get(m)! }))
}
```
Run → PASS.

- [ ] **Step 3: Commit** — `git add niriksha/src/lib/modalityMix.ts niriksha/src/lib/__tests__/modalityMix.test.ts && git commit -m "feat(niriksha): derive modality mix from collection details"`

---

## Task 5: Primitives (Card, StatTile, ModalityChip, ConnectionPill, EmptyState)

**Files:** Create the five components under `src/components/primitives/`; add to `src/components/primitives/index.ts`; tests under `src/components/primitives/__tests__/`.

- [ ] **Step 1: Failing tests** — `__tests__/primitives.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { Card, StatTile, ModalityChip, ConnectionPill, EmptyState } from '../index'
test('Card renders children', () => { render(<Card>hi</Card>); expect(screen.getByText('hi')).toBeInTheDocument() })
test('StatTile shows label + value', () => {
  render(<StatTile label="Collections" value="12" />)
  expect(screen.getByText('Collections')).toBeInTheDocument(); expect(screen.getByText('12')).toBeInTheDocument()
})
test('ModalityChip labels modality', () => { render(<ModalityChip modality="image" />); expect(screen.getByText('Image')).toBeInTheDocument() })
test('ConnectionPill shows status', () => { render(<ConnectionPill ok label="connected" />); expect(screen.getByText('connected')).toBeInTheDocument() })
test('EmptyState shows title', () => { render(<EmptyState title="Nothing here" />); expect(screen.getByText('Nothing here')).toBeInTheDocument() })
```
Run → FAIL.

- [ ] **Step 2: Implement components** (token classes only — never raw oklch):

`Card.tsx`:
```tsx
import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
export function Card({ children, className, glow }: { children: ReactNode; className?: string; glow?: boolean }) {
  return (
    <div className={cn('rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]', glow && 'relative overflow-hidden', className)}>
      {glow && <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'var(--glow-accent)' }} />}
      <div className="relative">{children}</div>
    </div>
  )
}
```
`StatTile.tsx`:
```tsx
import { type ReactNode } from 'react'
export function StatTile({ label, value, icon, foot }: { label: string; value: ReactNode; icon?: ReactNode; foot?: ReactNode }) {
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 text-[var(--text-xs)] text-[var(--color-text-muted)] uppercase tracking-[0.06em]">
        {icon}<span>{label}</span>
      </div>
      <div className="mt-2 text-[var(--text-3xl)] font-semibold text-[var(--color-text)] tabular-nums">{value}</div>
      {foot && <div className="mt-1 text-[var(--text-xs)] text-[var(--color-text-subtle)]">{foot}</div>}
    </div>
  )
}
```
`ModalityChip.tsx`:
```tsx
import { type Modality, MODALITY_LABEL, modalityColorVar } from '../../lib/modality'
export function ModalityChip({ modality }: { modality: Modality }) {
  const c = modalityColorVar(modality)
  return (
    <span className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[var(--text-2xs)] font-medium border"
      style={{ color: c, borderColor: `color-mix(in oklch, ${c} 40%, transparent)`, background: `color-mix(in oklch, ${c} 12%, transparent)` }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />{MODALITY_LABEL[modality]}
    </span>
  )
}
```
`ConnectionPill.tsx`:
```tsx
export function ConnectionPill({ ok, label }: { ok: boolean; label: string }) {
  const c = ok ? 'var(--color-success)' : 'var(--color-error)'
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[var(--text-xs)] text-[var(--color-text-muted)] border border-[var(--color-border)]">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />{label}
    </span>
  )
}
```
`EmptyState.tsx`:
```tsx
import { type ReactNode } from 'react'
export function EmptyState({ title, description, action, icon }: { title: string; description?: string; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="dot-grid rounded-[var(--radius-lg)] border border-[var(--color-border)] flex flex-col items-center justify-center text-center py-12 px-6">
      {icon && <div className="mb-3 text-[var(--color-text-subtle)]">{icon}</div>}
      <div className="text-[var(--text-md)] font-medium text-[var(--color-text)]">{title}</div>
      {description && <div className="mt-1 text-[var(--text-sm)] text-[var(--color-text-muted)] max-w-sm">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
```

- [ ] **Step 3: Export** — add to `src/components/primitives/index.ts`: `export { Card } from './Card'` etc. for all five.

- [ ] **Step 4: Run** `npm test` → PASS; `npm run build` → succeeds. (Note: the legacy `src/components/EmptyState.tsx` stays until callers migrate; new code imports from `primitives`.)

- [ ] **Step 5: Commit** — `git add niriksha/src/components/primitives && git commit -m "feat(niriksha): Card, StatTile, ModalityChip, ConnectionPill, EmptyState primitives"`

---

## Task 6: Vendor reactbits CountUp

**Files:** Create `src/components/reactbits/CountUp.tsx`.

- [ ] **Step 1: Fetch source** — call the reactbits MCP `get_component` with `name: "count-up"`, `style: "tailwind"`, and `get_component_demo` for usage. Save the component into `src/components/reactbits/CountUp.tsx`.

- [ ] **Step 2: Re-tokenize** — CountUp animates a number and carries no theme colors; remove any hardcoded color/className that conflicts (it should inherit `currentColor`). Ensure props at minimum: `to: number`, optional `durationMs`. Keep framer-motion (already installed).

- [ ] **Step 3: Test** — `src/components/reactbits/__tests__/countup.test.tsx`: render `<CountUp to={42} />` and assert the container eventually shows `42` (use `findByText('42')` with a generous timeout; under reduced-motion it renders final value immediately).

- [ ] **Step 4: Run** `npm test` → PASS.

- [ ] **Step 5: Commit** — `git add niriksha/src/components/reactbits/CountUp.tsx niriksha/src/components/reactbits/__tests__/countup.test.tsx && git commit -m "feat(niriksha): vendor reactbits CountUp"`

---

## Task 7: Vendor reactbits SpotlightCard

**Files:** Create `src/components/reactbits/SpotlightCard.tsx`.

- [ ] **Step 1: Fetch source** — reactbits MCP `get_component` `name: "spotlight-card"`, `style: "tailwind"` (+ demo). Save to `src/components/reactbits/SpotlightCard.tsx`.

- [ ] **Step 2: Re-tokenize** — replace the hardcoded spotlight color with our accent: the radial highlight color becomes `color-mix(in oklch, var(--color-accent) 25%, transparent)`; the card surface uses `bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-[var(--radius-lg)]`. Keep the mouse-position spotlight effect. Props: `children`, optional `className`.

- [ ] **Step 3: Test** — render `<SpotlightCard>content</SpotlightCard>`; assert child text present (effect itself is not asserted).

- [ ] **Step 4: Run** `npm test` → PASS; `npm run build` → succeeds.

- [ ] **Step 5: Commit** — `git add niriksha/src/components/reactbits/SpotlightCard.tsx niriksha/src/components/reactbits/__tests__ && git commit -m "feat(niriksha): vendor + re-tokenize reactbits SpotlightCard"`

---

## Task 8: Themed Donut chart

**Files:** Create `src/components/charts/DonutChart.tsx` + test. Install `recharts`.

- [ ] **Step 1: Install** — `npm i recharts@^2` (in `niriksha/`, nvm active).

- [ ] **Step 2: Failing test** — `src/components/charts/__tests__/donut.test.tsx`:
```tsx
import { render } from '@testing-library/react'
import { DonutChart } from '../DonutChart'
test('renders without crashing', () => {
  const { container } = render(<DonutChart data={[{ label: 'A', value: 3, color: 'var(--color-chart-1)' }, { label: 'B', value: 1, color: 'var(--color-chart-2)' }]} />)
  expect(container).toBeTruthy()
})
```
(Recharts needs width; wrap chart in a fixed-size div in the component so jsdom renders.)

- [ ] **Step 3: Implement** — `src/components/charts/DonutChart.tsx`:
```tsx
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
export interface DonutDatum { label: string; value: number; color: string }
export function DonutChart({ data, size = 160 }: { data: DonutDatum[]; size?: number }) {
  return (
    <div style={{ width: '100%', height: size }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="label" innerRadius="62%" outerRadius="100%" paddingAngle={2} stroke="var(--color-bg-elevated)">
            {data.map((d, i) => <Cell key={i} fill={d.color} />)}
          </Pie>
          <Tooltip contentStyle={{ background: 'var(--color-bg-overlay)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-sm)', fontSize: 12, color: 'var(--color-text)' }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 4: Run** `npm test` → PASS; `npm run build` → succeeds.

- [ ] **Step 5: Commit** — `git add niriksha/src/components/charts niriksha/package.json niriksha/package-lock.json && git commit -m "feat(niriksha): themed Recharts donut"`

---

## Task 9: collectionStore — details + fetchAllDetails

**Files:** Modify `src/stores/collectionStore.ts` + test `src/stores/__tests__/collectionStore.test.ts`.

- [ ] **Step 1: Read** the current `collectionStore.ts` to match its axios/client pattern and existing `collections`/`health` shape.

- [ ] **Step 2: Failing test** (mock the api client used by the store; assert `fetchAllDetails` populates `details` from `GET /v1/collections/{name}`). Use `vi.mock` on `../api/client`.

- [ ] **Step 3: Implement** — add `details: CollectionDetail[]` state and `fetchAllDetails()` that, given the current `collections`, fetches each `GET /v1/collections/{name}` and stores `{name, fields}`. Reuse the existing axios client (`src/api/client.ts`). Keep it resilient: a failed detail is skipped, not fatal.

- [ ] **Step 4: Run** `npm test` → PASS.

- [ ] **Step 5: Commit** — `git add niriksha/src/stores/collectionStore.ts niriksha/src/stores/__tests__ && git commit -m "feat(niriksha): collectionStore details + fetchAllDetails"`

---

## Task 10: Nav regroup + Home route

**Files:** Modify `src/components/shell/Sidebar.tsx`, `src/App.tsx`; test `src/components/shell/__tests__/sidebar.test.tsx`.

- [ ] **Step 1: Failing test** — render `<MemoryRouter><Sidebar onOpenPalette={()=>{}}/></MemoryRouter>`; assert `Home` link present and group labels `OVERVIEW`, `DATA`, `SEARCH`, `OBSERVE` present.

- [ ] **Step 2: Implement** — replace `GROUPS` in `Sidebar.tsx`:
```ts
const GROUPS: NavGroup[] = [
  { label: 'Overview', items: [{ to: '/home', label: 'Home', icon: LayoutDashboard, hint: 'H' }] },
  { label: 'Data', items: [
    { to: '/collections', label: 'Collections', icon: Database, hint: 'C' },
    { to: '/pipelines', label: 'Pipelines', icon: GitBranch },
    { to: '/chunker', label: 'Chunker A/B', icon: Beaker },
  ] },
  { label: 'Search', items: [
    { to: '/search', label: 'Search', icon: Search, hint: 'S' },
    { to: '/atlas', label: 'Atlas', icon: Atom, hint: 'A' },
  ] },
  { label: 'Quality', items: [
    { to: '/evaluation', label: 'Evaluation', icon: Beaker },
    { to: '/quality', label: 'Quality Lab', icon: FlaskConical },
    { to: '/dashboards', label: 'Dashboards', icon: BarChart3 },
  ] },
  { label: 'Observe', items: [
    { to: '/observability', label: 'Observability', icon: BarChart3 },
    { to: '/health', label: 'Health', icon: Activity },
    { to: '/indexes', label: 'Indexes', icon: Layers },
  ] },
  { label: 'Developer', items: [
    { to: '/console', label: 'Dev Console', icon: TerminalSquare },
    { to: '/settings', label: 'Settings', icon: Settings },
  ] },
]
```
(Group label rendering already uppercases via CSS; keep capitalization here as-is — the existing `Group` component lowercases visually. Match the existing render. Assert against the rendered text accordingly.)

- [ ] **Step 3: Default route** — in `App.tsx`, change both `Navigate` targets from `/collections` to `/home`.

- [ ] **Step 4: Run** `npm test` → PASS; `npm run build` → succeeds.

- [ ] **Step 5: Commit** — `git add niriksha/src/components/shell/Sidebar.tsx niriksha/src/App.tsx niriksha/src/components/shell/__tests__ && git commit -m "feat(niriksha): regroup nav + land on Home"`

---

## Task 11: Dashboard Home rebuild

**Files:** Rewrite `src/pages/DashboardHome.tsx`; test `src/pages/__tests__/dashboardHome.test.tsx`.

- [ ] **Step 1: Failing test** — mock `useCollectionStore` to return 6 collections + healthy health + details; render within `MemoryRouter`; assert: a KPI label `Collections` and the value `6` appear, the collections table shows a known collection name (`articles`), and a `Multimodal` chip renders for `product_catalog`.

- [ ] **Step 2: Implement** the page composing: page header (title + `ConnectionPill` + "＋ New collection" action stub), a KPI grid of `SpotlightCard`+`StatTile` (Collections, Vectors [from `sumPromMetric` if metrics fetched else "—"], Fields, Shards, Version) with `CountUp` for numeric values, a two-column row of `Card`(DonutChart modality mix via `deriveModalityMix(details)` with a legend using `ModalityChip`) + `Card`(top collections list), and an upgraded collections table (modality chip per row via `deriveModality`, field/shard counts, click-through to `/collections/{name}`). Wire to `collectionStore` (`fetchCollections`, `fetchHealth`, `fetchAllDetails`). Graceful states: loading → `Skeleton`; empty → `EmptyState` with a create action; error → error card. Use the existing motion entrance. Subtle premium: page wrapped with a gated radial-glow via `Card glow` on the KPI band only.

- [ ] **Step 3: Run** `npm test` → PASS.

- [ ] **Step 4: Build + lint** — `npm run build` and `npm run lint` → clean (fix any lint).

- [ ] **Step 5: Commit** — `git add niriksha/src/pages/DashboardHome.tsx niriksha/src/pages/__tests__ && git commit -m "feat(niriksha): premium Dashboard Home showcase"`

---

## Task 12: Verify against the live app + screenshot

- [ ] **Step 1: Rebuild/reload** — dev server hot-reloads; if needed restart per Task 0 Step 4.

- [ ] **Step 2: Playwright** — navigate to `http://gpu-host-a.internal.example:5174/niriksha/home`; confirm KPIs show real counts (6 collections), the modality donut renders with legend, the collections table lists the seeded collections with modality chips. Screenshot `home-redesigned.png`. Also screenshot sidebar (new groups) and `⌘K` palette.

- [ ] **Step 3: A11y/motion checks** — tab through; focus rings visible; emulate `prefers-reduced-motion: reduce` and confirm CountUp shows final values without animation.

- [ ] **Step 4: Final gate** — `npm test`, `npm run build`, `npm run lint` all green on gpu-host-a.

- [ ] **Step 5: Commit screenshots note** (optional) — record evidence path in the PR/notes; no source change.

---

## Self-review notes
- **Spec coverage:** tokens (accent ramp, modality, chart, premium) → Task 2; IA Home-in-nav + default route → Task 10; primitives subset needed by showcase → Task 5; charts layer → Task 8; reactbits integration → Tasks 6–7; Dashboard Home showcase → Task 11; verification/DoD → Task 12; seed/backend → Task 0. (Full primitive set, collapsible sidebar, top-bar cluster, command-palette upgrade, Login, EmptyState-everywhere are Plans 1.2/1.3 per the Phase-1 spec.)
- **No fabricated data:** "Vectors" KPI is real (from `veda_records_total` metric) or shows "—"; activity feed intentionally omitted (no events endpoint) — replaced by a real "top collections" list.
- **Type consistency:** `Modality`, `SchemaField`, `CollectionDetail`, `ModalitySlice`, `DonutDatum` are defined once and reused across modality/modalityMix/DonutChart/Home.
