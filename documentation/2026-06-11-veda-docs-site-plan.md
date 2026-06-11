# Veda Docs Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline; visual
> build verified with build + Playwright, not unit tests). Steps use `- [ ]`.

**Goal:** Build the 10-page documentation site at `/docs/*` inside `website/`, per the spec
(`2026-06-11-veda-docs-site-design.md`).

**Architecture:** registry-driven markdown docs. `docs/registry.ts` orders groups/pages and
binds each slug to a `?raw`-imported markdown file. `DocPage` resolves the slug and renders
through `DocMarkdown` (react-markdown + remark-gfm, fully component-mapped to the obsidian
prose system). The shell (`DocsLayout`) is a three-pane grid: sidebar / article / TOC rail.
`slugify.ts` + `headings.ts` are the single source of heading ids for renderer, TOC, and
search. Search is a dependency-free Ctrl-K modal over titles + headings.

**Tech:** + `react-markdown@^10`, `remark-gfm@^4` (only new deps; ESM, React-19-safe).
`vite/client` types make `*.md?raw` imports `string`. **Branch:** `feat/website-docs`
(spec `34248ed`). Dev on :5175; edit local → scp → build over ssh → Playwright local.

---

## Task 1: Deps, routes, docs shell (sidebar + layout) with 2 stub pages

**Files:** modify `website/package.json` (add deps), `website/src/App.tsx` (routes),
`website/src/components/Nav.tsx` (add `solid` prop; links → `/#features`, `/#architecture`);
create `src/docs/registry.ts`, `src/docs/content/{introduction,quickstart}.md` (stubs),
`src/components/docs/DocsLayout.tsx`, `src/components/docs/DocsSidebar.tsx`,
`src/pages/DocPage.tsx`.

- [ ] **Step 1 — deps:** `npm i react-markdown@^10 remark-gfm@^4` on gpu-host-a.
- [ ] **Step 2 — `registry.ts`:**

```ts
import introduction from './content/introduction.md?raw'
import quickstart from './content/quickstart.md?raw'
// ...one import per page (10 total by Task 4)

export interface DocPageEntry { slug: string; title: string; md: string }
export interface DocGroup { label: string; items: DocPageEntry[] }

export const DOCS: DocGroup[] = [
  { label: 'Getting started', items: [
    { slug: 'introduction', title: 'Introduction', md: introduction },
    { slug: 'quickstart', title: 'Quick start', md: quickstart },
    // installation
  ]},
  // Core concepts: collections, indexes, search, document-pipeline
  // Reference: rest-api, python-sdk, observability
]

export const FLAT: DocPageEntry[] = DOCS.flatMap((g) => g.items)
export const bySlug = (slug: string) => FLAT.find((p) => p.slug === slug)
```

- [ ] **Step 3 — routes in `App.tsx`:** `/docs` → `<Navigate to="/docs/introduction" replace/>`,
  `/docs/:slug` → `<DocPage/>` (DocPage redirects unknown slugs the same way).
- [ ] **Step 4 — `Nav` `solid?: boolean`** (always `.glass border-b` when set) + absolute links.
- [ ] **Step 5 — `DocsLayout`:** `<Nav solid/>` + `pt-16` grid `lg:grid-cols-[240px_minmax(0,1fr)]
  xl:grid-cols-[240px_minmax(0,1fr)_200px]` in a `max-w-[1320px]` container; sidebar sticky
  `top-16`; article `py-12 max-w-[760px]`. Mobile: sidebar behind a "Docs menu" toggle button.
- [ ] **Step 6 — `DocsSidebar`:** groups with `.mono-label` headers; items 14px, active =
  saffron text + 2px left saffron bar + `bg-[var(--color-surface)]`; NavLink-driven.
- [ ] **Step 7 — `DocPage` (v1):** resolve slug; render `<h1>` + raw md in a `<pre>` placeholder.
- [ ] **Step 8 — scp; build; Playwright smoke** (`/docs` redirects, sidebar navigates, 0 errors).
- [ ] **Step 9 — commit** `feat(website): docs shell (routes, registry, sidebar layout)`.

## Task 2: Markdown prose system

**Files:** create `src/docs/slugify.ts`, `src/docs/headings.ts`,
`src/components/docs/CodeBlock.tsx`, `src/components/docs/DocMarkdown.tsx`; update `DocPage`.

- [ ] **Step 1 — `slugify.ts`:**

```ts
export const slugify = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9\s-]/g, '').trim().replace(/\s+/g, '-')
```

- [ ] **Step 2 — `headings.ts`:** parse raw md lines for `^##\s` / `^###\s` (skip fenced code
  blocks by toggling on ```` ``` ````), return `{ depth: 2|3, text, id: slugify(text) }[]`.
- [ ] **Step 3 — `CodeBlock.tsx`:** obsidian surface (`--color-code`), hairline border,
  radius-md, mono 13px; header row only when a language is present (mono-label lang +
  copy button); copy = `navigator.clipboard.writeText` with 1.5s "Copied" state.
- [ ] **Step 4 — `DocMarkdown.tsx`:** `<ReactMarkdown remarkPlugins={[remarkGfm]}
  components={{...}}>`: h1 display 36px; h2 display 24px `id=slugify(text)` with `pt-8 mt-10
  border-t border-[var(--color-border)]`; h3 17px semibold `id`; p 15.5px/1.75 muted; a saffron;
  ul/ol with saffron markers; blockquote saffron left-rule callout on surface; table hairline
  (mono-label thead); inline code mono chip on `--color-surface-2`; `pre>code` → CodeBlock
  (language from `className="language-x"`). Extract heading text via a `textOf(children)`
  helper (handles nested nodes).
- [ ] **Step 5 — `DocPage` v2:** `<h1>` from title is NOT rendered separately — pages start
  with their own `# Title`; DocPage just renders `<DocMarkdown md={page.md}/>`.
- [ ] **Step 6 — scp; build; Playwright:** quickstart stub (give it real headings/code/table)
  renders styled; copy works. Commit `feat(website): docs markdown prose system + code blocks`.

## Task 3: TOC, pager, search

**Files:** create `src/components/docs/DocsToc.tsx`, `DocsPager.tsx`, `DocsSearch.tsx`;
wire into `DocsLayout`/`DocPage`.

- [ ] **Step 1 — `DocsToc`:** props `{ md }` → `headings(md)` (h2 only + h3 indented);
  sticky right rail (`xl:` only), mono-label "On this page"; scroll-spy: IntersectionObserver
  on heading elements (`rootMargin: '-80px 0px -70% 0px'`), active item saffron. Click →
  `scrollIntoView({ behavior: 'smooth' })` + history hash.
- [ ] **Step 2 — `DocsPager`:** index in `FLAT`; prev/next cards (mono-label "Previous/Next"
  + title) with hairline top border.
- [ ] **Step 3 — `DocsSearch`:** index = `FLAT.flatMap(p => [{type:'page',...}, ...headings(p.md)
  .map(h => ({type:'heading', ...}))])`. Global Ctrl/Cmd-K listener (in DocsLayout) toggles a
  fixed modal (obsidian surface, hairline, input + result list ≤12); substring match on
  lowercased text; ↑/↓ selection, Enter navigates (`/docs/slug#id`), Esc closes; clicking a
  result too. Also a sidebar trigger button showing `⌘K`.
- [ ] **Step 4 — hash scrolling:** on route/hash change in DocPage, `getElementById(hash)` →
  scrollIntoView; else scroll top.
- [ ] **Step 5 — scp; build; Playwright:** TOC follows scroll; search finds a heading and
  navigates; pager walks. Commit `feat(website): docs TOC, search palette, pager`.

## Task 4: The 10 content pages

**Files:** create the remaining 8 `src/docs/content/*.md` + finalize the 2 stubs; complete
`registry.ts` imports/groups.

- [ ] **Step 1 —** Fetch sources from gpu-host-a (`README.md`, `sdk/python/README.md`,
  `documentation/openapi.yaml` paths list, `competitive_advantages.md` already known) and
  adapt per the spec's content table. Commands/JSON stay verbatim; prose rewritten concise.
  Each page starts `# Title`, uses `##`/`###` for the TOC, gfm tables where tabular.
- [ ] **Step 2 —** scp all; build; Playwright sweep of ALL 10 pages (render + no overflow).
- [ ] **Step 3 —** commit `docs(website): all ten documentation pages`.

## Task 5: Final verification

- [ ] Build green; Playwright at 1440 + 1920: full nav walk, TOC spy, search→heading,
  copy button, pager ends, marketing links (`Read the docs`, nav, footer) land correctly;
  0 console errors; no horizontal overflow. Fix-ups; commit; finishing-a-development-branch.

## Self-review

Spec coverage: routes/registry/markdown (T1–2), TOC/search/pager/copy (T2–3), 10 pages (T4),
verification + marketing links (T5). Shared slugify keeps renderer/TOC/search ids consistent
(T2). Mobile sidebar = toggle (T1). No placeholders; code given for novel logic. ✓
