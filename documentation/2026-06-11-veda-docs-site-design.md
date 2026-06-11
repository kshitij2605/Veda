# Veda Docs Site — Design Spec

**Date:** 2026-06-11
**Status:** Approved direction; pending spec review → implementation plan
**Goal:** A real documentation site at `/docs/*` inside the existing `website/` app, in the
approved Obsidian + Saffron language, with content assembled from the repo's actual
documentation (README, sdk/python/README, documentation/openapi.yaml, CONTRIBUTING) — no
invented claims.

## Approved decisions

1. **Scope — full v1 set (10 pages)** in three sidebar groups:
   - *Getting started:* Introduction · Quick start · Installation & config
   - *Core concepts:* Collections & fields · Indexes · Search & fusion · Document pipeline
   - *Reference:* REST API · Python SDK · Observability
2. **Markdown-driven:** content lives as `website/src/docs/content/*.md`, imported with
   Vite `?raw` and rendered by **react-markdown + remark-gfm** (the only new deps), with
   every element mapped to obsidian-styled components.
3. **Shell features (all):** right-rail "On this page" TOC with scroll-spy · **Ctrl/Cmd-K
   quick search** (client-side over page titles + headings, no deps) · prev/next footer
   pagination · copy buttons on all code blocks.
4. Same app/brand: obsidian tokens, Space Grotesk/IBM Plex Sans/JetBrains Mono, saffron
   accent, `.mono-label` group headers. Dark-only.

## Architecture

```
website/src/
  docs/
    registry.ts            # ordered groups → {slug, title, file(raw md)}; flatten() for prev/next
    content/*.md           # 10 pages (?raw imports)
    slugify.ts             # heading text → anchor id (shared by renderer + TOC + search)
    headings.ts            # parse ##/### from raw md → {depth, text, id}[]
  components/docs/
    DocsLayout.tsx         # route shell: Nav (solid) + sidebar + content + TOC rail
    DocsSidebar.tsx        # grouped nav, saffron active state; <lg: toggleable panel
    DocMarkdown.tsx        # react-markdown with component map (prose system)
    CodeBlock.tsx          # obsidian code surface + language label + copy button
    DocsToc.tsx            # right rail, IntersectionObserver scroll-spy
    DocsSearch.tsx         # Ctrl/Cmd-K modal: filter titles+headings, keyboard nav
    DocsPager.tsx          # prev/next links from registry order
  pages/DocPage.tsx        # /docs/:slug → resolve registry → render; unknown → redirect
App.tsx                    # routes: /docs → redirect introduction; /docs/:slug → DocPage
components/Nav.tsx         # gains `solid` prop (always-glass on docs); links absolute (/#features)
```

- **Routing:** `/docs` redirects to `/docs/introduction`; `/docs/:slug` renders; unknown
  slugs redirect to introduction. Route changes scroll to top; `#hash` scrolls to heading.
- **Prose mapping (DocMarkdown):** h1 display 36px · h2 display 26px with slugified `id` +
  top hairline rhythm · h3 18px semibold · p 15.5px/1.75 `--color-text-muted` (≈7:1) ·
  inline code mono 13px on `--color-surface-2` chip · pre → CodeBlock · gfm table → hairline
  table (mono-label header) · blockquote → saffron left-rule callout · links saffron.
- **TOC:** parsed from raw markdown (not the DOM) via `headings.ts`; ids must match the
  renderer's — both use `slugify.ts`. Scroll-spy highlights the active heading.
- **Search:** index built once from registry (page titles + h2/h3). Modal opens on
  Ctrl/Cmd-K or sidebar button; substring match; ↑/↓/Enter/Esc; navigates to
  `/docs/slug#id`. ~120 lines, no dependency.
- **CodeBlock:** `--color-code` surface, hairline border, mono 13px, language label
  (mono-label), copy button (clipboard API, "Copied" flash, saffron focus ring).

## Content plan (source → page)

| Page | Source |
|---|---|
| introduction | README intro/features + competitive_advantages.md (condensed) |
| quickstart | README Quick Start (docker compose + profiles) + first collection/search calls |
| installation | README Build from source + Configuration (embedding models, blob storage) + memory scaling |
| collections | README API examples (create/insert/embed-insert) + field types incl. sparse/multi-vector/blob |
| indexes | README/competitive_advantages 7-index material + quantization (RaBitQ/SQ8) |
| search | README search/filtered + multi-vector fusion (RRF/weighted/intersection) + rerank + explain + `?at=` snapshot reads |
| document-pipeline | README Document Processing (OCR setup, formats, 7 chunking strategies, single + batch) |
| rest-api | Endpoint reference assembled from documentation/openapi.yaml + README examples |
| python-sdk | sdk/python/README.md (install, quickstart, auth, fusion, auto-embed) |
| observability | README Monitoring & Observability (metrics, health, OTel, explain) |

Adaptation rules: rewrite for docs voice (concise, imperative), keep all commands/JSON
verbatim from the sources, no feature claims not present in the repo docs.

## Non-goals

- No server-side search, no versioned docs, no MDX, no i18n.
- No edit-on-GitHub links yet (repo paths may change).
- Marketing page untouched except the Nav `solid` prop + absolute links.

## Verification

- Build green (the 2 new deps install cleanly with React 19).
- Playwright at 1440 + 1920: sidebar navigation across all 10 pages renders; TOC scroll-spy
  follows; Ctrl-K search finds + navigates to a heading; copy button copies; prev/next walk
  the full order; 0 console errors; no horizontal overflow.
- Marketing → Docs links (nav, footer, CTA "Read the docs") land on `/docs/introduction`.

## Rollout

1. Deps + docs shell (layout/sidebar/routes/registry with 2 stub pages) → verify.
2. Markdown renderer + CodeBlock/copy + prose system → verify on a real page.
3. TOC + pager + mobile sidebar + search palette.
4. All 10 content pages (adapted from sources).
5. Full Playwright pass → commit per task on `feat/website-docs`.
