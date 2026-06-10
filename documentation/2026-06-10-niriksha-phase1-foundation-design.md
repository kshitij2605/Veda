# Niriksha Redesign — Phase 1: Foundation (Design System + App Shell)

**Date:** 2026-06-10
**Status:** Approved (design) — pending implementation plan
**Component:** `niriksha/` (React 19 + Vite 8 + Tailwind 4 dashboard bundled with Veda)
**Related research:** `research/frontend/2026-06-10-competitive-research.md`

---

## 1. Context & goals

Niriksha is the React dashboard bundled with the Veda multimodal vector database. It already ships ~14 pages, a token-driven design system ("Phase 0"), a grouped sidebar, a `⌘K` command palette, and lazy-loaded routes. We are doing a full visual + UX overhaul plus flagship features, sequenced **foundation-first**.

This spec covers **Phase 1 only: the design-system + app-shell foundation that every later page inherits**, ending with one screenshot-worthy showcase surface to prove the aesthetic end to end.

### Decisions locked during brainstorming
- **Build order:** Niriksha first, then marketing site, then docs site (each its own spec → plan → build).
- **Ambition:** maximum tier — visual overhaul + UX overhaul + flagship features (delivered across Phases 1–4).
- **Personality:** *disciplined premium* — flat/dense/calm on data surfaces; depth, glow, and animated backgrounds reserved for hero / empty / login / Atlas; targeted micro-interactions; reactbits + gsap used purposefully, never behind dense data.
- **Sequencing:** foundation-first.
- **Theme:** dark-first for v1; light-theme tokens stay wired but light polish is deferred to Phase 4.
- **Brand accent:** keep indigo-violet (OKLCH hue 282); it is premium and unowned in this market.

### Overall decomposition (context; only Phase 1 is in scope here)
- **Phase 1 — Foundation:** design system + shell + showcase (this spec).
- **Phase 2 — Core data surfaces:** Collections, Search Playground v2 (Dense/Sparse/BM25/Hybrid/ColPali/Rerank), embed-insert + document-processing pipeline, System Health.
- **Phase 3 — Flagship viz + observability:** Atlas v2 (UMAP/t-SNE, 2D/3D, density, clusters, similarity graph), Observability v2 (queryless drilldown, explain-as-waterfall, BubbleUp-style analysis), Index Manager / Evaluation / Quality / Pipelines / Chunker / Dashboards / Console.
- **Phase 4 — Onboarding + polish:** first-run tutorial, light theme, RBAC/settings, a11y/perf pass.

---

## 2. Phase 1 scope

**In scope**
1. Design-token system (evolve Phase-0 `global.css`).
2. App shell + information architecture (sidebar, top bar, command palette, page-header pattern).
3. Core primitive component library (fill the gaps).
4. Charting + reactbits integration layers.
5. Showcase surfaces: **Dashboard Home**, **Login**, global **EmptyState**.

**Out of scope (later phases):** all data pages, Search v2, Atlas v2, Observability v2, light-theme polish, marketing/docs sites.

---

## 3. Visual system (tokens)

Extend the existing `@theme` block in `niriksha/src/styles/global.css`. Keep the OKLCH base scale (hue 252), Inter + JetBrains Mono, 4px spacing grid, radii, motion, and focus-ring rules unchanged. Add:

### 3.1 Complete the accent ramp + gradient
The accent (indigo-violet, hue 282) currently has steps 0,2,4,5,7,9,11. Fill 1,3,6,8,10,12 so it is a full 0–12 scale matching the base scale. Add one reserved gradient token pair for showcase surfaces only:
```
--gradient-hero: linear-gradient(135deg, oklch(0.62 0.22 282), oklch(0.66 0.21 320));
```

### 3.2 Modality palette (signature — multimodal-native identity)
Categorical, colorblind-aware OKLCH; used for badges/chips, Atlas point coloring, result-type indicators, and modality charts. Final values tuned in implementation; target:
```
--color-modality-text:   oklch(0.70 0.15 250);  /* blue    */
--color-modality-image:  oklch(0.72 0.18 320);  /* magenta */
--color-modality-audio:  oklch(0.74 0.16 160);  /* emerald */
--color-modality-video:  oklch(0.76 0.15 60);   /* amber   */
--color-modality-doc:    oklch(0.70 0.14 25);   /* coral   */
--color-modality-mixed:  oklch(0.72 0.02 252);  /* neutral */
```
Each adjacent pair ≥15 OKLCH L-units and hue-distinct for grayscale/colorblind legibility.

### 3.3 Chart theme
An 8-color categorical viz ramp (anchored on the existing teal data hue 195 + curated OKLCH steps ≥15 L-units apart), plus a sequential ramp and a diverging ramp, and axis/grid/tooltip/crosshair tokens. All chart components read these — no chart defines its own colors.

### 3.4 Premium-surface tokens
Refined elevation set (keep `--shadow-*`), keep `.glass` and `.dot-grid`, and add **radial-glow** + **grain** utility tokens **gated to showcase surfaces** (hero/login/empty/Atlas). Never applied behind tables or dense panels.

### 3.5 Token discipline
Component code references semantic aliases (`--color-bg`, `--color-text`, `--color-accent`, `--color-modality-*`) — never raw `oklch()`. Retire the legacy aliases (`--color-bg-secondary`, `text-text-secondary`, etc.) as pages are touched; `DashboardHome` is migrated in this phase.

---

## 4. Information architecture + shell

### 4.1 Navigation grouping
Refine the existing groups and **add Home** (today `DashboardHome` exists but is not linked in the sidebar):
```
OVERVIEW    Home
DATA        Collections · Pipelines (Ingest) · Chunker A/B
SEARCH      Search · Atlas
QUALITY     Evaluation · Quality Lab · Dashboards
OBSERVE     Observability · Health · Indexes
DEVELOPER   Console
──────────  (footer) Settings · theme toggle · connection pill
```
Default route `/` redirects to `/home` (currently `/collections`).

### 4.2 Shell upgrades
- **Sidebar** (`components/shell/Sidebar.tsx`): same 220px rail, refined active state (accent left-bar + subtle bg), **collapsible to a ~56px icon rail** with tooltips, real `g`-then-key shortcuts (hints already present).
- **Top bar** (`AppShell` `Header`): keep breadcrumbs; add a right cluster — connection/health pill, theme toggle, and a context-action slot (e.g. "＋ New collection"). Keep TrustBadges.
- **Command palette** (`components/palette/CommandPalette.tsx`): upgrade from quick-jump to **actions + navigation + recent collections** (cmdk already installed).
- **Page-header pattern**: standardize `title / description / actions / tabs` so every page is visibly one system. Replace the ad-hoc `components/PageHeader.tsx`.

---

## 5. Component & library strategy

### 5.1 Primitives
Keep the existing set (`components/primitives/`: Badge, Button, CopyButton, Drawer, FilterChip, Input, KBD, Modal, Select, Skeleton, Sparkline, Toast, Tooltip). **Add:** Card/Panel, StatTile, Tabs (wrap Radix), SegmentedControl, Switch, ProgressBar, Stepper, Toolbar, CodeBlock, ModalityChip, ConnectionPill, an upgraded **Table** shell (TanStack table + virtual, already installed), and an upgraded **EmptyState**. All exported from `components/primitives/index.ts`.

### 5.2 Charting libraries (add)
- **Recharts** — standard panels (bar/line/donut).
- **uPlot** — dense/streaming time-series (used heavily in Observe, Phase 3).
- Wrapped in a thin token-themed layer at `components/charts/` (or `lib/charts/`) so every chart is consistent.
- *(Atlas WebGL scatter — Apple Embedding Atlas / regl-scatterplot — is Phase 3; not installed now.)*

### 5.3 reactbits integration layer
Fetch each component's source via the reactbits MCP into `components/reactbits/`, **re-tokenize** it (replace hardcoded colors with `var(--color-*)`), and wrap it behind a local API. **Do not use** the Buttons/Forms categories (quality 2/10 placeholders) — use our primitives. Usage map (restraint-first):

| Where | reactbits / gsap |
|---|---|
| Login + empty states + (Phase 3) Atlas backdrop | Backgrounds: Aurora / Beams / Dark Veil / Particles / Dot Grid — lazy, paused offscreen, disabled under reduced-motion |
| Hero / overview only | Text: Count Up (KPIs), Gradient/Shiny Text (headline), Split Text (reveals) |
| Overview + cards | Components: Magic Bento (KPI grid), Spotlight Card, Animated List (activity), Stepper (Phase-2 doc wizard) |
| Transitions / timelines | gsap for complex reveals/number tweens; framer-motion (installed) for the rest |

### 5.4 Dependencies to add
`recharts`, `uplot`, `gsap`. WebGL backgrounds (Aurora/Beams/Particles) pull in `ogl`/`three` — added only if those specific backgrounds are used, and confined to the login/showcase routes so they never enter the main bundle.

---

## 6. Showcase surfaces

### 6.1 Dashboard Home (`pages/DashboardHome.tsx`)
Premium overview that proves the system and is finally linked in nav. Replaces today's plain metric-grid + table.
```
┌──────────────────────────────────────────────────────────────┐
│  Veda · Overview                        [healthy ●] [＋ New]  │  page header
├──────────────────────────────────────────────────────────────┤
│  ┌ Magic Bento KPI grid (CountUp) ─────────────────────────┐ │
│  │ COLLECTIONS  VECTORS  FIELDS  SHARDS  VERSION  HEALTH    │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌ Modality mix (donut) ─┐  ┌ Recent activity (AnimatedList)┐ │
│  └───────────────────────┘  └───────────────────────────────┘ │
│  ┌ Collections (TanStack: modality chips · spark · counts) ─┐ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
        subtle radial-glow bg (gated · reduced-motion-safe)
```
**Data sources:** `/v1/admin/health`, `/v1/admin/metrics`, `/v1/collections` (summary). Any KPI not yet served degrades gracefully to "—" (no fabricated values). Modality mix is derived from collection field types where available.

### 6.2 Login (`pages/Login.tsx`, new)
The one fully expressive surface: animated background + brand mark + Gradient-Text headline + a clean credential form (our primitives) posting to `/v1/auth/login`, storing the JWT in `localStorage.veda_token`. Fills a real gap — the AUDIT notes there is no login page today (token is read but never set). Unauthenticated users are routed here when auth is enabled; auth-disabled deployments skip it.

### 6.3 Global EmptyState (`components/primitives/EmptyState.tsx` upgrade)
Dot-grid / subtle-glow background + icon + title/description + primary action; one reusable pattern.

---

## 7. Architecture / module boundaries

| Unit | Path | Responsibility |
|---|---|---|
| Tokens | `src/styles/global.css` `@theme` | single source of truth for color/type/space/motion |
| Primitives | `src/components/primitives/` | unstyled-logic + token styling; no app data |
| reactbits wrappers | `src/components/reactbits/` | vendored + re-tokenized decorative components |
| Charts | `src/components/charts/` | token-themed Recharts/uPlot wrappers |
| Shell | `src/components/shell/` | Sidebar, Header, Breadcrumbs, layout |
| Palette | `src/components/palette/` | command palette (actions + nav + recent) |
| Showcase pages | `src/pages/DashboardHome.tsx`, `src/pages/Login.tsx` | compose the above |

Each unit is independently understandable and testable; pages compose primitives/charts/reactbits and own data fetching via existing stores (`src/stores/`).

---

## 8. Verification / definition of done

- **Tokens:** complete accent ramp, modality palette, chart theme, premium-surface tokens defined; no Phase-1 file uses raw `oklch()`; `DashboardHome` migrated off legacy aliases.
- **Shell:** nav regrouped with Home; sidebar collapsible; top bar with connection pill + theme toggle + action slot; palette does actions + nav + recent.
- **Primitives:** new primitives implemented and exported; Storybook-free manual matrix verified in-app.
- **Charts + reactbits layers:** established; reactbits components used by the showcase are re-tokenized.
- **Showcase:** Dashboard Home redesigned and wired to real endpoints with graceful degradation; Login functional against `/v1/auth/login`; global EmptyState in place.
- **Quality gates:** `npm run build` (tsc + vite) and `npm run lint` clean; `prefers-reduced-motion` disables decorative motion; focus rings + keyboard nav intact; decorative WebGL backgrounds lazy-loaded off the main bundle.
- **Evidence:** Playwright screenshots of Home, Login, and shell states (sidebar expanded/collapsed, `⌘K` open) captured against the running dev server (`npm run dev -- --host 0.0.0.0 --port 5174` → `http://iitgpu11.hon.otsuka-shokai.co.jp:5174/niriksha`).

---

## 9. Risks & open items
- **Backend + data:** development/verification needs `veda-server` (:6340) running with a few seeded collections (including an image collection to demonstrate multimodal). Confirm what is live on iitgpu11; seed sample data if absent.
- **WebGL perf:** animated backgrounds are confined to login/showcase routes, lazy-loaded, paused offscreen, and disabled under reduced-motion to protect the main bundle and low-power devices.
- **Light theme:** intentionally deferred to Phase 4; dark is canonical for v1.
- **reactbits dependencies:** some backgrounds add `ogl`/`three`; only pulled in if that specific background ships, and never in the shared bundle.

---

## 10. Out of scope
Data pages, Search v2, Atlas v2, Observability v2, marketing site, docs site, light-theme polish, RBAC management UI — all addressed in later phases per the decomposition in §1.
