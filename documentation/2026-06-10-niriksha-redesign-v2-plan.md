# Niriksha Redesign v2 (Light-first, Open & Airy) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline,
> recommended for this visual work) or superpowers:subagent-driven-development. Steps use
> checkbox (`- [ ]`) syntax. Visual tasks are verified live with Playwright on the dev
> server (light **and** dark) + screenshots, not unit tests; unit tests cover pure logic only.

**Goal:** Pivot Niriksha to a light-first, open/airy design language (Kibana-inspired,
Niriksha identity) across the shared tokens + showcase/shell surfaces, keeping dark at parity.

**Architecture:** The token system is a `--color-base-0..12` scale that inverts per theme;
all surface/text aliases reference it. We make **light canonical** (default), give light
first-class palette/elevation, expand the type scale + add a display font (Geist), loosen
spacing via a taller PageHeader + airier primitives, then rebuild Home/Login/Atlas/empty
states. Components inherit tokens, so most pages flip to light automatically.

**Tech Stack:** React 19, Tailwind 4 (`@theme`), Vite 8, framer-motion, **gsap** (already a
dep, currently unused), vendored **reactbits** components, Google Fonts (Inter / JetBrains
Mono / **Geist**), Playwright (verification), vitest (unit logic).

**Branch:** `feat/niriksha-redesign-v2` (already created, spec committed `24c0af3`).

**Dev env:** backend `:6340` (live, collections seeded) + Vite dev `:5174`
(`http://iitgpu11.hon.otsuka-shokai.co.jp:5174/niriksha`). Workflow: edit local → scp up →
`npm run build` / `npm test` over ssh → Playwright from the local box.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `niriksha/index.html` (remote) | modify | default theme = light, pre-paint bg, add Geist link |
| `niriksha/src/lib/theme.ts` | modify | default `'light'`; `data-theme="dark"` is the override |
| `niriksha/src/styles/global.css` | modify | light-canonical palette, light modality/chart/shadows, type scale (+Display), Geist, bumped radii |
| `niriksha/src/lib/greeting.ts` | create | time-aware greeting (pure, unit-tested) |
| `niriksha/src/lib/greeting.test.ts` | create | greeting unit tests |
| `niriksha/src/lib/useReveal.ts` | create | gsap scroll-reveal hook, reduced-motion gated |
| `niriksha/src/components/primitives/{Card,PageHeader,StatTile,Button,EmptyState}.tsx` | modify | airier padding, bigger type, soft elevation, hover lift |
| `niriksha/src/components/shell/{Sidebar,AppShell}.tsx` | modify | fully-light shell, generous spacing, active accent bar |
| `niriksha/src/components/reactbits/BentoCard.tsx` | create (vendor) | bento/spotlight grid card for showcase |
| `niriksha/src/pages/DashboardHome.tsx` | modify | hero band + airy KPI band + bento grid |
| `niriksha/src/pages/Login.tsx` | modify | refresh to new type/spacing/buttons |
| `niriksha/src/pages/atlas/AtlasPage.tsx` | modify | taller PageHeader, light controls |

---

## Task 1: Foundation — light-canonical tokens, fonts, theme default

**Files:** Modify `niriksha/src/styles/global.css`, `niriksha/src/lib/theme.ts`,
`niriksha/index.html` (remote).

- [ ] **Step 1 — global.css: add the display tier + bigger base, Geist, bumped radii.**
  In the `@theme` block, update typography + radii:

```css
  --font-sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --font-display: "Geist", "Inter", sans-serif;

  --text-2xs: 10px;
  --text-xs:  11px;
  --text-sm:  12px;
  --text-base: 14px;   /* was 13 — body comfortable */
  --text-md:  15px;
  --text-lg:  17px;
  --text-xl:  20px;
  --text-2xl: 24px;
  --text-3xl: 30px;    /* H1 */
  --text-4xl: 38px;
  --text-5xl: 46px;    /* Display */

  --radius-xs: 6px;    /* was 3 */
  --radius-sm: 8px;    /* was 5 */
  --radius-md: 10px;   /* was 7 */
  --radius-lg: 14px;   /* was 10 — card */
  --radius-xl: 18px;   /* was 14 */
```

- [ ] **Step 2 — global.css: make light first-class.** Replace the existing
  `[data-theme="light"]` block (which only redefines base-0..12) with a block that also
  overrides the surface/text aliases (white cards on soft-gray canvas), light-tuned
  modality + chart, and soft light shadows:

```css
[data-theme="light"] {
  /* base scale — cool near-white → near-black ink (for components using base-N directly) */
  --color-base-0:  oklch(0.985 0.004 255);
  --color-base-1:  oklch(1.000 0     0);
  --color-base-2:  oklch(0.970 0.004 255);
  --color-base-3:  oklch(0.918 0.006 255);
  --color-base-4:  oklch(0.870 0.007 256);
  --color-base-5:  oklch(0.800 0.008 258);
  --color-base-6:  oklch(0.650 0.012 260);
  --color-base-7:  oklch(0.560 0.014 262);
  --color-base-8:  oklch(0.500 0.015 264);
  --color-base-9:  oklch(0.400 0.018 265);
  --color-base-10: oklch(0.320 0.020 265);
  --color-base-11: oklch(0.270 0.020 266);
  --color-base-12: oklch(0.150 0.020 266);

  /* surface aliases — white cards float on a soft cool-gray canvas */
  --color-bg:           oklch(0.985 0.004 255);
  --color-bg-elevated:  oklch(1 0 0);
  --color-bg-overlay:   oklch(1 0 0);
  --color-bg-hover:     oklch(0.955 0.005 255);
  --color-bg-active:    oklch(0.930 0.006 256);
  --color-bg-selected:  oklch(0.955 0.020 282);

  --color-border:        oklch(0.912 0.006 256);
  --color-border-strong: oklch(0.855 0.008 258);

  --color-text:          oklch(0.270 0.020 266);
  --color-text-muted:    oklch(0.500 0.016 264);
  --color-text-subtle:   oklch(0.620 0.013 262);
  --color-text-disabled: oklch(0.720 0.010 260);
  --color-text-secondary: oklch(0.400 0.018 265);

  /* accent on light — slightly deeper for AA on white */
  --color-accent:        oklch(0.530 0.205 282);
  --color-accent-hover:  oklch(0.480 0.210 282);
  --color-accent-subtle: oklch(0.955 0.030 282);

  /* modality — legible on white (~0.55–0.62 L) */
  --color-modality-text:  oklch(0.550 0.180 264);
  --color-modality-image: oklch(0.580 0.150 155);
  --color-modality-audio: oklch(0.600 0.140 70);
  --color-modality-video: oklch(0.560 0.200 18);
  --color-modality-doc:   oklch(0.540 0.180 300);
  --color-modality-mixed: oklch(0.600 0.130 220);

  /* chart ramp — light-tuned */
  --color-chart-1: oklch(0.560 0.140 195);
  --color-chart-2: oklch(0.560 0.190 282);
  --color-chart-3: oklch(0.580 0.150 160);
  --color-chart-4: oklch(0.640 0.150 60);
  --color-chart-5: oklch(0.580 0.190 320);
  --color-chart-6: oklch(0.600 0.160 30);
  --color-chart-7: oklch(0.560 0.130 230);
  --color-chart-8: oklch(0.640 0.150 110);

  /* soft light elevation */
  --shadow-xs: 0 0 0 1px oklch(0.2 0.02 266 / 0.05);
  --shadow-sm: 0 1px 2px oklch(0.2 0.02 266 / 0.06), 0 0 0 1px oklch(0.2 0.02 266 / 0.04);
  --shadow-md: 0 2px 8px -1px oklch(0.2 0.02 266 / 0.08), 0 1px 2px oklch(0.2 0.02 266 / 0.05);
  --shadow-lg: 0 10px 30px -8px oklch(0.2 0.02 266 / 0.12), 0 2px 6px oklch(0.2 0.02 266 / 0.06);
  --shadow-glow: 0 0 0 3px oklch(0.53 0.205 282 / 0.18);

  color-scheme: light;
}
```

- [ ] **Step 3 — global.css: light pre-paint + reduced-motion.** Leave the dark `@theme`
  values as the fallback. (Light is applied by default via theme.ts + index.html in steps
  4–5.) No change to the `@layer base` block except confirm `body` still uses tokens.

- [ ] **Step 4 — theme.ts: default to light.** Replace the file body:

```ts
export type Theme = 'dark' | 'light'

const KEY = 'veda_theme'

/** Persisted theme, defaulting to light (the canonical theme). */
export function getStoredTheme(): Theme {
  const v = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null
  return v === 'dark' ? 'dark' : 'light'
}

/** Apply a theme via `data-theme` on <html> (light = default attribute; dark sets it). */
export function applyTheme(t: Theme): void {
  const el = document.documentElement
  el.dataset.theme = t
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* storage unavailable — non-fatal */
  }
}
```

  Note: we always set `data-theme` explicitly (light or dark). The CSS `[data-theme="light"]`
  block now carries the full light palette, and `@theme` is the dark fallback (matched by
  `[data-theme="dark"]` falling through to `:root`). Verify in step 7 that toggling works
  both ways.

- [ ] **Step 5 — index.html (remote): light default + Geist.** Change `<body data-theme="dark">`
  → `<body data-theme="light">`; change the pre-paint style background to
  `oklch(0.985 0.004 255)`; change `<meta name="theme-color" content="#0a0a10">` →
  `content="#f7f8fb"`; append Geist to the font link:

```html
    <link
      href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
      rel="stylesheet"
    />
```

- [ ] **Step 6 — scp + build.** scp the three files; on iitgpu11:
  `npm run build` → expect `tsc` + `vite` clean.

- [ ] **Step 7 — Playwright verify light default + toggle.** Restart dev server (warm cache);
  load `/niriksha/home`. Expect: app renders **light** by default (soft-gray canvas, white
  panels), 0 console errors. Click the theme toggle → dark still renders cleanly → toggle
  back to light. Screenshot both.

- [ ] **Step 8 — Commit.**

```bash
git add niriksha/src/styles/global.css niriksha/src/lib/theme.ts niriksha/index.html
git commit -m "feat(niriksha): light-first design tokens (palette, type scale, Geist, radii)"
```

---

## Task 2: Motion + greeting infra (gsap reveal hook, time-aware greeting)

**Files:** Create `niriksha/src/lib/greeting.ts`, `niriksha/src/lib/greeting.test.ts`,
`niriksha/src/lib/useReveal.ts`.

- [ ] **Step 1 — Write the failing greeting test.** `niriksha/src/lib/greeting.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { greetingFor } from './greeting'

describe('greetingFor', () => {
  it('returns Good morning before noon', () => {
    expect(greetingFor(new Date('2026-06-10T08:00:00'))).toBe('Good morning')
  })
  it('returns Good afternoon midday', () => {
    expect(greetingFor(new Date('2026-06-10T14:00:00'))).toBe('Good afternoon')
  })
  it('returns Good evening at night', () => {
    expect(greetingFor(new Date('2026-06-10T21:00:00'))).toBe('Good evening')
  })
})
```

- [ ] **Step 2 — Run it; expect FAIL** (`greetingFor` not exported):
  `npm test -- greeting` → FAIL.

- [ ] **Step 3 — Implement `greeting.ts`:**

```ts
/** Time-aware greeting for the Home hero. Boundaries: <12 morning, <18 afternoon, else evening. */
export function greetingFor(now: Date): string {
  const h = now.getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}
```

- [ ] **Step 4 — Run it; expect PASS.** `npm test -- greeting` → 3 pass.

- [ ] **Step 5 — Implement `useReveal.ts`** (gsap scroll-reveal, reduced-motion gated):

```ts
import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * Reveal direct children of the returned ref on scroll: fade + rise, staggered.
 * No-ops under prefers-reduced-motion (elements stay at their final state).
 */
export function useReveal<T extends HTMLElement = HTMLDivElement>(deps: unknown[] = []) {
  const ref = useRef<T>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const items = Array.from(el.children) as HTMLElement[]
    const ctx = gsap.context(() => {
      gsap.from(items, {
        opacity: 0,
        y: 14,
        duration: 0.5,
        ease: 'power2.out',
        stagger: 0.07,
        scrollTrigger: { trigger: el, start: 'top 85%' },
      })
    }, el)
    return () => ctx.revert()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}
```

- [ ] **Step 6 — scp; build + test.** scp the 3 files; `npm run build` clean;
  `npm test` → existing + greeting tests green.

- [ ] **Step 7 — Commit.**

```bash
git add niriksha/src/lib/greeting.ts niriksha/src/lib/greeting.test.ts niriksha/src/lib/useReveal.ts
git commit -m "feat(niriksha): greeting util + gsap reveal hook (reduced-motion gated)"
```

---

## Task 3: Core primitives — airier, softer, bigger

**Files:** Modify `Card.tsx`, `PageHeader.tsx`, `StatTile.tsx`, `Button.tsx`,
`EmptyState.tsx` in `niriksha/src/components/primitives/`.

**Concrete changes (read each file first; preserve its public props/API):**
- **Card:** default padding `p-5` (was tighter); `rounded-[var(--radius-lg)]`;
  `shadow-[var(--shadow-md)]` on `--color-bg-elevated`; add an `interactive` path that adds
  `hover:shadow-[var(--shadow-lg)] hover:-translate-y-0.5 transition`.
- **PageHeader:** taller block — title at `text-[30px] font-[var(--font-display)] font-semibold
  tracking-[-0.015em]`, optional subtitle `text-[var(--text-md)] text-[var(--color-text-muted)]`,
  actions right-aligned; vertical padding `py-6`/`py-7`; hairline bottom border. Keep existing
  props (`title`, `subtitle?`, `actions?`).
- **StatTile / KpiCard:** number `text-[30px] font-[var(--font-mono)] tabular-nums`, label as
  Overline (`text-[11px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)]`), more
  padding (`p-5`), keep sparkline/trend slots.
- **Button:** primary = accent fill (`bg-[var(--color-accent)] text-[var(--color-accent-fg, white)]`),
  secondary = border, ghost; sizes bumped (sm h-8, md h-9), `rounded-[var(--radius-md)]`.
  Verify `--color-accent-fg` exists; if not, use white. Keep variant/size props.
- **EmptyState:** large centered panel — icon in a tinted circle
  (`bg-[var(--color-accent-subtle)]`), title `text-[var(--text-xl)] font-semibold`, muted body
  `text-[var(--text-md)]`, optional CTA. Keep props (`icon`, `title`, `description`, `action?`).

- [ ] **Step 1 — Edit the five primitives** per the above; match existing className idioms.
- [ ] **Step 2 — scp; build.** `npm run build` clean.
- [ ] **Step 3 — Run existing tests.** `npm test` → green (primitives have light tests; update
  any snapshot/text assertions only if they break on the new structure — do not weaken them).
- [ ] **Step 4 — Playwright spot-check.** Load `/niriksha/home` (light) — Cards/PageHeader/
  StatTiles render airier with soft shadows; 0 console errors.
- [ ] **Step 5 — Commit.**

```bash
git add niriksha/src/components/primitives/Card.tsx niriksha/src/components/primitives/PageHeader.tsx niriksha/src/components/primitives/StatTile.tsx niriksha/src/components/primitives/Button.tsx niriksha/src/components/primitives/EmptyState.tsx
git commit -m "feat(niriksha): airier, softer core primitives (card/header/stat/button/empty)"
```

---

## Task 4: Fully-light shell (Sidebar, AppShell/TopBar)

**Files:** Modify `niriksha/src/components/shell/Sidebar.tsx`,
`niriksha/src/components/shell/AppShell.tsx`.

**Concrete changes (read first; preserve collapse + nav behavior):**
- **Sidebar:** light surface (`bg-[var(--color-bg-elevated)]` or canvas), hairline right
  border, Overline group labels, generous item spacing (`py-2`/`gap`), active item =
  `bg-[var(--color-accent-subtle)] text-[var(--color-accent)]` + a 2px left accent bar; hover
  `bg-[var(--color-bg-hover)]`. Keep the collapse/icon-rail logic.
- **AppShell / TopBar:** light top bar, taller (`h-14`), keep ConnectionPill + ThemeToggle;
  content area on `--color-bg` (soft gray) with wider gutters and a centered max-width wrapper
  (`max-w-[1280px] mx-auto px-6/px-8`) applied to page content. Keep theme init.

- [ ] **Step 1 — Edit Sidebar + AppShell** per the above.
- [ ] **Step 2 — scp; build.** clean.
- [ ] **Step 3 — Run tests.** `npm test` → green (Sidebar has a test asserting group labels;
  keep those labels).
- [ ] **Step 4 — Playwright.** Load app — light shell, active nav has accent bar, collapse
  still works, theme toggle persists; light + dark both clean; 0 errors.
- [ ] **Step 5 — Commit.**

```bash
git add niriksha/src/components/shell/Sidebar.tsx niriksha/src/components/shell/AppShell.tsx
git commit -m "feat(niriksha): fully-light app shell (sidebar + top bar)"
```

---

## Task 5: Home redesign (hero → KPI band → bento) — CHECKPOINT

**Files:** Modify `niriksha/src/pages/DashboardHome.tsx`; create
`niriksha/src/components/reactbits/BentoCard.tsx` (vendor + re-tokenize a reactbits
spotlight/bento card).

**Concrete design:**
- **Hero band:** `greetingFor(new Date())` (Display 46, `--font-display`) + cluster summary
  line (muted Body-lg) + primary "New collection" Button; a restrained accent gradient wash
  (reuse `--gradient-hero`/Aurora subtly, light). GradientText (reactbits) may accent the
  product/cluster name.
- **KPI band:** 5 StatTiles (Collections / Vectors / Fields / Shards / Version) in a responsive
  grid with generous gaps, CountUp numbers, gsap staggered entrance via `useReveal`.
- **Bento grid:** modality-mix donut card + "All collections" table card (comfortable rows,
  modality chips) using BentoCard/SpotlightCard; airy gaps; reveal on scroll.

- [ ] **Step 1 — Vendor BentoCard** from reactbits (via `mcp__reactbits` get_component);
  re-tokenize colors to design tokens; keep it a thin presentational wrapper.
- [ ] **Step 2 — Rebuild DashboardHome** with hero + KPI band + bento, wired to the existing
  data hooks (`useCollectionStore` / overview metrics — reuse current data fetching, only the
  presentation changes). Use `useReveal` for entrance.
- [ ] **Step 3 — scp; build + test.** clean; existing tests green.
- [ ] **Step 4 — Playwright + screenshots (light & dark).** Capture `niriksha-home-after.png`
  (light) and dark. Verify real data renders, 0 console errors, animations play once.
- [ ] **Step 5 — CHECKPOINT: present before/after screenshots to the user for sign-off**,
  including the **final display-font pick (Geist vs General Sans)**. Adjust per feedback
  before proceeding.
- [ ] **Step 6 — Commit.**

```bash
git add niriksha/src/pages/DashboardHome.tsx niriksha/src/components/reactbits/BentoCard.tsx
git commit -m "feat(niriksha): redesigned open/airy Dashboard Home (hero + KPI band + bento)"
```

---

## Task 6: Login + Atlas chrome + empty states

**Files:** Modify `niriksha/src/pages/Login.tsx`, `niriksha/src/pages/atlas/AtlasPage.tsx`,
and apply the new `EmptyState` where empty panels appear (Atlas already uses an empty panel).

**Concrete changes:**
- **Login:** keep the dramatic Aurora hero; update headings to `--font-display`, the new type
  scale + spacing, and the new Button styles. Ensure the standalone route still works.
- **Atlas:** swap the cramped 48px header for the taller PageHeader pattern; restyle the
  control row (selects/buttons) to the light tokens; the scatter canvas stays as-is.
- **Empty states:** ensure Atlas's empty panel uses the new `EmptyState` primitive (large,
  centered).

- [ ] **Step 1 — Edit Login, AtlasPage, empty panels.**
- [ ] **Step 2 — scp; build + test.** clean; green.
- [ ] **Step 3 — Playwright (light & dark):** `/login`, `/atlas` (generate a projection to
  confirm UMAP still works post-restyle), an empty state. 0 console errors. Screenshots.
- [ ] **Step 4 — Commit.**

```bash
git add niriksha/src/pages/Login.tsx niriksha/src/pages/atlas/AtlasPage.tsx
git commit -m "feat(niriksha): light/open Login + Atlas chrome + empty-state pattern"
```

---

## Task 7: Final verification + handoff

- [ ] **Step 1 — Full build + tests.** `npm run build` clean; `npm test` → all green.
- [ ] **Step 2 — Playwright sweep (light + dark):** Home, Login, Atlas, Collections (inherits
  tokens), Search, Observability — confirm no broken/illegible surfaces, 0 new console errors.
  Note any dense page that looks off (logged for the later "dense pages" pass — out of scope
  here).
- [ ] **Step 3 — Contrast check:** verify AA on light (text vs canvas/cards; accent buttons).
- [ ] **Step 4 — Capture before/after** Home screenshots in the PR/report.
- [ ] **Step 5 — finishing-a-development-branch** to merge/push per user choice.

---

## Self-review (plan vs spec)

- **Spec coverage:** tokens/palette (T1), type scale + Geist (T1), spacing/radii (T1, T3, T4),
  elevation (T1, T3), motion/gsap (T2, T5), reactbits (T5), light shell (T4), Home/Login/Atlas/
  empty (T5, T6), comfortable-but-efficient data (inherited tokens + T3 primitives), dark parity
  (T1 keeps `@theme` dark fallback + toggle, verified each task), verification (T1–T7). ✓
- **Out of scope (as speced):** dense-page *layouts* (Collections browser, Search, Pipelines,
  Observability, Quality) — they inherit tokens but keep layouts; flagged in T7 for a later pass.
- **Display font:** default Geist (Google Fonts); General Sans is the live alternative decided at
  the T5 checkpoint (if chosen, swap the font link to Fontshare / self-host + `--font-display`).
- **Risk:** components referencing `--color-base-N` directly (scrollbar, dot-grid, RecordCell)
  rely on the light base scale (T1 step 2 sets it) — spot-check in T7.
