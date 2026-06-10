# Veda Marketing Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline,
> recommended for this visual build). Steps use checkbox (`- [ ]`). Visual surfaces are
> verified live with Playwright (light **and** dark) + screenshots, not unit tests.

**Goal:** Build a polished marketing landing page for Veda as a new `website/` Vite app that
reuses Niriksha's design system, so the brand reads as one product.

**Architecture:** A standalone Vite + React 19 + Tailwind 4 app under `website/`. It copies
Niriksha's `global.css` tokens, fonts (Geist/Inter/JetBrains), theme system, `useReveal` gsap
hook, and the vendored reactbits components (Aurora/GradientText/SpotlightCard/CountUp) +
Button. A single-scroll `Landing` page composes 9 independent section components, with an
animated vector-scatter hero centerpiece. Light-first with a dark toggle.

**Tech Stack:** Vite 8, React 19.2.4, Tailwind 4.2.2 (`@tailwindcss/vite`), react-router-dom 7,
gsap 3.15, ogl 1.0.11, framer-motion, lucide-react, class-variance-authority. `verbatimModuleSyntax`
is on → use `import type` for type-only imports; `noUnusedLocals/Parameters` on → no unused vars.

**Branch:** `feat/veda-website` (created; spec committed `f0596a6`).

**Dev env:** new Vite dev server on **port 5175** (`http://gpu-host-a.internal.example:5175/`).
Workflow: edit local → scp up (or remote `cp` for files copied from niriksha) → `npm run build`
over ssh → Playwright from the local box. node v24.3.0 via nvm.

---

## File structure

```
website/
  package.json, vite.config.ts, tsconfig.json, tsconfig.app.json, tsconfig.node.json
  index.html                       # fonts + theme init + light pre-paint
  src/
    main.tsx, App.tsx              # router: '/' -> Landing
    styles/global.css              # COPIED from niriksha (tokens)
    lib/cn.ts, theme.ts, useReveal.ts   # COPIED from niriksha
    components/
      reactbits/{Aurora,GradientText,SpotlightCard,CountUp}.tsx  # COPIED from niriksha
      primitives/{Button,Badge,ThemeToggle}.tsx                  # COPIED/adapted from niriksha
      Nav.tsx, Footer.tsx, Section.tsx, HeroScatter.tsx, CodeShowcase.tsx
    sections/{Hero,Stats,Differentiators,CodeSection,HowItWorks,Architecture,CTA}.tsx
    pages/Landing.tsx
    content/{features,stats,code,pipeline}.ts
```

---

## Task 1: Scaffold the `website/` app

**Files:** Create all config + entry files; copy the reusable design-system files from niriksha.

- [ ] **Step 1 — Create `website/package.json`:**

```json
{
  "name": "veda-website",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": { "dev": "vite", "build": "tsc -b && vite build", "preview": "vite preview" },
  "dependencies": {
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "framer-motion": "^12.38.0",
    "gsap": "^3.15.0",
    "lucide-react": "^1.16.0",
    "ogl": "^1.0.11",
    "react": "^19.2.4",
    "react-dom": "^19.2.4",
    "react-router-dom": "^7.14.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.2.2",
    "@types/node": "^24.12.2",
    "@types/react": "^19.2.14",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^6.0.1",
    "tailwindcss": "^4.2.2",
    "typescript": "~6.0.2",
    "vite": "^8.0.4"
  }
}
```

- [ ] **Step 2 — `website/vite.config.ts`:**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { allowedHosts: ['.internal.example', 'localhost'] },
})
```

- [ ] **Step 3 — `website/tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`:** copy
  niriksha's three verbatim, except in `tsconfig.app.json` drop the test-related `exclude`
  (no tests here) — keep `"include": ["src"]`, `verbatimModuleSyntax: true`, `jsx: react-jsx`,
  `noUnusedLocals/Parameters: true`, `types: ["vite/client"]`.

- [ ] **Step 4 — `website/index.html`** (mirror niriksha's, retitled; light pre-paint + theme
  init script + Geist/Inter/JetBrains font link):

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="color-scheme" content="light dark" />
    <meta name="theme-color" content="#f7f8fb" />
    <meta name="description" content="Veda — a next-generation multimodal vector database for the era of RAG." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
    <title>Veda — Multimodal Vector Database</title>
    <script>
      try { var t = localStorage.getItem('veda_theme'); document.documentElement.dataset.theme = t === 'dark' ? 'dark' : 'light' }
      catch (e) { document.documentElement.dataset.theme = 'light' }
    </script>
    <style>
      html { margin: 0 } body { margin: 0 }
      html[data-theme='light'], html[data-theme='light'] body { background: oklch(0.985 0.004 255) }
      html[data-theme='dark'], html[data-theme='dark'] body { background: oklch(0.105 0.004 252) }
      #root { min-height: 100vh }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5 — Copy the design-system files from niriksha (remote `cp` on gpu-host-a):**
  `global.css`, `lib/cn.ts`, `lib/theme.ts`, `lib/useReveal.ts`, `components/reactbits/{Aurora,
  GradientText,SpotlightCard,CountUp}.tsx`, `components/primitives/Button.tsx`. (Button imports
  only `cn` + `class-variance-authority` — both available. Drop any import it doesn't need.)

- [ ] **Step 6 — `website/src/main.tsx`:**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/global.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 7 — `website/src/App.tsx`** (router + theme init):

```tsx
import { useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { applyTheme, getStoredTheme } from './lib/theme'
import { Landing } from './pages/Landing'

export default function App() {
  useEffect(() => { applyTheme(getStoredTheme()) }, [])
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  )
}
```

- [ ] **Step 8 — Temporary `website/src/pages/Landing.tsx`** (placeholder to boot):

```tsx
export function Landing() {
  return <div className="p-10 text-[var(--color-text)]">Veda — coming together…</div>
}
```

- [ ] **Step 9 — Install + build + run.** On gpu-host-a: `cd website && npm install`; `npm run build`
  → tsc + vite clean; start dev `npm run dev -- --host 0.0.0.0 --port 5175 --strictPort`.
- [ ] **Step 10 — Playwright:** load `http://gpu-host-a.internal.example:5175/` → light
  canvas, placeholder text renders, 0 console errors.
- [ ] **Step 11 — Commit.**

```bash
git add website
git commit -m "chore(website): scaffold veda marketing site (vite+react+tailwind, niriksha tokens)"
```

---

## Task 2: Site shell — Nav, Footer, Section wrapper

**Files:** Create `components/Nav.tsx`, `components/Footer.tsx`, `components/Section.tsx`,
`components/primitives/ThemeToggle.tsx` (copy/adapt from niriksha), update `pages/Landing.tsx`.

- [ ] **Step 1 — `Section.tsx`:** a wrapper — `<section>` with centered `max-w-[1200px] mx-auto
  px-6 md:px-8 py-20 md:py-28`, optional overline (`text-[12px] uppercase tracking-[0.14em]
  font-semibold text-[var(--color-accent)]`) + heading (`text-[clamp(28px,4vw,40px)]
  [font-family:var(--font-display)] font-semibold tracking-[-0.02em]`) + optional lede.
- [ ] **Step 2 — `Nav.tsx`:** fixed top bar; Veda shard mark + wordmark (Geist), links
  (Features/Architecture/Docs/GitHub), `Button` "Get started", `ThemeToggle`. Transparent at
  top → on scroll (`window.scrollY > 8`) add `.glass` bg + bottom border (track via a scroll
  listener + state).
- [ ] **Step 3 — `Footer.tsx`:** 3 columns (Product / Developers / brand line) + copyright,
  hairline top border, muted text.
- [ ] **Step 4 — `Landing.tsx`:** `<Nav/>` + `<main>` (placeholder sections come next) + `<Footer/>`.
- [ ] **Step 5 — scp; build; Playwright** (light + dark) — nav + footer render, scroll glass
  works, theme toggle persists, 0 errors.
- [ ] **Step 6 — Commit:** `feat(website): site shell (nav, footer, section primitive)`.

---

## Task 3: Hero + HeroScatter + Stats — CHECKPOINT

**Files:** Create `components/HeroScatter.tsx`, `sections/Hero.tsx`, `sections/Stats.tsx`,
`content/stats.ts`; update `Landing.tsx`.

- [ ] **Step 1 — `HeroScatter.tsx`:** a `<canvas>` sized via `ResizeObserver`, drawn at
  `devicePixelRatio`. Generate ~120 points with random positions + small velocities + a
  modality-palette color (read the 6 `--color-modality-*` vars via `getComputedStyle`). A
  `requestAnimationFrame` loop drifts them (wrap at edges) and draws faint links between points
  within a radius (cap links for perf). Under `matchMedia('(prefers-reduced-motion: reduce)')`,
  draw one static frame and don't start the loop. Clean up RAF + observer on unmount.
- [ ] **Step 2 — `content/stats.ts`:** `export const STATS = [{value:7,label:'index algorithms'},
  {value:3,label:'service deploy'},{value:6,label:'distance metrics'},{value:5,label:'modalities'}]`.
- [ ] **Step 3 — `Hero.tsx`:** full-viewport-ish band. `Aurora` wash (lazy) + `HeroScatter`
  layered behind/beside; overline "Multimodal vector database"; headline (Geist
  `clamp(40px,7vw,76px)` semibold, with a `GradientText` accent word); one-sentence subhead;
  dual CTA (`Button` primary "Get started", secondary "View on GitHub"). gsap reveal on the
  text column.
- [ ] **Step 4 — `Stats.tsx`:** the 4 stats as `CountUp` numbers (Geist) + labels, in a row,
  `useReveal` stagger.
- [ ] **Step 5 — scp; build; Playwright (light + dark):** capture `website-hero-light.png` /
  `-dark.png`. Scatter animates; 0 console errors.
- [ ] **Step 6 — CHECKPOINT: present hero screenshots to the user for sign-off** (it anchors the
  page). Adjust per feedback.
- [ ] **Step 7 — Commit:** `feat(website): animated vector-scatter hero + stats band`.

---

## Task 4: Differentiators + Code showcase

**Files:** Create `content/features.ts`, `content/code.ts`, `components/CodeShowcase.tsx`,
`sections/Differentiators.tsx`, `sections/CodeSection.tsx`; update `Landing.tsx`.

- [ ] **Step 1 — `content/features.ts`:** the 7 differentiators (from `competitive_advantages.md`)
  as `{title, blurb, icon (lucide name), modality}` — Multimodal-native pipeline (Boxes, image),
  Pure-Rust + SIMD (Cpu, text), 7 index algorithms (Layers, doc), Multi-vector fusion (GitMerge,
  mixed), Document intelligence (FileText, doc), Lightweight 3-service (Server, audio),
  OTel-first observability (Activity, video). Keep blurbs to 1–2 accurate sentences.
- [ ] **Step 2 — `Differentiators.tsx`:** `Section` ("Why Veda") → a responsive grid of
  `SpotlightCard`s; each card: lucide icon in a modality-tinted chip
  (`bg-[color-mix(in_oklch,var(--color-modality-X)_14%,transparent)]`), title (H3), blurb.
  `useReveal` stagger.
- [ ] **Step 3 — `content/code.ts`:** 3 samples (strings) — create multimodal collection,
  multi-vector fusion search (the RRF JSON from `competitive_advantages.md`), document upload.
- [ ] **Step 4 — `CodeShowcase.tsx`:** tabbed (3 tabs) `<pre>` blocks, JetBrains Mono, a rounded
  dark-or-tinted code surface that reads in both themes; tab state via `useState`.
- [ ] **Step 5 — `CodeSection.tsx`:** `Section` ("One API call") + `CodeShowcase` + a short lede.
- [ ] **Step 6 — scp; build; Playwright (light + dark)** — 0 errors; cards + code legible.
- [ ] **Step 7 — Commit:** `feat(website): differentiators grid + API code showcase`.

---

## Task 5: How-it-works + Architecture + CTA

**Files:** Create `content/pipeline.ts`, `sections/HowItWorks.tsx`, `sections/Architecture.tsx`,
`sections/CTA.tsx`; update `Landing.tsx`.

- [ ] **Step 1 — `content/pipeline.ts`:** the 6 pipeline steps (Upload, OCR, Chunk, Embed, Index,
  Search) with a one-line caption + lucide icon each.
- [ ] **Step 2 — `HowItWorks.tsx`:** `Section` ("From raw files to search, in one pipeline") →
  horizontal step flow (icon chips + arrow connectors, wraps on mobile), `useReveal` stagger.
- [ ] **Step 3 — `Architecture.tsx`:** `Section` ("Lightweight by default") → two beats: a
  "3 mandatory services vs 5+" comparison (Veda/MinIO/NATS vs etcd+Kafka+MinIO+…), and the
  7-index table (index · best-for · advantage) from `competitive_advantages.md`. Clean table
  styling (overline header, hairline rows), matching the dashboard's data tables.
- [ ] **Step 4 — `CTA.tsx`:** Aurora/gradient band — "Up and running in 30 seconds", a
  `docker compose up` mono line, buttons (Read the docs / Star on GitHub / Open the dashboard).
- [ ] **Step 5 — scp; build; Playwright (light + dark)** — 0 errors.
- [ ] **Step 6 — Commit:** `feat(website): how-it-works + architecture + CTA band`.

---

## Task 6: Final verification + polish

- [ ] **Step 1 — Full `npm run build`** clean.
- [ ] **Step 2 — Playwright full-page screenshots** (`website-landing-light.png` / `-dark.png`),
  desktop width; scan for overflow/contrast/illegibility; 0 console errors.
- [ ] **Step 3 — AA contrast check** on light (hero, CTA buttons, code surface).
- [ ] **Step 4 — Consistency check** vs the dashboard (fonts, accent, modality colors, feel).
- [ ] **Step 5 — Polish pass** for any rough spots found.
- [ ] **Step 6 — finishing-a-development-branch** to merge/push per user choice.

---

## Self-review (plan vs spec)

- **Spec coverage:** stack/structure (T1), tokens/fonts/theme reuse (T1), shell/nav/footer (T2),
  hero + animated scatter + stats (T3), differentiators + code showcase (T4), how-it-works +
  architecture + CTA (T5), footer (T2), light+dark + verification (T1–T6), content sourced from
  competitive_advantages/README (T3–T5). ✓
- **Out of scope (as speced):** docs site (next cycle), deployment.
- **Risk:** reading `--color-modality-*` via `getComputedStyle` in HeroScatter must run after
  mount (vars resolved) — handle in an effect; re-read on theme change.
- **Risk:** `verbatimModuleSyntax` → all type-only imports need `import type`; `noUnusedLocals`
  → prune unused. Build will catch both.
