# Veda Website Redesign (Obsidian + Saffron) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline — visual
> build with a live Playwright session + a hero checkpoint). Steps use `- [ ]`. Visual tasks
> verify via build + Playwright at 1440/1920/2560, not unit tests.

**Goal:** Reskin the `website/` app to the approved Obsidian + Saffron direction, fix the
full-bleed layout bug, and upgrade the hero graph to an interactive query-vector visualization.

**Architecture:** Same app/structure (`website/`), new token layer (dark-only, replaces the
copied Niriksha tokens), new fonts (Space Grotesk / IBM Plex Sans / JetBrains Mono), restyled
components in place. Two new vendored reactbits pieces (DotGrid, DecryptedText) + ShinyText.
`useReveal` upgraded to gsap ScrollTrigger (the site scrolls the window). HeroScatter gains
pointer interaction (kNN links to cursor + stir impulses).

**Tech stack:** unchanged (Vite 8, React 19, Tailwind 4, gsap, lucide). `verbatimModuleSyntax`
+ `noUnusedLocals` are on. **Branch:** `feat/website-redesign-obsidian` (spec `84031e1`).
**Dev:** port 5175 on gpu-host-a; edit local → scp → build over ssh → Playwright local.

---

## Task 1: Foundation — dark-only tokens, fonts, primitives

**Files:** Rewrite `website/src/styles/global.css`; modify `website/index.html`,
`website/src/App.tsx`, `website/src/components/primitives/Button.tsx`; delete theme toggle
usage (Nav edited in Task 2; remove `ThemeToggle.tsx` + `lib/theme.ts` when no importers).

- [ ] **Step 1 — `global.css`:** replace entirely with the spec's token table under `@theme`
  (bg `#0A0C10`, raised `#0F1218`, surface `#12151C`/`#171B23`, hairline borders
  `rgba(255,255,255,0.08|0.16)`, text `#F2F4F8`/`#9BA3B0`/`#6B7280`, accent `#FFB224`,
  accent-hot `#FF8A00`, accent-ink `#161002`, accent-dim `rgba(255,178,36,0.14)`, positive
  `#3DD68C`, danger `#F4564A`; graph palette `--color-node-1..6`; radii 4/6/8/12; fonts
  `--font-display:"Space Grotesk"`, `--font-sans:"IBM Plex Sans"`, `--font-mono:"JetBrains
  Mono"`; durations 150/240/400ms). Base layer: body bg/text/sans 16px/1.6; saffron 2px
  focus-visible; saffron-tinted selection; thin dark scrollbar; `prefers-reduced-motion`
  global damp; keep `.glass` (retuned to obsidian) ; add `.section-rule` (full-width hairline)
  and `.mono-label` (mono 12px uppercase +0.08em) utilities. No `[data-theme]` blocks.
- [ ] **Step 2 — `index.html`:** fonts link → `Space+Grotesk:wght@500;600;700`,
  `IBM+Plex+Sans:wght@400;500;600`, `JetBrains+Mono:wght@400;500;700`; drop the theme-init
  script; pre-paint `background:#0A0C10`; `meta theme-color #0A0C10`; `color-scheme: dark`.
- [ ] **Step 3 — `App.tsx`:** remove `applyTheme`/`getStoredTheme` effect + imports.
- [ ] **Step 4 — `Button.tsx`:** variants → primary: saffron bg, `--color-accent-ink` text,
  hover `--color-accent-hot` + glow `0 0 24px var(--color-accent-dim)`; secondary: transparent,
  `--color-border-strong` border, hover saffron border/text; ghost: muted→text. Radius-md,
  mono-friendly weight 600.
- [ ] **Step 5 — scp; build** (expect TS errors only if ThemeToggle still imported — Nav fixed
  next task; if blocking, stub-remove the import now).
- [ ] **Step 6 — Commit** `feat(website): obsidian+saffron token foundation (dark-only, new type)`.

## Task 2: Shell — Nav, Footer, Section

**Files:** `Nav.tsx`, `Footer.tsx`, `Section.tsx`; delete
`components/primitives/ThemeToggle.tsx` and `lib/theme.ts`.

- [ ] **Step 1 — Nav:** remove ThemeToggle; wordmark in `--font-display` 600; links in
  `.mono-label` style; saffron primary CTA; scrolled state = obsidian glass + hairline.
- [ ] **Step 2 — Footer:** `.mono-label` column titles, hairline top rule, saffron link hover.
- [ ] **Step 3 — Section:** props gain `number` (e.g. `"02"`); overline renders
  `<span class="mono-label" style="color:var(--color-accent)">{number} / {overline}</span>`;
  title in `--font-display` `clamp(30px,4vw,48px)`; container `max-w-[1320px] px-6 md:px-10`.
- [ ] **Step 4 — Delete** ThemeToggle.tsx + theme.ts (grep first: no remaining importers).
- [ ] **Step 5 — scp; build; Playwright smoke** (renders dark, no toggle, 0 errors). Commit
  `feat(website): obsidian shell (nav/footer/numbered sections)`.

## Task 3: Hero v2 (interactive graph, full-bleed) + Stats — CHECKPOINT

**Files:** Rewrite `HeroScatter.tsx`; rewrite `sections/Hero.tsx`; restyle
`sections/Stats.tsx`; vendor `components/reactbits/DotGrid.tsx` +
`components/reactbits/DecryptedText.tsx` (via reactbits MCP `get_component`, re-tokenized);
replace `lib/useReveal.ts` with a ScrollTrigger version. Delete `Aurora.tsx` + `GradientText.tsx`
usage (hero no longer uses them; remove files if unreferenced).

- [ ] **Step 1 — `useReveal.ts` (ScrollTrigger):**

```ts
import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
gsap.registerPlugin(ScrollTrigger)

/** Staggered fade-rise of direct children when the container scrolls into view. */
export function useReveal<T extends HTMLElement = HTMLDivElement>(deps: unknown[] = []) {
  const ref = useRef<T>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const items = Array.from(el.children) as HTMLElement[]
    if (!items.length) return
    const ctx = gsap.context(() => {
      gsap.from(items, {
        opacity: 0, y: 18, duration: 0.55, ease: 'power2.out', stagger: 0.07,
        scrollTrigger: { trigger: el, start: 'top 82%', once: true },
      })
    }, el)
    return () => ctx.revert()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return ref
}
```

- [ ] **Step 2 — `HeroScatter.tsx` v2 (interactive).** Keep DPR/ResizeObserver plumbing. New:
  node palette from `--color-node-1..6`; pointer state + kNN links + stir:

```ts
// inside the effect, after seed():
const pointer = { x: -9999, y: -9999, px: -9999, py: -9999, active: false }
const onMove = (e: PointerEvent) => {
  const r = canvas.getBoundingClientRect()
  pointer.px = pointer.x; pointer.py = pointer.y
  pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top
  pointer.active = pointer.x >= 0 && pointer.x <= r.width && pointer.y >= 0 && pointer.y <= r.height
  if (!pointer.active) return
  const dxv = pointer.x - pointer.px, dyv = pointer.y - pointer.py
  for (const p of pts) {                       // stir: impulse ∝ proximity, clamped
    const dx = p.x - pointer.x, dy = p.y - pointer.y
    const d2 = dx * dx + dy * dy
    if (d2 < 120 * 120) {
      const f = (1 - Math.sqrt(d2) / 120) * 0.05
      p.ix = Math.max(-1.6, Math.min(1.6, p.ix + dxv * f))
      p.iy = Math.max(-1.6, Math.min(1.6, p.iy + dyv * f))
    }
  }
}
const onLeave = () => { pointer.active = false; pointer.x = -9999; pointer.y = -9999 }
canvas.addEventListener('pointermove', onMove)
canvas.addEventListener('pointerleave', onLeave)
// step(): p.x += p.vx + p.ix; p.y += p.vy + p.iy; p.ix *= 0.94; p.iy *= 0.94 (Pt gains ix,iy=0)
// draw(): after base pass — if pointer.active:
//   const R = 260, K = 7
//   const near = pts.map(p => ({ p, d: Math.hypot(p.x - pointer.x, p.y - pointer.y) }))
//     .filter(o => o.d < R).sort((a, b) => a.d - b.d).slice(0, K)
//   for (const { p, d } of near):  saffron edge cursor→p, lineWidth 1.2,
//     alpha (1 - d / R) * 0.6;  re-draw p at r * 1.6 with '#FFB224'
//   cursor ring: stroke '#FFB224' alpha 0.9, radius 9 + crosshair ticks 4px
```

  Reduced-motion: current static-frame behavior, no listeners.
- [ ] **Step 3 — Vendor `DotGrid` + `DecryptedText`** via reactbits MCP; re-tokenize: DotGrid
  base dot `rgba(255,255,255,0.05)`, active `rgba(255,178,36,0.35)`, gap ~28, square-off; trim
  to tailwind/TS variant; remove unneeded deps. DecryptedText: speed ~40ms, reveal once on
  mount, chars from mono set.
- [ ] **Step 4 — `Hero.tsx` rewrite:** full-bleed `<section>` (relative, min-h ~92vh, overflow
  hidden): DotGrid absolute layer (opacity ~0.5) edge-to-edge → graph canvas absolutely
  positioned `left-[44%] right-0 top-0 bottom-0` (so it touches the viewport edge; under
  content on mobile via lower opacity) → content container (1320px): mono overline
  `01 / MULTIMODAL VECTOR DATABASE` (saffron), display headline `Search everything you can
  <DecryptedText>embed</DecryptedText>.` (Space Grotesk, clamp 44–92px), Plex subhead, saffron
  CTA + ghost GitHub button, `$ docker compose up` mono chip. Left fade overlay
  (`linear-gradient(90deg, var(--color-bg) 0%, transparent 55%)`) so text stays readable.
- [ ] **Step 5 — `Stats.tsx`:** hairline-ruled columns (divide-x), numerals Space Grotesk 56px
  + mono labels; keep CountUp; `bg-[var(--color-bg-raised)]`.
- [ ] **Step 6 — scp; build; Playwright at 1440 AND 2560:** graph touches right edge at both;
  move mouse over hero → kNN saffron links + ring render (screenshot mid-hover); 0 errors.
- [ ] **Step 7 — CHECKPOINT: show hero screenshots to the user.**
- [ ] **Step 8 — Commit** `feat(website): interactive query-vector hero, full-bleed layout, stats`.

## Task 4: Differentiators keyline grid + terminal code showcase

**Files:** `sections/Differentiators.tsx`, `components/CodeShowcase.tsx`,
`sections/CodeSection.tsx`; vendor `components/reactbits/ShinyText.tsx`; `content/features.ts`
(icon tint per node color).

- [ ] **Step 1 — Differentiators:** replace SpotlightCard grid with a keyline grid — one
  bordered container, cells separated by hairlines (`grid sm:grid-cols-2 lg:grid-cols-3` with
  gap-px on a border-color background, cells `bg-[var(--color-bg)] p-7`); cell = mono number
  `01`…`07` (subtle) + icon (node-color) + Space Grotesk H3 + Plex blurb; hover: saffron icon
  + inset hairline glow. Remove SpotlightCard import (delete file if unreferenced).
- [ ] **Step 2 — CodeShowcase:** terminal chrome — titlebar with three dim dots + mono
  filename, tabs as mono labels w/ saffron active underline, body `bg-[#0D1015]` mono 13px,
  saffron `$` prompts (wrap prompt lines), `border-[var(--color-border)]`.
- [ ] **Step 3 — CodeSection heading** uses ShinyText (saffron sheen) on a key phrase.
- [ ] **Step 4 — scp; build; Playwright;** commit `feat(website): keyline features + terminal code showcase`.

## Task 5: Pipeline rail, architecture, CTA, footer polish

**Files:** `sections/HowItWorks.tsx`, `sections/Architecture.tsx`, `sections/CTA.tsx`.

- [ ] **Step 1 — HowItWorks:** connected rail — steps in a row joined by a hairline with
  saffron progress accents; mono step numbers, icons, Plex captions; ScrollTrigger stagger.
- [ ] **Step 2 — Architecture:** service card + index table restyled (mono index names already;
  hairline rows on `--color-surface`, saffron checks, terminal chip for compose line).
- [ ] **Step 3 — CTA:** full-bleed DotGrid band; display heading; saffron CTA w/ glow; chips.
- [ ] **Step 4 — scp; build; Playwright;** commit `feat(website): pipeline rail + architecture + CTA (obsidian)`.

## Task 6: Multi-width verification + polish

- [ ] **Step 1 —** Build green. Playwright at **1440 / 1920 / 2560**: full-page sweep — no dead
  right band anywhere; section backgrounds full-bleed; 0 console errors.
- [ ] **Step 2 —** Interaction: hover/drag the hero (mouse-move scripted) — links/ring/stir;
  reduced-motion run renders static.
- [ ] **Step 3 —** Contrast spot-checks (saffron/obsidian, muted text); focus rings visible.
- [ ] **Step 4 —** Fix anything found; final screenshots; commit
  `chore(website): redesign verification + polish`.
- [ ] **Step 5 —** finishing-a-development-branch (merge per user choice).

## Self-review

- Spec coverage: tokens/fonts/dark-only (T1), shell (T2), full-bleed + interactive hero +
  DotGrid + DecryptedText + stats (T3), keyline grid + terminal code + ShinyText (T4), rail/
  arch/CTA (T5), multi-width + interaction verification (T6). Bug fix = T3 layout + T6 proof. ✓
- Orphan cleanup: ThemeToggle/theme.ts (T2), Aurora/GradientText/SpotlightCard when
  unreferenced (T3/T4) — grep before delete.
- Types: `Pt` gains `ix, iy` (init 0) — update seed() accordingly.
