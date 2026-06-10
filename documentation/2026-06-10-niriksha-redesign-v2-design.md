# Niriksha Redesign v2 — "Open & Airy" (light-first) Design Spec

**Date:** 2026-06-10
**Status:** Approved direction; pending spec review → implementation plan
**Motivation:** Benchmarked against Kibana (Elastic UI). Kibana reads as modern and
inviting because of *openness* — generous whitespace, a strong multi-size type
hierarchy, purposeful color, and a light/airy canvas. Niriksha's current
"disciplined premium, dark-first, dense" language now reads as dated by
comparison. This spec pivots the design language toward open/airy **while keeping
Niriksha's identity** (indigo-violet accent + multimodal palette).

## Approved decisions (from brainstorming)

1. **Theme:** Light-first. A light, airy theme becomes the default; dark mode is kept
   at full parity as a toggle.
2. **Approach:** "B — Modern airy, Niriksha identity." Adopt Kibana's *principles*
   (space, hierarchy, color), not its exact look. Keep indigo-violet (hue ~282) +
   the modality palette as the signature. Tasteful gsap motion + reactbits accents.
3. **Scope (this pass):** Shared design tokens + the showcase/shell surfaces — Home,
   Login, Atlas, empty/landing states, and the shell (sidebar / top bar / page
   header) + the core primitives they use. Dense data pages (Collections browser,
   Search, Pipelines, Observability, Quality) inherit the new tokens automatically
   but keep their layouts for a later pass.
4. **Data density:** "Comfortable but efficient." Working surfaces get more breathing
   room + stronger hierarchy + color, but stay scannable (not as airy as the
   landing).
5. **Shell:** Fully light shell — light sidebar + light top bar + soft-gray content.
   Accent used sparingly (active nav).
6. **Display type:** Inter (UI/body) + a **modern display sans** for large titles +
   JetBrains Mono (numerals/code). Default is **Geist** (self-hosted woff2,
   SIL OFL); **General Sans** is the live alternative shown in the Home prototype
   for the final pick.

## Non-goals

- Not redesigning every page in this pass (see scope #3).
- Not changing app behavior, data flow, or API usage — this is presentational.
- Not removing dark mode; it stays first-class via the toggle.
- No new heavy dependencies beyond the agreed reactbits components, gsap, and one
  display webfont (self-hosted).

## Foundations — design tokens

All tokens are CSS custom properties in `niriksha/src/styles/global.css` under
Tailwind 4 `@theme`. Light is the default `:root`; dark overrides under
`.dark` / `[data-theme="dark"]` (existing mechanism). Values below are the light
theme; dark keeps the current dark palette remapped to the same variable names.

### Color — light theme

| Token | Value (oklch) | Use |
|---|---|---|
| `--color-bg` | `0.99 0.004 255` | app canvas (soft cool near-white) |
| `--color-bg-subtle` | `0.975 0.005 255` | section / inset backgrounds |
| `--color-surface` | `1 0 0` | cards / panels (white) |
| `--color-surface-2` | `0.985 0.004 255` | nested surfaces, table header |
| `--color-border` | `0.92 0.005 255` | hairline borders |
| `--color-border-strong` | `0.86 0.006 255` | emphasized borders, inputs |
| `--color-text` | `0.27 0.02 265` | primary ink (near-slate) |
| `--color-text-muted` | `0.50 0.015 265` | secondary text |
| `--color-text-subtle` | `0.62 0.012 265` | tertiary / captions |
| `--color-accent` | `0.55 0.20 282` | primary buttons, links, active |
| `--color-accent-hover` | `0.50 0.21 282` | hover |
| `--color-accent-fg` | `1 0 0` | text on accent |
| `--color-accent-tint` | `0.96 0.03 282` | accent fills (active nav bg) |
| `--color-accent-tint-2` | `0.93 0.05 282` | stronger tint, badges |

**Modality palette (light-tuned, legible on white):**
text `0.55 0.18 264` · image `0.60 0.16 155` · audio `0.62 0.15 70` ·
video `0.58 0.20 18` · doc `0.55 0.18 300` · mixed `0.62 0.13 220`.

**Semantic:** success `0.60 0.15 150` · warning `0.70 0.15 75` ·
error `0.58 0.20 25` · info = accent.

**Chart ramp (`--color-chart-1..8`):** 8 distinct hues spread across the wheel,
light-tuned for ~0.60 L / moderate chroma so they read on white.

**Elevation (light):**
- `--shadow-sm`: `0 1px 2px oklch(0.2 0.02 265 / 0.06)`
- `--shadow-md`: `0 2px 8px -1px oklch(0.2 0.02 265 / 0.08), 0 1px 2px oklch(0.2 0.02 265 / 0.06)`
- `--shadow-lg`: `0 10px 30px -8px oklch(0.2 0.02 265 / 0.12)`
- `--shadow-glow`: accent-tinted glow, **showcase only** (hero/login/atlas).

**Radii:** `--radius-xs` 6 · `--radius-sm` 8 · `--radius-md` 10 · `--radius-lg` 14
(cards) · `--radius-xl` 18 · pill 999.

### Typography

Families: `--font-sans` = Inter; `--font-display` = modern display sans
(default Geist, General Sans as the live alternative; self-hosted woff2);
`--font-mono` = JetBrains Mono.

| Role | size/line | weight | tracking | family |
|---|---|---|---|---|
| Display | 44 / 1.05 | 600 | -0.02em | display |
| H1 | 30 / 1.15 | 600 | -0.015em | display |
| H2 | 22 / 1.25 | 600 | -0.01em | sans |
| H3 | 18 / 1.35 | 600 | — | sans |
| Body-lg | 16 / 1.6 | 400 | — | sans |
| Body | 15 / 1.55 | 400 | — | sans |
| Small | 13 / 1.5 | 400 | — | sans |
| Caption | 12 / 1.4 | 500 | 0.01em | sans |
| Overline | 11 / 1.3 | 600 | 0.08em UPPER | sans |

Numerals (KPIs, metrics, ids): JetBrains Mono, tabular-nums. This is the
"different typographies / font sizes" the redesign is after — today the app tops
out near 24px and is visually flat.

### Spacing & layout

Base unit 4px. Steps: 1=4 2=8 3=12 4=16 5=20 6=24 8=32 10=40 12=48 16=64.
- Page gutter: 32 (lg screens), 20 (md), 16 (sm).
- Section vertical gap: 28–32.
- Card padding: 20–24.
- Grid gaps: 16–24.
- Centered content max-width ~1280 so big screens breathe.

### Motion

`--duration-fast` 120ms · `--duration-base` 200ms · `--duration-slow` 360ms.
Easing: `--ease-out` cubic-bezier(0.22,0.61,0.36,1); `--ease-emphasized`
cubic-bezier(0.2,0.8,0.2,1). gsap reveal = opacity 0→1 + translateY 12→0,
stagger 60–80ms. **All motion gated on `prefers-reduced-motion`** (skip → final
state).

## Components / primitives (updated)

- **Card:** white surface, radius-lg, `--shadow-md`, padding-5/6. Interactive
  variant lifts on hover (`--shadow-lg` + translateY -2).
- **PageHeader (new, taller):** Display/H1 title + muted subtitle + right-aligned
  actions; ~py-7/8; hairline bottom border. Replaces today's cramped 48px bar.
- **StatTile / KpiCard:** large mono number (H1/Display), Overline label, optional
  sparkline + trend, CountUp entrance; more padding.
- **Sidebar (light):** white/subtle bg, Overline group labels, generous item
  spacing, active = accent-tint bg + accent text + 2px left accent bar.
- **TopBar (light):** taller, search + health pill + theme toggle.
- **Button:** primary (accent fill), secondary (border), ghost; larger hit areas,
  radius-md.
- **EmptyState (new pattern):** large centered panel — icon in a tinted circle, H2
  title, muted body, CTA. Mirrors Kibana's welcome panels.
- **ModalityChip / Badge / Tabs / SegmentedControl / Switch:** restyled to light
  tokens; unchanged API.

## Surfaces (this pass)

### Home / Overview
1. **Hero band:** time-aware greeting (Display) + cluster name/summary line +
   primary CTA ("New collection"); subtle accent gradient wash (reactbits Aurora
   or a soft gradient) kept light and restrained.
2. **KPI band:** 5 airy cards (Collections / Vectors / Fields / Shards / Version)
   with CountUp + sparkline, generous padding, gsap staggered entrance.
3. **Bento grid:** Modality-mix donut card + "All collections" table card
   (comfortable rows, modality chips), with breathing room.

### Login
Keep the dramatic hero (Aurora) — a dark, striking login reads well as a contrast
moment even in a light app — refreshed to the new type scale + spacing + button
styles.

### Atlas
Apply light tokens, the taller PageHeader, and more legible controls. The scatter
canvas itself stays functional (its own dark plotting surface is fine); the
chrome around it gets the new language.

### Empty states (app-wide pattern)
Adopt the new EmptyState everywhere an empty/landing panel appears (Atlas,
Collections, Search, etc.) — large, centered, friendly.

## Tooling

- **reactbits:** SpotlightCard (KPI/feature cards), GradientText (hero accent),
  Aurora (login/home hero) — already vendored — plus pull a Bento/Magic-Bento grid
  and a border-gradient/"shiny" card for the showcase. Vendor + re-tokenize (avoid
  reactbits' low-quality form/button components).
- **gsap:** ScrollTrigger reveals, staggered KPI/row entrances, hero text reveal,
  hover lift/tilt micro-interactions. Lazy-load on showcase routes only.
- **ui-ux principles:** clear hierarchy, one primary action per view, generous
  targets, AA contrast on light, consistent spacing rhythm.

## Verification / success criteria

- Build green (`tsc` + `vite`); existing vitest suite stays green; unit tests for
  any new pure logic (e.g. greeting/time helpers).
- Playwright visual pass on **Home, Login, Atlas, an empty state** in **both light
  and dark**, 0 new console errors.
- Before/after screenshots captured for Home (the headline surface).
- AA contrast verified on light surfaces (text vs bg, accent buttons).
- `prefers-reduced-motion` honored (animations skipped).
- The Home prototype is built first and shown for visual sign-off (incl. the final
  display-font pick) before rolling the language out to Login/Atlas/empty states.

## Rollout order (implementation)

1. Tokens + fonts (global.css) + dark-parity remap.
2. Core primitives (Card, PageHeader, StatTile, Button, EmptyState, Sidebar,
   TopBar).
3. **Home** → screenshot sign-off (font + direction).
4. Login + Atlas chrome + empty states.
5. Visual verification (light + dark) + before/after.
