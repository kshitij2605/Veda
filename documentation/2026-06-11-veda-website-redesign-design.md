# Veda Website Redesign — "Obsidian + Saffron" Design Spec

**Date:** 2026-06-11
**Status:** Approved direction; pending spec review → implementation plan
**Motivation:** The v1 landing (violet gradient on soft white, Inter/Geist-everywhere, Aurora
wash) reads as generic "AI-generated" design, and at full-screen widths the page visually
stops ~360–680px before the right edge (everything locked in a centered `max-w-[1200px]` with
no full-bleed treatment). This is the first customer touchpoint — it must be distinctive,
credible dev-infra branding. Direction researched via ui-ux-pro-max (style: Dark/OLED dev-tool;
landing: scroll-triggered storytelling) and chosen by the user from three options.

## Approved decisions

1. **Direction A — Obsidian + Saffron.** Near-black obsidian canvas, hairline borders, ONE hot
   accent: saffron amber (nod to Veda's Sanskrit name; no vector-DB competitor owns amber —
   Supabase/Weaviate own green, Milvus/Pinecone blue, ClickHouse yellow).
2. **Dark-only site.** Per the OLED dev-tool style recommendation: no light mode, no theme
   toggle on the marketing site (the dashboard keeps its own theming; the user noted the site
   need not match Niriksha — that can converge later).
3. **Typography:** **Space Grotesk** (display, 500–700, tight tracking) + **IBM Plex Sans**
   (body) + **JetBrains Mono** (labels, numerals, code, section numbering). Kills the
   Inter-everywhere look; all on Google Fonts.
4. **Interactive hero graph (kept + upgraded):** the cursor acts as a **query vector** —
   nearest-neighbor points light up saffron and link to the cursor; drag stirs the field.
5. **Full-bleed layout fix:** backgrounds/visuals run edge-to-edge; the hero graph spans to
   the viewport's right edge; content max-width raised to 1320px with fluid gutters.
6. Tools: **gsap** (ScrollTrigger reveals — the site scrolls the window, so ScrollTrigger is
   safe here), **reactbits** (DotGrid, DecryptedText, ShinyText, CountUp), **Playwright**
   verification at 1440 *and* 1920/2560 widths.

## Non-goals

- Docs site (still the next cycle; it will adopt this language).
- Changing Niriksha (later, if desired).
- New copy claims — content stays sourced from `competitive_advantages.md` / README.
- Light mode.

## Design tokens (replace the copied Niriksha tokens in `website/src/styles/global.css`)

| Token | Value | Use |
|---|---|---|
| `--color-bg` | `#0A0C10` | page canvas (obsidian) |
| `--color-bg-raised` | `#0F1218` | raised bands |
| `--color-surface` | `#12151C` | cards/panels |
| `--color-surface-2` | `#171B23` | hover / nested |
| `--color-border` | `rgba(255,255,255,0.08)` | hairlines |
| `--color-border-strong` | `rgba(255,255,255,0.16)` | emphasized |
| `--color-text` | `#F2F4F8` | primary |
| `--color-text-muted` | `#9BA3B0` | secondary |
| `--color-text-subtle` | `#6B7280` | tertiary / labels |
| `--color-accent` | `#FFB224` | saffron — CTAs, links, highlights |
| `--color-accent-hot` | `#FF8A00` | hover / gradient end |
| `--color-accent-ink` | `#161002` | text on saffron |
| `--color-accent-dim` | `rgba(255,178,36,0.14)` | tints/glows |
| `--color-positive` | `#3DD68C` | success only |
| `--color-danger` | `#F4564A` | errors only |
| graph node colors | `#FFB224 #5AD8FF #B886FF #3DD68C #FF7A59 #F2F4F8` | hero graph + categorical |

Radii: xs 4 / sm 6 / md 8 / lg 12 (sharper than v1 — less "soft AI card").
Shadows: minimal; depth comes from surface steps + hairlines + saffron glow
(`0 0 24px rgba(255,178,36,0.18)`) reserved for CTAs/hero.

Type scale: Display `clamp(44px, 7.5vw, 92px)` (Space Grotesk 700, tracking −0.03em, UPPERCASE
optional per section), H2 `clamp(30px, 4vw, 48px)`, H3 20, body 16/1.6 (Plex Sans), label/mono
12–13 (JetBrains Mono, uppercase, +0.08em). Section numbering `01 /`, `02 /` in mono saffron.

## Layout system (the bug fix)

- Sections are **full-bleed**: backgrounds, hairline rules, and textures span 100vw.
- Content container: `max-w-[1320px]` with `px-6 md:px-10` gutters.
- **Hero:** two-layer — content container holds the headline/CTAs on the left; the graph
  canvas is absolutely positioned from the container's midpoint **to the viewport's right
  edge** (`right: 0` of the section, not the container), fading under the headline on the
  left. At 2560px the graph still touches the screen edge.
- A subtle interactive **DotGrid** (reactbits) texture underlays the hero + CTA bands,
  edge-to-edge.
- Full-width hairline rules separate sections (blueprint-meets-terminal rhythm).

## Interactive hero graph — `HeroScatter` v2

- ~140 points drifting slowly (as now), recolored to the new node palette on obsidian.
- **Query-vector cursor:** the pointer is rendered as a saffron ring; the k (≈7) nearest
  points connect to it with saffron edges (distance-weighted opacity) and brighten/grow.
  This *is* nearest-neighbor search, visualized — motion with meaning.
- **Stir on drag:** pointer velocity imparts gentle impulse to nearby points (damped).
- Idle behavior unchanged (slow drift + faint neutral links). Touch: tap acts as a momentary
  query point. `prefers-reduced-motion`: static frame, no cursor effects.
- Canvas DPR-aware, ResizeObserver-sized (unchanged plumbing).

## Page structure (content preserved, re-skinned + sharpened)

1. **Nav** — obsidian glass on scroll; Space Grotesk wordmark; saffron "Get started".
2. **Hero** — mono overline `01 / MULTIMODAL VECTOR DATABASE`; Space Grotesk display headline
   **"Search everything you can embed."** with a DecryptedText reveal on the key word;
   subhead (current pipeline sentence); CTAs: saffron primary + ghost GitHub; an inline
   `$ docker compose up` mono chip; interactive graph full-bleed right.
3. **Stats band** — CountUp numerals in Space Grotesk/Mono, hairline-ruled columns (7/3/6/5).
4. **Differentiators (`02 /` Why Veda)** — keyline grid (hairline-bordered cells, not floating
   cards): mono number, icon, title, blurb; saffron hover state (border + icon). The 7 features.
5. **Code showcase (`03 /`)** — terminal-chrome block (mono titlebar dots, saffron prompt),
   tabs in mono; ShinyText accent on the section heading.
6. **How it works (`04 /`)** — 6-step pipeline as a connected horizontal rail with mono step
   numbers, gsap ScrollTrigger stagger.
7. **Architecture (`05 /`)** — 3-service card + 7-index table (mono index names, hairline rows).
8. **CTA band** — DotGrid texture, display-size "Up and running in 30 seconds", saffron CTA,
   `docker compose up` chip.
9. **Footer** — mono labels, hairline top rule, saffron hover links.

Motion: gsap ScrollTrigger section reveals (fade-rise, 0.5s, stagger 60–80ms), nav glass
transition, CTA glow pulse on hover; everything `prefers-reduced-motion`-gated; durations
150–400ms per UX guidance.

## Accessibility

- Saffron `#FFB224` on `#0A0C10` ≈ 11:1; body `#F2F4F8` on `#0A0C10` ≈ 17:1; muted `#9BA3B0`
  ≥ 7:1 — all AA+. Buttons: `--color-accent-ink` on saffron (≥ 12:1).
- Focus rings saffron 2px; touch targets ≥ 44px; reduced-motion respected everywhere.

## Verification

- Build green; Playwright at **1440, 1920, 2560** widths: no dead right band — hero graph and
  section backgrounds reach the viewport edge; 0 console errors.
- Interaction check: cursor kNN links render + drag stirs (verified via Playwright mouse moves
  + screenshot); reduced-motion renders static.
- Contrast spot-checks per tokens above. Full-page screenshots for sign-off.

## Rollout

1. Tokens + fonts + dark-only base (remove theme toggle) → shell (nav/footer/section).
2. Hero v2 (full-bleed + interactive graph) + stats → **screenshot checkpoint**.
3. Differentiators keyline grid + terminal code showcase.
4. Pipeline rail + architecture + CTA + footer polish.
5. Multi-width Playwright verification + final sign-off.
