# Veda Marketing Site — Design Spec

**Date:** 2026-06-10
**Status:** Approved direction; pending spec review → implementation plan
**Goal:** A polished, customer-facing marketing landing page for Veda that converts
developers, built as a *marketing-grade expression of the Niriksha design system* so the
brand reads as one product. Docs site is a separate follow-up cycle (out of scope here).

## Approved decisions (from brainstorming)

1. **Stack:** Reuse Niriksha's stack — a new **Vite + React 19 + Tailwind 4** app that copies
   Niriksha's exact OKLCH design tokens, **Geist + Inter + JetBrains Mono**, vendored
   **reactbits** components, and **gsap** for motion. Maximum visual consistency, same
   toolchain (node v24.3.0 on gpu-host-a), Playwright-verifiable.
2. **Structure:** **One `website/` app** — marketing landing at `/`, docs at `/docs/*` in a
   later cycle. Single design system, single build.
3. **Scope (this pass):** the **marketing landing**, a solid v1 (nav → hero → stats →
   differentiators → code showcase → how-it-works → architecture → CTA → footer).
4. **Theme:** light-first (matches the dashboard), dark mode at parity via a toggle.
5. **Hero centerpiece:** an **animated vector scatter** — a live, slowly-drifting embedding
   point cloud (the Atlas motif) beside the tagline, over an Aurora wash. Distinctive,
   on-brand for a vector DB, and visually tied to the dashboard's Atlas.

## Non-goals

- Not the docs site (next cycle).
- No deployment/hosting setup (build + dev-server preview, like Niriksha).
- No backend — purely static marketing content sourced from `competitive_advantages.md` +
  `README.md`. Links point at the real GitHub repo, the docs (when built), and the dashboard.
- No new heavy deps beyond what Niriksha already uses (react-router, gsap, ogl, framer-motion,
  lucide-react, tailwind, reactbits-vendored).

## Design system (reuse + marketing-grade expression)

Copy Niriksha's `niriksha/src/styles/global.css` tokens verbatim (light-first OKLCH palette,
indigo-violet accent hue 282, modality palette, chart ramp, soft elevation, radii, motion,
`--font-display: Geist`, Inter, JetBrains Mono). The marketing site **expresses them more
boldly** than the dashboard:

- **Display type:** hero headline at Geist `clamp(40px, 7vw, 76px)`, section headings 32–40px
  (vs the dashboard's 30px max). Same family, bigger stage.
- **Color:** more gradient + Aurora on the hero/CTA bands; modality colors used as the
  categorical accent across the 7 differentiator cards.
- **Motion:** gsap scroll-reveals + staggered entrances + stat count-ups; the animated hero
  scatter. All gated on `prefers-reduced-motion`.
- **Restraint:** premium, not a circus — dense technical sections (code, tables) stay calm and
  legible, matching the dashboard's "comfortable but efficient" data surfaces.

## App structure

```
website/
  index.html                 # Geist/Inter/JetBrains font links; light pre-paint + theme init script
  package.json               # vite, react, react-dom, react-router-dom, tailwindcss, @tailwindcss/vite,
                             #   gsap, ogl, framer-motion, lucide-react
  vite.config.ts             # base '/', dev server host 0.0.0.0 port 5175 strictPort
  tsconfig.json / tsconfig.app.json / tsconfig.node.json
  src/
    main.tsx                 # mounts <App/>
    App.tsx                  # react-router: '/' -> <Landing/> (route stub for '/docs/*' later)
    styles/global.css        # copied Niriksha tokens (light-first + dark)
    lib/cn.ts                # copied
    lib/theme.ts             # copied (light default + dark toggle)
    lib/useReveal.ts         # copied (gsap on-mount stagger, reduced-motion gated)
    components/
      reactbits/Aurora.tsx, GradientText.tsx, SpotlightCard.tsx, CountUp.tsx   # vendored from niriksha
      primitives/Button.tsx, Badge.tsx, ThemeToggle.tsx                        # copied/adapted
      Nav.tsx                # sticky glass nav + theme toggle
      HeroScatter.tsx        # animated vector-scatter canvas (the hero centerpiece)
      CodeShowcase.tsx       # tabbed code blocks (JetBrains mono, custom light syntax tint)
      Footer.tsx
      Section.tsx            # shared section wrapper (centered max-w, overline + heading)
    sections/
      Hero.tsx, Stats.tsx, Differentiators.tsx, CodeSection.tsx,
      HowItWorks.tsx, Architecture.tsx, CTA.tsx
    pages/Landing.tsx        # composes the sections in order
    content/
      features.ts            # the 7 differentiators (title, blurb, icon, modality accent)
      stats.ts               # hero stats (7 indexes / 3 services / 6 metrics / 5 modalities)
      code.ts                # the 3 code samples (create / fusion search / document upload)
      pipeline.ts            # the how-it-works steps
```

Each file has one responsibility; sections are independent and compose in `Landing.tsx`.

## Sections (content sourced from the repo)

1. **Nav** — Veda wordmark + shard mark; links Features / Architecture / Docs / GitHub;
   primary "Get started" button; theme toggle. Transparent at top, glass + border on scroll.
2. **Hero** — overline "Multimodal vector database"; Geist headline **"The multimodal vector
   database for the era of RAG"**; subhead (one sentence on the end-to-end pipeline); dual CTA
   ("Get started" → docs/quickstart anchor, "View on GitHub"); `HeroScatter` centerpiece over
   an Aurora wash; GradientText accent on a key word.
3. **Stats band** — CountUp: **7** index algorithms · **3**-service deploy · **6** distance
   metrics · **5** modalities. gsap stagger.
4. **Differentiators** — the 7 advantages as a bento/SpotlightCard grid: Multimodal-native
   pipeline · Pure-Rust + SIMD · 7 index algorithms · Multi-vector fusion · Built-in document
   intelligence · Lightweight 3-service architecture · OTel-first observability. Each: lucide
   icon in a modality-tinted chip, title, 1–2 line blurb.
5. **Code showcase** — "One API call" — tabbed code blocks: (a) create a multimodal collection,
   (b) multi-vector fusion search (the RRF example from `competitive_advantages.md`),
   (c) document upload → searchable vectors. JetBrains Mono, subtle syntax tint.
6. **How it works** — the document pipeline as a horizontal flow: Upload → OCR (FlashParse/
   GLM-OCR) → Chunk (7 strategies) → Embed (auto-routed) → Index → Search. Icons + connectors,
   gsap reveal.
7. **Architecture** — two beats: "Lightweight by default" (3 mandatory services vs Milvus's
   5+), and the **7-index table** (index · best-for · advantage) from `competitive_advantages.md`.
8. **CTA band** — "Up and running in 30 seconds" + a `docker compose up` code line + buttons
   (Read the docs / Star on GitHub / Open the dashboard). Aurora/gradient band.
9. **Footer** — columns: Product (Features, Architecture, Dashboard), Developers (Docs, Python
   SDK, GitHub, Contributing), and a brand line. Copyright.

## Hero scatter (the centerpiece)

`HeroScatter.tsx`: a `<canvas>` rendering ~120 small points that drift slowly (gentle
per-point orbital + noise motion via `requestAnimationFrame`), colored from the modality
palette, over the Aurora wash. Faint short links appear between nearby points (evoking vector
similarity), capped for performance. Sized to its container via `ResizeObserver`, drawn at
`devicePixelRatio`. Under `prefers-reduced-motion` it renders a single static frame. Pure
presentational canvas — no data dependency.

## Tooling

- **reactbits:** Aurora (hero/CTA wash), GradientText (hero accent), SpotlightCard (feature
  cards), CountUp (stats) — vendored + re-tokenized (already done for Niriksha; copy those).
  Pull at most one additional component (e.g. a tilt/border-gradient card) only if a section
  clearly benefits.
- **gsap:** `useReveal` (on-mount staggered fade-rise for section children), hero scatter
  drift, hover micro-interactions. Reduced-motion gated throughout.
- **ui-ux principles:** one primary CTA per band, clear hierarchy, AA contrast on light,
  consistent vertical rhythm, generous section padding.

## Verification / success criteria

- `npm run build` (tsc + vite) clean; the app runs on the dev server (new port).
- Playwright pass on the landing in **light and dark**, desktop width, **0 console errors**;
  full-page screenshot captured for sign-off.
- Hero scatter animates (and is static under reduced-motion).
- AA contrast verified on light surfaces; nav/CTA buttons legible.
- Visual consistency with the dashboard confirmed (same fonts, accent, modality colors, feel).
- Content is accurate to `competitive_advantages.md` / `README.md` (no invented claims).

## Rollout order (implementation)

1. Scaffold `website/` Vite app + copy tokens/fonts/lib + theme + base layout (nav/footer/
   section shell). Verify it builds + runs light/dark.
2. Hero + HeroScatter + Stats band → screenshot sign-off (hero is the make-or-break).
3. Differentiators + Code showcase.
4. How-it-works + Architecture + CTA band.
5. Full light+dark Playwright pass + before/after-style screenshots; polish.
