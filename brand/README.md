# Veda + Niriksha — Brand Marks

**Status: FINAL.** The selected marks live in `final/`:

| | Mark | Favicon | Identity |
|---|---|---|---|
| **Veda** | `final/veda-mark.svg` | `final/veda-favicon.svg` | **Constellation V** — a V traced through a nearest-neighbor graph, saffron `#FFB224` (light: `#E8960B`) |
| **Niriksha** | `final/niriksha-mark.svg` | `final/niriksha-favicon.svg` | **Eye of nodes** — the watching eye drawn as a closed walk through six nodes, violet `#9D8CFF` (light: `#6A57E8`; in the dashboard it uses `var(--color-accent)`) |

Applied in: `website/src/components/BrandMark.tsx` + `website/public/favicon.svg`
(Veda), `niriksha/src/components/shell/Sidebar.tsx` (Niriksha mark),
`niriksha/src/pages/Login.tsx` (Veda mark — the login is the doorway to Veda),
`niriksha/public/favicon.svg`.

---

The exploration below is kept for history. Boards: `logo-concepts.html` (v1),
`logo-concepts-v2.html` (refined A + ancient-India), `logo-concepts-v3.html`
(Niriksha iterations), `logo-concepts-v4.html` (Krishna-set Veda iterations).

## Concepts

| | Veda | Niriksha | Idea |
|---|---|---|---|
| **A — kNN Constellation** | `concept-a/veda-mark.svg` | `concept-a/niriksha-mark.svg` | The database is a graph (a V traced through a nearest-neighbor walk); the dashboard is the eye watching it (node-lens with the query-ring pupil from the website's interactive hero). |
| **B — Shard Aperture** | `concept-b/veda-mark.svg` | `concept-b/niriksha-mark.svg` | Evolution of the original three-shard mark: stepped storage shards for Veda, a four-shard aperture/iris for Niriksha. |
| **C — Typographic + Bindu** | `concept-c/veda-mark.svg` | `concept-c/niriksha-mark.svg` | Bold geometric letterforms with a Sanskrit bindu: V + saffron dot for Veda; an N whose diagonal is a trail of observation points for Niriksha. Crispest at 16 px. |

## Colorways

The committed SVGs are the **dark-background** colorway (the obsidian brand):

| Token | Dark bg | Light bg |
|---|---|---|
| Saffron | `#FFB224` | `#E8960B` |
| Violet (Niriksha) | `#9D8CFF` | `#6A57E8` |
| Ink/letterform | `#F2F4F8` | `#15181E` |

Each SVG carries a comment noting its light-background substitutions.

## Usage notes

- Wordmark lockups are composed in-app (mark SVG + product name in **Space Grotesk**
  700, tracking −0.02em), matching the website/dashboard `BrandMark` components.
- Once a concept is selected: simplify a favicon variant (16/32 px), swap the
  `BrandMark` components in `website/` and `niriksha/`, and regenerate `favicon.svg`
  for both apps.
