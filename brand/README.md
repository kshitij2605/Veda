# Veda + Niriksha — Brand Marks

Three candidate logo families, each a matched Veda/Niriksha pair. Open
`logo-concepts.html` in a browser for the full board (marks, wordmark lockups,
light-background and favicon-size tests). **Status: awaiting final selection.**

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
