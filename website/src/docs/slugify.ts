/** Heading text → anchor id. The single source of ids for renderer, TOC, and search. */
export const slugify = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
