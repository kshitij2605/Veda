import { slugify } from './slugify'

export interface DocHeading {
  depth: 2 | 3
  text: string
  id: string
}

/** Parse h2/h3 headings from raw markdown, skipping fenced code blocks. */
export function parseHeadings(md: string): DocHeading[] {
  const out: DocHeading[] = []
  let fenced = false
  for (const line of md.split('\n')) {
    if (line.startsWith('```')) {
      fenced = !fenced
      continue
    }
    if (fenced) continue
    const m = /^(##|###)\s+(.+)$/.exec(line)
    if (m) {
      const text = m[2]!.trim()
      out.push({ depth: m[1]!.length as 2 | 3, text, id: slugify(text) })
    }
  }
  return out
}
