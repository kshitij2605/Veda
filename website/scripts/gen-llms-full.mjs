// Generates dist/llms-full.txt: every doc's raw markdown concatenated into one
// file, for agents that want the whole corpus in a single fetch. Reads the
// content/*.md files directly (not via Vite's ?raw import — this runs as a
// plain node script). The GROUP STRUCTURE comes from docs/registry.ts via the
// SSR bundle: this script used to hardcode its own slug list, which drifted and
// silently dropped four pages from llms-full.txt despite the comment above.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { docGroups } from '../dist-ssr/entry-server.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const contentDir = join(__dirname, '..', 'src', 'docs', 'content')
const dist = join(__dirname, '..', 'dist')

let out = '# Veda — full documentation\n\n> Every Veda doc page concatenated for agents/LLMs that want the whole corpus in one fetch. See /llms.txt for a curated index with links instead.\n'

for (const g of docGroups) {
  out += `\n---\n\n# ${g.label}\n`
  for (const { slug } of g.items) {
    const md = readFileSync(join(contentDir, `${slug}.md`), 'utf8')
    out += `\n<!-- slug: ${slug} -->\n\n${md.trim()}\n`
  }
}

mkdirSync(dist, { recursive: true })
writeFileSync(join(dist, 'llms-full.txt'), out)
console.log('wrote dist/llms-full.txt (', out.length, 'bytes,', docGroups.flatMap((g) => g.items).length, 'docs )')
