// Generates dist/llms-full.txt: every doc's raw markdown concatenated into one
// file, for agents that want the whole corpus in a single fetch. Reads the
// content/*.md files directly (not via Vite's ?raw import — this runs as a
// plain node script) so it can never drift from what's actually published.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const contentDir = join(__dirname, '..', 'src', 'docs', 'content')
const dist = join(__dirname, '..', 'dist')

const GROUPS = [
  {
    label: 'Getting started',
    slugs: ['introduction', 'quickstart', 'installation'],
  },
  {
    label: 'Core concepts',
    slugs: [
      'collections', 'indexes', 'search', 'multimodal',
      'document-pipeline', 'agents-mcp', 'web-connectors',
    ],
  },
  {
    label: 'Reference',
    slugs: [
      'rest-api', 'python-sdk', 'observability', 'apm-observability',
      'llm-observability', 'aiops', 'ops-advisor', 'log-insights',
      'resource-signals', 'service-reliability', 'rag-aiops',
    ],
  },
  {
    label: 'Compare',
    slugs: ['vs-elasticsearch', 'vs-pinecone'],
  },
  {
    label: 'Community',
    slugs: ['contributing'],
  },
]

let out = '# Veda — full documentation\n\n> Every Veda doc page concatenated for agents/LLMs that want the whole corpus in one fetch. See /llms.txt for a curated index with links instead.\n'

for (const g of GROUPS) {
  out += `\n---\n\n# ${g.label}\n`
  for (const slug of g.slugs) {
    const md = readFileSync(join(contentDir, `${slug}.md`), 'utf8')
    out += `\n<!-- slug: ${slug} -->\n\n${md.trim()}\n`
  }
}

mkdirSync(dist, { recursive: true })
writeFileSync(join(dist, 'llms-full.txt'), out)
console.log('wrote dist/llms-full.txt (', out.length, 'bytes,', GROUPS.flatMap((g) => g.slugs).length, 'docs )')
