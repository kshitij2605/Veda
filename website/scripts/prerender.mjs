// Static prerender: render each route to HTML and write dist/<route>/index.html.
// Runs after `vite build` (client) + `vite build --ssr src/entry-server.tsx`.
// vite-react-ssg can't run on this Vite 8 / react-router 7 stack, so this is a
// small custom SSR-prerender instead. The client (main.tsx) hydrates the markup.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { render, docPages } from '../dist-ssr/entry-server.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const dist = join(__dirname, '..', 'dist')
const SITE_URL = 'https://veda.example' // placeholder — see src/lib/site.ts


// Comparison pages get their own title/description (marketing intent, not the
// generic "<title> — Veda Docs" doc template) so search snippets read well.
const SEO = {
  'vs-elasticsearch': {
    title: 'Veda vs Elasticsearch — Multimodal Vector Search Compared',
    desc: 'A benchmarked, honest comparison of Veda and Elasticsearch for vector and multimodal search: recall, latency, document ingestion, and RAG quality.',
  },
  'vs-pinecone': {
    title: 'Veda vs Pinecone — Open-Source Alternative to a Managed Vector DB',
    desc: 'How Veda, an open-source self-hostable multimodal vector database, compares to Pinecone, a managed proprietary vector store.',
  },
}

const routes = [
  {
    path: '/',
    title: 'Veda — Multimodal Vector Database & RAG Platform (Rust)',
    desc: 'Veda is an open-source, Rust-native multimodal vector database and RAG platform: one unified vector space for text, images, audio, video and documents, with hybrid search, reranking, and a built-in MCP server.',
  },
  ...docPages.map(({ slug, title }) => ({
    path: `/docs/${slug}`,
    title: SEO[slug]?.title ?? `${title} — Veda Docs`,
    desc:
      SEO[slug]?.desc ??
      `${title} — documentation for Veda, the open-source multimodal vector database and RAG platform.`,
  })),
]

// Every markdown file under docs/content must be registered, or it is built
// into the bundle, linked from the prose, and served as a 404. Deriving routes
// from the registry fixes the drift; this catches the next new page that is
// added to content/ but never registered.
const onDisk = readdirSync(join(__dirname, '..', 'src', 'docs', 'content'))
  .filter((f) => f.endsWith('.md'))
  .map((f) => f.replace(/\.md$/, ''))
const unrouted = onDisk.filter((slug) => !docPages.some((p) => p.slug === slug))
if (unrouted.length > 0) {
  console.error(
    'PRERENDER_FAIL: content file(s) with no route — add them to src/docs/registry.ts:',
    unrouted.join(', '),
  )
  process.exit(1)
}

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

function headBlock({ path, title, desc }) {
  const url = SITE_URL + path
  return `<!--ssg:head-->
    <title>${esc(title)}</title>
    <meta name="description" content="${esc(desc)}" />
    <link rel="canonical" href="${esc(url)}" />
    <meta property="og:title" content="${esc(title)}" />
    <meta property="og:description" content="${esc(desc)}" />
    <meta property="og:url" content="${esc(url)}" />
    <meta name="twitter:title" content="${esc(title)}" />
    <meta name="twitter:description" content="${esc(desc)}" />
    <!--/ssg:head-->`
}

const template = readFileSync(join(dist, 'index.html'), 'utf8')
const reHead = /<!--ssg:head-->[\s\S]*?<!--\/ssg:head-->/
if (!reHead.test(template)) {
  console.error('PRERENDER_FAIL: ssg:head markers not found in dist/index.html')
  process.exit(1)
}

let n = 0
for (const r of routes) {
  const app = render(r.path)
  if (!app || app.length < 50) {
    console.error('PRERENDER_FAIL: empty render for', r.path)
    process.exit(1)
  }
  let html = template.replace(reHead, headBlock(r))
  html = html.replace('<div id="root"></div>', `<div id="root">${app}</div>`)
  const outDir = r.path === '/' ? dist : join(dist, r.path)
  mkdirSync(outDir, { recursive: true })
  writeFileSync(join(outDir, 'index.html'), html)
  console.log('prerendered', r.path, '(', app.length, 'bytes )')
  n++
}
console.log('PRERENDER_DONE', n, 'routes')
