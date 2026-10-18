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
// `/docs` is a client-side redirect to `/docs/introduction` (App.tsx), so a
// static host has nothing to serve for it: a direct hit, an external link or a
// crawler gets a 404 — and both the nav and the footer link there. Emit a real
// redirect stub. SITE_BASE is the same variable vite.config.ts builds `base`
// from, so the target carries the deploy sub-path.
const base = (process.env.SITE_BASE || '/').replace(/\/+$/, '')
const docsTarget = `${base}/docs/introduction`
mkdirSync(join(dist, 'docs'), { recursive: true })
writeFileSync(
  join(dist, 'docs', 'index.html'),
  `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Veda Docs</title>
    <link rel="canonical" href="${SITE_URL}/docs/introduction" />
    <meta http-equiv="refresh" content="0; url=${docsTarget}" />
    <meta name="robots" content="noindex" />
  </head>
  <body>
    <p>Redirecting to <a href="${docsTarget}">the Veda documentation</a>.</p>
  </body>
</html>
`,
)
console.log('prerendered /docs (redirect stub -> ' + docsTarget + ')')

// sitemap.xml was a hand-written file in public/ and had drifted to 24 of 28
// pages — the four unrouted ones were also unindexable. Generated from the same
// registry-derived route list so the two cannot disagree again. No <lastmod> or
// <priority>: Google ignores priority, and a fabricated lastmod is worse than
// none (a CI checkout would stamp every page with the clone time).
writeFileSync(
  join(dist, 'sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by scripts/prerender.mjs from src/docs/registry.ts. Do not hand-edit. -->
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${routes.map((r) => `  <url><loc>${SITE_URL}${r.path}</loc></url>`).join('\n')}
</urlset>
`,
)
console.log('wrote dist/sitemap.xml (', routes.length, 'urls )')

// llms.txt keeps hand-written one-line descriptions, so it is not generated —
// but it drifted the same way. Fail the build if a page is missing from it.
const llms = readFileSync(join(dist, 'llms.txt'), 'utf8')
const unlisted = docPages.filter((p) => !llms.includes(`/docs/${p.slug})`)).map((p) => p.slug)
if (unlisted.length > 0) {
  console.error(
    'PRERENDER_FAIL: page(s) missing from public/llms.txt — add a line for each:',
    unlisted.join(', '),
  )
  process.exit(1)
}

console.log('PRERENDER_DONE', n, 'routes')
