// Static prerender: render each route to HTML and write dist/<route>/index.html.
// Runs after `vite build` (client) + `vite build --ssr src/entry-server.tsx`.
// vite-react-ssg can't run on this Vite 8 / react-router 7 stack, so this is a
// small custom SSR-prerender instead. The client (main.tsx) hydrates the markup.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { render } from '../dist-ssr/entry-server.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const dist = join(__dirname, '..', 'dist')
const SITE_URL = 'https://veda.example' // placeholder — see src/lib/site.ts

const docs = [
  ['introduction', 'Introduction'],
  ['quickstart', 'Quick start'],
  ['installation', 'Installation & config'],
  ['collections', 'Collections & fields'],
  ['indexes', 'Indexes'],
  ['search', 'Search & fusion'],
  ['multimodal', 'Multimodal & cross-modal'],
  ['document-pipeline', 'Document pipeline'],
  ['agents-mcp', 'Agents & MCP'],
  ['web-connectors', 'Web Connectors'],
  ['rest-api', 'REST API'],
  ['python-sdk', 'Python SDK'],
  ['observability', 'Observability'],
  ['apm-observability', 'APM & Performance'],
  ['llm-observability', 'LLM Observability'],
  ['aiops', 'AIOps'],
  ['ops-advisor', 'Ops Advisor'],
  ['log-insights', 'Log Insights'],
  ['resource-signals', 'Node Resource Signals'],
  ['service-reliability', 'Service Reliability'],
  ['rag-aiops', 'RAG & AIOps Intelligence'],
  ['contributing', 'Contributing'],
]

const routes = [
  {
    path: '/',
    title: 'Veda — Multimodal Vector Database & RAG Platform (Rust)',
    desc: 'Veda is an open-source, Rust-native multimodal vector database and RAG platform: one unified vector space for text, images, audio, video and documents, with hybrid search, reranking, and a built-in MCP server.',
  },
  ...docs.map(([slug, title]) => ({
    path: `/docs/${slug}`,
    title: `${title} — Veda Docs`,
    desc: `${title} — documentation for Veda, the open-source multimodal vector database and RAG platform.`,
  })),
]

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
