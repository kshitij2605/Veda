#!/usr/bin/env node
//
// Rewrite the placeholder origin across the built site.
//
// The domain appears in more places than a single constant can cover:
// canonical tags and Open Graph URLs (emitted by `site.ts` through the
// prerender), plus `robots.txt`, `sitemap.xml` and `llms.txt`, which are
// static files in `public/` with absolute URLs baked in. Swapping only
// `site.ts` leaves the last three pointing at a dead host — and a wrong
// canonical still renders perfectly, so nothing surfaces the mistake.
//
// This runs last and rewrites every occurrence in `dist`, then asserts none
// survive. Set SITE_URL to the full origin plus any sub-path, with no
// trailing slash:
//
//   SITE_URL=https://kshitij2605.github.io/Veda   (project Pages site)
//   SITE_URL=https://veda.dev                     (custom domain / user site)
//
// Unset SITE_URL and this is a no-op: local `npm run build` keeps the
// placeholder and does not fail.

import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const PLACEHOLDER = 'https://veda.example'
const DIST = 'dist'
const EXTENSIONS = ['.html', '.txt', '.xml', '.json', '.js', '.css']

const siteUrl = process.env.SITE_URL?.replace(/\/+$/, '')
if (!siteUrl) {
  console.log('[set-site-url] SITE_URL unset — leaving the placeholder in place')
  process.exit(0)
}
if (!/^https?:\/\//.test(siteUrl)) {
  console.error(`[set-site-url] SITE_URL must be an absolute URL, got: ${siteUrl}`)
  process.exit(1)
}

function walk(dir) {
  const out = []
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry)
    if (statSync(p).isDirectory()) out.push(...walk(p))
    else if (EXTENSIONS.some((e) => p.endsWith(e))) out.push(p)
  }
  return out
}

let filesChanged = 0
let occurrences = 0
for (const file of walk(DIST)) {
  const before = readFileSync(file, 'utf8')
  if (!before.includes(PLACEHOLDER)) continue
  const hits = before.split(PLACEHOLDER).length - 1
  writeFileSync(file, before.split(PLACEHOLDER).join(siteUrl))
  filesChanged += 1
  occurrences += hits
}

console.log(
  `[set-site-url] ${PLACEHOLDER} -> ${siteUrl} (${occurrences} occurrence(s) in ${filesChanged} file(s))`,
)

// A surviving placeholder means a file type this script does not scan, or a
// build step that ran after it. Either way the deployed site would advertise
// a dead host, so fail rather than ship it.
const leftovers = walk(DIST).filter((f) => readFileSync(f, 'utf8').includes(PLACEHOLDER))
if (leftovers.length > 0) {
  console.error(`[set-site-url] placeholder still present in:\n  ${leftovers.join('\n  ')}`)
  process.exit(1)
}
