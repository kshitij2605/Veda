# Veda Discoverability Playbook — being found by humans *and* agents (2026)

> Goal: when a developer (or their coding agent) searches for "multimodal vector
> database", "open-source RAG platform", "Elasticsearch alternative for vectors",
> "self-hosted vector DB with reranking", etc., Veda shows up — in Google, in GitHub
> search, in LLM answers, and in agentic tool/registry lookups.
>
> This is the consolidated, actionable version of the discoverability guidance from the
> README/positioning work. Items are tagged **[DONE]**, **[TODO]**, or **[ONGOING]** and
> the final section is a prioritized checklist.

---

## 0. Why this is different in 2026

Two distinct discovery surfaces now matter, and they reward different things:

1. **Human search (Google / GitHub / Bing).** Classic SEO: keyword-aligned titles,
   structured headings, fast pages, backlinks, fresh content, schema.org markup.
2. **Agentic search (LLMs + coding agents).** When someone asks Claude/ChatGPT/Cursor
   "what's a good open-source multimodal vector DB?", the answer is shaped by: training
   data (GitHub, docs, blog posts, HN/Reddit), live web/tool retrieval, `llms.txt`,
   MCP/tool registries, and machine-readable docs. Agents reward **clean, declarative,
   plain-text/markdown content** and **canonical machine-readable manifests** — not JS-
   heavy marketing pages.

The thesis: optimize for **both** — keyword-clean prose that humans skim and crawlers
index, plus plain-text/JSON artifacts that agents can ingest verbatim.

---

## 1. Keyword strategy

Lead with the terms people actually type. Primary + secondary clusters:

**Primary (must appear in title, H1, meta description, GitHub description, first paragraph):**
- multimodal vector database
- open-source vector database (Rust)
- RAG platform / retrieval-augmented generation
- vector search engine

**Secondary (sprinkle naturally across README, docs, website):**
- Elasticsearch alternative (for vector/semantic search)
- self-hosted vector database
- hybrid search (dense + sparse + BM25)
- cross-modal search (text · image · audio · video)
- reranking, ColPali, SPLADE, HNSW, quantization (RaBitQ/SQ8)
- embeddings, semantic search, similarity search
- AGPL / source-available vector database
- MCP server for vector search / agent memory

**Long-tail (great for docs pages + comparison pages — low competition, high intent):**
- "vector database with built-in OCR / document pipeline"
- "self-hosted Pinecone alternative"
- "multimodal RAG with one embedding space"
- "vector DB with observability / AIOps built in"
- "how to search images and text in one index"

Rule of thumb: **one primary keyword per page**, used in the URL slug, H1, first 100
words, and one H2. Don't keyword-stuff — agents and Google both penalize it.

---

## 2. Repository & README (the single highest-leverage asset)

GitHub is both a human destination and a top training/retrieval source for agents.

- **[DONE]** Keyword-rich H1: *"Veda — Multimodal Vector Database & RAG Platform (Rust)"*.
- **[DONE]** One-line value prop in the first paragraph (what + who it's for + why).
- **[DONE]** Quickstart that works in <5 min (docker compose up → insert → search).
- **[DONE]** Comparison table vs Elasticsearch / Pinecone / Qdrant / Weaviate / Milvus.
- **[DONE]** LLM-readable structure (clear headings, fenced code, no essential info locked
  in images/diagrams only).
- **[TODO]** Set the GitHub **repo description** + **website link** to the primary keywords
  (this is what GitHub search and Google index first).
- **[TODO]** Add **GitHub Topics** (the repo's tag chips — strong GitHub-search signal):
  `vector-database`, `rag`, `multimodal`, `vector-search`, `embeddings`, `semantic-search`,
  `rust`, `hnsw`, `reranking`, `llm`, `ai-search`, `elasticsearch-alternative`,
  `mcp`, `retrieval-augmented-generation`.
- **[TODO]** Pin a **social preview image** (Settings → Social preview) — drives CTR when
  the repo is shared on X/LinkedIn/Slack.
- **[TODO]** A short **"Why Veda" / positioning** block above the fold linking to the docs
  site and the comparison pages.
- **[ONGOING]** Keep a tight **CHANGELOG** and tagged **releases** — recency is a ranking
  signal and a trust signal for agents summarizing "is this maintained?".
- **[TODO]** Add badges (build, license, crates.io if/when published, docs) — humans read
  them as health signals.

---

## 3. Website — technical SEO checklist

The marketing site (`website/`) and docs (`/docs/*`) are the canonical destinations. Make
them crawlable and machine-summarizable.

- **[TODO] Per-page `<title>` + `<meta name="description">`** — unique, keyword-led, ≤155
  chars for the description. Today the SPA likely ships one title for all routes; fix this
  first.
- **[TODO] Open Graph + Twitter Card tags** (`og:title`, `og:description`, `og:image`,
  `og:url`, `twitter:card=summary_large_image`) so shared links render rich cards.
- **[TODO] JSON-LD schema.org markup**: `SoftwareApplication` (or `SoftwareSourceCode`) on
  the home page, `TechArticle` on docs pages, `FAQPage` on any FAQ. This is what Google
  rich results and many agents parse for structured facts.
- **[TODO] `sitemap.xml` + `robots.txt`** (robots should *allow* crawling + point to the
  sitemap). Submit the sitemap in Google Search Console + Bing Webmaster Tools.
- **[TODO] Server-side rendering / static pre-render (SSG).** A pure client-rendered SPA is
  weakly indexed and nearly invisible to non-JS agent fetchers. Pre-render the marketing +
  docs routes to static HTML (Vite SSG / a prerender step) so the content is in the initial
  payload. **This is the biggest single SEO win for the site.**
- **[TODO] Canonical URLs** (`<link rel="canonical">`) to avoid duplicate-content dilution.
- **[ONGOING] Performance** — fast LCP/CLS (Core Web Vitals) is a ranking factor; the site
  is already lightweight, keep bundles small.
- **[TODO] Internal linking** — every docs page links to related pages + the home page;
  the home page links to docs + comparison pages. Crawl depth ≤3.

---

## 4. Agentic / LLM discoverability (the 2026 differentiator)

This is where most projects under-invest. Make Veda trivially ingestible by agents.

- **[TODO] `llms.txt`** at the site root — a curated, plain-markdown index of the project
  for LLMs (what Veda is, key concepts, links to the most important docs as raw markdown).
  Emerging de-facto standard; cheap to ship, increasingly consumed.
- **[TODO] `llms-full.txt`** — the full docs concatenated into one plain-text file for
  agents that want the whole corpus in one fetch.
- **[TODO] Raw-markdown doc endpoints** — serve each docs page also as `.md` (e.g.
  `/docs/quickstart.md`) so an agent fetching the page gets clean text, not rendered HTML.
- **[TODO] MCP discoverability** — Veda ships an MCP server. Publish an MCP **`server.json`**
  / register it in the public MCP server registries and `mcp.so`-style directories so
  agents can find and wire up Veda as a tool. Document the stdio + Streamable-HTTP configs
  in the README (Claude Desktop / Cursor / VS Code snippets).
- **[TODO] Clear, declarative docs prose.** Agents quote what is unambiguous. Prefer
  "Veda supports X. To do X, call `POST /v1/...`." over narrative/marketing voice.
- **[ONGOING] Be present where training data is gathered** — GitHub (code + docs + README),
  technical blog posts, HN/Reddit/Lobsters threads, Stack Overflow answers, and comparison
  articles. These become the substrate LLMs answer from.

---

## 5. Comparison & landing pages (high-intent, low-competition)

Dedicated pages targeting one long-tail query each — these rank fast and convert.

- **[TODO] vs-Elasticsearch** — "Veda vs Elasticsearch for vector / multimodal search"
  (recall + latency numbers from the benchmark report; the document-format + cross-modal
  advantages). High intent, you already have the data.
- **[TODO] vs-Pinecone** — "Open-source / self-hosted Pinecone alternative".
- **[TODO] vs-Qdrant / vs-Weaviate / vs-Milvus** — feature/positioning matrix.
- **[TODO] "Multimodal RAG" landing page** — the unified cross-modal embedding-space story.
- **[TODO] "AGPL / source-available" explainer** — what the license means for users +
  the commercial option (reduces friction for evaluators worried about licensing).
- **[TODO] "MCP server for vector search" page** — captures the agent-builder audience.

Each: one keyword in the slug + H1, an honest comparison table, a CTA to the quickstart.

---

## 6. Distribution & off-site (backlinks + reach)

Rankings (human and agentic) are downstream of being talked about.

- **[TODO] Launch posts**: Show HN, r/MachineLearning, r/Rust, r/vectordatabase,
  Lobsters, the LocalLLaMA community. A good launch thread is both traffic and a durable
  citation source for agents.
- **[TODO] Listings**: get Veda onto the vector-DB comparison/landscape sites and "awesome"
  lists (awesome-vector-search, awesome-rag, awesome-rust, the LLM-tooling landscapes).
  These are high-authority backlinks crawlers + agents trust.
- **[TODO] Dev directories**: Product Hunt, AlternativeTo (as an "Elasticsearch/Pinecone
  alternative"), libhunt, OpenSauced, StackShare.
- **[ONGOING] Technical content**: 2–4 deep posts (e.g. "How we matched Elasticsearch
  recall at scale", "One embedding space for text, image, audio, video", "Building agentic
  RAG with Veda + MCP"). Cross-post to dev.to / Medium / the project blog; link back to docs.
- **[ONGOING] Engage** where the audience is (HN comments, Discord/Slack communities,
  conference CFPs) — earned mentions compound.

---

## 7. Measurement (so you know what's working)

- **[TODO]** Google Search Console + Bing Webmaster Tools — verify the domain, submit the
  sitemap, watch impressions/clicks per query.
- **[TODO]** Plausible/Umami (privacy-friendly) analytics on the site — track referrers and
  top landing pages.
- **[ONGOING]** GitHub stars/forks/traffic (Insights → Traffic) — referrers + popular content.
- **[ONGOING]** Periodically **ask the agents**: query Claude/ChatGPT/Perplexity with
  "best open-source multimodal vector database" and see whether Veda appears + how it's
  described. That answer *is* the agentic-SEO scoreboard; track it monthly.

---

## 8. Prioritized action checklist

**P0 — do first (highest leverage, low effort):**
1. Set GitHub repo **description + topics + website link** (§2).
2. Add **per-page title/meta + Open Graph + JSON-LD** to the website (§3).
3. **Pre-render (SSG)** the marketing + docs routes to static HTML (§3) — the biggest SEO unlock.
4. Ship **`sitemap.xml` + `robots.txt`**; submit to Search Console + Bing (§3, §7).
5. Publish **`llms.txt`** (+ raw `.md` doc endpoints) (§4).

**P1 — do next (compounding):**
6. **Comparison pages** (vs Elasticsearch / Pinecone / Qdrant), AGPL explainer (§5).
7. **MCP `server.json`** + register in MCP directories; README client snippets (§4).
8. **Launch posts** (Show HN, Reddit, Lobsters) + **awesome-list / AlternativeTo** listings (§6).
9. **`llms-full.txt`** + canonical URLs + internal linking pass (§3, §4).

**P2 — ongoing:**
10. 2–4 **technical blog posts** with backlinks (§6).
11. **Search Console / analytics** review cadence; monthly **agent-answer check** (§7).
12. Keep **releases + CHANGELOG** current (§2).

---

### One-paragraph summary
Veda's content is already keyword-clean and LLM-readable (README done). The gap is
**machine-crawlability + machine-ingestibility + distribution**: pre-render the site to
static HTML, add titles/meta/OG/JSON-LD/sitemap, publish `llms.txt` and raw-markdown docs,
register the MCP server, ship a handful of comparison pages, and seed the launch on HN/
Reddit/awesome-lists. Do the five P0 items and Veda becomes visible to both Google and the
agents that increasingly mediate "what tool should I use?".
