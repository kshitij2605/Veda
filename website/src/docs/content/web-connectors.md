# Web Connectors & Agentic Web Search

Data-source connectors are one of Elastic's headline integration stories — 30-plus SaaS connectors (SharePoint, Confluence, Salesforce, …) that pull external content into the index. Most of them phone home to a third-party SaaS, and chasing that catalog is an out-of-lane race for a vector database. Veda's in-lane answer is the one source that actually matters for RAG and that nobody self-hosts well: **the open web**.

This is a web-native RAG stack built entirely on **open-source, self-hosted services** — `crawl4ai` for clean page-to-markdown crawling and `searxng` for meta-search — wired into Veda so you can (1) **ingest** live web pages into a collection, (2) let an agent **search → fetch → ground** over the open web, and (3) **manage** web data sources from Niriksha. No external API, no data leaving your trust boundary. Everything reuses an existing primitive (the chunk → embed → insert path, the agent [tool catalog](/docs/agents-mcp#the-tool-catalog), saved-object persistence) and is **opt-in** — both services are off by default (endpoints unset → a clean `503`, no behavior change), and the web agent tools are excluded from the default agent.

## Web ingestion

RAG over the web needs *clean* content — boilerplate-stripped markdown, not raw HTML. `crawl4ai` (Apache-2.0) does exactly that: give it a URL, get back clean markdown.

`POST /v1/collections/{name}/ingest-url` takes `{ url | urls, deep_crawl?, max_pages?, chunker?, ttl_secs? }`:

```bash
curl -X POST http://localhost:6340/v1/collections/kb/ingest-url \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{ "url": "https://example.com/docs", "deep_crawl": true, "max_pages": 25 }'
```

- It **crawls** the URL(s) — or, with `deep_crawl`, follows in-domain links bounded by `max_pages` — into clean markdown; then
- it hands the markdown to a shared `ingest_markdown` helper that does the rest: **chunk → embed → insert**.

That helper is the same chunk/embed/insert tail the [document pipeline](/docs/document-pipeline) uses — `process-document` was refactored to route through it, so web ingestion and document ingestion share one code path (no OCR regression). Inserted records carry **`source_url`**, **`title`**, and **`chunk_index`** metadata, so a retrieved chunk traces back to its page. `ttl_secs` sets a [record TTL](/docs/collections#record-ttl). The endpoint is **tenant/RLS-scoped** — a crawl writes into the caller's tenant only.

## Agentic web search

Ingestion pulls a *known* URL into a collection. The agentic case is open-ended: an agent that needs a fact it doesn't have should **search the web, fetch the best result, and ground its answer in it** — the same search → fetch → read loop a human does. `searxng` is a self-hosted meta-search aggregator that federates many engines and returns JSON. Two new agent tools sit on top:

| Tool | What it does |
|---|---|
| `web_search { query, limit }` | Runs a searxng search and returns the top results (title, url, snippet) |
| `fetch_url { url }` | Fetches a URL through crawl4ai and returns its markdown, **truncated** to a bounded length so one page can't blow the context window |

Together they give an agent the full loop: `web_search` to find candidate pages, `fetch_url` to read the best one, then ground the answer in the fetched content. A built-in **`veda-web`** agent ships with these whitelisted — its toolset is `web_search` + `fetch_url` + `search` (Veda's own collection search), so the agent can blend live-web and in-corpus retrieval:

```bash
curl -X POST http://localhost:6340/v1/agents/veda-web/converse \
  -H "Content-Type: application/json" \
  -d '{ "input": "What changed in the latest release of <project>? Search the web." }'
```

**Opt-in by design.** The two web tools are deliberately **excluded from the default `veda-agent`** — web access is a trust decision, not a default. The tool catalog has **15 built-ins**, but the default agent resolves to **13**: the web tools are present in the catalog (so `veda-web` and explicitly-configured agents can use them) yet absent from the default agent's toolset.

## Connectors framework

Ingesting a URL on demand is the imperative case; a **connector** is the declarative one — a named, persisted web data source you can re-run:

```json
{ "id": "conn-…", "kind": "web", "name": "Docs site", "config": { "urls": ["…"], "collection": "kb" }, "enabled": true }
```

Connectors persist as tenant-scoped saved objects (`id` is optional on create — the server assigns `conn-{uuid}`). The CRUD surface is admin-only:

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/admin/connectors` | List / create a connector |
| `GET/PUT/DELETE` | `/v1/admin/connectors/{id}` | Read / update / delete a connector |
| `POST` | `/v1/admin/connectors/{id}/run` | **Run now** — a `web` connector crawls its configured URLs into its target collection |

`run` reuses the same `crawl_and_ingest` path that backs `/ingest-url` — no duplicate ingestion logic. In Niriksha, the **Settings** page gains a **Data Source Connectors** section to add a web source and trigger **"Crawl now"**, alongside the existing read-only infra-service status.

## Deployment

Both services are optional and self-hosted, gated behind a Compose profile:

```bash
docker compose --profile web up
```

This brings up **`crawl4ai`** (`unclecode/crawl4ai`, page → clean markdown) and **`searxng`** (`searxng/searxng`, meta-search, configured for JSON output via a mounted `settings.yml`). The `veda` service is wired with the two endpoints, **default empty** — so a plain `docker compose up` is unchanged and the web features cleanly report "not configured" / `503`.

| Variable | Default | Notes |
|---|---|---|
| `VEDA_CRAWL4AI_ENDPOINT` | *(empty)* | crawl4ai base URL; unset → `ingest-url` / `fetch_url` / connector `run` return `503` |
| `VEDA_SEARXNG_ENDPOINT` | *(empty)* | searxng base URL; unset → `web_search` is unavailable |

## Security & trust boundary

`ingest-url` and the `fetch_url` tool fetch **arbitrary user-provided URLs** — a classic SSRF surface (internal addresses, cloud metadata endpoints). Today that is mitigated by the endpoints being **admin/authenticated and tenant-scoped** within a **self-hosted trust boundary**: the operator controls who can call them. A URL allowlist / private-range denylist is a sensible follow-up before exposing these to lower-trust callers.

**Licensing.** `crawl4ai` is Apache-2.0 (fully compatible). `searxng` is AGPL-3.0, but Veda only **HTTP-clients a separately-run instance** over its JSON API — AGPL's network-copyleft binds modifications to searxng *itself*, not a client that calls its API. Veda ships a client, not a fork, and the service stays opt-in and self-hosted (you run the upstream image unmodified), so there's no contamination of Veda's Apache-2.0 codebase.

## Current limitations

- **SSRF — allowlist is a follow-up.** Arbitrary-URL fetch is currently gated by the admin/authenticated, tenant-scoped, self-hosted trust boundary; a URL allowlist / private-range denylist is the planned hardening.
- **Full SaaS connectors are out of scope by design.** This is web-native RAG, not a SharePoint/Confluence/Salesforce connector catalog — depth on the open web over breadth of proprietary SaaS sources.
- **Deep crawl is bounded by `max_pages`** — it follows in-domain links only, not a full-site spider, to bound cost.
- **searxng must be configured for JSON output** (enabled via the mounted `settings.yml` in the compose profile).
- **Live validation requires `--profile web`** — the mock tests cover the wiring and the ingestion path, but the end-to-end run against the real crawl4ai/searxng images is not run in CI (the crawl4ai image is large).
