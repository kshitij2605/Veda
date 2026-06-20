# Veda Web Connectors & Agentic Web Search — Crawl4ai Web Ingestion, Searxng `web_search` / `fetch_url`, the Connectors Framework + Niriksha Page

**Date:** 2026-06-19
**Status:** Shipped on `feat/web-connectors` (queued for `main`), v0.4.0. Unit/mock-tested (in-test crawl4ai + searxng servers, including a connector `run` that mock-crawls and does a real chunk/embed/insert); **live validation requires `docker compose --profile web`** (not run in CI — the crawl4ai image is large).

Data-source connectors are one of Elastic's headline integration stories — 30-plus SaaS connectors (SharePoint, Confluence, S3, Salesforce, …) that pull external content into the index. Chasing that catalog is a losing, out-of-lane race for a vector DB, and most of those connectors phone home to a third-party SaaS. Veda's in-lane, differentiating answer is the one source that actually matters for RAG and that nobody self-hosts well: **the open web**. This cycle ships a web-native RAG stack built entirely on **open-source, self-hosted services** — `crawl4ai` for clean page-to-markdown crawling and `searxng` for meta-search — wired into Veda so you can (1) **ingest** live web pages into a collection, (2) let an agent **search → fetch → ground** over the open web, and (3) **manage** web data sources from Niriksha. No external API, no data leaving your trust boundary. Everything reuses an existing primitive (the chunk → embed → insert path, the agent ToolCatalog, SavedObjects persistence) and is **opt-in** — both services are off by default (endpoints unset → a clean "not configured"/503, no behavior change), and the web agent tools are excluded from the default agent.

---

## 1. Web ingestion connector (crawl4ai)

RAG over the web needs *clean* content — boilerplate-stripped, readable markdown, not raw HTML. `crawl4ai` (Apache-2.0) does exactly that: give it a URL, get back clean markdown. Veda adds a thin async client and a single ingestion endpoint.

`Crawl4aiClient` is an async `reqwest` client that calls **`POST {VEDA_CRAWL4AI_ENDPOINT}/md`** (single `url` + `f:"fit"` → clean, boilerplate-stripped markdown) and returns clean markdown (plus the page title). It is **opt-in** — when `VEDA_CRAWL4AI_ENDPOINT` is unset the client is `None`, and the endpoints that need it return **503** rather than changing behavior.

> **crawl4ai endpoint:** the real crawl4ai REST API is `POST /md`, not `/crawl` (which does not exist — an early client bug targeting `/crawl` 500'd at routing before any crawl ran). Recursive/deep crawl uses crawl4ai's async `/crawl/job` endpoint, which is a documented follow-up; today `deep_crawl` / `max_pages` are accepted but each URL is fetched per-URL via `/md`.

> **Exposing crawl4ai requires `VEDA_CRAWL4AI_TOKEN`.** The crawl4ai container binds **loopback-only** unless `CRAWL4AI_API_TOKEN` is set — once it is, the container binds all interfaces **and** requires that token as a bearer on every request. The compose wires `CRAWL4AI_API_TOKEN: ${VEDA_CRAWL4AI_TOKEN:-}` (the same value Veda sends as its bearer to crawl4ai), so you **must set `VEDA_CRAWL4AI_TOKEN`** to use crawl4ai at all: without it crawl4ai is unreachable from Veda even when the container reports "healthy". Set it on **both** the `crawl4ai` service and the `veda` service.

`POST /v1/collections/{name}/ingest-url` takes `{ url | urls, deep_crawl?, max_pages?, chunker?, ttl_secs? }`:

- it **crawls** the URL(s) — or, with `deep_crawl`, follows in-domain links bounded by `max_pages` — into clean markdown; then
- it hands the markdown to a **factored shared `ingest_markdown` helper** that does the rest of the work everything else in Veda uses: **chunk → embed → insert**.

The key refactor here is `ingest_markdown`: the document pipeline's `process-document` (OCR → markdown → chunk → embed → insert) was refactored to route its chunk/embed/insert tail through the *same* helper, so web ingestion and document ingestion share one code path — **with no OCR regression** (the OCR front-half is untouched). Inserted records carry **`source_url`**, **`title`**, and **`chunk_index`** metadata so a retrieved chunk traces back to its page. The endpoint is **tenant/RLS-scoped** — a crawl writes into the caller's tenant only.

---

## 2. Agentic web search (searxng)

Ingestion pulls a known URL into a collection. The *agentic* case is the open-ended one: an agent that needs a fact it doesn't have should be able to **search the web, fetch the best result, and ground its answer in it** — the same search → fetch → read loop a human does. Veda already has an agent executor with a tool catalog, so this is two new tools and a built-in agent.

`SearxngClient` calls `GET {VEDA_SEARXNG_ENDPOINT}/search?format=json` (searxng is a meta-search aggregator — it federates many engines and returns JSON when configured for it). Two new **opt-in** ToolCatalog tools sit on top:

- **`web_search { query, limit }`** — runs a searxng search and returns the top results (title, url, snippet); and
- **`fetch_url { url }`** — fetches a URL through crawl4ai and returns its markdown, **truncated** to a bounded length so a single page can't blow the context window.

Together they give an agent the full loop: `web_search` to find candidate pages, `fetch_url` to read the best one, then ground the answer in the fetched content. A built-in **`veda-web`** agent ships with these whitelisted — its toolset is `web_search` + `fetch_url` + `search` (the last being Veda's own collection search, so the agent can blend live-web and in-corpus retrieval).

**Opt-in enforcement.** The two web tools are deliberately **excluded from the default `veda-agent`**: web access is a trust decision, not a default. Concretely, the tool catalog now has **15 builtins**, but the default agent still **resolves to 13** — the web tools are present in the catalog (so `veda-web` and explicitly-configured agents can use them) yet absent from the default agent's resolved toolset. A test pins both counts so the exclusion can't silently regress.

---

## 3. Connectors framework + Niriksha page

Ingesting a URL on demand is the imperative case; a **connector** is the declarative one — a named, persisted web data source you can re-run. `crates/veda-gateway/src/rest/connectors.rs` adds:

```
Connector { id, kind: web|websearch, name, config, enabled }
```

persisted via **SavedObjects** (so it lives in the data dir, **tenant-scoped**); `id` is **optional on create** — when omitted the server assigns `conn-{uuid}`. The CRUD surface is admin-only:

- `GET` / `POST /v1/admin/connectors` — list / create;
- `GET` / `PUT` / `DELETE /v1/admin/connectors/{id}` — read / update / delete; and
- `POST /v1/admin/connectors/{id}/run` — **run now**: a `web` connector crawls its configured URLs into its target collection, **reusing the Phase-1 `crawl_and_ingest`** path that backs `/ingest-url` (no duplicate ingestion logic).

On the Niriksha side, `Settings.tsx` gained a **Data Source Connectors** CRUD section — add a web source and trigger **"Crawl now"** — sitting alongside the existing **read-only infra-service status** (the page already showed service health; it now also manages connectors).

---

## 4. Deployment (`--profile web`)

Both web services are optional and self-hosted, gated behind a new Docker Compose profile:

```bash
docker compose --profile web up
```

This brings up two containers:

- **`crawl4ai`** (`unclecode/crawl4ai`) — page → clean markdown crawler; and
- **`searxng`** (`searxng/searxng`) — meta-search aggregator, configured for **JSON output via a mounted `settings.yml`**.

The `veda` service is wired with `VEDA_CRAWL4AI_ENDPOINT` / `VEDA_SEARXNG_ENDPOINT` (plus `VEDA_CRAWL4AI_TOKEN`), **default empty** — so a default `docker compose up` (without `--profile web`) is unchanged and the web features cleanly report "not configured"/503. Bring up the profile, set the endpoints **and `VEDA_CRAWL4AI_TOKEN`** (see §1 — crawl4ai is unreachable without it), and ingestion + the web tools light up.

**SSRF URL guard (default-secure).** Every user-provided URL — on `ingest-url`, the connector `run` path, and the `fetch_url` agent tool — is run through a guard **before any crawl**. The guard **resolves the host and checks the RESOLVED IP** (so it is DNS-rebinding-safe, not a string match on the hostname) and **rejects private/internal/reserved addresses by default**: loopback, RFC1918, link-local (including the `169.254.169.254` cloud-metadata address), and IPv6 ULA. A blocked URL returns **403 `URL_BLOCKED_PRIVATE`** (or, for the `fetch_url` tool, a `{blocked:true}` tool result) **without crawling**. Tune it with host-suffix `VEDA_WEB_ALLOWLIST` / `VEDA_WEB_BLOCKLIST`, or open the guard entirely with `VEDA_WEB_ALLOW_PRIVATE=1` (an escape hatch for trusted internal networks). Deep-crawl `max_pages` is independently clamped to a server cap (`MAX_PAGES_CAP=50`, env `VEDA_WEB_MAX_PAGES`).

| Variable | Default | Notes |
|---|---|---|
| `VEDA_CRAWL4AI_ENDPOINT` | *(empty)* | crawl4ai base URL; unset → `ingest-url` / `fetch_url` / connector `run` return 503 |
| `VEDA_SEARXNG_ENDPOINT` | *(empty)* | searxng base URL; unset → `web_search` is unavailable |
| `VEDA_CRAWL4AI_TOKEN` | *(empty)* | crawl4ai bearer (also `CRAWL4AI_API_TOKEN` on the crawl4ai container); **required** — crawl4ai binds loopback-only and is unreachable from Veda when unset |
| `VEDA_WEB_ALLOWLIST` | *(empty)* | host-suffix allowlist for the SSRF guard (comma-separated) |
| `VEDA_WEB_BLOCKLIST` | *(empty)* | host-suffix blocklist for the SSRF guard (comma-separated) |
| `VEDA_WEB_ALLOW_PRIVATE` | `0` | `1` disables the private/internal-address SSRF guard (escape hatch for trusted networks) |
| `VEDA_WEB_MAX_PAGES` | `50` | server cap on deep-crawl `max_pages` (`MAX_PAGES_CAP`); requested values are clamped to this |

---

## 5. License

The two services have different licenses, and the boundary matters:

- **crawl4ai is Apache-2.0** — fully compatible with Veda's Apache-2.0; nothing to manage.
- **searxng is AGPL-3.0.** Veda only **HTTP-clients a separately-run searxng instance** (an optional `--profile web` service) over its JSON API. AGPL's network-copyleft binds **modifications to searxng itself** — a service whose modified source you must offer to users — it does **not** reach a client that merely calls searxng's API over HTTP. So there is **no contamination** of Veda's Apache-2.0 codebase: Veda ships a client, not a fork. Keeping searxng **opt-in and self-hosted** (you run the upstream image unmodified) keeps it cleanly on the right side of that line.

---

## 6. Testing & validation

- **Unit/mock-tested.** The tests stand up **in-test crawl4ai and searxng servers** (mock HTTP) and exercise the client + endpoints against them, including a connector **`run` that mock-crawls and does a real chunk/embed/insert** end-to-end (so the `ingest_markdown` path is exercised for real, only the network fetch is mocked). The builtin-count test pins the 15-catalog / 13-default-agent split.
- **Live validation requires `docker compose --profile web`.** The mock servers cover the wiring and the ingestion path, but a true end-to-end run against the real `unclecode/crawl4ai` + `searxng/searxng` images **is not run in CI** — the crawl4ai image is large. To validate live: bring up `--profile web`, set the two endpoints, then `POST /v1/collections/{name}/ingest-url` a real URL and converse with the `veda-web` agent. Stated honestly: the live path is not exercised in CI.

---

## 7. Limitations & follow-ups

- **SSRF / arbitrary-URL fetch — guarded by default (v0.4.1, DONE).** `ingest-url`, the connector `run` path, and the `fetch_url` tool fetch **arbitrary user-provided URLs**, a classic SSRF surface (internal addresses, cloud metadata, etc.). This is now **default-secure**: a guard resolves the host and checks the RESOLVED IP (DNS-rebinding-safe) and **rejects loopback / RFC1918 / link-local (incl. `169.254.169.254`) / IPv6 ULA**, returning **403 `URL_BLOCKED_PRIVATE`** (or a `{blocked:true}` tool result) without crawling — tunable via `VEDA_WEB_ALLOWLIST` / `VEDA_WEB_BLOCKLIST` / `VEDA_WEB_ALLOW_PRIVATE`. The v0.4.0 "allowlist is a follow-up" gap is closed.
- **Deep/recursive crawl via `/crawl/job` is the next follow-up.** Today each URL is fetched per-URL via crawl4ai's `POST /md`; `deep_crawl` / `max_pages` are accepted (and `max_pages` is capped at **50**, env `VEDA_WEB_MAX_PAGES`) but true in-domain recursive crawling uses crawl4ai's async `/crawl/job` endpoint, which is not yet wired — that is the next follow-up.
- **Full SaaS connectors are deliberately out of scope.** This is web-native RAG, not a SharePoint/Confluence/Salesforce connector catalog. Matching Elastic's 30+ SaaS connectors is an intentional non-goal — Veda's bet is depth on the open web over breadth of proprietary SaaS sources.
- **searxng must be configured for JSON output.** The `web_search` tool depends on searxng's JSON format, enabled via the mounted `settings.yml` in the compose profile; a searxng instance without JSON output enabled will not work.

### Live-container e2e (2026-06-20)

Brought up `docker compose --profile web` (real `unclecode/crawl4ai` + `searxng/searxng`) and ran Veda against them. Honest results:

- **searxng `web_search` — WORKS live.** The `veda-web` agent called `web_search` and returned a grounded answer from live searxng results (8 real results federated from Google/Bing/etc.).
- **SSRF guard — WORKS live.** `ingest-url` of `http://169.254.169.254/latest/meta-data/` and `http://127.0.0.1/...` both returned **403 `URL_BLOCKED_PRIVATE`** without crawling.
- **crawl4ai `/md` — reached live.** After setting `VEDA_CRAWL4AI_TOKEN`, Veda reached crawl4ai and it launched headless Chromium and attempted to navigate the target (`_crawl_web` / `Page.goto` in the crawl4ai logs), proving the `/md` fix is correct — the old `/crawl` 500'd at routing before any crawl.
- **Live external crawl could NOT complete on this host — an environment limitation, not a Veda bug.** crawl4ai's headless **Chromium cannot egress through this test host's corporate proxy** (`net::ERR_CONNECTION_CLOSED`, "Proxy direct failed"): the container's HTTP clients go through the proxy, but Chromium tries a direct connection. On a normal-egress network (or with crawl4ai's browser configured for the proxy) ingestion works; the Veda side — endpoint, auth, SSRF, chunk/embed/insert — is correct and mock-tested, including a real-insert path.
