# Veda + Niriksha — Frontend Competitive Research

**Date:** 2026-06-10
**Author:** Research pass (8 parallel agents) ahead of the Niriksha UI/UX redesign + marketing/docs site builds.
**Purpose:** Durable reference for three frontend deliverables: (1) redesign Niriksha's observability/console UI, (2) a marketing website for Veda/Niriksha, (3) a documentation website. Covers competitor observability consoles, vector-DB GUIs, marketing sites, pricing models, docs sites/tooling, and visual-design-language + data-viz best practices.

---

## 0. Strategic throughline (read this first)

Three unowned opportunities surfaced repeatedly across every research stream:

1. **Best-in-class visualization is wide-open white space.** Only Qdrant ships an embedding projection + a similarity graph. Nobody offers true 3D, multimodal thumbnail browsing, or a hybrid/fusion playground. Niriksha's Atlas (Apple Embedding Atlas: WebGPU, density contours, in-browser clustering, lasso) can leapfrog the entire category.
2. **A bundled dashboard as a headline product.** No competitor markets a built-in exploration UI as a feature; Pinecone/Chroma users resort to third-party desktop GUIs. Niriksha is a screenshot-able differentiator that logos and GitHub stars cannot replicate.
3. **Multimodal-native positioning + radical pricing transparency.** Competitors lead on speed (Pinecone/Qdrant/Milvus), consolidation (Weaviate/MongoDB), or object-storage cost (Chroma/turbopuffer). None lead with true multimodal (text+image+audio+video in one unified space). Both that lane and the "most transparent pricing in the category" lane are open.

---

## PART 1 — NIRIKSHA: OBSERVABILITY & CONSOLE UX

### 1.1 Vector-DB consoles (most direct competitors)

| Product | Console exploration | Vector visualization | In-console monitoring |
|---|---|---|---|
| **Qdrant** (bundled at :6333/dashboard) | REST playground w/ live validation + Ctrl+Enter; interactive tutorial w/ sample data | **2D scatter (t-SNE/UMAP)** in a web worker w/ progress; `color_by` payload -> auto-legend; **force-directed similarity graph** (click node -> expand nearest neighbors; Kruskal tree mode) — genuinely novel | minimal |
| **Weaviate Cloud** | No-code **Explorer** (cluster->collection->object, color-coded expandable property groups, named-vectors-by-name + copy, raw-JSON toggle) **+** a GraphQL IDE (split browse/query) | none | per-cluster |
| **Attu (Milvus/Zilliz)** | Most feature-complete admin; visual schema designer; CSV/JSON/Parquet import-export; embedding-provider-configurable search; **embedded AI agent w/ 50+ tools** | none | **16+ Prometheus metrics, interactive topology, slow-query analysis** |
| **Pinecone** (hosted) | Basic record **Browser** tab (not a rich playground; 3rd-party "Pinecone Explorer" fills the gap) | none | Metrics tab (capacity, ops, latency) — **tier-gated**; some serverless metrics empty |
| **Chroma Cloud** | CLI TUI (`chroma browse`) richer than the thin web UI | none | p50/p90/p99 latency, request/storage volume, spend |

**Pattern to adopt:** Weaviate's deliberate **split of "browse" (no-code object inspector) from "query" (power playground)**. Qdrant's **web-worker projection with progress** (never blocks UI) and **color_by -> auto-legend**. Pinecone's **one-time-reveal API key + RBAC roles UI**. Qdrant's **first-run interactive tutorial with sample data** for empty states.

**White space to beat them:** true **2D/3D** atlas + Qdrant-style **click-to-expand-neighbors graph** + density/cluster overlays + lasso->bulk-inspect; **multimodal thumbnail rendering** inline in browse + results (every competitor renders text/JSON rows only); **dense/sparse/hybrid + fusion-weight playground with side-by-side result diffing**; **always-on (not paywalled) monitoring** unified into the console; consider an **embedded AI agent** (Attu) as emerging table-stakes.

### 1.2 Kibana / Elastic (analytics + observability UI)

- **#1 lesson — navigation:** Kibana's legacy 100+-item flat nav is the root of its "overwhelming" reputation. Elastic's fix = **solution-oriented navigation** (Search / Observability / Security / Classic) scoped per **Space**. Keep observability surfaces lean; hide unrelated tooling.
- **Discover** (the gold-standard exploration idiom): **field sidebar ordered by popularity**, click-a-field -> **top-10 values + record counts + one-click "Visualize"**, KQL query bar w/ autocomplete, **collapsible filter pills** w/ rich context menus (pin/invert/disable/delete), a **collapsible histogram**, and **Saved Sessions** (queries+filters+columns+time, embeddable).
- **Lens** = drag-and-drop viz w/ auto-suggested chart type. **Drilldowns** (dashboard->dashboard, URL, ->Discover) preserve filters/time/selection context.
- **APM:** Service Map (force-directed deps), **trace waterfall** (spans color-coded by service, click->SQL/command + % of trace time, library-vs-user-code framing), an **"Investigate" button** that jumps to correlated logs/metrics without losing context.
- **Contextual alert creation** pre-filled from the view you're in (service/trace/metric), not a disconnected rules form. Flapping detection, snooze, investigation guides.
- **EUI design system:** semantic tokens whose **names are identical across light/dark — only values change**; **10 colorblind-safe dataviz tokens** + a 5-step severity palette; dark mode enforces 3:1 contrast on dataviz; dense, flat, 14px body. **Elastic Charts** is their dedicated React charting lib.
- **Criticisms:** steep learning curve, tight Elasticsearch coupling/lock-in, limited multi-source dashboard flexibility, historically slow rendering, confusing destructive micro-flows.

### 1.3 Grafana (gold-standard OSS observability dashboarding)

- **Queryless "Drilldown" apps** (Metrics/Logs/Traces/Profiles) are Grafana's biggest UX bet: point-and-click exploration, auto-generated per-label breakdowns, **no query syntax**. This is the clearest pattern to borrow for a focused single-DB dashboard.
- **Dual-mode query builders** (Builder <-> Code w/ two-way sync + Explain) lower the barrier while keeping a power escape hatch.
- **Explore mode** w/ split view (compare ranges/queries/sources); **exemplars** (diamonds on metric graphs link a point -> the contributing trace); **correlations** for metrics->logs->traces pivots.
- **Encoded viz best practices** (adopt as defaults): thresholds, units, tooltip Single/All modes, legend calculations (min/max/mean/last), annotations for events, null-handling.
- **Unified alerting** cleanly separates Rules / Contact points / Notification policies (label-matcher routing tree).
- **Saga design system:** tokens via Token Studio + Figma; **Inter** typeface; theming in TS via `createTheme()` -> `GrafanaTheme2` (colors/spacing/shape/typography/viz palette) shared TS<->Sass.
- **Criticisms:** steep curve for advanced dashboards/alerting; **dashboard sprawl + monolithic diff-hostile JSON**; perf degradation on large dashboards.
- **Differentiate:** ship opinionated, code-defined, versioned views (no blank-canvas builder, no DSL exposure). Build **domain-native viz Grafana lacks** — vector-space projections, ANN recall-vs-latency frontier curves, segment/shard topology, per-modality breakdowns.

### 1.4 Modern observability SaaS (Datadog / Honeycomb / New Relic / Axiom)

- **Honeycomb structured query builder** = "query-language power without writing one": clickable clauses (VISUALIZE/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT) w/ autocomplete; GROUP BY auto-splits series; top-50-series density cap.
- **Honeycomb BubbleUp = the single highest-value pattern to borrow.** SELECT HEATMAP -> user **draws a box** around anomalous points -> tool computes **Selection (yellow) vs Baseline (blue)** across hundreds of dimensions, ranked by difference (bar charts + donut %). Visual-selection-as-query: point at the symptom, it tells you the cause. (Datadog **Watchdog** is the AI counterpart — a feed of "stories" + "Explains".)
- **Datadog unified search bar:** key:value + free text in one input, **syntax highlighting** (keys/values/free-text/control chars), autocomplete, **inline error validation that suggests fixes**; facets NOT required up front.
- **Datadog trace view:** four toggleable views of one trace (Flame Graph / Waterfall / Span List / Map); **Trace Preview mode** for >100MB traces (numbered pills expand hidden groups) — smart density control. Cross-signal correlation tabs (Logs/Infra/Profiles/Span Links) on span detail.
- **New Relic:** visual Data Explorer **generates the underlying NRQL** (teaches the query language through use); omnipresent context-preserving NRQL console.
- **Axiom:** raw-APL-first but adds **Cmd-K natural-language->query** and AI-generated dashboards from schema.
- **Density discipline + saved views + live tail** recur everywhere. Documented pitfalls to avoid (Honeycomb review): broken log-scale histogram buckets, missing unit labels ("100k" w/o ms), query/trace views being disconnected.

### 1.5 Niriksha observability — consolidated takeaways

1. **Queryless-first** exploration over Veda-native dimensions (collection, index type, modality, shard/segment). Never expose PromQL/LogQL.
2. **Reuse Veda's `"explain": true` per-phase timing as a mini waterfall** (filter->ANN->fusion->rerank) — the right-sized analog to a distributed trace; no competitor exposes this in-console.
3. **BubbleUp-style "what's different?"** on a slow/low-recall query histogram (box-select -> Selection-vs-Baseline across k / filter selectivity / index type / modality).
4. **One smart search bar** (key:value + free text, highlighting, autocomplete, inline error hints).
5. **Structured clause builder** over a DSL; if a DSL appears later, reveal the generated query.
6. **Saved views + live tail + faceted sidebar + toggleable views** (list/timeseries/table) over one result set.
7. **Solution-scoped nav** (avoid Kibana's flat-100). **Density discipline** (top-N caps, progressive disclosure, small fixed severity palette, units on every axis).
8. **Drilldowns/Investigate links** that preserve filter/time/selection context across views.

---

## PART 2 — MARKETING WEBSITE

### 2.1 Vector-DB marketing sites — hero positioning

| Company | Lead message | Primary CTA | The ONE thing |
|---|---|---|---|
| **Pinecone** | "Search through billions of items... an API call away" | Start Building | Speed + scale, zero-ops, managed |
| **Weaviate** | "For AI engineers... one foundation for search, RAG, agents" | Get Started / Try Cloud | Unified AI-app platform |
| **Qdrant** | "High-Performance Vector Search at Scale" | Start Free in Cloud | Performance + Rust + deployment flexibility |
| **Milvus/Zilliz** | "High-Performance Vector Database Built for Scale" | Try Zilliz Cloud | OSS scale + maturity (40k+ stars, LF AI & Data) |
| **Chroma** | "Open-source search infrastructure for AI... Apache 2.0" | Start free on Cloud | OSS dev-first + object-storage cost ("10x cheaper") |
| **LanceDB** | "Build Better Models, Faster... AI-native Multimodal Lakehouse" | Get Started | Multimodal lakehouse (closest to Veda) |
| **turbopuffer** | "search every byte... built on object storage: 10x cheaper" | Sign up | Cost via object storage (Cursor/Anthropic/Notion logos) |
| **MongoDB Atlas** | "Build intelligent applications... native vector capabilities" | Get started free | "Avoid the synchronization tax" |

Two hero strategies: **performance/scale** vs **platform/workflow consolidation**; newest entrants lead on **object-storage cost**.

### 2.2 Developer-facing first scroll
- **Weaviate** is strongest: live Python snippets for vector/semantic/hybrid + a before/after complexity-reduction comparison.
- **Chroma / turbopuffer**: TS snippet + terminal output w/ latency + **ASCII architecture diagrams** ("systems engineer" credibility).
- **Pinecone / Qdrant**: no code in first scroll (value-prop cards instead). **Milvus**: `pip install` as the hook.

### 2.3 Social proof & benchmarks
- Logo walls; **GitHub stars as a badge** (Qdrant 30k+, Chroma 27k + 15M downloads, Milvus 40k); Pinecone uses "9,000+ customers" (closed-source).
- **Benchmark framing** dominated by **VectorDBBench** (Zilliz's OSS tool: QPS / p99 / recall / cost on Cohere-10M). Vendors lead with whichever axis they win (Qdrant "up to 64x less memory" via quantization; Chroma "p50 20ms warm, 90-100% recall").

### 2.4 OSS-vs-cloud positioning
Everyone except Pinecone is **open-core**. Qdrant offers a 4-deployment matrix (Cloud/Hybrid/Private/Edge). Weaviate frames OSS + cloud as **parallel equals**. Chroma is the OSS-purist ("no licensing limits, no lock-in"). Milvus = Apache-2.0 under LF AI & Data; Zilliz Cloud = commercial. **The pgvector threat:** 2026 discourse ("vectors are a data type, not a database category") makes a "why not just pgvector / a text-only DB?" rebuttal mandatory.

### 2.5 Pricing models — compact comparison

| Company | Model | Free tier | Paid entry | Transparency |
|---|---|---|---|---|
| Pinecone | Serverless usage + monthly minimums | 2GB/2M writes/1M reads | Builder $20/mo; Standard $50 min; Ent $500 min | Moderate-high (calculator, many SKUs) |
| Qdrant | Resource-based (vCPU/RAM/disk), hourly | 0.5 vCPU/1GB/4GB forever | usage / min-spend | **Low** ($ hidden behind calculator) |
| Weaviate | **Dimension-based** usage | 14-day sandbox | Flex $45/mo; Premium $400/mo | High (per-dimension calculator) |
| Zilliz | Compute-Unit (CU) | 5GB + ~100 CU-hrs | Serverless $0; Dedicated ~$99+ | Moderate; storage cut to $0.04/GB-mo (Jan 2026) |
| Chroma | Serverless usage | $5 credits/mo | Team $250/mo + usage | **High** (publishes a sample bill) |
| turbopuffer | Usage, object-storage cheap | calculator | comparative "10x cheaper" | Low (comparative) |

Universal shape: **free tier + interactive calculator + "Contact Sales" enterprise**. Market splitting between **serverless/usage-based** and **resource/node-based**.

### 2.6 Multi-product platform marketing (Elastic / Grafana / Datadog / Honeycomb / MongoDB)
- **Use Elastic's "one engine, few solutions" model, NOT Datadog's 30-product sprawl.** Elastic collapses everything into Search/Observability/Security over one engine — clean for a small company.
- **Three-rung ladder is universal:** OSS self-managed (free, the acquisition wedge) -> managed cloud (usage-based) -> enterprise (contact sales).
- **License = trust + marketing.** Elastic's Apache->SSPL->AGPL saga + the OpenSearch fork prove it. Veda's **Apache-2.0 is an asset — lead with it.**
- **Pricing transparency is a positioning lever.** Honeycomb's single predictable event-based unit is marketed as anti-penalty; **Datadog is the cautionary tale** (compounding per-host + per-GB-ingest + per-event-index + custom-metric tax -> "bills surprise teams"; the container trap, high-water-mark host billing, ingest-then-index double charge).
- **Grafana's free tier is the market's most generous** — a deliberate acquisition wedge.
- Stand up a **Trust Center** early (SOC2/ISO/compliance), logo wall, ROI framing to unlock the enterprise rung. Run a **dual-audience** split: dev-first authentic content (Honeycomb) feeding a buyer-facing solutions/ROI/compliance track.

### 2.7 Veda marketing site — consolidated takeaways
1. **Hero leads with multimodal-native** ("One vector database for every modality") — the unowned lane.
2. **Runnable code in the first scroll showing multimodal** (image + text into one collection, queried across both, terminal latency). Pair with a clean architecture diagram.
3. **Niriksha as a headline pillar** with its own screenshot section.
4. **Apache-2.0 front-and-center**; OSS + cloud as parallel equals; mirror Qdrant's deployment matrix.
5. **Most transparent pricing in the category:** one predictable primary unit (stored vectors/dims + storage GB + query/compute units), a calculator, and a **worked sample bill**. Avoid Datadog's compounding traps.
6. **Honest benchmarks** (VectorDBBench-style) + a **multimodal-recall axis competitors can't match** + a "when you've outgrown pgvector" section.
7. Social-proof scaffolding even pre-logos: stars/downloads/Discord + 2-3 use-case pages (RAG, multimodal search, agent memory).

---

## PART 3 — DOCUMENTATION WEBSITE

### 3.1 Information architecture
- **Diataxis-inspired split** is dominant: Tutorials / How-to / Reference / Concepts.
- **Pinecone:** multiple **quickstarts by use case** + intent-split (integrated-embedding vs bring-your-own-vectors) — directly relevant to Veda's auto-embedding vs BYO duality.
- **Qdrant:** skill-laddered tutorials (Basics -> Search Engineering -> Operations & Scale) — strong model.
- **Supabase:** front-loads framework quickstarts (10+) + per-language client libs + competitor migration guides.

### 3.2 API reference
- **Stripe three-column layout** is the benchmark: left nav, center prose, **right column of live copy-paste code**; one-click language switching; auto-injected test keys when logged in; synchronized hover-highlighting; embedded shell for live calls.
- **Spec-driven generation** (Mintlify, Fern) auto-builds reference + interactive "try it" playground from OpenAPI -> no doc/code drift. Veda already has `documentation/openapi.yaml`.

### 3.3 Search & AI assistants (2025-2026 shift)
- Conversational **"Ask AI"** is now standard: Weaviate + Supabase (Cmd-K "Ask AI or Search"), Grafana "Ask Grot", Milvus/Zilliz "Ask AI", **Algolia DocSearch v4** bundles grounded/cited Ask-AI.
- **AI-readiness convention:** publish **`/llms.txt` + `/llms-full.txt` + clean Markdown** (confirmed on Pinecone, Grafana). Veda can **dogfood its own vector DB** to power docs search — a genuine differentiator + live demo.

### 3.4 Docs tooling landscape 2026
- **Mintlify** — beautiful OOTB, docs-as-code (MDX/Git), auto OpenAPI playground, AI assistant w/ citations, auto llms.txt/MCP. Hobby $0 (incl. playground + semantic search); Pro ~$250-300/mo adds AI assistant. **What Pinecone uses.** Con: limited deep customization.
- **Docusaurus** — free, infinitely customizable (React/MDX, versioning, i18n, self-host). **What Weaviate uses.** Con: generic default look; you build AI/search/playground yourself.
- **Fern** — best when the OpenAPI IS the product: generates type-safe SDKs (9 langs) + reference + interactive Explorer from one spec. Powers Square/ElevenLabs.
- **Nextra** (Next.js/MDX) — good middle ground if already in React/Next.
- **Hugo** — fastest builds (Qdrant) but manual for API ref + modern AI/search.

### 3.5 Veda docs — recommendation
- **Adopt Diataxis**; ladder tutorials like Qdrant; mirror Pinecone's auto-embedding-vs-BYO intent split; add a Niriksha section + a multimodal/document-pipeline guide.
- **Multiple quickstarts** (curl, Python `veda-client`, Docker Compose) showing the seal-after-insert flow end-to-end.
- **Generate API reference from `openapi.yaml`** + interactive console + multi-language switchers.
- **Ship `/llms.txt` + `/llms-full.txt`** + an "Ask AI" assistant powered by Veda itself.
- Must-haves: version picker, dark mode, on-this-page TOC, breadcrumbs, "Was this helpful?", copy buttons.
- **Recommended stack:** **Mintlify** to launch best-in-class fast (matches Pinecone), OR **Docusaurus** for zero recurring cost + full control (matches Weaviate) w/ Algolia DocSearch+AskAI bolted on. Evaluate **Fern** if SDKs become the centerpiece.

---

## PART 4 — VISUAL DESIGN LANGUAGE & DATA-VIZ

### 4.1 Premium developer-tool aesthetics
- **Linear:** strict **8px spacing** scale; standardized motion (fixed durations + easing); 2025 refresh **cut color aggressively** to near-monochrome + few accents; high contrast, subtle gradients/glass. Restraint + consistency.
- **Vercel Geist:** "tight type, sharp edges, restrained color, generous space." Pure black/white, Geist Sans + Geist Mono, **minimal/near-sharp radius**. Display 48-64px @ -0.04em tracking, ~1.15 line-height. Same-family Sans/Mono unifies copy + code.
- **Stripe:** signature purple gradients + light weight-300 elegance; token-driven multi-theme (dark + "darker" for dev apps).

### 4.2 Design systems / token philosophy
- **Radix Colors** — canonical **12-step scale** (solid + alpha): 1-2 app/subtle bg, 3-5 component bg default->hover->active, 6-8 borders subtle->interactive->focus-ring, **9** purest accent (solid fills), 10 hover, **11** low-contrast text, **12** high-contrast text; guaranteed APCA contrast vs step-2. Re-skin by remapping step 9. **shadcn-ui builds on Radix + Tailwind.** Most directly reusable model.
- **Geist colors** — 10 scales (gray + 9 accents) w/ state-based 1-10 progression; P3 on supported displays.
- **Grafana Saga** — observability-domain benchmark; dark+light kits; Token Studio + Figma.
- **EUI** — mature data-dense system; good secondary reference for table/filter-heavy layouts.

### 4.3 OKLCH (the consensus direction; Tailwind's default palette now uses it)
- Splits color into **Lightness / Chroma / Hue** with **perceptual uniformity** — equal numeric steps look equal across hues (unlike HSL).
- Payoffs: predictable contrast; **trivial dark mode** (adjust only L, keep C/H); relative syntax `oklch(from var(--accent) calc(l + 0.1) c h)` for hover states; **P3 wide-gamut** (~30% more colors).
- For categorical dashboard palettes: **>=15 OKLCH L-units between adjacent colors** for grayscale/colorblind distinguishability.

### 4.4 Embedding-visualization tech — how they scatter millions of points
- **Apple Embedding Atlas** (the stated inspiration): **WebGPU w/ WebGL2 fallback**, **order-independent transparency** (overlaps don't turn to mud), **KDE + density contours**, clustering on the projected 2D density map (sub-second labels), UMAP in WASM, density clustering in Rust->WASM. Ships npm w/ **React + Svelte bindings**, Jupyter widget, Python CLI. Handles a few million points in-browser. Repo: github.com/apple/embedding-atlas
- **Nomic deepscatter:** **>1B points** via Apache Arrow feather + quadtree tiled at 50k pts/tile, GPU transforms (Regl), static-hostable; zoom-level topic labels.
- **regl-scatterplot** (Fritz Lekschas): **up to 20M points**, WebGL/Regl, **fast lasso** (freeform/rect/brush), KDBush spatial index, density-based opacity, performance mode. Most production-ready standalone engine.
- **TF Embedding Projector / scatter-gl** (Google PAIR): nice 2D/3D pan/rotate/zoom UX but only ~tens of thousands of points.

### 4.5 React charting libraries
- **Recharts** — sensible default (SVG/D3, low code); powers **Tremor** (beautiful-by-default dashboard blocks — good for metric panels).
- **visx** (Airbnb) — D3 primitives as React components; for bespoke chart types.
- **Apache ECharts** — Canvas + WebGL; perf pick for 100k+ points + streaming.
- **uPlot** — tiny, extremely fast Canvas time-series (high-frequency latency/throughput lines).
- **Observable Plot** — concise exploratory grammar-of-graphics.

### 4.6 Niriksha — actionable design tokens & patterns
- **Color:** Radix-style **12-step scale defined in OKLCH**, dark-first. Low-chroma blue-gray neutral base; brand hue at step 9. Dark mode by lowering only L; ship P3. Keep chrome restrained — reserve saturated color for **data, not chrome**.
- **Categorical viz palette:** 8-10 OKLCH colors w/ >=15 L-units separation (colorblind-safe); sequential ramps by walking L at fixed C/H.
- **Type:** Inter for UI (display tracking -0.02 to -0.04em, line-height ~1.15), **JetBrains Mono** for IDs/vectors/metrics/code. Tokenize size+line-height+tracking+weight together (Geist-style).
- **Spacing/radius/motion:** 8px base; small radii (4-8px); standardized motion (~150-250ms, one easing curve, e.g. cubicInOut); subtle, never decorative.
- **Marketing vs docs vs app:** marketing may lean into a restrained gradient/grain hero (Stripe-lite) over the same tokens; docs + app stay flat, high-contrast, dense (Saga/EUI discipline).
- **Charting + scatter recommendation:**
  - **Atlas page (the differentiator):** build on **`@apple/embedding-atlas`** React binding (WebGPU/WebGL2, density contours, in-browser clustering, lasso, NN search — matches inspiration, zero custom GLSL). Drop to **regl-scatterplot** (20M pts) if exceeding its ceiling; **deepscatter** only for billion-point tiled maps.
  - **Dashboard charts:** **Recharts/Tremor** standard panels -> **visx** bespoke -> **ECharts/uPlot** for high-cardinality/streaming.

---

## APPENDIX — KEY URLS

**Observability/console:** elastic.co/docs (Discover, Lens, APM), eui.elastic.co (tokens/dataviz), grafana.com/docs (Drilldown, Scenes, time-series viz, alerting), grafana.com/developers/saga, docs.honeycomb.io + honeycomb.io/platform/bubbleup, docs.datadoghq.com (trace_view, service map, log explorer, search syntax, watchdog), docs.newrelic.com (query builder), axiom.co/docs/apl.

**Vector-DB consoles:** qdrant.tech/documentation/web-ui, github.com/qdrant/qdrant-web-ui, zilliz.com/attu + github.com/zilliztech/attu, docs.weaviate.io/cloud/tools/explorer-tool, docs.pinecone.io/docs/monitoring, trychroma.com.

**Marketing/pricing:** pinecone.io, weaviate.io/pricing, qdrant.tech/pricing, milvus.io + zilliz.com/pricing, zilliz.com/vdbbench-leaderboard, trychroma.com/pricing, lancedb.com, turbopuffer.com, mongodb.com/products/platform/atlas-vector-search; datadoghq.com/pricing, grafana.com/pricing, honeycomb.io pricing, elastic.co/pricing.

**Docs:** docs.pinecone.io, docs.weaviate.io, qdrant.tech/documentation, milvus.io/docs, docs.stripe.com/api, supabase.com/docs, diataxis.fr, mintlify.com, docusaurus.io, buildwithfern.com, docsearch.algolia.com.

**Design/data-viz:** linear.app/brand, vercel.com/geist (colors/typography), radix-ui.com/colors, grafana.com/developers/saga, evilmartians.com/chronicles/oklch-in-css-why-quit-rgb-hsl, github.com/apple/embedding-atlas + apple.github.io/embedding-atlas, github.com/nomic-ai/deepscatter, github.com/flekschas/regl-scatterplot, blog.logrocket.com/best-react-chart-libraries-2025.
