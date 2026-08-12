# New entrants and adjacent players — the two contested lanes

**Research date:** 2026-08-12. These players are absent from the April 2026 comparison docs.
Weighted toward the two lanes the May SOTA audit called contested: **object-storage-native
architecture** and **multimodal-native positioning**.

---

## Market context

| Metric | Figure | Source |
|---|---|---|
| Market size 2025 | ~$2.7–2.8 B | [Ailog](https://app.ailog.fr/en/blog/news/vector-database-trends-2026), [Medium/Rupareliya](https://medium.com/@pratik-rupareliya/top-15-vector-databases-in-2026-a-production-decision-guide-from-100-enterprise-deployments-dd58a04f51a5) |
| Projection | ~$8.5 B by 2028 / ~$9 B by 2030 (27.5% CAGR) | same |
| Primary driver | **RAG is now the dominant use case** | same |
| 2026–27 outlook | consolidation by cloud providers, multimodal support emerging, RAG-as-a-Service, price competition favouring open-source and cost-efficient newcomers | Ailog |

The stated 2026 architectural trends are: hybrid dense+sparse as standard, aggressive
quantization (4×–32×), multi-tenancy by namespace/partition, and native generative integration.
**Veda has all four.** None of them is a differentiator any more.

---

## Lane 1 — Object-storage-native

This is the lane the May audit flagged and it has now been *commercially* validated, not just
architecturally.

### Turbopuffer

| Attribute | Detail |
|---|---|
| Architecture | Object storage (S3/GCS/Azure Blob) as source of truth + intelligent caching; no SSD-resident vectors |
| Scale in production | **1 T+ documents, 10 M+ writes/s, 25 k+ queries/s** |
| Latency | sub-10 ms p50 |
| Search modes | dense ANN, BM25 full-text, **sparse vectors (added April 2026)**, nested boolean attribute filters, or any combination in one request |
| Revenue | ~**$100 M annualized** (Mar 2026), up from ~$75 M end-2025 |
| Funding | seed from Thrive Capital, Dec 2025 |
| Anchor customers | **Cursor** (1 T+ code chunks across 80 M+ namespaces), Notion |

Sources: [turbopuffer](https://turbopuffer.com/) ·
[Sacra](https://sacra.com/c/turbopuffer/) ·
[vector guide](https://turbopuffer.com/docs/vector)

**Read:** the "cheap object storage + smart caching" thesis is proven at a scale Veda has not
approached. Turbopuffer does *not* do multimodal, OCR, RAG eval or agents — it is deliberately a
fast, cheap retrieval primitive. It is not a feature competitor; it is a **cost-per-vector
competitor**, and it sets the price expectation buyers bring to us.

### Amazon S3 Vectors

- **GA December 2025** with 40× the preview scale: up to **2 B vectors per index**, 10 000
  indexes per vector bucket.
- Claims **up to 90 % lower total cost** to upload/store/query vs specialized vector DBs.
- **June 2026:** query charges cut a further **up to 80 %** for indexes above 10 M vectors.
- **March 2026:** expanded to 17 more regions — **31 regions** total.
- Latency: sub-second for infrequent queries, ~100 ms or better for frequent ones.
- AWS positions it as **"complementary"** to vector databases; analysts are split.

Sources: [AWS GA](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/) ·
[query price cut](https://aws.amazon.com/about-aws/whats-new/2026/06/s3-vectors-reduces-query-charges-80-percent-large-indexes/) ·
[region expansion](https://aws.amazon.com/about-aws/whats-new/2026/03/s3-vectors-expands-17-regions) ·
[VentureBeat analysis](https://venturebeat.com/infrastructure/aws-claims-90-vector-cost-savings-with-s3-vectors-ga-calls-it-complementary)

**Read:** the floor on "just store and query vectors" is now ~100 ms at near-zero cost, from the
hyperscaler, in 31 regions. Any Veda pitch that leads with raw vector storage economics loses.
The counter is everything *around* the vector — which is where Veda actually lives.

### Chroma Cloud, Milvus 3.0 Loon

Covered in `01`. Both moved to object-storage-native in 2026. Together with Turbopuffer and S3
Vectors this is now the **default architecture of the category**, not a differentiator.

---

## Lane 2 — Multimodal-native

The May audit and the June frontend research both recorded multimodal-native positioning as an
**unclaimed lane**. It is now contested by at least five players.

### Marqo — the closest thing to a direct pitch collision

> "Marqo specializes in multi-modal AI: **text, images, audio, and video in one search system**.
> The end-to-end approach means you send raw content (images, text, video) to a single API.
> Marqo handles **embedding generation, storage, and retrieval**."

Open-source, single REST API, tensor + lexical + hybrid search, index lifecycle, model
lifecycle, recommendations, telemetry and health endpoints. Proprietary ecommerce models claim
up to **88 %** better than Amazon Titan Multimodal for product search (+94.5 % NDCG@10 in-domain).

Sources: [Marqo](https://www.marqo.ai/) · [GitHub](https://github.com/api-evangelist/marqo) ·
[Simplyblock](https://simplyblock.io/glossary/what-is-marqo/)

**Read:** this is the single most direct competitor to Veda's **Part 2** claim ("pure vector DBs
require you to run embedding/OCR/chunking externally"). Marqo does end-to-end multimodal with
built-in embedding, in the open. Its focus is **ecommerce/product discovery**, not enterprise
document RAG — that vertical difference, plus Veda's OCR/document pipeline and RAG-quality
layer, is what remains defensible.

### MongoDB Atlas Vector Search + Voyage AI

- **Voyage 4** family (voyage-4, -large, -lite, -nano) for retrieval accuracy/cost/latency tiers.
- **`voyage-multimodal-3.5`** — adds **video** to text and images, and explicitly targets
  "tables, graphics, figures, slides, **PDFs**".
- **Automated embeddings**: vectors generated automatically on write/update — "eliminates the
  need for separate embedding pipelines".
- Lexical prefilters for vector search; AI assistant in Compass / Atlas Data Explorer.

Sources: [PRNewswire](https://www.prnewswire.com/news-releases/mongodb-sets-a-new-standard-for-retrieval-accuracy-with-voyage-4-models-for-production-ready-ai-applications-302662558.html) ·
[TechTarget](https://www.techtarget.com/searchdatamanagement/news/366637414/MongoDB-launches-latest-Voyage-models-to-aid-AI-development) ·
[MongoDB AI search](https://www.mongodb.com/products/platform/ai-search-and-retrieval)

**Read:** MongoDB now bundles a frontier multimodal embedder **and** auto-embedding-on-write into
a database enterprises already run. "Multimodal + no separate embedding pipeline" is exactly
Veda's `auto_embed` pitch, shipped by a vendor with an existing enterprise footprint.

### Vespa

Multiple embeddings per document enabling **ColBERT / ColPali-style retrieval, image–text
matching and cross-modal search in one schema**. Native ColBERT embedder is GA with an
asymmetric compression technique giving **32× compression** of token-level vectors without
significant ranking loss; long-context ColBERT also shipped.

Sources: [multi-vector HNSW](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) ·
[ColBERT embedder](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/) ·
[long-context ColBERT](https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/)

**Read:** Vespa remains the most technically serious late-interaction implementation. Veda's
MUVERA + PLAID/WARP path is defensible but Vespa's 32× ColBERT compression is a specific,
published number we cannot currently match with a citation.

### Cohere Embed v4

Described as the first embedding model to vectorize **text, images and interleaved documents**
together. Reinforces that unified multimodal embedding is now a *model-layer* commodity supplied
by several vendors (Jina, Voyage, Cohere, Qwen) rather than a database differentiator.

### Nexus and KnowQL

Two new entrants launched **May 2026**. Low signal so far — no substantial primary sources
surfaced beyond listing coverage. **Flagged for the next refresh** rather than assessed here;
treating a listicle mention as evidence of a competitor would be exactly the marketing-vs-
verified error this document is meant to avoid.

Source: [Ailog trends](https://app.ailog.fr/en/blog/news/vector-database-trends-2026)

### Redis

Vector search remains an **add-on** to a primary datastore, in the same category as pgvector —
a different buyer from Veda's.

Source: [Firecrawl comparison](https://www.firecrawl.dev/blog/best-vector-databases)

---

## What this means for Veda

| Lane | Status in April/May | Status now |
|---|---|---|
| Object-storage-native | "the architecturally distinctive 2026 move" (May audit) | **Category default.** Turbopuffer ($100 M ARR), S3 Vectors (GA, 31 regions), Chroma Cloud, Milvus Loon |
| Multimodal-native positioning | "unclaimed lane" (June research) | **Contested by ≥5**: Elastic (Jina omni on-prem), Marqo, MongoDB (voyage-multimodal-3.5), LanceDB, Weaviate (+Marengo video) |
| Unified embedding across modalities | Veda's wedge | **Model-layer commodity** — Jina v5 omni, voyage-multimodal-3.5, Cohere Embed v4, Qwen3-VL |
| End-to-end (raw content → searchable) | Veda + few others | **Marqo** (open, multimodal), **MongoDB** (auto-embed on write), **Pinecone** (inference API) |

The honest position: **the multimodal wedge has closed at the model and storage layers.** What
has *not* been replicated by any single competitor is the **combination** Veda ships — OCR/document
pipeline **plus** unified cross-modal retrieval **plus** the RAG-quality layer (eval harness,
faithfulness/citations, chunker A/B, auto-tuning, semantic cache, guardrails) **plus** vector-native
AIOps, all self-hostable. See `07-veda-position-and-stale-claims.md` for what that means for the
positioning doc.
