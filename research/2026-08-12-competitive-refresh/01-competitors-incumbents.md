# Incumbent competitors — delta since April/May 2026

**Research date:** 2026-08-12. Baseline documents this updates:
`research/vector-database-comparison.md` (Milvus/Weaviate/Qdrant),
`research/vector-db-comparison.md` (Chroma/Pinecone/pgvector),
`research/vector-database-research.md` (OpenSearch/FAISS/LanceDB) — all compiled April 2026 —
plus the Elasticsearch positioning in `competitive_advantages.md` (2026-07-02).

Every row carries a source URL. The **"Erodes a Veda claim?"** column refers to the numbered
claims in `competitive_advantages.md`.

---

## Headline: the two claims that moved

Two things happened in late July / early August 2026 that change our positioning materially.

1. **Elastic shipped unified multimodal — self-hosted.** On 2026-07-27 Elastic released
   *Jina On-Prem*: 28 Jina models including **`jina-embeddings-v5-omni`** (text, image, audio,
   video in **one** embedding space) and `jina-reranker-v3`, running **entirely air-gapped**
   with "no calls to the outside once deployed". Elastic Stack 9.5 (2026-08-04) added a
   `semantic` field that accepts **text, images, video, audio and documents**.
   → This is the same model family Veda's `jina-v5-omni-small` preset uses.
2. **Milvus went lake-native.** Milvus 3.0 (2026-07-27/29) reads Parquet / **Lance** / Iceberg /
   **Vortex** in place, zero-copy, and replaced its storage engine with a manifest-based
   columnar engine on object storage.

Both land squarely on lanes the May SOTA audit identified as Veda's differentiators or open gaps.

---

## 1. Elasticsearch — the primary comparator

| Attribute | April/May 2026 baseline | Today (2026-08-12) |
|---|---|---|
| Latest version | 8.19 (the version our June benchmark used) | **9.5**, released 2026-08-04 |
| Multimodal | text + `dense_vector`/kNN + ELSER; non-text as opaque attachments | **`semantic` field supports text, images, video, audio, documents** (9.5+) |
| Multimodal model | — | `jina-clip-v2` (text+image, one space, 89 languages, 2026-06-23); **`jina-embeddings-v5-omni`** (all four modalities, ~100 languages) |
| Self-hosting the embedder | Elastic Inference Service (cloud) | **Jina On-Prem** (2026-07-27) — all 28 Jina models air-gapped; **single-command install, no outbound calls at all: no license server, no telemetry, no model registry**; CPU/GPU auto-detect, small models on a single 8 GB GPU; drop-in EIS replacement so existing integrations work unchanged |
| Quantization | BBQ (1-bit), GA in 9.0 | **BBQ 2/4/7-bit** (9.4); DiskBBQ **auto-calibration** of depth/preconditioning/oversampling (9.5) |
| GPU | — | **cuVS GPU indexing GA** in 9.4 (tech preview in 9.3): up to **12× indexing throughput**, 7× faster force-merge |
| Dense search perf | our benchmark: ~7 ms p50 | 9.4 adds native SIMD kernels (~3× vector-op throughput) and **≥3× lower latency on restrictive filters**; 9.5 adds a `VectorDB` index mode with pre-optimized defaults |
| Late interaction | ColPali / ColBERT support (9.0/8.18) | unchanged, plus Jina reranker v3 |
| Storage | — | **Columnar Mode** (tech preview, 9.5) — column store, no inverted index by default |
| Agents | Agent Builder | Workflows **GA** (9.4); Agent Builder + Skills/Attachments/Connectors/Plugins; agent observability as OTel traces (preview, 9.5); MCP apps |

**Erodes:** claims **#1** (multimodal wedge), **#2** (document ingestion — `semantic` accepts PDFs),
**#5** (self-hosted/private), **#7** (agent-native interop), and the **performance table**
(benchmarked against 8.19; 9.4/9.5 changed the dense path substantially).

Sources: [Elastic 9.4](https://www.elastic.co/blog/whats-new-elastic-9-4-0) ·
[Elastic 9.5](https://www.elastic.co/blog/whats-new-elastic-9-5-0) ·
[jina-clip-v2 on EIS](https://www.elastic.co/search-labs/blog/multimodal-search-elasticsearch-jina-clip-v2) ·
[semantic_text reference](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text) ·
**[Jina On-Prem — Elastic Labs (primary)](https://www.elastic.co/search-labs/blog/on-prem-ai-jina-embedding-models)** ·
[Jina On-Prem announcement](https://www.stocktitan.net/news/ESTC/elastic-brings-jina-multimodal-and-multilingual-semantic-search-to-cf0n1g76mofh.html) ·
[BigDATAwire coverage](https://www.hpcwire.com/bigdatawire/this-just-in/elastic-brings-jina-multimodal-and-multilingual-semantic-search-to-on-prem-and-air-gapped-environments/) ·
[DiskBBQ SIMD](https://www.elastic.co/search-labs/blog/vector-search-diskbbq-simd-block-scoring)

*Spot-checked 2026-08-12: the Jina On-Prem and GLM-OCR findings were each confirmed against a
second independent source before being written up, since both reverse a prior position.*

> **Nuance worth keeping.** `semantic_text` is still **text-only**; the multimodal path is the
> separate `semantic` field plus "a compatible multimodal embedding endpoint". Elastic is
> assembling the same components Veda integrates (Jina omni + reranker + a semantic field), but
> as configuration rather than as one opinionated pipeline. That distinction — *integrated
> pipeline vs assembled parts* — is what survives of claims #1/#2, and it is a much narrower
> claim than the one the doc currently makes.

---

## 2. Milvus

| Attribute | April 2026 baseline | Today |
|---|---|---|
| Latest version | v2.6.10 (Mar 2026) | **v3.0.0**, released 2026-07-27 (docs say 2026-07-29) |
| Architecture | cloud-native, Woodpecker object-storage WAL (Jan 2026) | **Lake-native.** "External Collection" queries lakehouse data (Parquet, **Lance**, Iceberg, **Vortex**) **in place: zero-copy, read-only** |
| Storage engine | — | **Storage V3 "Loon"** — manifest-based columnar on S3-compatible object storage; **Vortex** default format (dictionary/RLE/bit-packing/float compression) |
| Sparse | sparse vectors | **SINDI + Block-Max WAND + Block-Max MaxScore**, inverted-list compression + quantization; index ~**3× smaller** than 2.6 at comparable recall |
| Multi-vector | — | **EmbList + DiskANN**: variable-length vector list per entity (chunk sets, late-interaction per-token vectors); **StructArray** for nested doc/chunk + ColBERT vectors |
| Query engine | ANN + filters | server-side **sorting, aggregation, faceted search**; null values, bitmap indexes, dynamic field addition on live collections |
| WAL | embedded | **Woodpecker deployable as an independent service** |
| Governance | LF AI & Data | unchanged |

**Erodes:** the "index breadth / 11 index types" advantage is less distinctive when Milvus adds
EmbList + a redesigned sparse stack; **Veda's OLAP aggregation differentiator** is directly
matched by server-side sorting/aggregation/faceting; and **Lance/Vortex adoption makes Veda's
JSONL/Columnar storage the clearest remaining technical gap**.

Sources: [Milvus release notes](https://milvus.io/docs/release_notes.md) ·
[Milvus 3.0 blog](https://milvus.io/blog/announcing-milvus-3-lake-native-vector-search-and-a-more-powerful-retrieval-engine.md) ·
[LF AI & Data](https://lfaidata.foundation/blog/2026/07/29/milvus-3-0-is-here-lf-ai-datas-open-source-vector-database-goes-lake-native/) ·
[Zilliz newsroom](https://zilliz.com/news/milvus-3-0-lake-native-vector-database)

---

## 3. Qdrant

| Version | Date | Headline |
|---|---|---|
| v1.18.0 | 2026-05-11 | **TurboQuant** — 8× vector compression "without the recall tax"; named-vectors API; deep memory reporting; low-memory mode; strict mode |
| v1.18.1–1.18.3 | May–Jul 2026 | consensus / indexed-range-filter / resharding fixes |
| **v1.19.0** | **2026-08-05** | **TurboQuant 4-bit as a primary-storage datatype** (store only 4-bit codes, drop the originals); **memory tiers** (`cold`/`cached`/`pinned`); prefix matching; **per-query IDF for sparse**; slice filtering + deterministic sampling; global quotas; routing tokens |

Cloud/enterprise in 2026: **GPU-accelerated indexing**, Multi-AZ clusters, audit logging.
2026 roadmap: 4-bit quantization (shipped), read-write segregation, block storage, relevance
feedback, fully scalable multitenancy, read-only replicas.

**Erodes:** the quantization lead. The May audit said "RaBitQ 1-bit still SOTA recall-per-bit"
and Veda shipped it early; Qdrant's TurboQuant 4-bit-as-primary-storage plus Elastic's BBQ
2/4/7-bit means the *multi-bit ladder* is now table stakes — which is exactly the gap the May
audit flagged and Veda has since closed with `RaBitQExtRerank`. **Par, not ahead.**

Sources: [Qdrant releases](https://github.com/qdrant/qdrant/releases) ·
[Qdrant roadmap](https://github.com/qdrant/qdrant/blob/master/docs/roadmap/README.md) ·
[Qdrant Cloud launch](https://siliconangle.com/2026/04/28/qdrant-cloud-launches-high-performance-vector-database-features-ai-workloads/)

---

## 4. Weaviate

| Version | Date | Headline |
|---|---|---|
| v1.37 | Apr 2026 | **First native MCP server in any vector database**; incremental backups; **Diversity Search (MMR)**; query profiling; BlobHash |
| v1.38.x | Jul–Aug 2026 | HFresh disk-based vector index + MCP server to GA; async replication rebuilt on a single scheduler; Boost API + nested object filtering (preview); **TwelveLabs Marengo multimodal vectorizer** (v1.38.9, 2026-08-06) |
| **v1.39.0** | **2026-08-04** | **Namespaces** (control-plane + data isolation on shared clusters); alter-schema reindex/drop-vector-index (preview); gRPC-web; dedicated search REST endpoints; **Hybrid MMR**; **RQ4 — 4-bit rotational quantization**; BM25 **BlockMax WAND** |

Multimodal: text, images and audio in the same vector space; multi-vector visual-document
retrieval via **ModernVBERT / ColModernVBERT**; TwelveLabs Marengo adds video.

**Erodes:** claim **#7** — Weaviate shipped the first native MCP server in a vector DB (Apr 2026)
and took it GA, so "agent-native interop" is no longer distinctive, only its combination with
RLS/tenant enforcement is. Weaviate's multimodal space (text+image+audio, plus video via
Marengo) also narrows claim **#1**.

Sources: [Weaviate releases](https://github.com/weaviate/weaviate/releases) ·
[Weaviate multi-vector docs](https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors) ·
[Apr 2026 MCP coverage](https://ranksquire.com/2026/05/01/vector-database-news-april-2026/)

---

## 5. Chroma

| Attribute | April 2026 baseline | Today |
|---|---|---|
| Latest version | v1.5.5 (2026-03-10) | **v1.5.9** (2026-05-05) — block functions on topology DBs, Tilt fault-injection CLI, sharded-collection rebuilds, group-by with sharding |
| Cloud | — | **Chroma Cloud** launched Q1 2026: serverless, multi-tenant, **runs directly on object storage**, usage-based pricing, $5 free credits |
| Search modalities | vector | **four in one system: vector, full-text, regex, metadata filtering** |
| Perf | — | 500K+ vector collections "noticeably faster" on insert and query YoY |

**Erodes:** little directly — Chroma remains prototype-to-mid-scale positioned. Its **regex
search** modality is a capability Veda does not have (Veda has a suffix-array index, which is
adjacent). The object-storage-native cloud reinforces the architectural trend.

Sources: [Chroma releases](https://github.com/chroma-core/chroma/releases) ·
[Chroma Cloud](https://www.joinnextdev.com/a/chroma/chroma-cloud-one-vector-db-to-rule-them-all)

---

## 6. Pinecone

2026 additions: new data-plane endpoints under `/namespaces/{namespace}/documents/`;
**`$match_phrase` filter operator** composable with any `score_by`; **Pinecone Inference API**
with built-in embedding generation (removes a round trip); flexible deployment with on-demand
**and** dedicated read capacity on serverless; new regions (eu-central-1, ap-southeast-1).
Multimodal embeddings are stored like text embeddings — dimensionality and metric fixed at
index creation, i.e. **no unified auto-routing across modalities**.

**Erodes:** nothing structural. Pinecone's built-in inference narrows the "you must run your own
embedder" critique in `competitive_advantages.md` Part 2, but only for hosted users.

Source: [Pinecone 2026 release notes](https://docs.pinecone.io/release-notes/2026)

---

## 7. pgvector

Still on the **0.8 line** — no 0.9 release found. 0.8.0 brought iterative scan for filtered
queries, parallel/faster HNSW builds, and streaming-insert improvements. The ecosystem story
(pgvectorscale, Neon, Timescale) carries most of the momentum rather than pgvector core.

**Erodes:** nothing. Remains the "good enough, already in your Postgres" option — a different
buyer from Veda's.

Sources: [pgvector CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md) ·
[pgvectorscale releases](https://github.com/timescale/pgvectorscale/releases)

---

## 8. OpenSearch

3.0 introduced cuVS **GPU acceleration** (preview) — 9.3–9.5× faster indexing, ~3.75× lower cost
vs CPU. **3.1** took GPU index builds **GA** and added semantic fields plus **star-tree indexes**
(aggregation acceleration up to 100×). **3.5** (Feb 2026) added **agentic conversation memory**,
hook-based context management for LLM token optimization, and expanded Prometheus support.

**Erodes:** the AIOps/observability adjacency — OpenSearch is building agent memory and
observability into the same engine. Star-tree aggregation acceleration is another point against
Veda's OLAP-aggregation differentiator.

Sources: [OpenSearch GPU blog](https://opensearch.org/blog/gpu-accelerated-vector-search-opensearch-new-frontier/) ·
[3.0 release notes](https://github.com/opensearch-project/opensearch-build/blob/main/release-notes/opensearch-release-notes-3.0.0.md)

---

## 9. FAISS

Library, not a competitor product, but the reference implementation. v1.10 integrated
**NVIDIA cuVS**: `GpuIndexFlat` / `GpuIndexIVFFlat` / `GpuIndexIVFPQ` on cuVS plus a
**`GpuIndexCagra`** graph index. Reported: IVF build up to **4.7×** faster and search latency up
to **8.1×** lower; CAGRA build up to **12.3×** faster than CPU HNSW with **4.7×** lower search
latency. `faiss-gpu-cuvs` **1.14.1** as of March 2026.

**Erodes:** nothing directly, but it confirms **CAGRA-via-cuVS is the industry-standard GPU
path** — the one Veda adopted after the May audit. Veda is now *with* the field here, not ahead.

Sources: [Meta engineering](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/) ·
[FAISS releases](https://github.com/facebookresearch/faiss/releases)

---

## 10. LanceDB

Repositioned as a **"multimodal lakehouse for AI"** — images, video, audio, text, annotations,
embeddings and model features in one schema-enforced table serving curation, feature backfills,
experiment splits, retrieval **and training**. 2026: Lance-native SQL retrieval via **DuckDB**,
Uber-scale multi-bucket storage, 1.5M IOPS benchmarks, **Lance format 2.2** (Blob V2, nested
schema evolution, native Map type, more compression), HF Hub dataset upload, and OpenClaw
positioning Lance as a default **agent memory layer**.

**Erodes:** claim **#1**'s framing. LanceDB markets *multimodal* as its headline, and Milvus 3.0
now reads the Lance format natively — so "multimodal-native" is no longer an unclaimed lane, it
is LanceDB's stated identity. Veda's differentiation has to shift from *storing* multimodal data
to *retrieving across* it (cross-modal query) plus the RAG-quality layer.

Sources: [LanceDB](https://www.lancedb.com/) ·
[Multimodal Lakehouse](https://www.lancedb.com/blog/multimodal-lakehouse) ·
[Lance releases](https://github.com/lance-format/lance/releases)

---

## Scorecard

| Competitor | Version now | Biggest move since April/May | Erodes |
|---|---|---|---|
| Elasticsearch | 9.5 (2026-08-04) | Jina On-Prem omni (air-gapped 4-modality space) + `semantic` field + BBQ multi-bit + cuVS GA | **#1, #2, #5, #7, perf table** |
| Milvus | 3.0.0 (2026-07-27) | Lake-native external collections; Loon/Vortex; sparse overhaul; server-side aggregation | index breadth, **aggregations**, storage format |
| Qdrant | 1.19.0 (2026-08-05) | TurboQuant 4-bit as primary storage; memory tiers | quantization lead |
| Weaviate | 1.39.0 (2026-08-04) | Namespaces; RQ4; Hybrid MMR; native MCP GA; Marengo video | **#7**, narrows **#1** |
| Chroma | 1.5.9 (2026-05-05) | Chroma Cloud on object storage; 4 search modalities incl. regex | — |
| Pinecone | rolling | built-in inference; `$match_phrase`; namespace document endpoints | Part 2 framing |
| pgvector | 0.8.x | no major release | — |
| OpenSearch | 3.5 (Feb 2026) | GPU GA; star-tree aggregations; agentic memory | AIOps adjacency, aggregations |
| FAISS | 1.14.1 pkg (Mar 2026) | cuVS + `GpuIndexCagra` | confirms CAGRA as standard |
| LanceDB | Lance 2.2 | "multimodal lakehouse" identity; DuckDB SQL; agent memory | **#1** framing |
