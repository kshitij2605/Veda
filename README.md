<h1 align="center">Veda — Multimodal Vector Database &amp; RAG Platform (Rust)</h1>

<p align="center">
  <strong>Search and reason over text, images, audio, video, and any document — in one self-hosted vector database.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#why-veda">Why Veda</a> ·
  <a href="#features">Features</a> ·
  <a href="#comparison">Comparison</a> ·
  <a href="#niriksha-dashboard">Niriksha</a> ·
  <a href="https://docs.veda.example">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
  <img alt="Built with Rust" src="https://img.shields.io/badge/built%20with-Rust-orange">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-server-green">
</p>

> **Veda** is an **open-source multimodal vector database and RAG platform** written in **Rust**.
> It stores text, image, audio, and video embeddings in **one unified vector space**, ingests **any
> document** (PDF, DOCX, PPTX, XLSX, scans) through a built-in **OCR → chunk → embed** pipeline, and
> serves **hybrid + cross-modal search**, an **MCP server** for AI agents, and built-in **RAG
> evaluation, guardrails, and observability** — all **self-hosted**, no vendor lock-in.

Veda is a vector database for the multimodal, agentic era of **Retrieval-Augmented Generation (RAG)**.
Most vector databases store only text embeddings and leave document parsing, chunking, embedding, and
RAG evaluation to you. Veda owns the whole pipeline and treats **multimodal as native, not a bolt-on**:
index your corpus once and query it with text, an image, an audio clip, or a video — across a single
unified embedding space.

> _Demo: drop a screenshot/GIF of the Niriksha dashboard here (descriptive alt text, e.g._
> _"Niriksha dashboard — cross-modal search playground over a multimodal collection")._

## Why Veda

- 🎬 **Multimodal &amp; cross-modal search** — text · image · audio · video in **one** 1024-d space.
  Query with any modality (e.g. text → audio/image/video). Most vector DBs are text-only or treat
  non-text as opaque attachments.
- 📄 **Ingest any document** — PDF, DOCX, PPTX, XLSX, and scanned images via a built-in OCR +
  layout pipeline (tables and figures extracted) → chunk → embed → searchable, in one API call.
  No external Tika / Unstructured stack to wire up.
- 🤖 **RAG-native &amp; agent-native** — faithfulness/citation scoring, a RAG eval harness, chunker
  A/B, semantic response cache, embedding-based guardrails, plus an **MCP server**, **A2A**, and a
  `/converse` agent API with full step traces.
- 🏠 **Self-hosted &amp; private** — runs entirely on your infrastructure (embedder, OCR, reranker,
  LLM endpoint are all swappable / BYO-model). Air-gap friendly. Truly open source (**AGPL**, not SSPL).
- ⚡ **Vector-native engine** — pure Rust, SIMD distance kernels, mmap zero-copy reads, **11+ ANN /
  quantization index types** (HNSW, DiskANN, IVF-PQ, RaBitQ, SQ4/SQ8), hybrid dense + sparse (SPLADE)
  + BM25, and vector-native AIOps &amp; LLM observability.

## Quick start

```bash
# Core stack (Veda + MinIO + NATS)
docker compose up -d

# Optional profiles: embedding models, OCR, web connectors, monitoring
docker compose --profile embedding --profile ocr up -d
```

Veda's REST API listens on **`:6340`**. Create a collection, add data, and search:

```bash
# 1. Create a multimodal collection (1024-d cosine vector field, HNSW index)
curl -X POST localhost:6340/v1/collections -H 'content-type: application/json' -d '{
  "name": "kb",
  "shard_count": 1,
  "fields": [
    { "name": "content", "type": "string", "bm25": true },
    { "name": "modality", "type": "string", "filterable": true },
    { "name": "vec", "type": "vector", "dimension": 1024, "metric": "cosine", "index": "hnsw" }
  ]
}'

# 2. Ingest any document (PDF/DOCX/PPTX/XLSX/scan) → OCR → chunk → embed → insert
curl -X POST localhost:6340/v1/collections/kb/process-document -F file=@report.pdf

# 3. (or) ingest an image / audio / video file directly
curl -X POST localhost:6340/v1/collections/kb/ingest-media -F file=@clip.mp4 -F class=demo

# 4. Cross-modal search — a TEXT query retrieves matching text, images, audio, and video
curl -X POST localhost:6340/v1/collections/kb/search -H 'content-type: application/json' -d '{
  "query": { "vector": { "field": "vec", "values": [/* 1024-d query embedding */] } },
  "limit": 10, "explain": true
}'
```

Prefer a UI? Open **[Niriksha](#niriksha-dashboard)** and use the search playground. Python users:
`pip install veda-client` (see the [Python SDK docs](https://docs.veda.example/docs/python-sdk)).

> Veda speaks its own API — it is **not** a drop-in Pinecone/Qdrant clone. See the
> [REST API reference](https://docs.veda.example/docs/rest-api) for exact request shapes.

## Features

### Multimodal &amp; cross-modal retrieval
Unified text · image · audio · video embedding space. Index once, query with any modality. Auto-modality
selection routes each input to the right embedder; image/audio/video search returns hits from the same
space as text — true cross-modal retrieval, not a separate per-modality index.

### Document &amp; OCR ingestion pipeline (any format)
`POST /process-document(s)`: PDF / DOCX / PPTX / XLSX / scanned images → OCR + layout analysis (tables
&amp; figures extracted) → chunk (multiple strategies: markdown-recursive, semantic, sentence, fixed,
page…) → embed → insert. `ingest-media` does the same for raw image/audio/video; `ingest-url` crawls the
web into the same pipeline.

### Hybrid search, indexes &amp; filtering
Dense vectors, sparse **SPLADE** vectors, and **BM25** full-text (Lucene-style analyzer: stemming,
stopwords, char filters) — fused with RRF / weighted-sum. `match_phrase` proximity, MMR, reranking.
**11+ index types** (HNSW, DiskANN, IVF-PQ, RaBitQ family, SQ4/SQ8, Flat) + a SQL-like filter DSL with
nested-JSON paths, row-level security, MVCC snapshot reads (`?at=ts`), and OLAP aggregations.

### Agents, MCP &amp; A2A
A built-in **Model Context Protocol (MCP) server** exposes a 15-tool catalog (cross-modal `search`,
`aggregate`, `route`, …) to any MCP client (Claude, Cursor, VS Code). Plus **A2A** agent cards, a
`/converse` agent API with step traces, and opt-in web-search / fetch tools.

### RAG quality, AIOps &amp; LLM observability
Faithfulness + citation scoring, a RAG eval harness, chunker A/B, coverage/orphan detection, a semantic
response cache, and embedding-based guardrails. Vector-native AIOps (semantic log clustering, incident
correlation, forecasting, LLM-grounded RCA) and LLM observability (TTFT, token/cost, quality drift).

### Distributed &amp; durable
Sharded, FoundationDB-shape data plane (per-shard append-only log, MVCC, Percolator snapshot-isolation
transactions), S3/MinIO blob storage for originals, Prometheus/OTel metrics, and multi-tenancy + RBAC.

## Comparison

| | **Veda** | Pinecone | Qdrant | Milvus | Weaviate |
|---|---|---|---|---|---|
| Open source / self-hostable | ✅ AGPL | ❌ | ✅ | ✅ | ✅ |
| Text + **image + audio + video** in one space | ✅ | text-only | text/image* | text/image* | text/image* |
| Built-in **document/OCR ingestion** (PDF/Office/scan) | ✅ | ❌ | ❌ | ❌ | partial |
| Hybrid dense + sparse + BM25 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Built-in **RAG eval / faithfulness / guardrails** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **MCP server** for agents | ✅ | ✅ | ✅ | partial | ✅ |
| Index types | 11+ | managed | HNSW(+quant) | many | HNSW/Flat |

<sub>*via external/bolt-on image embeddings; no native audio/video retrieval. Comparison reflects
publicly documented capabilities as of 2026; corrections welcome via PR.</sub>

## Niriksha dashboard

**Niriksha** is Veda's RAG-native console: a search playground (dense/sparse/hybrid/cross-modal), a
vector atlas, a chunker A/B lab, a RAG quality lab (eval / faithfulness), document-pipeline UI, and
AIOps insights. (React 19 / TypeScript; source-available under FSL-1.1.)

## Documentation

- [Quick start &amp; installation](https://docs.veda.example/docs/quickstart)
- [Multimodal &amp; cross-modal search](https://docs.veda.example/docs/multimodal)
- [Document pipeline](https://docs.veda.example/docs/document-pipeline) ·
  [Indexes](https://docs.veda.example/docs/indexes) ·
  [Search &amp; fusion](https://docs.veda.example/docs/search)
- [Agents &amp; MCP](https://docs.veda.example/docs/agents-mcp) ·
  [REST API](https://docs.veda.example/docs/rest-api) ·
  [Python SDK](https://docs.veda.example/docs/python-sdk)

## License

Veda is **dual-licensed**. The engine is open source under [**AGPL-3.0**](LICENSE); the
[Niriksha](niriksha/) dashboard is source-available under [**FSL-1.1-ALv2**](niriksha/LICENSE)
(converts to Apache-2.0 after two years). A **commercial license** is available for proprietary
embedding or managed-service use — see [LICENSING.md](LICENSING.md), [PRICING.md](PRICING.md), and
[TRADEMARK.md](TRADEMARK.md). Contributions require the [CLA](CLA.md).

## Contributing

We welcome contributions of every size — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/kshitij2605/Veda/labels/good%20first%20issue) label.
