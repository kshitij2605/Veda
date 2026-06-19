# Introduction

**Veda** is a next-generation multimodal vector database for the era of RAG. It stores, indexes, and searches embeddings from any modality — text, images, audio, video, and documents — and owns the entire path from raw files to fused, cross-modal search.

## What makes Veda different

Most vector databases just store vectors: you bring your own embedding services, OCR pipeline, and chunking glue. Veda handles the whole pipeline natively, in a single pure-Rust engine.

- **Multimodal native** — first-class support for text, image, audio, video, and PDF embeddings, with automatic modality-based model routing.
- **Nine index algorithms** — HNSW, DiskANN (Vamana graph), IVF-PQ, Flat, RaBitQ, Sq4 + Sq8 scalar quantization, a suffix array for exact text patterns, and scalar indexes for metadata filtering.
- **SIMD-optimized distances** — SimSIMD kernels (AVX-512, ARM NEON, SVE) on by default, with an AVX2/scalar fallback, across 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard).
- **Multi-vector fusion** — RRF, WeightedSum, and Intersection strategies for cross-modal retrieval in a single API call.
- **Agents & MCP** — a built-in agentic layer and an MCP server, so any MCP client (Claude Desktop, Cursor, VS Code) gets cross-modal retrieval over your data, secured by JWT auth and row-level security — with answer [citations + faithfulness scoring](/docs/rag-aiops#faithfulness-citations) because Veda owns the retriever.
- **Built-in document intelligence** — FlashParse OCR integration plus seven chunking strategies turn raw documents into searchable vectors with zero external tooling.
- **Self-hosted web RAG** — [crawl the open web](/docs/web-connectors) into a collection and let agents `search → fetch → ground` over live pages, on open-source self-hosted services (crawl4ai + searxng) — no external API, no data leaving your trust boundary.
- **SQL-like filters** — combine vector search with structured metadata predicates.
- **Blob storage** — store the raw files alongside their embeddings via MinIO or any S3-compatible backend.
- **Lightweight by default** — three mandatory services (Veda, MinIO, NATS); `docker compose up` and you have a vector database in under 30 seconds.

## Beyond the basics

Veda also ships a distributed data plane (per-shard append-only logs with quorum replication, MVCC snapshot reads at `?at=<ts>`, Percolator-style snapshot-isolation transactions, atomic reindex with alias cutover, cross-region WAN log shipping), JWT authentication with role-based access control and per-record [row-level security](/docs/collections#row-level-security), [JSON-path filtering](/docs/search#json-path-filtering), [record TTL](/docs/collections#record-ttl) with a [cron scheduler](/docs/rest-api#scheduled-jobs), [analytics aggregations](/docs/rest-api#analytics), an OpenTelemetry-first observability stack with [APM & performance observability](/docs/apm-observability) (distributed tracing, profiling, GenAI/RAG traces with cost accounting, an index advisor), a vector-native [AIOps](/docs/aiops) layer (an automated [Ops Advisor](/docs/ops-advisor) over 34 rules + [node resource signals](/docs/resource-signals); semantic log categorization, spike attribution, incident correlation & novelty; change-point detection; capacity forecasting; and a cross-signal RCA bundle with proactive root-cause drafts), [Log Insights](/docs/log-insights), [Service Reliability](/docs/service-reliability) (SLOs, synthetic checks, AI triage), and [RAG & AIOps Intelligence](/docs/rag-aiops) (answer citations + faithfulness scoring, an eval harness, agent memory, streaming), the **Niriksha** visualization dashboard, and a Python SDK.

## How these docs are organized

- **Getting started** — [Quick start](/docs/quickstart) gets a cluster running and serving queries in minutes; [Installation & config](/docs/installation) covers building from source, cluster modes, and configuration.
- **Core concepts** — [Collections & fields](/docs/collections), [Indexes](/docs/indexes), [Search & fusion](/docs/search), the [Document pipeline](/docs/document-pipeline), [Agents & MCP](/docs/agents-mcp), and [Web Connectors](/docs/web-connectors).
- **Reference** — the [REST API](/docs/rest-api), the [Python SDK](/docs/python-sdk), [Observability](/docs/observability), [APM & Performance](/docs/apm-observability), [AIOps](/docs/aiops) (and its deep dives: the [Ops Advisor](/docs/ops-advisor), [Log Insights](/docs/log-insights), [Node Resource Signals](/docs/resource-signals), [Service Reliability](/docs/service-reliability), and [RAG & AIOps Intelligence](/docs/rag-aiops)).

## License

Veda is open source under the Apache License 2.0.
