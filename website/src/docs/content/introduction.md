# Introduction

**Veda** is a next-generation multimodal vector database for the era of RAG. It stores, indexes, and searches embeddings from any modality — text, images, audio, video, and documents — and owns the entire path from raw files to fused, cross-modal search.

## What makes Veda different

Most vector databases just store vectors: you bring your own embedding services, OCR pipeline, and chunking glue. Veda handles the whole pipeline natively, in a single pure-Rust engine.

- **Multimodal native** — first-class support for text, image, audio, video, and PDF, with audio and video [fully wired through the pipeline](/docs/multimodal) and automatic modality-based routing; point Veda at a unified omni embedding model and all modalities share one cross-modal space, so a text query retrieves audio, video, and images directly.
- **11 vector index types** — HNSW, DiskANN (Vamana graph), IVF-PQ, Flat, three RaBitQ variants, Sq4 + Sq8 scalar quantization, and SPANN (posting lists in object storage), so you can [trade recall, latency and memory per collection](/docs/indexes). A suffix array for exact text patterns and scalar indexes for metadata filtering sit alongside them, plus a GPU CAGRA index on cuVS builds.
- **Multi-hop questions** — the agent [searches again when one lookup is not enough](/docs/multi-hop), joining facts across documents with no knowledge graph and no query planner.
- **Guardrails on retrieved documents** — [hostile instructions hiding inside your corpus](/docs/retrieval-guardrails) are screened at ingest and dropped before an LLM sees them, at the tool-dispatch choke point that every agent surface shares.
- **Query Parquet in place** — [register a collection over existing Parquet files](/docs/external-collections) and search them with no ingestion and no second copy.
- **SIMD-optimized distances** — SimSIMD kernels (AVX-512, ARM NEON, SVE) on by default, with an AVX2/scalar fallback, across 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard).
- **Multi-vector fusion** — RRF, WeightedSum, and Intersection strategies for cross-modal retrieval in a single API call.
- **Agents & MCP** — a built-in agentic layer and an MCP server, so any MCP client (Claude Desktop, Cursor, VS Code) gets cross-modal retrieval over your data, secured by JWT auth and row-level security — with answer [citations + faithfulness scoring](/docs/rag-aiops#faithfulness-citations) because Veda owns the retriever.
- **Self-tuning RAG** — Veda [generates a labeled test set from your own corpus and auto-tunes the pipeline](/docs/auto-tuning) (chunker, index, fusion, rerank) for the best recall/faithfulness/latency, then applies the winner reversibly — optionally re-tuning every time the collection grows.
- **Built-in document intelligence** — FlashParse OCR integration plus seven chunking strategies turn raw documents into searchable vectors with zero external tooling.
- **Self-hosted web RAG** — [crawl the open web](/docs/web-connectors) into a collection and let agents `search → fetch → ground` over live pages, on open-source self-hosted services (crawl4ai + searxng) — no external API, no data leaving your trust boundary.
- **SQL-like filters** — combine vector search with structured metadata predicates.
- **Blob storage** — store the raw files alongside their embeddings via MinIO or any S3-compatible backend.
- **Lightweight by default** — three mandatory services (Veda, MinIO, NATS); `docker compose up` and you have a vector database in under 30 seconds.

## Beyond the basics

Veda also ships a distributed data plane (per-shard append-only logs with quorum replication, MVCC snapshot reads at `?at=<ts>`, Percolator-style snapshot-isolation transactions, atomic reindex with alias cutover, cross-region WAN log shipping), JWT authentication with role-based access control and per-record [row-level security](/docs/collections#row-level-security), [JSON-path filtering](/docs/search#json-path-filtering), [record TTL](/docs/collections#record-ttl) with a [cron scheduler](/docs/rest-api#scheduled-jobs), [analytics aggregations](/docs/rest-api#analytics), an OpenTelemetry-first observability stack with [APM & performance observability](/docs/apm-observability) (distributed tracing, profiling, GenAI/RAG traces with cost accounting, an index advisor), a vector-native [AIOps](/docs/aiops) layer (an automated [Ops Advisor](/docs/ops-advisor) over 36 rules + [node resource signals](/docs/resource-signals); semantic log categorization, spike attribution, incident correlation & novelty; change-point detection; capacity forecasting; and a cross-signal RCA bundle with proactive root-cause drafts), [Log Insights](/docs/log-insights), [Service Reliability](/docs/service-reliability) (SLOs, synthetic checks, AI triage), [RAG & AIOps Intelligence](/docs/rag-aiops) (answer citations + faithfulness scoring, an eval harness, agent memory, streaming), and [LLM Observability & Optimization](/docs/llm-observability) (a semantic response cache, embedding-based guardrails, and a searchable trace store), the **Niriksha** visualization dashboard, and a Python SDK.

## How these docs are organized

- **Getting started** — [Quick start](/docs/quickstart) gets a cluster running and serving queries in minutes; [Installation & config](/docs/installation) covers building from source, cluster modes, and configuration.
- **Core concepts** — [Collections & fields](/docs/collections), [External collections](/docs/external-collections), [Indexes](/docs/indexes), [Search & fusion](/docs/search), [Multimodal & cross-modal](/docs/multimodal), the [Document pipeline](/docs/document-pipeline), [Agents & MCP](/docs/agents-mcp), [Multi-hop retrieval](/docs/multi-hop), and [Web Connectors](/docs/web-connectors).
- **Reference** — the [REST API](/docs/rest-api), the [Python SDK](/docs/python-sdk), [Observability](/docs/observability), [APM & Performance](/docs/apm-observability), [LLM Observability](/docs/llm-observability), [Retrieval guardrails](/docs/retrieval-guardrails), [AIOps](/docs/aiops) (and its deep dives: the [Ops Advisor](/docs/ops-advisor), [Log Insights](/docs/log-insights), [Node Resource Signals](/docs/resource-signals), [Service Reliability](/docs/service-reliability), and [RAG & AIOps Intelligence](/docs/rag-aiops)), and [RAG Auto-Tuning](/docs/auto-tuning).

## License

Veda is open source under the Apache License 2.0.
