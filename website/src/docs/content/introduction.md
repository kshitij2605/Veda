# Introduction

**Veda** is a next-generation multimodal vector database for the era of RAG. It stores, indexes, and searches embeddings from any modality — text, images, audio, video, and documents — and owns the entire path from raw files to fused, cross-modal search.

## What makes Veda different

Most vector databases just store vectors: you bring your own embedding services, OCR pipeline, and chunking glue. Veda handles the whole pipeline natively, in a single pure-Rust engine.

- **Multimodal native** — first-class support for text, image, audio, video, and PDF embeddings, with automatic modality-based model routing.
- **Nine index algorithms** — HNSW, DiskANN (Vamana graph), IVF-PQ, Flat, RaBitQ, Sq4 + Sq8 scalar quantization, a suffix array for exact text patterns, and scalar indexes for metadata filtering.
- **SIMD-optimized distances** — SimSIMD kernels (AVX-512, ARM NEON, SVE) on by default, with an AVX2/scalar fallback, across 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard).
- **Multi-vector fusion** — RRF, WeightedSum, and Intersection strategies for cross-modal retrieval in a single API call.
- **Built-in document intelligence** — FlashParse OCR integration plus seven chunking strategies turn raw documents into searchable vectors with zero external tooling.
- **SQL-like filters** — combine vector search with structured metadata predicates.
- **Blob storage** — store the raw files alongside their embeddings via MinIO or any S3-compatible backend.
- **Lightweight by default** — three mandatory services (Veda, MinIO, NATS); `docker compose up` and you have a vector database in under 30 seconds.

## Beyond the basics

Veda also ships a distributed data plane (per-shard append-only logs with quorum replication, MVCC snapshot reads at `?at=<ts>`, Percolator-style snapshot-isolation transactions, atomic reindex with alias cutover, cross-region WAN log shipping), JWT authentication with role-based access control, an OpenTelemetry-first observability stack, the **Niriksha** visualization dashboard, and a Python SDK.

## How these docs are organized

- **Getting started** — [Quick start](/docs/quickstart) gets a cluster running and serving queries in minutes; [Installation & config](/docs/installation) covers building from source, cluster modes, and configuration.
- **Core concepts** — [Collections & fields](/docs/collections), [Indexes](/docs/indexes), [Search & fusion](/docs/search), and the [Document pipeline](/docs/document-pipeline).
- **Reference** — the [REST API](/docs/rest-api), the [Python SDK](/docs/python-sdk), and [Observability](/docs/observability).

## License

Veda is open source under the Apache License 2.0.
