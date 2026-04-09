# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-07

### Added
- Core vector database engine with 7 Rust crates
- HNSW index with adaptive filtered search (95.5% recall@10)
- DiskANN (Vamana graph) index (99.9% recall@10)
- IVF-PQ index with residual product quantization
- RaBitQ and scalar quantization
- SIMD-optimized distance functions (AVX2 + scalar fallback)
- 6 distance metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard
- mmap-backed vector storage format
- Scalar indexes: inverted, range, bitmap
- Growing + sealed segment architecture with compaction
- SQL-like filter expression parser and evaluator
- Multi-vector fusion search (RRF, WeightedSum, Intersection)
- Distributed query with shard fan-out and merge
- Suffix array pattern search with fuzzy matching
- REST API gateway (axum) with full CRUD
- Auto-embedding service with modality validation
- Blob storage with MinIO/S3-compatible and local filesystem backends
- JWT authentication and RBAC (Admin/Editor/Viewer)
- Full observability stack: Prometheus metrics (16 metric types), OpenTelemetry tracing and log export, health/readiness endpoints, query explain with per-phase timing
- Pre-built Grafana dashboard with provisioned datasources (Mimir, Tempo, Loki)
- Docker Compose `monitoring` profile (OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana)
- W3C Trace Context propagation for incoming traceparent headers
- HTTP request metrics middleware with automatic error counting
- `#[instrument]` tracing spans on handlers, embedding client, blob store, segment manager, and search executor
- Cluster coordination with shard routing
- WAL abstraction with in-memory backend
- Niriksha dashboard with 7 pages (React + TypeScript)
- Python SDK (veda-client)
- 7 configurable chunking strategies: MarkdownRecursive (default), MarkdownHeading (levels 1/2/3), Semantic, Sentence, FixedSize, Page, None
- Semantic chunking strategy with word-overlap Jaccard similarity for topic-based splitting
- Markdown heading-level chunking (split on H1, H2, or H3 depth)
- FlashParse OCR integration with Docker Compose profiles (`docker compose --profile ocr up`)
- End-to-end document processing pipeline (parse, chunk, embed, insert) via `/v1/collections/{name}/process-document` and `/v1/collections/{name}/process-documents`
- `.env.example` configuration template for VLM and service settings
- Docker and Docker Compose deployment
- ANN benchmark suite
- Comprehensive integration and stress tests
