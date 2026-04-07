# Architectural Decision Records

This document captures the key architectural decisions made during the design and implementation of Veda.

---

## ADR-001: Rust as Implementation Language

**Status:** Accepted

### Context
A vector database has strict performance requirements: low-latency similarity search, efficient memory management for large vector sets, and safe concurrency for multi-threaded indexing and queries. We evaluated Rust, C++, Go, and Java.

### Decision
Use Rust as the primary implementation language for the core engine.

### Consequences
- **Positive:** Memory safety without garbage collection pauses, zero-cost abstractions, excellent SIMD intrinsics support, strong type system catches bugs at compile time, and Cargo provides a mature build and dependency system.
- **Negative:** Steeper learning curve for contributors, longer compile times compared to Go/Java, smaller ecosystem for some database-specific libraries compared to C++.

---

## ADR-002: Full Disaggregation Architecture

**Status:** Accepted

### Context
We needed to decide between a monolithic architecture and a disaggregated one where gateway, query, index, storage, and coordination are separate crates with clean interfaces.

### Decision
Adopt a fully disaggregated architecture with 7 crates: `veda-core`, `veda-index`, `veda-storage`, `veda-query`, `veda-gateway`, `veda-coordinator`, and `veda-server`.

### Consequences
- **Positive:** Each component can be developed and tested independently, clear ownership boundaries, easier to scale individual layers, enables future deployment as separate services.
- **Negative:** More complex dependency graph, cross-crate refactoring is more involved, potential overhead from trait-based abstractions at crate boundaries.

---

## ADR-003: HNSW as Primary Index

**Status:** Accepted

### Context
Multiple ANN index algorithms exist (HNSW, NSG, IVF variants, tree-based). We needed a default index type that offers the best balance of recall, speed, and simplicity for the common case.

### Decision
Use HNSW (Hierarchical Navigable Small World) as the primary and default index type, with DiskANN, IVF-PQ, and Flat as alternatives.

### Consequences
- **Positive:** HNSW is well-studied with predictable performance characteristics, achieves 95.5% recall@10 on our benchmarks, supports incremental insertions without full rebuilds, and integrates naturally with filtered search.
- **Negative:** Higher memory overhead compared to IVF-PQ due to graph structure, not ideal for extremely large datasets that cannot fit in memory (DiskANN fills this gap).

---

## ADR-004: RaBitQ for Quantization

**Status:** Accepted

### Context
Vector quantization is essential for reducing memory consumption. Common options include Product Quantization (PQ), Scalar Quantization (SQ), and the newer RaBitQ (Random Bit Quantization) approach.

### Decision
Implement RaBitQ as the primary quantization method alongside SQ8 (8-bit scalar quantization) as a simpler alternative.

### Consequences
- **Positive:** RaBitQ achieves strong compression ratios with minimal recall loss, works well with binary distance operations for fast candidate filtering, and pairs well with HNSW re-ranking.
- **Negative:** RaBitQ is less widely adopted than PQ, so fewer reference implementations and community resources. We mitigate this by also offering SQ8 for users who want a more conventional approach.

---

## ADR-005: Suffix Array Pattern Search

**Status:** Accepted

### Context
Users need to search text fields by substring patterns in addition to vector similarity. Traditional approaches include inverted indexes with n-grams, trigram indexes, or suffix-based structures.

### Decision
Use suffix arrays for pattern search, with support for fuzzy matching via edit distance.

### Consequences
- **Positive:** Suffix arrays provide O(m log n) substring search, lower memory overhead than suffix trees, and integrate cleanly with the existing filter pipeline. Fuzzy matching extends utility beyond exact substring search.
- **Negative:** Suffix array construction requires O(n) time and the array itself consumes O(n) space. Updates require rebuilding the array for the affected segment, which is acceptable under the growing/sealed segment model.

---

## ADR-006: No CLI Client (REST + SDK + Niriksha)

**Status:** Accepted

### Context
We considered whether to build a dedicated CLI client for interacting with Veda. Alternatives are the REST API directly (via curl or similar), a language SDK, or the Niriksha dashboard.

### Decision
Do not build a dedicated CLI client. Provide REST API, Python SDK, and the Niriksha dashboard as the three access methods.

### Consequences
- **Positive:** Less code to maintain, the REST API is universally accessible via curl/httpie, the Python SDK covers programmatic use cases, and Niriksha provides a visual interface for exploration and administration.
- **Negative:** Power users who prefer CLI workflows must use curl or write shell wrappers. We may revisit this decision if user demand is strong.

---

## ADR-007: MinIO as Default Blob Storage

**Status:** Accepted

### Context
Veda supports storing raw files (images, PDFs, audio, video) alongside their embeddings. We needed a blob storage backend that is self-contained for development and compatible with cloud deployments.

### Decision
Use MinIO as the default blob storage backend, accessed via the S3-compatible API. Also support a local filesystem backend for development.

### Consequences
- **Positive:** MinIO is S3-compatible, so the same code works with AWS S3, GCS (via compatibility layer), and other S3-compatible stores. MinIO is easy to run locally via Docker. The local filesystem backend keeps development simple.
- **Negative:** Adds MinIO as a runtime dependency for production deployments with blob storage. Users who do not need blob storage can disable it entirely.

---

## ADR-008: Channel-Based Coordination (Raft-Ready)

**Status:** Accepted

### Context
Cluster coordination requires consensus on shard assignments, leader election, and configuration changes. Full Raft consensus is complex to implement correctly but necessary for production-grade multi-node deployments.

### Decision
Implement coordination using Rust async channels for intra-process communication, with the architecture designed so that Raft can be plugged in later for multi-node consensus.

### Consequences
- **Positive:** Channel-based coordination is simple, correct for single-node and embedded multi-shard scenarios, and easy to test. The coordinator trait boundary makes future Raft integration a localized change.
- **Negative:** Multi-node deployments are not supported until Raft (or equivalent) is implemented. This is acceptable for the initial release.

---

## ADR-009: Axum for REST API

**Status:** Accepted

### Context
We evaluated Rust web frameworks for the REST API layer: axum, actix-web, warp, and rocket.

### Decision
Use axum as the HTTP framework for the gateway crate.

### Consequences
- **Positive:** Axum is built on tokio and tower, which we already use throughout the stack. Its extractor-based design produces clean handler signatures. Strong community adoption and active maintenance. Tower middleware (compression, rate limiting, tracing) works out of the box.
- **Negative:** Less "batteries included" than actix-web for some features (e.g., WebSocket support requires additional setup). This is a minor tradeoff given our REST-first API design.

---

## ADR-010: Pure Python SDK (Not PyO3)

**Status:** Accepted

### Context
The Python SDK could be implemented as a pure Python HTTP client or as a PyO3/Rust-backed native extension that embeds parts of the engine.

### Decision
Implement the Python SDK as a pure Python package (`veda-client`) that communicates with Veda over REST.

### Consequences
- **Positive:** Simple to install (`pip install veda-client`), no compilation step or Rust toolchain required for users, easy to distribute across all platforms, straightforward to maintain and debug.
- **Negative:** All operations go through HTTP, so there is no option for in-process embedding or direct index access. Latency is bounded by network round-trips. This is the correct tradeoff for a client SDK -- users who need in-process access can use the Rust crates directly.

---

## ADR-011: MarkdownRecursive as Default Chunking Strategy

**Status:** Accepted

### Context
The document processing pipeline needs to chunk OCR output into pieces suitable for embedding and vector search. Many chunking strategies exist, ranging from simple fixed-size splitting to model-dependent semantic chunking. The choice of default strategy affects retrieval quality, processing cost, and operational complexity.

### Decision
Use MarkdownRecursive as the default chunking strategy with 512 tokens (~2000 characters) chunk size and 15% overlap (~300 characters). Research shows that recursive markdown splitting outperforms semantic chunking in end-to-end RAG benchmarks while being approximately 10x cheaper, since it requires no embedding model calls during the chunking phase.

### Consequences
- **Positive:** Simple, fast, and deterministic. No model dependency during chunking. Respects document structure (headings, paragraphs) for natural break points. Works well with markdown output from FlashParse OCR.
- **Negative:** May miss some topic boundaries that semantic chunking catches, particularly in unstructured text without clear markdown formatting. Users can switch to semantic chunking via the `chunking_strategy` parameter when needed.
